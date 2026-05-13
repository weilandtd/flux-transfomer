"""
HSIC-based global sensitivity analysis for metabolic flux analysis.

Implements De Veiga (2015) Sobol-like sensitivity indices using the unbiased
HSIC U-statistic of Gretton et al. (2012).

References:
    De Veiga, S. (2015). Global sensitivity analysis with dependence measures.
    J. Stat. Comput. Simul. 85(7):1283-1305.

    Gretton, A. et al. (2012). A Kernel Two-Sample Test. JMLR 13:723-773.
"""
from __future__ import annotations

import concurrent.futures
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist


def rbf_kernel(x: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """RBF kernel matrix with median bandwidth heuristic.

    Args:
        x: (n,) or (n, d) input array.
        bandwidth: σ² for exp(-‖·‖²/2σ²). If None, uses median heuristic:
            σ² = median(lower-triangle pairwise squared distances) / 2.
            Constant inputs (bandwidth < 1e-30) return the all-ones matrix,
            which yields HSIC = 0 under the U-statistic — correct behaviour.

    Returns:
        K: (n, n) symmetric PSD kernel matrix with ones on the diagonal.

    Raises:
        ValueError: if n < 4 (U-statistic denominator n(n-3) is non-positive).
    """
    x = np.atleast_2d(x) if x.ndim > 1 else x.reshape(-1, 1)
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"rbf_kernel requires n >= 4, got n={n}")

    if bandwidth is None:
        sq_dists = pdist(x, metric="sqeuclidean")
        bandwidth = float(np.median(sq_dists) / 2.0)

    if bandwidth < 1e-30:
        return np.ones((n, n))

    return np.exp(-cdist(x, x, metric="sqeuclidean") / (2.0 * bandwidth))


def hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC U-statistic (Gretton et al. 2012, Eq. 5).

    Args:
        K: (n, n) kernel matrix for variable X.
        L: (n, n) kernel matrix for variable Y.

    Returns:
        Scalar HSIC estimate (can be negative for small n or near-independent vars).

    Raises:
        ValueError: if n < 4.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"hsic_unbiased requires n >= 4, got n={n}")

    # Zero the diagonals: K̃_ij = K_ij * (1 - δ_ij)
    Kt = K - np.diag(np.diag(K))
    Lt = L - np.diag(np.diag(L))

    # term1 = tr(K̃L̃) = Σ_ij K̃_ij L̃_ij  (Frobenius inner product, O(n²) not O(n³))
    term1 = float((Kt * Lt).sum())

    # term2 = (Σ_ij K̃_ij)(Σ_ij L̃_ij) / ((n-1)(n-2))
    term2 = float(Kt.sum() * Lt.sum() / ((n - 1) * (n - 2)))

    # term3 = -(2/(n-2)) * Σ_ij (K̃L̃)_ij
    # Identity: Σ_ij (A·B)_ij = A.sum(axis=0) · B.sum(axis=0)  when A,B symmetric
    term3 = float(-(2.0 / (n - 2)) * (Kt.sum(axis=0) @ Lt.sum(axis=0)))

    return (term1 + term2 + term3) / (n * (n - 3))


# ── Private helpers ────────────────────────────────────────────────────────────

def _compute_bandwidths(X: np.ndarray, max_pairs: int = 50_000) -> np.ndarray:
    """Median-heuristic bandwidths for every column of X in one vectorised pass.

    Instead of p separate pdist calls, samples at most max_pairs lower-triangle
    pairs and computes all p medians simultaneously.

    Returns:
        bw: (p,) bandwidth array — zero entries signal constant columns.
    """
    n, p = X.shape
    ii, jj = np.tril_indices(n, k=-1)
    n_pairs = ii.shape[0]
    if n_pairs > max_pairs:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_pairs, size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    sq = (X[ii, :] - X[jj, :]) ** 2     # (pairs, p)
    return np.median(sq, axis=0) / 2.0   # (p,)


def _auto_chunk(n: int, mem_bytes: int = 1 << 30) -> int:
    """Largest B such that a single (n, n, B) float32 buffer fits in mem_bytes."""
    return max(1, mem_bytes // (n * n * 4))


def _hsic_chunk(
    X_chunk: np.ndarray,
    bw_chunk: np.ndarray,
    Lt: np.ndarray,
    Lt_row_sums: np.ndarray,
    Lt_total: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised HSIC(X_i, Y) and HSIC(X_i, X_i) for a batch of scalar inputs.

    Builds one (n, n, B) float32 kernel tensor in-place to minimise peak memory,
    then computes all HSIC quantities with a single pass over that tensor.

    Args:
        X_chunk:     (n, B) float32 — B scalar inputs.
        bw_chunk:    (B,) float64 — per-input bandwidths; zero = constant input.
        Lt:          (n, n) float32 — Y kernel with diagonal zeroed.
        Lt_row_sums: (n,) float32 — off-diagonal row sums of Lt.
        Lt_total:    scalar — sum of Lt.
        n:           number of samples.

    Returns:
        hsic_xiY:  (B,) float64
        hsic_xixi: (B,) float64
    """
    constant = bw_chunk < 1e-30

    # Build K in-place: diff → sq_dist → -sq/(2σ²) → exp  (one n²B buffer)
    K = X_chunk[:, None, :] - X_chunk[None, :, :]    # (n, n, B)
    K **= 2
    safe_bw = np.where(constant, 1.0, bw_chunk).astype(np.float32)
    K /= 2.0 * safe_bw[None, None, :]
    np.negative(K, out=K)
    np.exp(K, out=K)
    # Constant inputs: all sq_dists == 0 → K == 1 everywhere → HSIC = 0 (zeroed below)

    denom = float(n * (n - 3))
    c1 = 1.0 / float((n - 1) * (n - 2))
    c2 = 2.0 / float(n - 2)

    # Off-diagonal row sums: K.sum(axis=1)[j,b] - 1  (subtract diagonal which is 1)
    K_rs = K.sum(axis=1) - 1.0                               # (n, B)
    K_total = K.sum(axis=(0, 1)) - n                         # (B,) — exclude diagonal

    # HSIC(X_i, Y) ─────────────────────────────────────────────────────────────
    # T1 = Σ_{j≠k} K[j,k,b]*L[j,k]  — Lt diagonal is 0 so no correction needed
    T1_xY = (K * Lt[:, :, None]).sum(axis=(0, 1))            # (B,)
    T3_xY = (K_rs * Lt_row_sums[:, None]).sum(axis=0)        # (B,)
    hsic_xiY = (T1_xY + K_total * Lt_total * c1 - c2 * T3_xY) / denom

    # HSIC(X_i, X_i) ───────────────────────────────────────────────────────────
    # T1_xx = Σ_{j≠k} K[j,k]²  =  (K²).sum() - n  (diagonal is 1² = 1)
    T1_xx = (K * K).sum(axis=(0, 1)) - n                     # (B,)
    T3_xx = (K_rs * K_rs).sum(axis=0)                        # (B,)
    hsic_xixi = (T1_xx + K_total ** 2 * c1 - c2 * T3_xx) / denom

    # Constant inputs contribute zero dependence
    hsic_xiY[constant] = 0.0
    hsic_xixi[constant] = 0.0

    return hsic_xiY.astype(np.float64), hsic_xixi.astype(np.float64)


# ── Public API ─────────────────────────────────────────────────────────────────

def hsic_indices(
    X: np.ndarray,
    Y: np.ndarray,
    input_names: list[str] | None = None,
    bandwidth_X: float | list[float] | None = None,
    bandwidth_Y: float | None = None,
    chunk_size: int | None = None,
    max_samples: int | None = None,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """De Veiga (2015) HSIC sensitivity indices for each column of X vs Y.

    For each scalar input X_i the following are computed:
        hsic  — HSIC_u(X_i, Y): raw dependence measure.
        eta   — HSIC_u(X_i, Y) / HSIC_u(Y, Y): Sobol-like index (can exceed 1).
        R     — HSIC_u(X_i, Y) / sqrt(HSIC_u(X_i,X_i) * HSIC_u(Y,Y)):
                kernel distance correlation ∈ [−1, 1].

    A single multivariate RBF kernel is used for Y (the full labeling vector),
    capturing joint dependence on all outputs simultaneously.

    Performance notes:
        - All kernel matrices use float32 (half the memory of float64).
        - Bandwidths for all p inputs are estimated in one vectorised pass.
        - X kernels are built in chunks of ``chunk_size`` columns simultaneously
          using a single (n, n, B) float32 buffer; auto-sized to ~1 GB by default.

    Args:
        X: (n_samples, p) input matrix — each column is one scalar input.
        Y: (n_samples, q) output matrix — treated as one multivariate output.
        input_names: length-p list of names for the DataFrame index.
            Defaults to ['X_0', 'X_1', ...].
        bandwidth_X: σ² for the X kernels. Scalar applies to all columns;
            list of length p applies per column; None uses median heuristic.
        bandwidth_Y: σ² for the Y kernel. None uses median heuristic.
        chunk_size: number of X columns processed simultaneously.
            None auto-sizes to fit ~1 GB of float32 kernel matrices.
        max_samples: if given and n > max_samples, randomly subsample this
            many rows before any kernel computation. Reduces O(n²) cost to
            O(max_samples²). Subsampling is reproducible via ``seed``.
        seed: integer seed for np.random.default_rng used by max_samples
            subsampling. Default 0. Pass None for non-deterministic.
        n_jobs: parallel workers for the column-chunk loop. 1 (default) runs
            sequentially. -1 uses all CPU cores (os.cpu_count()). Values ≥ 2
            set the ThreadPoolExecutor max_workers directly.

    Returns:
        DataFrame with columns ['hsic', 'eta', 'R'], indexed by input_names,
        sorted by eta descending. R is np.nan when the kernel is degenerate
        (constant input or near-zero HSIC(Y,Y)).
    """
    n, p = X.shape
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same n_samples; got {X.shape[0]} vs {Y.shape[0]}")
    if n < 4:
        raise ValueError(f"hsic_indices requires n >= 4, got n={n}")
    if input_names is None:
        input_names = [f"X_{i}" for i in range(p)]
    if len(input_names) != p:
        raise ValueError(f"len(input_names)={len(input_names)} != X.shape[1]={p}")

    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs}")

    # ── Optional row subsampling ──────────────────────────────────────────────
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        n = max_samples

    B = chunk_size if chunk_size is not None else _auto_chunk(n)

    # ── Bandwidths for all X columns in one pass ───────────────────────────────
    if bandwidth_X is None:
        bw_X = _compute_bandwidths(X.astype(np.float64))
    elif np.isscalar(bandwidth_X):
        bw_X = np.full(p, float(bandwidth_X))
    else:
        bw_X = np.asarray(bandwidth_X, dtype=np.float64)
        if bw_X.shape[0] != p:
            raise ValueError(f"len(bandwidth_X)={bw_X.shape[0]} != X.shape[1]={p}")

    # ── Y kernel — built once, stored as float32 ───────────────────────────────
    L = rbf_kernel(Y, bandwidth_Y).astype(np.float32)
    Lt = (L - np.diag(np.diag(L)))
    Lt_row_sums = Lt.sum(axis=1)    # (n,) float32
    Lt_total = float(Lt.sum())
    hsic_YY = hsic_unbiased(L.astype(np.float64), L.astype(np.float64))

    # ── Chunked loop over X columns ────────────────────────────────────────────
    all_hsic_xiY  = np.empty(p, dtype=np.float64)
    all_hsic_xixi = np.empty(p, dtype=np.float64)

    chunks = [(start, min(start + B, p)) for start in range(0, p, B)]

    if n_jobs == 1:
        for start, end in chunks:
            hiY, hixi = _hsic_chunk(
                X[:, start:end].astype(np.float32),
                bw_X[start:end],
                Lt, Lt_row_sums, Lt_total, n,
            )
            all_hsic_xiY[start:end]  = hiY
            all_hsic_xixi[start:end] = hixi
    else:
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _hsic_chunk,
                    X[:, start:end].astype(np.float32),
                    bw_X[start:end],
                    Lt, Lt_row_sums, Lt_total, n,
                )
                for start, end in chunks
            ]
            for (start, end), fut in zip(chunks, futures):
                hiY, hixi = fut.result()
                all_hsic_xiY[start:end]  = hiY
                all_hsic_xixi[start:end] = hixi

    # ── Assemble results ───────────────────────────────────────────────────────
    eta = np.where(hsic_YY != 0.0, all_hsic_xiY / hsic_YY, np.nan)
    denom_R = all_hsic_xixi * hsic_YY
    R = np.full(p, np.nan)
    valid = denom_R > 0.0
    R[valid] = all_hsic_xiY[valid] / np.sqrt(denom_R[valid])

    return (
        pd.DataFrame({"hsic": all_hsic_xiY, "eta": eta, "R": R}, index=input_names)
        .sort_values("eta", ascending=False)
    )
