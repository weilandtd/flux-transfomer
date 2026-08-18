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
    """RBF kernel matrix exp(-‖·‖²/2σ²); bandwidth defaults to the median pairwise-squared-distance heuristic.

    A constant input (bandwidth < 1e-30) returns the all-ones matrix, correctly
    yielding HSIC = 0. Raises if n < 4 (the U-statistic denominator needs it).
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
    """Unbiased HSIC U-statistic between kernel matrices K and L (Gretton et al. 2012, Eq. 5); can be negative for small n or near-independent variables. Raises if n < 4."""
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
    """Median-heuristic bandwidths for every column of X in one vectorised pass, sampling at most max_pairs lower-triangle pairs instead of p separate pdist calls. Zero entries signal constant columns."""
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
    """Vectorised HSIC(X_i, Y) and HSIC(X_i, X_i) for a batch of B scalar inputs, building one (n, n, B) float32 kernel tensor in-place to minimise peak memory."""
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
    """De Veiga (2015) HSIC sensitivity indices for each column of X against the (possibly multivariate) output Y, via one joint RBF kernel over Y.

    Returns a DataFrame with columns 'hsic' (raw dependence), 'eta' (Sobol-like
    index, can exceed 1), and 'R' (kernel distance correlation in [-1, 1], NaN when
    degenerate), indexed by input_names and sorted by eta descending. max_samples
    subsamples rows for O(n^2) cost control; chunk_size/n_jobs control the
    memory/parallelism of the per-column kernel computation.
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
