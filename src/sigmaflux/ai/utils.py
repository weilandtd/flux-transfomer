import warnings

import numpy as np
import pandas as pd
import scipy.linalg
import torch


def compute_token_scores(L_c, method='pca', n_components=50, y_c=None, y_std_c=None):
    """Score labeling tokens by their informativeness for training-time selection.

    Args:
        L_c:          (n_train, n_tokens) centered labeling matrix
        method:       'pca' | 'maxr'
        n_components: PCA components to retain (method='pca' only)
        y_c:          (n_train, n_free) centered basis targets (method='maxr' only)
        y_std_c:      (n_free,) std of basis targets (method='maxr' only)

    Returns:
        scores: (n_tokens,) float tensor — higher means more informative
    """
    if method == 'pca':
        _U, _S, _Vh = torch.linalg.svd(L_c, full_matrices=False)
        k = min(n_components, _S.shape[0])
        scores = (_Vh[:k] ** 2 * _S[:k, None] ** 2).sum(dim=0)
        var_k = (_S[:k] ** 2).sum().item()
        var_total = (_S ** 2).sum().item()
        print(f"Token scores (PCA): top-{k} PCs explain {100 * var_k / var_total:.1f}% "
              f"of labeling variance")
    elif method == 'maxr':
        if y_c is None or y_std_c is None:
            raise ValueError("y_c and y_std_c are required for method='maxr'")
        L_std_c = L_c.std(dim=0).clamp_min(1e-10)
        cov     = (y_c.T @ L_c) / y_c.shape[0]
        corr    = cov / y_std_c.unsqueeze(1) / L_std_c.unsqueeze(0)
        scores  = corr.abs().max(dim=0).values
        print(f"Token scores (max|r|): min={scores.min():.3f}, max={scores.max():.3f}, "
              f"median={scores.median():.3f}")
    else:
        raise ValueError(f"Unknown token selection method: {method!r}. Choose 'pca' or 'maxr'.")
    return scores


def load_fisher_token_scores(
    features_csv: str, labeling_df
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-computed Fisher scores and inclusion mask aligned to labeling_df row order.

    Reads features.csv produced by extract_informative_features.py and returns:
      - (n_tokens,) tensor of marginal_nats scores
      - (n_tokens,) bool tensor marking the tokens included for training

    Tokens absent from the CSV receive score 0.0 and included=False.

    Args:
        features_csv: path to features.csv
        labeling_df:  DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer)

    Returns:
        scores:        (n_tokens,) float tensor of marginal_nats scores
        included_mask: (n_tokens,) bool tensor — True for tokens passed to training
    """
    feat = pd.read_csv(features_csv)
    score_map: dict = {}
    included_set: set = set()
    for _, row in feat.iterrows():
        key = (row["metabolite_id"], row["isotopomer"])
        v = row["marginal_nats"]
        score_map[key] = 0.0 if v != v else float(v)
        if row.get("included", False):
            included_set.add(key)

    mets = labeling_df.index.get_level_values('Metabolite')
    isos = labeling_df.index.get_level_values('Isotopomer')
    scores = torch.tensor(
        [score_map.get((m, i), 0.0) for m, i in zip(mets, isos)],
        dtype=torch.float32,
    )
    included_mask = torch.tensor(
        [(m, i) in included_set for m, i in zip(mets, isos)],
        dtype=torch.bool,
    )
    n_matched = sum(1 for m, i in zip(mets, isos) if (m, i) in score_map)
    n_inc = int(included_mask.sum().item())
    print(f"Fisher scores loaded from {features_csv}: "
          f"{n_matched}/{len(scores)} tokens matched, "
          f"{n_inc} included")
    return scores, included_mask


def compute_sparse_nullspace(
    S: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute a sparse null-space basis of stoichiometric matrix S via
    SVD followed by Varimax (Kaiser 1958) orthogonal rotation.

    Background
    ----------
    The null space of S contains all steady-state flux distributions v
    satisfying S @ v = 0. scipy.linalg.null_space returns an orthonormal
    basis Z via SVD, but each basis vector typically has non-zero weight
    on most reactions, making the basis hard to interpret biologically.

    Varimax rotation finds an orthogonal matrix R such that the columns
    of Z @ R span the same null space but are *sparser*: each rotated
    basis vector concentrates its weight on a smaller subset of reactions.
    Sparsity is achieved by maximising the sum of per-column variances of
    squared loadings — the Kaiser criterion:

        obj = Σ_j [ n * Σ_i z_{ij}^4  −  (Σ_i z_{ij}^2)^2 ]

    where the sum is over basis vectors j and reactions i, and n is the
    number of reactions.  This objective is maximised iteratively via
    pairwise Jacobi sweeps: at each step the optimal 2D rotation angle
    for a column pair (p, q) is solved in closed form.

    Because every Jacobi update is an orthogonal 2×2 rotation, column
    norms are preserved exactly at every step and the overall rotation
    R = Π R_{pq} remains orthogonal (R^T R = I).  The result is an
    orthonormal basis — no re-orthogonalisation is needed post-rotation.

    Convergence
    -----------
    A full sweep rotates every distinct column pair (p, q) once.
    Iteration stops when the absolute change in the Varimax objective
    between successive sweeps falls below `tol`, or after `max_iter`
    sweeps.

    Parameters
    ----------
    S : np.ndarray, shape (m, n)
        Stoichiometric matrix with m metabolites and n reactions.
    max_iter : int
        Maximum number of Jacobi sweeps. Each sweep performs k*(k-1)/2
        pairwise rotations, where k = dim(null space of S).
    tol : float
        Convergence threshold on the absolute change in the Varimax
        objective between consecutive sweeps.

    Returns
    -------
    Z_rotated : np.ndarray, shape (n, k)
        Orthonormal null-space basis after Varimax rotation. Columns
        span null(S) and are sparser than the raw SVD basis.
    """
    # --- initial orthonormal null-space basis from SVD ---
    Z = scipy.linalg.null_space(S)   # (n_rxn, k)
    n_rxn, k = Z.shape

    # rotation is a no-op for a 1-D (or empty) null space
    if k <= 1:
        return Z

    prev_obj = -np.inf

    for _ in range(max_iter):
        # --- one full Jacobi sweep over all column pairs ---
        for p in range(k - 1):
            for q in range(p + 1, k):
                # quantities needed for the closed-form Kaiser angle
                u = Z[:, p] ** 2 - Z[:, q] ** 2   # element-wise difference of squared loadings
                v = 2.0 * Z[:, p] * Z[:, q]        # element-wise cross term

                A = u.sum()
                B = v.sum()
                C = (u**2 - v**2).sum()
                D = (2.0 * u * v).sum()

                # Kaiser (1958) optimal rotation angle for this column pair;
                # the /n_rxn terms apply the column-normalisation correction
                num = D - 2.0 * A * B / n_rxn
                den = C - (A**2 - B**2) / n_rxn
                theta = np.arctan2(num, den) / 4.0

                c, s = np.cos(theta), np.sin(theta)

                # in-place 2×2 orthogonal Jacobi rotation;
                # preserves column norms and the null-space span
                rot2 = np.array([[c, -s],
                                 [s,  c]])
                Z[:, [p, q]] = Z[:, [p, q]] @ rot2

        # --- Varimax objective: sum of per-column variances of squared loadings ---
        # higher value ↔ sparser columns
        col_sq = Z**2                                          # (n_rxn, k)
        obj = (n_rxn * (col_sq**2).sum(axis=0)
               - col_sq.sum(axis=0)**2).sum()

        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

    return Z


def compute_targeted_nullspace(
    S: np.ndarray,
    reaction_names: list,
    target_reactions: list,
    tol: float = 1e-10,
    warn_cond_threshold: float = 1e8,
):
    """Build a structured null-space basis that gives each target reaction its own DOF.

    For each target reaction t, the returned Z satisfies Z[t, j_t] = 1 and
    Z[t, j≠j_t] = 0 (identity block on targets), so the corresponding basis
    coefficient y[j_t] directly represents that reaction's flux with no
    cross-reaction amplification.

    Parameters
    ----------
    S : np.ndarray, shape (m, n)
        Stoichiometric matrix (metabolites × reactions).
    reaction_names : list[str], length n
        Ordered reaction identifiers matching S columns.
    target_reactions : list[str]
        Reactions that must each receive their own basis dimension.
    tol : float
        Relative tolerance for rank determination and mass-balance check.
    warn_cond_threshold : float
        Log a warning when cond(S_D) exceeds this value.

    Returns
    -------
    Z : np.ndarray, shape (n, n-r), or None if separability check fails
    status : dict with keys
        separable       bool   — False means targets are stoichiometrically coupled
        fallback_reason str    — set when separable=False
        cond_SD         float  — condition number of the dependent block (0 when failed)
        target_indices  list   — column indices of target reactions in S
        dep_cols        list   — column indices chosen as the dependent block
        free_cols       list   — column indices of free DOFs (targets first)
    """
    m, n = S.shape

    # Step 0 — validate inputs
    try:
        target_idx = [reaction_names.index(r) for r in target_reactions]
    except ValueError as e:
        raise ValueError(f"compute_targeted_nullspace: {e}") from None

    Z_raw = scipy.linalg.null_space(S)   # (n, k)
    k = Z_raw.shape[1]
    if k == 0:
        return None, {
            'separable': False,
            'fallback_reason': 'null space is empty (S is square and full-rank)',
            'cond_SD': 0.0, 'target_indices': target_idx,
            'dep_cols': [], 'free_cols': [],
        }
    if len(target_reactions) > k:
        raise ValueError(
            f"compute_targeted_nullspace: {len(target_reactions)} targets but "
            f"null-space dimension is only {k}"
        )

    # Step 1 — separability check: target rows must span a full-rank subspace in null(S)
    G = Z_raw[target_idx, :]             # (|T|, k)
    rank_G = np.linalg.matrix_rank(G, tol=tol * np.linalg.norm(G))
    if rank_G < len(target_reactions):
        return None, {
            'separable': False,
            'fallback_reason': (
                f'targets are stoichiometrically coupled: '
                f'rank(G)={rank_G} < {len(target_reactions)}'
            ),
            'cond_SD': 0.0, 'target_indices': target_idx,
            'dep_cols': [], 'free_cols': [],
        }

    # Step 2 — identify r = rank(S) independent metabolite rows via QR on S.T
    # scipy.linalg.qr with pivoting=True: S.T[:, piv] = Q @ R  →  piv gives column
    # ordering of S.T, i.e. row ordering of S, by decreasing pivot magnitude.
    _, R_qr, piv_met = scipy.linalg.qr(S.T, pivoting=True)
    diag = np.abs(np.diag(R_qr))
    r = int(np.sum(diag > tol * diag[0]))      # rank(S)
    indep_met_idx = piv_met[:r]
    Sr = S[indep_met_idx, :]                   # (r, n)

    # Step 3 — among non-target columns, pick r most independent ones as the
    # dependent block S_D.  Column-pivoted QR on Sr restricted to non-target cols.
    non_target_cols = [i for i in range(n) if i not in set(target_idx)]
    Sr_nt = Sr[:, non_target_cols]             # (r, n - |T|)
    _, R2, piv2 = scipy.linalg.qr(Sr_nt, pivoting=True)
    dep_cols = [non_target_cols[piv2[i]] for i in range(r)]
    S_D = Sr[:, dep_cols]                      # (r, r)

    cond_SD = float(np.linalg.cond(S_D))
    if cond_SD > warn_cond_threshold:
        warnings.warn(
            f"compute_targeted_nullspace: cond(S_D) = {cond_SD:.2e} exceeds "
            f"warn_cond_threshold={warn_cond_threshold:.2e}. "
            f"Predictions for some reactions may be numerically unreliable."
        )

    # Step 4 — free DOFs: targets first, then remaining non-pivot non-target cols
    remaining_free = [non_target_cols[piv2[i]] for i in range(r, len(non_target_cols))]
    free_cols = target_idx + remaining_free    # length = n - r
    S_free = Sr[:, free_cols]                 # (r, n - r)

    # Step 5 — solve S_D @ X = -S_free  (LU with partial pivoting via np.linalg.solve)
    X = np.linalg.solve(S_D, -S_free)         # (r, n - r)

    # Step 6 — assemble Z: identity on free rows, LU solution on dependent rows
    Z = np.zeros((n, n - r))
    Z[dep_cols, :] = X
    Z[free_cols, :] = np.eye(n - r)

    residual = float(np.abs(S @ Z).max())
    if residual > 1e-6:
        warnings.warn(
            f"compute_targeted_nullspace: max |S @ Z| = {residual:.2e} — "
            f"mass balance may not hold to expected tolerance."
        )

    # Normalize columns to unit norm.  The target-block zeros are preserved exactly
    # (dividing by a positive scalar never turns a zero nonzero).  Column norms of
    # order 1 bound the Gram matrix G = Z.T @ Z to diagonal 1 / off-diagonal ≤ 1,
    # which controls both the initial loss scale and gradient magnitude during training.
    col_norms = np.linalg.norm(Z, axis=0)
    Z = Z / col_norms

    return Z, {
        'separable': True,
        'fallback_reason': '',
        'cond_SD': cond_SD,
        'col_norms': col_norms,
        'target_indices': target_idx,
        'dep_cols': dep_cols,
        'free_cols': free_cols,
    }
