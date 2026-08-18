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
    """Load pre-computed Fisher scores and inclusion mask (from features.csv,
    produced by extract_informative_features.py) aligned to labeling_df row order.
    Tokens absent from the CSV get score 0.0 and included=False."""
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


def _varimax_rotate(Z: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Varimax (Kaiser 1958) orthogonal rotation applied in-place to columns of Z,
    maximising the sum of per-column variances of squared loadings via pairwise
    Jacobi sweeps. Input columns must already have unit norm."""
    n_rxn, k = Z.shape
    if k <= 1:
        return Z

    prev_obj = -np.inf
    for _ in range(max_iter):
        for p in range(k - 1):
            for q in range(p + 1, k):
                u = Z[:, p] ** 2 - Z[:, q] ** 2
                v = 2.0 * Z[:, p] * Z[:, q]
                A, B = u.sum(), v.sum()
                C = (u**2 - v**2).sum()
                D = (2.0 * u * v).sum()
                num = D - 2.0 * A * B / n_rxn
                den = C - (A**2 - B**2) / n_rxn
                theta = np.arctan2(num, den) / 4.0
                c, s = np.cos(theta), np.sin(theta)
                Z[:, [p, q]] = Z[:, [p, q]] @ np.array([[c, -s], [s, c]])

        col_sq = Z ** 2
        obj = (n_rxn * (col_sq ** 2).sum(axis=0) - col_sq.sum(axis=0) ** 2).sum()
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

    return Z


def compute_sparse_nullspace(
    S: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute a sparse null-space basis of stoichiometric matrix S via SVD followed
    by Varimax (Kaiser 1958) rotation, which finds an orthogonal rotation of the raw
    SVD basis that concentrates each basis vector's weight on fewer reactions while
    preserving orthonormality and the span of null(S)."""
    Z = scipy.linalg.null_space(S)   # (n_rxn, k) — unit-norm columns from SVD
    return _varimax_rotate(Z, max_iter=max_iter, tol=tol)


def compute_targeted_nullspace(
    S: np.ndarray,
    reaction_names: list,
    target_reactions: list,
    tol: float = 1e-10,
    warn_cond_threshold: float = 1e8,
):
    """Build a structured null-space basis that gives each target reaction its own DOF.

    For each target reaction t, Z[t, j_t] = 1 and Z[t, j≠j_t] = 0, so basis coefficient
    y[j_t] directly represents that reaction's flux. Redundant targets (exact linear
    combinations of others) are dropped rather than failing the whole construction; when
    choosing which of a redundant group keeps the DOF, the most-connected reaction
    (touching the most metabolites) wins, ties breaking on the caller's order.

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
    Z : np.ndarray, shape (n, n-r), or None if no basis could be built at all
    status : dict
        separable       — True if a targeted basis was built (possibly on a subset)
        fallback_reason — set when separable is False
        targets_kept    — targets that received their own DOF
        targets_dropped — redundant targets, each an exact combination of the kept
        cond_SD         — condition number of the dependent block (0 if failed)
        target_indices, dep_cols, free_cols — column indices into S
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
    # Asking for more targets than the null-space dimension is not an error: at most k
    # of them can be independent, so the surplus is redundant by construction and the
    # selection below drops it like any other redundancy.

    # Step 1 — keep a maximal independent subset of targets, most-connected first.
    # Adding a target that is already an exact combination of the kept ones would make
    # the identity block singular, so it is skipped; its flux is still determined, just
    # through the others rather than through a coordinate of its own.
    order = sorted(
        range(len(target_reactions)),
        key=lambda i: (-int(np.count_nonzero(S[:, target_idx[i]])), i),
    )
    kept, dropped = [], []
    for i in order:
        trial = kept + [target_idx[i]]
        G = Z_raw[trial, :]
        if np.linalg.matrix_rank(G, tol=tol * max(np.linalg.norm(G), 1.0)) == len(trial):
            kept.append(target_idx[i])
        else:
            dropped.append(target_reactions[i])
    # An empty target list is a legitimate request for an untargeted basis; only a
    # request that asked for targets and could keep none is a failure.
    if target_reactions and not kept:
        return None, {
            'separable': False,
            'fallback_reason': 'no target has a null-space component',
            'targets_kept': [], 'targets_dropped': list(target_reactions),
            'cond_SD': 0.0, 'target_indices': [],
            'dep_cols': [], 'free_cols': [],
        }
    target_idx = kept
    targets_kept = [reaction_names[i] for i in kept]

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

    # Apply Varimax rotation to the non-target free columns.
    # The target rows of those columns are all zero by construction (LU identity block),
    # so any orthogonal rotation within the free subspace keeps those zeros intact —
    # separability is fully preserved.  The rotation maximises sparsity of the free
    # dimensions so they concentrate on specific metabolic pathways rather than being
    # arbitrary LU artefacts.
    n_targets = len(target_idx)
    if n - r > n_targets + 1:
        Z[:, n_targets:] = _varimax_rotate(Z[:, n_targets:].copy())

    return Z, {
        'separable': True,
        'fallback_reason': '',
        'targets_kept': targets_kept,
        'targets_dropped': dropped,
        'cond_SD': cond_SD,
        'col_norms': col_norms,
        'target_indices': target_idx,
        'dep_cols': dep_cols,
        'free_cols': free_cols,
    }


def compute_ica_nullspace(
    S: np.ndarray,
    flux_samples: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Build a null-space basis informed by ICA on steady-state flux samples.

    The leading k columns are statistically independent flux modes from a single
    FastICA run on the variance-active null-space subspace (effective rank estimated
    from the sample covariance eigenspectrum via a machine-epsilon threshold); the
    remaining d-k columns complete the basis. Always satisfies S @ Z = 0, rank(Z) = d.

    Parameters
    ----------
    S : np.ndarray, shape (m, n)
        Stoichiometric matrix (metabolites × reactions).
    flux_samples : np.ndarray, shape (n_samples, n_reactions)
        Steady-state flux samples satisfying S @ v ≈ 0. Typically flux_df.values.T.
    seed : int
        Random seed for FastICA initialisation.

    Returns
    -------
    Z : np.ndarray, shape (n, d)
        Null-space basis; leading k columns are ICA modes, remaining d-k complete
        the basis. Columns orthonormal (Gram ≈ I).
    status : dict
        n_ica_components, n_complement, n_active (effective sampling rank), and
        fallback (True if FastICA failed and Varimax was used instead).
    """
    from sklearn.decomposition import FastICA  # local import — optional dependency

    n = S.shape[1]
    N = scipy.linalg.null_space(S)   # (n, d), orthonormal
    d = N.shape[1]

    if d == 0:
        return np.zeros((n, 0)), {
            'n_ica_components': 0, 'n_complement': 0, 'n_active': 0, 'fallback': False,
        }

    # Project training samples into null-space coordinates
    Y = flux_samples @ N    # (n_samples, d)
    n_samples = Y.shape[0]

    # Effective rank: count eigenvalues of the sample covariance above machine-epsilon.
    # Null-space directions with near-zero variance are unexplored by the sampler and
    # would become noise for ICA; the complement step fills those dimensions instead.
    Y_c = Y - Y.mean(axis=0)
    cov = (Y_c.T @ Y_c) / max(n_samples - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]   # descending
    eps_thresh = np.finfo(float).eps * d * eigvals[0]
    n_active = max(1, int(np.sum(eigvals > eps_thresh)))
    n_ica_req = min(n_active, n_samples - 1)
    print(f"ICA null-space: effective rank {n_active}/{d}")

    # Single FastICA run — sklearn's internal PCA whitening operates on the n_ica_req
    # leading principal components, giving ICA a full-rank well-conditioned input.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ica = FastICA(n_components=n_ica_req, random_state=seed)
            ica.fit(Y)
        modes = N @ ica.components_.T   # (n, n_ica_req), already in null(S) by construction
    except Exception:
        Z_fb = compute_sparse_nullspace(S)
        return Z_fb, {
            'n_ica_components': 0, 'n_complement': d, 'n_active': n_active, 'fallback': True,
        }

    # Orthonormalize ICA block via thin QR
    Q_ica, _ = np.linalg.qr(modes)      # (n, n_ica_req), orthonormal columns
    k_ica = np.linalg.matrix_rank(Q_ica)
    Q_ica = Q_ica[:, :k_ica]

    # Fill remaining d - k_ica dimensions with the orthogonal complement
    n_complement = d - k_ica
    if n_complement > 0:
        R = N - Q_ica @ (Q_ica.T @ N)      # (n, d); columns span complement in null(S)
        U_r, _, _ = np.linalg.svd(R, full_matrices=False)
        Q_complement = U_r[:, :n_complement]
        Z = np.hstack([Q_ica, Q_complement])
    else:
        Z = Q_ica.copy()

    # Sign convention: largest-|value| element positive per column
    for j in range(Z.shape[1]):
        if Z[np.argmax(np.abs(Z[:, j])), j] < 0:
            Z[:, j] = -Z[:, j]
    col_norms = np.linalg.norm(Z, axis=0)
    col_norms[col_norms < 1e-14] = 1.0
    Z = Z / col_norms

    residual_max = float(np.abs(S @ Z).max())
    assert residual_max <= 1e-8, (
        f"compute_ica_nullspace: max|S @ Z| = {residual_max:.2e}"
    )
    assert np.linalg.matrix_rank(Z) == d, (
        f"compute_ica_nullspace: rank(Z) = {np.linalg.matrix_rank(Z)} != d = {d}"
    )

    print(f"ICA null-space: {k_ica} ICA modes, {n_complement} complement")
    return Z, {
        'n_ica_components': k_ica,
        'n_complement': n_complement,
        'n_active': n_active,
        'fallback': False,
    }



def detect_parallel_reaction_groups(
    S: np.ndarray,
    reaction_names: list,
    cofactor_metabolites: list | None = None,
    cos_threshold: float = 0.99,
) -> list[list[str]]:
    """Find groups of reactions with stoichiometrically equivalent carbon backbones.

    Two reactions are parallel if their stoichiometric columns are nearly proportional
    (cosine similarity >= cos_threshold) after zeroing out cofactor rows; such
    reactions carry the same carbon flux pattern and cannot be individually resolved
    from ¹³C labeling data — only their sum (net flux) is observable.

    Parameters
    ----------
    cofactor_metabolites : list[str] | None
        Metabolite IDs (row labels of S) to mask before computing similarities;
        when None, all rows are used.

    Returns
    -------
    groups : list[list[str]]
        Each inner list has >= 2 stoichiometrically parallel reaction names;
        reactions with no partner are omitted.
    """
    n = S.shape[1]
    S_masked = S.copy().astype(float)

    if cofactor_metabolites is not None:
        # Zero out cofactor rows so ATP/NAD usage differences are ignored
        met_to_row = {name: i for i, name in enumerate(
            getattr(S, '_metabolite_names', range(S.shape[0]))
        )}
        for met in cofactor_metabolites:
            row = met_to_row.get(met)
            if row is not None:
                S_masked[row, :] = 0.0

    # Column norms (avoid division by zero)
    col_norms = np.linalg.norm(S_masked, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    S_norm = S_masked / col_norms[np.newaxis, :]   # (m, n) — unit-norm columns

    # Pairwise cosine similarities (use matrix product: n×n)
    cos_sim = S_norm.T @ S_norm   # (n, n)

    # Union-find for grouping
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if abs(cos_sim[i, j]) >= cos_threshold:
                union(i, j)

    # Collect groups with ≥ 2 members
    from collections import defaultdict
    groups_dict: dict = defaultdict(list)
    for i in range(n):
        groups_dict[find(i)].append(reaction_names[i])

    return [sorted(members) for members in groups_dict.values() if len(members) >= 2]


def compute_targeted_ica_nullspace(
    S: np.ndarray,
    flux_samples: np.ndarray,
    reaction_names: list,
    target_reactions: list,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Build a null-space basis combining identity DOFs for measured reactions with
    ICA free modes: the first k columns give each target reaction its own DOF
    (identity block), and the remaining d-k columns are ICA-ordered statistically
    independent directions within the free subspace, sorted by descending
    variance-based observability score (ICA modes before the SVD complement).

    Raises RuntimeError if the target reactions are not separable (stoichiometrically
    coupled) — see :func:`compute_targeted_nullspace`. Falls back to the Varimax
    targeted basis (unrotated free columns) if FastICA itself fails.

    Parameters
    ----------
    S : np.ndarray, shape (m, n)
        Stoichiometric matrix (metabolites × reactions).
    flux_samples : np.ndarray, shape (n_samples, n_reactions)
        Steady-state flux samples satisfying S @ v ≈ 0. Typically flux_df.values.T.
    reaction_names : list[str], length n
        Ordered reaction identifiers matching S columns.
    target_reactions : list[str]
        Measured reactions that must each receive their own basis dimension.
    seed : int
        Random seed for FastICA initialisation.

    Returns
    -------
    Z : np.ndarray, shape (n, d)
        Null-space basis. First k columns have identity block on target rows.
        Columns k: are sorted by descending variance-obs score (ICA modes first,
        SVD complement last).
    status : dict
        separable, cond_SD, target_indices, n_ica_free, n_complement_free,
        n_active_free, and fallback (True if FastICA failed, Varimax free cols kept).
    """
    from sklearn.decomposition import FastICA

    # ── Step 1: targeted construction ────────────────────────────────────────
    Z_targ, tgt_status = compute_targeted_nullspace(S, reaction_names, target_reactions)
    if not tgt_status['separable']:
        # Deliberately not falling through to ica_informed. That substitution is
        # invisible in the resulting checkpoint -- the run reports null_space_type
        # targeted_ica while the measured fluxes hold no coordinate at all -- and it
        # cost the T cell its DNA_c and RNA_c predictions. Redundant targets no longer
        # trigger this; reaching it means no target has a null-space component.
        raise RuntimeError(
            f"targeted_ica basis could not be built: {tgt_status['fallback_reason']}. "
            f"Set null_space_type explicitly if that is what you want."
        )

    # The kept targets, not everything asked for: redundant ones share a DOF with a
    # more connected target rather than getting their own, so they are not in the block.
    n_targets = len(tgt_status['targets_kept'])
    if tgt_status['targets_dropped']:
        print(f"targeted_ica: {n_targets}/{len(target_reactions)} targets got their own DOF; "
              f"redundant: {', '.join(tgt_status['targets_dropped'])}")
    d = Z_targ.shape[1]
    n_free = d - n_targets

    # ── Step 2: trivial case — no free subspace to rotate ────────────────────
    if n_free < 2:
        return Z_targ, {
            'separable': True, 'cond_SD': tgt_status['cond_SD'],
            'targets_kept': tgt_status['targets_kept'],
            'targets_dropped': tgt_status['targets_dropped'],
            'target_indices': tgt_status['target_indices'],
            'n_ica_free': 0, 'n_complement_free': n_free,
            'n_active_free': n_free, 'fallback': False,
        }

    # ── Step 3: orthonormal basis of the free subspace ───────────────────────
    # Z_targ[:, n_targets:] has zeros on target rows; QR preserves that property.
    Q_free, _ = np.linalg.qr(Z_targ[:, n_targets:])   # (n, n_free), orthonormal

    # ── Step 4: free coordinates and eigendecomposition (shared across steps 5 & 6) ──
    Y_free = flux_samples @ Q_free            # (n_samples, n_free)
    Y_c_free = Y_free - Y_free.mean(axis=0)
    cov_free = (Y_c_free.T @ Y_c_free) / max(Y_c_free.shape[0] - 1, 1)
    eigvals_asc, eigvecs = np.linalg.eigh(cov_free)  # ascending order
    eigvals = eigvals_asc[::-1]                        # descending

    # ── Step 5: effective rank of free subspace ───────────────────────────────
    eps_thresh = np.finfo(float).eps * n_free * eigvals[0]
    n_active = max(1, int(np.sum(eigvals > eps_thresh)))
    n_ica_req = min(n_active, Y_free.shape[0] - 1)
    print(f"targeted_ica: free subspace effective rank {n_active}/{n_free}")

    # ── Step 6: manual PCA-whitening + FastICA ────────────────────────────────
    # Pre-whiten Y_free using the top n_ica_req eigenvectors so we can pass a
    # labeling-informed w_init in the whitened space (see plan §3C).
    top_order = np.argsort(eigvals_asc)[::-1][:n_ica_req]   # descending-variance indices
    V_pca = eigvecs[:, top_order]                             # (n_free, n_ica_req)
    D_pca = np.maximum(eigvals_asc[top_order], 1e-10)        # (n_ica_req,) positive
    W_wh  = V_pca / np.sqrt(D_pca)[np.newaxis, :]            # (n_free, n_ica_req) whitening
    Y_w   = Y_c_free @ W_wh                                  # (n_samples, n_ica_req)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ica = FastICA(n_components=n_ica_req, random_state=seed, whiten=False)
            ica.fit(Y_w)
        # ICA components_ is (n_ica_req, n_ica_req) un-mixing in whitened space.
        # Modes in reaction null-space: Q_free @ W_wh @ mixing_matrix
        modes = Q_free @ (W_wh @ ica.components_.T)   # (n, n_ica_req), zeros on target rows ✓
    except Exception:
        print("targeted_ica: FastICA failed; keeping Varimax free columns")
        return Z_targ, {
            'separable': True, 'cond_SD': tgt_status['cond_SD'],
            'targets_kept': tgt_status['targets_kept'],
            'targets_dropped': tgt_status['targets_dropped'],
            'target_indices': tgt_status['target_indices'],
            'n_ica_free': 0, 'n_complement_free': n_free,
            'n_active_free': n_active, 'fallback': True,
        }

    # ── Step 7: QR-orthonormalize ICA modes ──────────────────────────────────
    Q_ica_free, _ = np.linalg.qr(modes)
    k_ica = np.linalg.matrix_rank(Q_ica_free)
    Q_ica_free = Q_ica_free[:, :k_ica]

    # ── Step 8: SVD complement within the free subspace ──────────────────────
    n_comp = n_free - k_ica
    if n_comp > 0:
        R = Q_free - Q_ica_free @ (Q_ica_free.T @ Q_free)
        U_r, _, _ = np.linalg.svd(R, full_matrices=False)
        Z_free_block = np.hstack([Q_ica_free, U_r[:, :n_comp]])
    else:
        Z_free_block = Q_ica_free.copy()

    # ── Step 9: sign-fix on free block ───────────────────────────────────────
    for j in range(Z_free_block.shape[1]):
        if Z_free_block[np.argmax(np.abs(Z_free_block[:, j])), j] < 0:
            Z_free_block[:, j] *= -1

    # ── Step 10: assemble, normalize, validate ───────────────────────────────
    Z = np.hstack([Z_targ[:, :n_targets], Z_free_block])
    col_norms = np.linalg.norm(Z, axis=0)
    col_norms[col_norms < 1e-14] = 1.0
    Z = Z / col_norms

    assert np.abs(S @ Z).max() <= 1e-8, (
        f"compute_targeted_ica_nullspace: max|S @ Z| = {np.abs(S @ Z).max():.2e}"
    )
    assert np.linalg.matrix_rank(Z) == d, (
        f"compute_targeted_ica_nullspace: rank(Z) = {np.linalg.matrix_rank(Z)} != d = {d}"
    )

    return Z, {
        'separable': True,
        'cond_SD': tgt_status['cond_SD'],
        'targets_kept': tgt_status['targets_kept'],
        'targets_dropped': tgt_status['targets_dropped'],
        'target_indices': tgt_status['target_indices'],
        'n_ica_free': k_ica,
        'n_complement_free': n_comp,
        'n_active_free': n_active,
        'fallback': False,
    }
