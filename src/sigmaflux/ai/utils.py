import numpy as np
import scipy.linalg
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