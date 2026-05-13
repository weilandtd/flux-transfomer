import numpy as np
import scipy.linalg


def compute_sparse_nullspace(
    S: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute a sparse null-space basis of stoichiometric matrix S.

    SVD-based null_space returns an orthonormal basis whose columns are spread
    across many reactions. Varimax rotation (Kaiser 1958) applies pairwise Jacobi
    rotations to maximise the variance of squared loadings, concentrating each
    basis vector's weight onto fewer reactions. The rotated columns span the same
    null space but are sparser, making learned basis coefficients more
    interpretable. Varimax produces an oblique (non-orthogonal) basis — columns
    are intentionally NOT re-orthogonalised after rotation.

    Parameters
    ----------
    S : np.ndarray  (m, n)
    max_iter : int  — sweep limit; each sweep rotates every column pair once
    tol : float     — convergence threshold on change in varimax objective

    Returns
    -------
    Z_rotated : np.ndarray  (n, k)
    """
    Z = scipy.linalg.null_space(S)   # initial orthonormal basis (n, k)
    n, k = Z.shape

    if k <= 1:
        return Z

    prev_obj = -np.inf
    for _ in range(max_iter):
        for p in range(k - 1):
            for q in range(p + 1, k):
                u = Z[:, p] ** 2 - Z[:, q] ** 2
                v = 2.0 * Z[:, p] * Z[:, q]
                A, B = u.sum(), v.sum()
                C = (u ** 2 - v ** 2).sum()
                D = (2.0 * u * v).sum()
                # Kaiser varimax angle: maximises variance of squared loadings
                num = D - 2.0 * A * B / n
                den = C - (A ** 2 - B ** 2) / n
                theta = np.arctan2(num, den) / 4.0
                c, s = np.cos(theta), np.sin(theta)
                Z[:, [p, q]] = Z[:, [p, q]] @ np.array([[c, -s], [s, c]])

        # Varimax objective: sum of per-column variance of squared loadings
        obj = sum(
            n * (Z[:, j] ** 4).sum() - ((Z[:, j] ** 2).sum()) ** 2
            for j in range(k)
        )
        if abs(obj - prev_obj) < tol:
            break
        prev_obj = obj

    return Z
