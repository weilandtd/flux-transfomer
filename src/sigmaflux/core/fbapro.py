"""FBApro: the closest steady-state flux to a reference vector.

Projection from the FBApro paper (arXiv:2601.14577): given a reference flux ``vin``, find
the flux ``x`` closest to it that satisfies the steady-state condition ``S x = 0``,
optionally holding a set of reactions at fixed values -- a knockout is just a fixed value of
0, a measured exchange is fixed at its measured value. This is MoMA written as a plain
(optionally weighted) orthogonal projection.

The steady-state constraint is carried by a null-space basis ``Z`` (its columns span
``ker S``, so a feasible flux is ``x = Z a``). Pass whichever you have -- ``Z`` directly, or
``S`` (its null space is computed once). :func:`fbapro` enforces only ``S x = 0`` and the
fixed values, NOT reaction bounds/irreversibility; use :func:`bounded_moma` when those
matter.
"""
import numpy as np


def _null_space_basis(S=None, Z=None):
    """Null-space basis of ``S`` (columns span ``ker S``): return ``Z`` if given, else
    compute it from ``S``."""
    if Z is not None:
        return np.asarray(Z, dtype=np.float64)
    if S is None:
        raise ValueError("provide either S (stoichiometric matrix) or Z (null-space basis)")
    from scipy.linalg import null_space
    return null_space(np.asarray(S, dtype=np.float64))


def fbapro(vin, S=None, Z=None, vfixed=None, weights=None):
    """Closest steady-state flux to ``vin``, holding an optional set of reactions fixed.

    Solves ``min sum_i w_i (x_i - vin_i)^2`` subject to ``S x = 0`` and ``x[F] = vfixed[F]``
    for the fixed set ``F``, as a (weighted) orthogonal projection. ``vin`` need not be
    feasible -- it is first projected onto ``ker S``, so an ensemble median (which breaks
    ``S x = 0``) is fine. A knockout is a fixed value of 0; a measured exchange is a fixed
    value at its measured value; leave everything else free.

    Args:
        vin: reference flux(es), shape ``(n_reactions,)`` or ``(n_samples, n_reactions)``.
        S: stoichiometric matrix ``(n_metabolites, n_reactions)``; give this or ``Z``.
        Z: null-space basis of ``S`` ``(n_reactions, n_free)`` (``x = Z a``); give this or ``S``.
        vfixed: per-reaction fixed values as a full-length array, ``NaN`` where free and the
            pinned value where fixed (knockout = ``0.0``); ``None`` = no fixed reactions
            (plain projection onto ``{S x = 0}``).
        weights: optional per-reaction non-negative weights ``(n_reactions,)``; larger =
            held closer to ``vin``. ``None`` is the plain (unweighted) projection.

    Returns:
        Projected flux(es), same dimensionality as ``vin`` (1-D in -> 1-D out), satisfying
        ``S x = 0`` and the fixed values exactly.
    """
    Z = _null_space_basis(S, Z)
    single = np.asarray(vin).ndim == 1
    V = np.atleast_2d(np.asarray(vin, dtype=np.float64))
    n_rxn = Z.shape[0]
    w = (np.ones(n_rxn) if weights is None
         else np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None))

    # weighted projection onto ker S (exact for already-feasible vin); H = Z^T W Z.
    H_inv = np.linalg.pinv((Z.T * w) @ Z)
    X = ((Z @ H_inv @ (Z.T * w)) @ V.T).T

    if vfixed is not None:
        vfixed = np.asarray(vfixed, dtype=np.float64)
        F = np.where(~np.isnan(vfixed))[0]
        if F.size:
            # KKT correction pinning x[F] = vfixed[F]: one factorisation for the batch.
            G = Z[F, :]
            M = np.linalg.pinv(G @ H_inv @ G.T)
            resid = X[:, F] - vfixed[F]              # (n_samples, |F|)
            X = X - (Z @ H_inv @ G.T @ M @ resid.T).T
            X[:, F] = vfixed[F]                      # exact fixed values (clean round-off)

    return X[0] if single else X


def bounded_moma(vin, lower, upper, S=None, Z=None, vfixed=None, weights=None,
                 max_iter=10000, tol=1e-9):
    """Closest steady-state flux to ``vin`` that also respects reaction bounds (MoMA).

    Same objective as :func:`fbapro` with ``lower <= x <= upper`` added. Solved by Dykstra's
    alternating projection between ``ker S`` (an orthogonal projection built from ``Z`` or
    ``S``) and the bounds box (a clip); Dykstra's corrections make it converge to the
    closest point in the intersection. Fixed reactions in ``vfixed`` fold into the box as
    ``lower = upper = value`` (a knockout is ``value = 0``), so irreversibility keeps them
    from being revived. ``weights`` reweights the objective as in :func:`fbapro`.

    Args:
        vin: reference flux vector ``(n_reactions,)``.
        lower, upper: per-reaction bounds, each ``(n_reactions,)``.
        S / Z: stoichiometric matrix or null-space basis; give one (see :func:`fbapro`).
        vfixed: per-reaction fixed values, ``NaN`` where free (see :func:`fbapro`).
        weights: optional per-reaction non-negative weights.
        max_iter: maximum Dykstra iterations.
        tol: stop when successive iterates move less than this (L2).

    Returns:
        Predicted flux vector ``(n_reactions,)`` respecting the bounds and fixed values.
    """
    Z = _null_space_basis(S, Z)
    v = np.asarray(vin, dtype=np.float64)
    n = v.shape[0]
    lb = np.array(lower, dtype=np.float64)
    ub = np.array(upper, dtype=np.float64)
    if vfixed is not None:
        vfixed = np.asarray(vfixed, dtype=np.float64)
        F = ~np.isnan(vfixed)
        lb[F] = ub[F] = vfixed[F]

    # confidence weighting: min sum_i w_i (x_i - v_i)^2 is the unweighted problem in
    # u = sqrt(w) * x. The feasible set for u is col(sqrt(w) * Z) = ker(S / sqrt(w)); P
    # projects onto it, and the box is scaled likewise.
    w = (np.ones(n) if weights is None
         else np.clip(np.asarray(weights, dtype=np.float64), 1e-8, None))
    s = np.sqrt(w)
    B = s[:, None] * Z
    P = B @ np.linalg.pinv(B)
    u = s * v
    lbu, ubu = s * lb, s * ub
    p = np.zeros(n)
    q = np.zeros(n)
    for _ in range(max_iter):
        y = P @ (u + p)                              # project onto the feasible subspace
        p = (u + p) - y
        u_new = np.clip(y + q, lbu, ubu)             # project onto the (scaled) bounds box
        q = (y + q) - u_new
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        u = u_new
    return u / s
