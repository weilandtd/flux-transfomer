"""FBApro: the closest steady-state flux to a reference vector.

Projection from the FBApro paper (arXiv:2601.14577): given a reference flux vector ``v``,
find the flux ``x`` closest to ``v`` that satisfies the steady-state condition ``S x = 0``,
optionally holding a set of high-confidence reactions at exact values. This is MoMA
expressed as a plain orthogonal projection.

It enforces only ``S x = 0`` (and the fixed values) -- NOT reaction bounds or
irreversibility, so a predicted flux may run a reaction backwards or past its capacity.
Check that separately if it matters.

Main use here: predict a knockout flux as the minimal metabolic adjustment of a
wild-type SigmaFlux/MFA solution -- see :func:`predict_knockout_flux`.
"""
import numpy as np


def fbapro(S, v, fixed=(), fixed_values=None):
    """Closest steady-state flux to ``v``.

    Solves ``min ||x - v||^2`` subject to ``S x = 0`` and ``x[fixed] = fixed_values``,
    as an orthogonal projection of ``v`` onto that affine space.

    Args:
        S: stoichiometric matrix, (n_metabolites, n_reactions).
        v: reference flux vector, (n_reactions,).
        fixed: indices of reactions held at an exact value (the high-confidence set H).
        fixed_values: values for ``fixed`` (defaults to ``v[fixed]``).

    Returns the projected flux vector, (n_reactions,).
    """
    S = np.asarray(S, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    fixed = np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(n), fixed)

    if fixed_values is None:
        xf = v[fixed]
    else:
        xf = np.broadcast_to(np.asarray(fixed_values, dtype=np.float64), fixed.shape).copy()

    S_free = S[:, free]
    b = -S[:, fixed] @ xf if fixed.size else np.zeros(S.shape[0])
    # project v_free onto the affine space {x_free : S_free x_free = b}
    x_free = v[free] - np.linalg.pinv(S_free) @ (S_free @ v[free] - b)

    x = np.empty(n)
    x[fixed] = xf
    x[free] = x_free
    return x


def bounded_moma(S, v, lower, upper, fixed=(), fixed_values=None,
                 max_iter=10000, tol=1e-9):
    """Closest steady-state flux to ``v`` that also respects reaction bounds.

    Same objective as :func:`fbapro` (``min ||x - v||^2`` s.t. ``S x = 0``) but with
    ``lower <= x <= upper`` added -- i.e. the original MoMA. Solved by Dykstra's
    alternating projection between the balance space (``S x = 0``, an orthogonal
    projection) and the bounds box (a clip); Dykstra's corrections make it converge to
    the closest point in the intersection, not just some feasible point.

    Fixed reactions are folded into the box as ``lower = upper = value`` (a knockout is
    just ``value = 0``), so irreversibility keeps them from being revived.

    Args:
        S: stoichiometric matrix, (n_metabolites, n_reactions).
        v: reference flux vector, (n_reactions,).
        lower, upper: per-reaction bounds, each (n_reactions,).
        fixed: indices of reactions pinned to an exact value.
        fixed_values: values for ``fixed`` (defaults to ``v[fixed]``).
        max_iter: maximum Dykstra iterations.
        tol: stop when successive iterates move less than this (L2).

    Returns the predicted flux vector, (n_reactions,).
    """
    S = np.asarray(S, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lb = np.array(lower, dtype=np.float64)
    ub = np.array(upper, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=int)
    if fixed.size:
        xf = v[fixed] if fixed_values is None else np.broadcast_to(
            np.asarray(fixed_values, dtype=np.float64), fixed.shape)
        lb[fixed] = ub[fixed] = xf

    S_pinv = np.linalg.pinv(S)                       # projector onto {S x = 0}
    x = v.copy()
    p = np.zeros_like(v)
    q = np.zeros_like(v)
    for _ in range(max_iter):
        y = (x + p)
        y = y - S_pinv @ (S @ y)                     # project onto S x = 0
        p = (x + p) - y
        x_new = np.clip(y + q, lb, ub)               # project onto the bounds box
        q = (y + q) - x_new
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def predict_knockout_flux(S, v_wt, knockouts, fixed=(), lower=None, upper=None):
    """Predict a post-knockout flux as the minimal adjustment of a wild-type flux.

    Forces the knockout reactions to exactly zero and projects the wild-type flux back
    onto the steady-state space, moving the other reactions as little as possible (a
    MoMA-style prediction). Optionally also holds ``fixed`` reactions (e.g. measured
    exchange fluxes) at their wild-type values.

    If ``lower``/``upper`` are given, the bounded version (:func:`bounded_moma`) is used
    so the prediction respects irreversibility/capacity; otherwise the plain unbounded
    projection (:func:`fbapro`).

    Args:
        S: stoichiometric matrix, (n_metabolites, n_reactions).
        v_wt: wild-type flux vector, (n_reactions,).
        knockouts: indices of reactions to knock out (forced to 0).
        fixed: indices of additional reactions to hold at their wild-type value.
        lower, upper: optional per-reaction bounds; if given, use bounded MoMA.

    Returns the predicted knockout flux vector, (n_reactions,).
    """
    v_wt = np.asarray(v_wt, dtype=np.float64)
    knockouts = np.asarray(knockouts, dtype=int)
    fixed = np.asarray(fixed, dtype=int)
    H = np.concatenate([knockouts, fixed]).astype(int)
    values = np.concatenate([np.zeros(knockouts.size), v_wt[fixed]])
    if lower is None or upper is None:
        return fbapro(S, v_wt, fixed=H, fixed_values=values)
    return bounded_moma(S, v_wt, lower, upper, fixed=H, fixed_values=values)


def knockout_flux_from_model(model, v_wt, knockout_ids, fixed_ids=(), bounded=False):
    """Reaction-id wrapper around :func:`predict_knockout_flux` for a COBRA model.

    Args:
        model: a cobra Model; its reaction order defines the flux vector order.
        v_wt: wild-type fluxes as a dict/Series keyed by reaction id.
        knockout_ids: reaction ids to knock out.
        fixed_ids: reaction ids to hold at their wild-type value.
        bounded: if True, use bounded MoMA with the model's reaction bounds (respects
            irreversibility/capacity); if False, the plain unbounded projection.

    Returns the predicted knockout fluxes as a dict keyed by reaction id.
    """
    from cobra.util.array import create_stoichiometric_matrix

    reaction_ids = [r.id for r in model.reactions]
    idx = {rid: i for i, rid in enumerate(reaction_ids)}
    S = create_stoichiometric_matrix(model)
    v = np.array([float(v_wt[rid]) for rid in reaction_ids])
    ko = [idx[r] for r in knockout_ids]
    fx = [idx[r] for r in fixed_ids]
    if bounded:
        lb = np.array([r.lower_bound for r in model.reactions])
        ub = np.array([r.upper_bound for r in model.reactions])
        x = predict_knockout_flux(S, v, ko, fx, lower=lb, upper=ub)
    else:
        x = predict_knockout_flux(S, v, ko, fx)
    return {rid: float(x[i]) for i, rid in enumerate(reaction_ids)}
