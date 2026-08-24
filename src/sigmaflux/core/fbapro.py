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


def predict_knockout_flux(S, v_wt, knockouts, fixed=()):
    """Predict a post-knockout flux as the minimal adjustment of a wild-type flux.

    Forces the knockout reactions to exactly zero and projects the wild-type flux back
    onto the steady-state space, moving the other reactions as little as possible (a
    MoMA-style prediction). Optionally also holds ``fixed`` reactions (e.g. measured
    exchange fluxes) at their wild-type values.

    Args:
        S: stoichiometric matrix, (n_metabolites, n_reactions).
        v_wt: wild-type flux vector, (n_reactions,).
        knockouts: indices of reactions to knock out (forced to 0).
        fixed: indices of additional reactions to hold at their wild-type value.

    Returns the predicted knockout flux vector, (n_reactions,).
    """
    v_wt = np.asarray(v_wt, dtype=np.float64)
    knockouts = np.asarray(knockouts, dtype=int)
    fixed = np.asarray(fixed, dtype=int)
    H = np.concatenate([knockouts, fixed]).astype(int)
    values = np.concatenate([np.zeros(knockouts.size), v_wt[fixed]])
    return fbapro(S, v_wt, fixed=H, fixed_values=values)


def knockout_flux_from_model(model, v_wt, knockout_ids, fixed_ids=()):
    """Reaction-id wrapper around :func:`predict_knockout_flux` for a COBRA model.

    Args:
        model: a cobra Model; its reaction order defines the flux vector order.
        v_wt: wild-type fluxes as a dict/Series keyed by reaction id.
        knockout_ids: reaction ids to knock out.
        fixed_ids: reaction ids to hold at their wild-type value.

    Returns the predicted knockout fluxes as a dict keyed by reaction id.
    """
    from cobra.util.array import create_stoichiometric_matrix

    reaction_ids = [r.id for r in model.reactions]
    idx = {rid: i for i, rid in enumerate(reaction_ids)}
    S = create_stoichiometric_matrix(model)
    v = np.array([float(v_wt[rid]) for rid in reaction_ids])
    ko = [idx[r] for r in knockout_ids]
    fx = [idx[r] for r in fixed_ids]
    x = predict_knockout_flux(S, v, ko, fx)
    return {rid: float(x[i]) for i, rid in enumerate(reaction_ids)}
