"""FBApro projection (single function, S or Z) on small hand-checked networks."""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.core.fbapro import fbapro, bounded_moma  # noqa: E402

# Branched pathway:  v1: ->A,  v2: A->B,  v3: A->B (parallel),  v4: B->   (metabolites A, B)
S = np.array([[1.0, -1.0, -1.0, 0.0],
              [0.0, 1.0, 1.0, -1.0]])
V_WT = np.array([10.0, 6.0, 4.0, 10.0])   # steady state: S @ V_WT == 0
KO_V2 = np.array([np.nan, 0.0, np.nan, np.nan])   # knock out v2 (fixed to 0)
LB, UB = np.zeros(4), np.full(4, 1000.0)          # all reactions irreversible (>= 0)


def test_wildtype_is_a_fixed_point():
    # v already satisfies S v = 0, so the projection returns it unchanged
    assert np.allclose(fbapro(V_WT, S=S), V_WT)


def test_knockout_reroutes_minimally():
    # knock out v2; the parallel branch v3 must carry its flux
    x = fbapro(V_WT, S=S, vfixed=KO_V2)
    assert np.allclose(S @ x, 0.0)                   # still balanced
    assert abs(x[1]) < 1e-9                            # v2 exactly zero
    assert np.allclose(x, [8.0, 0.0, 8.0, 8.0])       # hand-computed minimal adjustment


def test_no_fixed_reactions_is_plain_projection():
    # vfixed=None: a feasible vector is returned unchanged, an infeasible one reprojected;
    # zeroing v2 in the *vector* then projecting does NOT keep it at zero (needs vfixed)
    assert np.allclose(fbapro(V_WT, S=S), V_WT)
    assert V_WT.ndim == fbapro(V_WT, S=S).ndim        # 1-D in -> 1-D out
    x = fbapro(np.array([10.0, 0.0, 4.0, 10.0]), S=S)  # v2 zeroed in the vector, no vfixed
    assert abs(x[1]) > 1e-6                             # projection revives it


def test_infeasible_input_is_reprojected():
    # an ensemble median can break S x = 0; fbapro must reproject onto ker S
    x = fbapro(np.array([5.0, 3.0, 3.0, 3.0]), S=S2)   # S2 @ v = 5-9 = -4 != 0
    assert np.allclose(S2 @ x, 0.0, atol=1e-9)


# one metabolite, three parallel outputs -> genuine post-KO freedom for weighting to use
#   v1: ->A,  v2/v3/v4: A->
S2 = np.array([[1.0, -1.0, -1.0, -1.0]])
V2 = np.array([9.0, 3.0, 3.0, 3.0])
# null-space basis of S2 (x1 = x2 + x3 + x4): columns span ker(S2), x = Z2 @ a
Z2 = np.array([[1.0, 1.0, 1.0],
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]])


def test_S_and_Z_give_the_same_projection():
    vfixed = np.array([np.nan, 0.0, np.nan, np.nan])
    x_s = fbapro(V2, S=S2, vfixed=vfixed)
    x_z = fbapro(V2, Z=Z2, vfixed=vfixed)
    assert np.allclose(x_s, x_z, atol=1e-9)
    assert np.allclose(x_s, [8.0, 0.0, 4.0, 4.0], atol=1e-6)


def test_missing_operator_errors():
    try:
        fbapro(V2)
    except ValueError:
        return
    raise AssertionError("expected ValueError when neither S nor Z is given")


def test_fixed_values_general():
    # pin v2 -> 0 (knockout) and v3 -> 5 (e.g. a measured flux); both held exactly
    x = fbapro(V2, S=S2, vfixed=np.array([np.nan, 0.0, 5.0, np.nan]))
    assert np.allclose(S2 @ x, 0.0, atol=1e-9)
    assert abs(x[1]) < 1e-12 and abs(x[2] - 5.0) < 1e-12


def test_weighting_holds_trusted_fluxes():
    vfixed = np.array([np.nan, 0.0, np.nan, np.nan])   # KO v2
    x_eq = fbapro(V2, S=S2, vfixed=vfixed)
    assert np.allclose(x_eq, [8.0, 0.0, 4.0, 4.0], atol=1e-6)   # equal weights spread evenly
    w = np.array([1.0, 1.0, 100.0, 0.01])              # hold v3 high, free v4
    x_w = fbapro(V2, S=S2, vfixed=vfixed, weights=w)
    assert np.allclose(S2 @ x_w, 0.0, atol=1e-6)
    assert abs(x_w[1]) < 1e-9                            # KO stays exactly 0
    assert abs(x_w[2] - 3.0) < abs(x_eq[2] - 3.0)       # v3 held closer than equal-weight
    assert abs(x_w[2] - 3.0) < abs(x_w[3] - 3.0)        # v3 moves less than v4


def test_batch_projection():
    V = np.stack([V2, np.array([12.0, 4.0, 4.0, 4.0])])   # two samples at once
    X = fbapro(V, Z=Z2, vfixed=np.array([np.nan, 0.0, np.nan, np.nan]))
    assert X.shape == (2, 4)
    assert np.allclose(X[:, 1], 0.0, atol=1e-12)          # KO zero in every sample
    for row in X:
        assert np.allclose(S2 @ row, 0.0, atol=1e-9)


# ── bounded MoMA (bounds-respecting sibling) ──────────────────────────────────

def test_bounded_matches_unbounded_when_no_bound_binds():
    # KO of v2 gives [8,0,8,8] -- all positive, so bounds don't bind
    x = bounded_moma(V_WT, LB, UB, S=S, vfixed=KO_V2)
    assert np.allclose(x, [8.0, 0.0, 8.0, 8.0], atol=1e-6)


def test_bounded_respects_a_binding_capacity():
    ub = UB.copy()
    ub[3] = 6.0                                          # cap secretion v4 at 6
    x = bounded_moma(V_WT, LB, ub, S=S, vfixed=KO_V2)
    assert np.allclose(S @ x, 0.0, atol=1e-6)
    assert x[3] <= 6.0 + 1e-6
    assert np.allclose(x, [6.0, 0.0, 6.0, 6.0], atol=1e-5)


def test_bounded_never_violates_bounds():
    x = bounded_moma(V_WT, LB, UB, S=S, vfixed=KO_V2)
    assert (x >= LB - 1e-6).all() and (x <= UB + 1e-6).all()


def test_bounded_weighting_holds_trusted_fluxes():
    lb2, ub2 = np.zeros(4), np.full(4, 1000.0)
    vfixed = np.array([np.nan, 0.0, np.nan, np.nan])    # KO v2
    x_eq = bounded_moma(V2, lb2, ub2, S=S2, vfixed=vfixed)
    assert np.allclose(x_eq, [8.0, 0.0, 4.0, 4.0], atol=1e-4)
    w = np.array([1.0, 1.0, 100.0, 0.01])              # hold v3, free v4
    x_w = bounded_moma(V2, lb2, ub2, S=S2, vfixed=vfixed, weights=w)
    assert np.allclose(S2 @ x_w, 0.0, atol=1e-4)
    assert abs(x_w[2] - 3.0) < abs(x_eq[2] - 3.0)       # v3 held closer than equal-weight
    assert abs(x_w[2] - 3.0) < abs(x_w[3] - 3.0)        # v3 moves less than v4
