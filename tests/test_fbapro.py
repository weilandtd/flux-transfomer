"""FBApro projection and knockout prediction on a small hand-checked network."""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.core.fbapro import fbapro, predict_knockout_flux, bounded_moma  # noqa: E402

# Branched pathway:  v1: ->A,  v2: A->B,  v3: A->B (parallel),  v4: B->
# metabolites A, B
S = np.array([[1.0, -1.0, -1.0, 0.0],
              [0.0, 1.0, 1.0, -1.0]])
V_WT = np.array([10.0, 6.0, 4.0, 10.0])   # steady state: S @ V_WT == 0


def test_wildtype_is_a_fixed_point():
    # v already satisfies S v = 0, so the projection returns it unchanged
    x = fbapro(S, V_WT)
    assert np.allclose(x, V_WT)


def test_knockout_reroutes_minimally():
    # knock out v2; the parallel branch v3 must carry its flux
    x = predict_knockout_flux(S, V_WT, knockouts=[1])
    assert np.allclose(S @ x, 0.0)                 # still balanced
    assert abs(x[1]) < 1e-9                          # v2 is exactly zero
    assert np.allclose(x, [8.0, 0.0, 8.0, 8.0])     # hand-computed minimal adjustment


def test_basic_projection_would_revive_the_knockout():
    # zeroing v2 in the vector then projecting (FBAproBasic) does NOT keep it at zero,
    # which is why the knockout must go in the fixed set
    v_zeroed = V_WT.copy()
    v_zeroed[1] = 0.0
    x = fbapro(S, v_zeroed)                          # no fixed set
    assert abs(x[1]) > 1e-6


LB = np.zeros(4)                    # all reactions irreversible (>= 0)
UB = np.full(4, 1000.0)


def test_bounded_matches_unbounded_when_no_bound_binds():
    # KO of v2 gives [8,0,8,8] -- all positive, so bounds don't bind
    x = predict_knockout_flux(S, V_WT, knockouts=[1], lower=LB, upper=UB)
    assert np.allclose(x, [8.0, 0.0, 8.0, 8.0], atol=1e-6)


def test_bounded_respects_a_binding_capacity():
    # cap secretion v4 at 6; the bounded solution must honor it and stay balanced
    ub = UB.copy()
    ub[3] = 6.0
    x = predict_knockout_flux(S, V_WT, knockouts=[1], lower=LB, upper=ub)
    assert np.allclose(S @ x, 0.0, atol=1e-6)
    assert x[3] <= 6.0 + 1e-6
    assert np.allclose(x, [6.0, 0.0, 6.0, 6.0], atol=1e-5)


def test_bounded_never_violates_bounds():
    x = bounded_moma(S, V_WT, LB, UB, fixed=[1], fixed_values=[0.0])
    assert (x >= LB - 1e-6).all() and (x <= UB + 1e-6).all()
