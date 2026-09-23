"""ReusableSampler builds one cobra sampler and only mutates bounds afterward.

Regression tests for the refactor that replaced "build a fresh cobra sampler per
knockout/stratum" with "build once, then mutate only the changed reaction's bounds in the
sampler's cached Problem arrays" (see reusable_sampler.py's module docstring for why).
"""

import os
import sys

import cobra
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.pipeline.reusable_sampler import ReusableSampler  # noqa: E402
from sigmaflux.pipeline.sample_flux import screen_nonessential  # noqa: E402


def _reaction(model, rid, stoich, lb, ub):
    r = cobra.Reaction(rid)
    r.bounds = (lb, ub)
    model.add_reactions([r])
    r.add_metabolites({model.metabolites.get_by_id(m): c for m, c in stoich.items()})
    return r


def _branched_model(with_alt_route=True):
    """IN -> A -> B -> C -> OUT, with a parallel B->C route (R3) unless disabled.

    OUT has a nonzero lower bound so the trivial all-zero flux isn't a feasible KO
    dodge -- knocking out the only B->C route actually makes the model infeasible.
    """
    m = cobra.Model("toy")
    m.add_metabolites([cobra.Metabolite(i) for i in ("A", "B", "C")])
    _reaction(m, "IN", {"A": 1}, 0, 10)
    _reaction(m, "R1", {"A": -1, "B": 1}, 0, 10)
    _reaction(m, "R2", {"B": -1, "C": 1}, 0, 10)
    if with_alt_route:
        _reaction(m, "R3", {"B": -1, "C": 1}, 0, 10)
    _reaction(m, "OUT", {"C": -1}, 1, 10)
    return m


def test_set_bounds_mutates_without_rebuild():
    sampler = ReusableSampler(_branched_model(), method="achr", thinning=1, seed=0)
    inner, problem = sampler._sampler, sampler._sampler.problem

    sampler.set_bounds({"R2": (0, 0)})

    assert sampler._sampler is inner, "set_bounds rebuilt the sampler instead of mutating it"
    assert sampler._sampler.problem is problem, "set_bounds rebuilt the Problem instead of mutating it"
    assert sampler.model.reactions.get_by_id("R2").bounds == (0, 0)
    fwd_idx, rev_idx = sampler._col["R2"]
    assert tuple(problem.variable_bounds[:, fwd_idx]) == (0, 0)
    assert tuple(problem.variable_bounds[:, rev_idx]) == (0, 0)
    assert problem.variable_fixed[fwd_idx] and problem.variable_fixed[rev_idx]


def test_ko_produces_near_zero_flux_for_that_reaction():
    sampler = ReusableSampler(_branched_model(), method="achr", thinning=1, seed=0)
    sampler.set_bounds({"R2": (0, 0)})

    samples = sampler.sample(200, seed=0)

    assert samples["R2"].abs().max() < 1e-6
    for rxn_id in ("R1", "R3", "OUT"):
        assert samples[rxn_id].std() > 1e-8, f"{rxn_id} shows no variance after R2's KO"


def test_temporary_bounds_restores_original_state():
    sampler = ReusableSampler(_branched_model(), method="achr", thinning=1, seed=0)
    orig = sampler.model.reactions.get_by_id("R2").bounds

    with sampler.temporary_bounds({"R2": (0, 0)}):
        assert sampler.model.reactions.get_by_id("R2").bounds == (0, 0)
    assert sampler.model.reactions.get_by_id("R2").bounds == orig

    with pytest.raises(RuntimeError):
        with sampler.temporary_bounds({"R2": (0, 0)}):
            raise RuntimeError("boom")
    assert sampler.model.reactions.get_by_id("R2").bounds == orig, \
        "bounds were not restored after an exception"


def test_sample_output_contract():
    model = _branched_model()
    sampler = ReusableSampler(model, method="achr", thinning=1, seed=0)

    samples = sampler.sample(50, seed=1)

    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (50, len(model.reactions))
    assert list(samples.columns) == [r.id for r in model.reactions]


def test_seed_reproducibility_contract():
    sampler = ReusableSampler(_branched_model(), method="achr", thinning=1, seed=0)

    a = sampler.sample(30, seed=42)
    b = sampler.sample(30, seed=42)
    pd.testing.assert_frame_equal(a, b)

    c = sampler.sample(30, seed=None)
    assert not a.equals(c), "seed=None should continue the chain, not repeat the last draw"


def _two_subnetwork_model():
    """Two independent branched subnetworks, so fixing one reaction in the A/B/C
    subnetwork to a nonzero value still leaves the rebuilt sampler enough independent
    warmup directions to construct (cobra's own ACHRSampler.__init__ refuses to build a
    sampler for a too-degenerate, direction-starved polytope -- unrelated to this
    wrapper, but it means the fallback-rebuild test needs more slack than the minimal
    branched model used elsewhere in this file).
    """
    m = _branched_model(with_alt_route=True)
    m.add_metabolites([cobra.Metabolite(i) for i in ("D", "E")])
    _reaction(m, "IN2", {"D": 1}, 0, 10)
    _reaction(m, "R4", {"D": -1, "E": 1}, 0, 10)
    _reaction(m, "R5", {"D": -1, "E": 1}, 0, 10)
    _reaction(m, "OUT2", {"E": -1}, 0, 10)
    return m


def test_nonzero_fixed_bound_falls_back_to_rebuild():
    """Fixing a reaction to a nonzero value can't be patched in place (it changes the
    equality-constraint shape) -- confirm the guard detects it, warns, and rebuilds
    rather than silently mutating a Problem whose shape no longer matches.
    """
    sampler = ReusableSampler(_two_subnetwork_model(), method="achr", thinning=1, seed=0)
    old_inner = sampler._sampler

    with pytest.warns(RuntimeWarning, match="nonzero bound"):
        sampler.set_bounds({"R2": (5, 5)})

    assert sampler._sampler is not old_inner, "fallback should have rebuilt the sampler"
    assert sampler.model.reactions.get_by_id("R2").bounds == (5, 5)
    assert not sampler._sampler.problem.homogeneous, \
        "fixing a reaction to a nonzero value should make the rebuilt problem inhomogeneous"


def test_screen_nonessential_unaffected_by_refactor():
    nonessential_model = _branched_model(with_alt_route=True)
    essential_model = _branched_model(with_alt_route=False)

    assert "R2" in screen_nonessential(nonessential_model)
    assert "R2" not in screen_nonessential(essential_model)
