"""Pools left without a consumer get a demand reaction, not silent deletion.

Regression tests for `drop_skip_constraints`, which used to decide "dead end"
from the stoichiometric sign (`all(r.metabolites[m] > 0 ...)`) and then repair it
with `remove_metabolites`. Both halves were wrong:

* the sign says nothing about whether a pool is consumed -- a reversible reaction
  consumes its products on reverse flux -- so well-balanced pools were stranded;
* `remove_metabolites` is non-destructive, so it stripped the pool out of its
  reactions and left them unbalanced. That is an invisible, unbounded sink.
"""

import os
import sys

import cobra
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.pipeline.sample_flux import can_consume, drop_skip_constraints  # noqa: E402


def _reaction(model, rid, stoich, lb, ub):
    r = cobra.Reaction(rid)
    r.bounds = (lb, ub)
    model.add_reactions([r])
    r.add_metabolites({model.metabolites.get_by_id(m): c for m, c in stoich.items()})
    return r


def _model(met_ids=("A", "B", "C")):
    m = cobra.Model("toy")
    m.add_metabolites([cobra.Metabolite(i) for i in met_ids])
    return m


def test_reversible_producer_counts_as_a_consumer():
    """B is a product of a reversible reaction, so reverse flux consumes it."""
    m = _model(("A", "B"))
    _reaction(m, "IN", {"A": 1}, 0, 10)
    rev = _reaction(m, "R", {"A": -1, "B": 1}, -10, 10)
    assert can_consume(rev, m.metabolites.B)


def test_irreversible_producer_is_not_a_consumer():
    m = _model(("A", "B"))
    fwd = _reaction(m, "R", {"A": -1, "B": 1}, 0, 10)
    assert not can_consume(fwd, m.metabolites.B)
    assert can_consume(fwd, m.metabolites.A)


def test_pool_kept_when_a_reversible_reaction_can_consume_it():
    """The old sign test deleted B here; it is perfectly well balanced."""
    m = _model(("A", "B"))
    _reaction(m, "IN", {"A": 1}, 0, 10)
    _reaction(m, "R", {"A": -1, "B": 1}, -10, 10)   # reversible: consumes B backwards
    _reaction(m, "BIOMASS", {"B": -1}, 0, 10)

    drop_skip_constraints(m, ["BIOMASS"])

    assert "B" in [x.id for x in m.metabolites], "reversible consumer was ignored"
    assert "DM_B" not in [r.id for r in m.reactions], "no demand should be needed"


def test_stranded_pool_gets_a_demand_and_keeps_its_reactions_balanced():
    """Dropping BIOMASS strands B; it must be drained, not deleted."""
    m = _model(("A", "B"))
    _reaction(m, "IN", {"A": 1}, 0, 10)
    producer = _reaction(m, "R", {"A": -1, "B": 1}, 0, 10)   # irreversible
    _reaction(m, "BIOMASS", {"B": -1}, 0, 10)

    drop_skip_constraints(m, ["BIOMASS"])

    assert "B" in [x.id for x in m.metabolites], "pool was deleted instead of drained"
    assert "DM_B" in [r.id for r in m.reactions], "no demand reaction added"
    assert m.metabolites.B in producer.metabolites, \
        "producer was silently unbalanced -- an invisible sink"


def test_stranded_pool_stays_solvable():
    """The point of the demand: the producer can still carry flux."""
    m = _model(("A", "B"))
    _reaction(m, "IN", {"A": 1}, 0, 10)
    _reaction(m, "R", {"A": -1, "B": 1}, 0, 10)
    _reaction(m, "BIOMASS", {"B": -1}, 0, 10)

    drop_skip_constraints(m, ["BIOMASS"])

    m.objective = m.reactions.R
    assert m.optimize().objective_value == pytest.approx(10.0), \
        "producer forced to zero -- the pool is still stranded"


def test_orphans_are_not_treated_as_stranded():
    """A metabolite with no reactions must not acquire a demand."""
    m = _model(("A", "B", "C"))
    _reaction(m, "IN", {"A": 1}, 0, 10)
    _reaction(m, "OUT", {"A": -1}, 0, 10)
    drop_skip_constraints(m, [])
    assert "DM_C" not in [r.id for r in m.reactions]
