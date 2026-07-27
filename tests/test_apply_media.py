"""``apply_media`` must support both boundary conventions.

For an organism in a defined medium, anything not in the medium is unavailable,
so every unlisted exchange is closed. For a whole-body or multi-tissue model the
``EX_`` reactions are internal physiology — inter-tissue exchange — and the
media block instead lists the fluxes that were *measured*. Closing the rest
there makes the model infeasible.
"""

import cobra
import pytest

from sigmaflux.pipeline.sample_flux import apply_media


@pytest.fixture
def model():
    """Minimal model: one medium exchange, one unrelated exchange, one internal."""
    m = cobra.Model("toy")
    a = cobra.Metabolite("a_e")
    b = cobra.Metabolite("b_e")
    m.add_metabolites([a, b])

    ex_a = cobra.Reaction("EX_a_e", lower_bound=-10, upper_bound=10)
    ex_a.add_metabolites({a: -1})
    ex_b = cobra.Reaction("EX_b_e", lower_bound=-7, upper_bound=7)
    ex_b.add_metabolites({b: -1})
    internal = cobra.Reaction("R_ab", lower_bound=-3, upper_bound=3)
    internal.add_metabolites({a: -1, b: 1})

    m.add_reactions([ex_a, ex_b, internal])
    return m


MEDIA = {"EX_a_e": {"lb": -5.0, "ub": 0.0}}


def test_closes_unlisted_exchanges_by_default(model):
    apply_media(model, MEDIA)
    assert model.reactions.EX_a_e.bounds == (-5.0, 0.0)
    assert model.reactions.EX_b_e.bounds == (0, 0), \
        "unlisted exchange should be closed under the defined-medium convention"


def test_close_unlisted_false_leaves_other_exchanges_alone(model):
    """The whole-body case: constrain what was measured, touch nothing else."""
    apply_media(model, MEDIA, close_unlisted=False)
    assert model.reactions.EX_a_e.bounds == (-5.0, 0.0)
    assert model.reactions.EX_b_e.bounds == (-7, 7), \
        "unlisted exchange must keep the bounds the model was built with"


def test_non_exchange_reactions_are_never_closed(model):
    """Only EX_-prefixed reactions are subject to the closing rule."""
    apply_media(model, MEDIA)
    assert model.reactions.R_ab.bounds == (-3, 3)


def test_media_may_constrain_non_exchange_reactions(model):
    """Measured fluxes are not all exchanges.

    Whole-body measurements include protein degradation and arterial delivery
    reactions that carry no EX_ prefix, so media entries must apply to any
    reaction id.
    """
    apply_media(model, {"R_ab": {"lb": 1.0, "ub": 2.0}}, close_unlisted=False)
    assert model.reactions.R_ab.bounds == (1.0, 2.0)


def test_unknown_media_reactions_are_reported_not_fatal(model, capsys):
    """A renamed reaction should warn rather than abort a long sampling run."""
    apply_media(model, {"EX_nope_e": {"lb": -1.0, "ub": 0.0}}, close_unlisted=False)
    assert "not in the model" in capsys.readouterr().out
