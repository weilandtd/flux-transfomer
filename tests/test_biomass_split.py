"""Pin the hierarchical biomass split for the two published yeast models.

``sc_annotated_hierarchical.json`` and ``imIsor_annotated_hierarchical.json`` were
produced out-of-band, before ``split_biomass_hierarchical`` existed, and they back
the released v1.0.0 ensembles. They are now rebuilt from their workbooks like every
other model, so the artifact on disk can no longer serve as its own reference: a
test that reads the fractions back out of the file it is checking would pass
whatever the builder produced.

So the reference lives in ``tests/data/biomass_split_v1.json``, captured from the
shipped artifacts *before* the rebuild. It holds the ``BIOMASS`` row and the five
macromolecule component rows for both organisms. A change to the split recipe now
fails here instead of silently regenerating a file.

The two organisms differ in a way worth knowing about: I. orientalis keeps an ATP
hydrolysis term in ``BIOMASS`` (atp + h2o -> amp + h + ppi) and S. cerevisiae does
not, which is why only the latter can assert "nothing but pseudo-metabolites".
"""
import json
import os

import pytest
from cobra.io import load_json_model

from sigmaflux.core.utils import (
    BIOMASS_COMPONENT_IDS,
    PUBLISHED_BIOMASS_FRACTIONS,
    _COMPONENT_TO_BM,
    derive_biomass_fractions,
    split_biomass_hierarchical,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_JSON = os.path.join(REPO_ROOT, "tests", "data", "biomass_split_v1.json")

FLAT = {
    "S_cerevisiae": os.path.join(REPO_ROOT, "models", "atomistic", "S_cerevisiae",
                                 "sc_annotated.json"),
    "I_orientalis": os.path.join(REPO_ROOT, "models", "atomistic", "I_orientalis",
                                 "imIsor_annotated.json"),
}
ORGANISMS = sorted(FLAT)


@pytest.fixture(scope="module")
def reference():
    """The pre-rebuild coefficients, per organism and reaction id."""
    with open(REFERENCE_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rebuilt():
    """The split as the builder produces it today, from the flat model."""
    out = {}
    for organism, path in FLAT.items():
        model = load_json_model(path)
        split_biomass_hierarchical(model, PUBLISHED_BIOMASS_FRACTIONS[organism])
        out[organism] = {
            rxn_id: {m.id: c for m, c in
                     model.reactions.get_by_id(rxn_id).metabolites.items()}
            for rxn_id in ("BIOMASS", *BIOMASS_COMPONENT_IDS)
        }
    return out


def test_split_reproduces_the_published_model(reference, rebuilt):
    """Rebuilding S. cerevisiae from its flat model reproduces what shipped, exactly.

    This is the justification for lifting the freeze: the workbook fixes land
    without the biomass stoichiometry moving underneath the released ensemble.
    I. orientalis is checked separately -- see
    :func:`test_i_orientalis_drops_the_hand_added_atp`.
    """
    for rxn_id in ("BIOMASS", *BIOMASS_COMPONENT_IDS):
        expected = reference["S_cerevisiae"][rxn_id]
        built = rebuilt["S_cerevisiae"][rxn_id]
        assert built.keys() == expected.keys(), rxn_id
        for met_id, coeff in expected.items():
            assert built[met_id] == pytest.approx(coeff, abs=1e-9), f"{rxn_id}:{met_id}"


@pytest.mark.parametrize("organism", ORGANISMS)
def test_macromolecule_composition_is_preserved(organism, reference, rebuilt):
    """The five mass fractions survive the rebuild for both organisms.

    Composition is what the released ensembles were trained against, so it is the
    property that must not move even where the rest of the row does.
    """
    for bm_id in _COMPONENT_TO_BM.values():
        assert rebuilt[organism]["BIOMASS"][bm_id] == pytest.approx(
            reference[organism]["BIOMASS"][bm_id], abs=1e-9), bm_id


def test_i_orientalis_drops_the_hand_added_atp(reference, rebuilt):
    """The shipped I. orientalis BIOMASS carried 1 ATP its own workbook does not.

    ``imIsor_annotated.json`` puts 217.659074 ATP into the macromolecule building
    blocks, and the component reactions the split creates carry exactly that. The
    shipped hierarchical model had a further ``atp_c + h2o_c -> amp_c + h_c +
    ppi_c`` on BIOMASS itself -- one unit, added out-of-band, derivable from
    nothing in the workbook. Rebuilding drops it, which makes growth 0.46% cheaper
    in ATP and makes the model agree with its own source.

    Pinned rather than papered over: this is the one stoichiometric difference
    between the v1.0.0 I. orientalis model and its v2.0.0 rebuild, and a reader
    comparing predictions across the two versions needs to know it is there.
    """
    shipped = reference["I_orientalis"]["BIOMASS"]
    built = rebuilt["I_orientalis"]["BIOMASS"]

    assert shipped["atp_c"] == pytest.approx(-1.0)
    assert set(shipped) - set(built) == {"atp_c", "h2o_c", "amp_c", "h_c", "ppi_c"}
    assert all(met_id.startswith("bm_") for met_id in built)

    # The building-block ATP itself is untouched: only the extra unit is gone.
    weighted = sum(rebuilt["I_orientalis"][c].get("atp_c", 0.0) * -built[bm]
                   for c, bm in _COMPONENT_TO_BM.items())
    assert weighted == pytest.approx(-217.659074, abs=1e-4)


@pytest.mark.parametrize("organism", ORGANISMS)
def test_published_fractions_match_the_shipped_biomass(organism, reference):
    """``PUBLISHED_BIOMASS_FRACTIONS`` is what the shipped BIOMASS actually says.

    The constant is the *input* to the rebuild, so if it ever drifts from the
    artifact it claims to describe, every rebuilt model silently changes
    composition. Checking it against the frozen reference costs nothing.
    """
    biomass = reference[organism]["BIOMASS"]
    for component, bm_id in _COMPONENT_TO_BM.items():
        assert -biomass[bm_id] == pytest.approx(
            PUBLISHED_BIOMASS_FRACTIONS[organism][component], abs=1e-9), component


def test_biomass_is_only_pseudo_metabolites(rebuilt):
    """S. cerevisiae's BIOMASS consumes the five bm_* pools and nothing else.

    I. orientalis is excluded on purpose: its BIOMASS also hydrolyses ATP, which
    the flat model carries and the split preserves.
    """
    biomass = rebuilt["S_cerevisiae"]["BIOMASS"]
    assert all(met_id.startswith("bm_") for met_id in biomass)


def test_derived_fractions_sum_to_one():
    model = load_json_model(FLAT["S_cerevisiae"])
    formulas = {"ala__L_c": "C3H7NO2", "atp_c": "C10H12N5O13P3", "utp_c": "C9H11N2O15P3"}
    fractions = derive_biomass_fractions(model, formulas)

    assert set(fractions) == set(BIOMASS_COMPONENT_IDS)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_split_rejects_an_already_split_model():
    model = load_json_model(FLAT["S_cerevisiae"])
    fractions = PUBLISHED_BIOMASS_FRACTIONS["S_cerevisiae"]
    split_biomass_hierarchical(model, fractions)

    with pytest.raises(ValueError, match="already has component reactions"):
        split_biomass_hierarchical(model, fractions)
