"""Pin the hierarchical biomass split against the shipped S. cerevisiae model.

sc_annotated_hierarchical.json was produced out-of-band before
`split_biomass_hierarchical` existed, and it backs the released s_cerevisiae
ensemble. Reconstructing it from the flat model is therefore the reference test for
the split: if the recipe drifts, a newly prepared organism would no longer be built
the same way as the two that ship.
"""
import json
import os

import pytest
from cobra.io import load_json_model

from sigmaflux.core.utils import (
    BIOMASS_COMPONENT_IDS,
    _COMPONENT_TO_BM,
    derive_biomass_fractions,
    split_biomass_hierarchical,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC_DIR = os.path.join(REPO_ROOT, "models", "atomistic", "S_cerevisiae")
SC_FLAT = os.path.join(SC_DIR, "sc_annotated.json")
SC_HIERARCHICAL = os.path.join(SC_DIR, "sc_annotated_hierarchical.json")


@pytest.fixture(scope="module")
def reference():
    with open(SC_HIERARCHICAL) as f:
        model = json.load(f)
    return {r["id"]: r["metabolites"] for r in model["reactions"]}


def test_split_reproduces_shipped_model(reference):
    fractions = {c: -reference["BIOMASS"][bm] for c, bm in _COMPONENT_TO_BM.items()}

    model = load_json_model(SC_FLAT)
    split_biomass_hierarchical(model, fractions)

    for rxn_id in list(BIOMASS_COMPONENT_IDS) + ["BIOMASS"]:
        built = {m.id: c for m, c in model.reactions.get_by_id(rxn_id).metabolites.items()}
        assert built.keys() == reference[rxn_id].keys(), rxn_id
        for met_id, coeff in reference[rxn_id].items():
            assert built[met_id] == pytest.approx(coeff, abs=1e-9), f"{rxn_id}:{met_id}"


def test_biomass_is_only_pseudo_metabolites(reference):
    model = load_json_model(SC_FLAT)
    split_biomass_hierarchical(model, {c: -reference["BIOMASS"][bm]
                                       for c, bm in _COMPONENT_TO_BM.items()})

    biomass = model.reactions.get_by_id("BIOMASS")
    assert all(m.id.startswith("bm_") for m in biomass.metabolites)


def test_derived_fractions_sum_to_one():
    model = load_json_model(SC_FLAT)
    formulas = {"ala__L_c": "C3H7NO2", "atp_c": "C10H12N5O13P3", "utp_c": "C9H11N2O15P3"}
    fractions = derive_biomass_fractions(model, formulas)

    assert set(fractions) == set(BIOMASS_COMPONENT_IDS)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_split_rejects_an_already_split_model(reference):
    model = load_json_model(SC_FLAT)
    fractions = {c: -reference["BIOMASS"][bm] for c, bm in _COMPONENT_TO_BM.items()}
    split_biomass_hierarchical(model, fractions)

    with pytest.raises(ValueError, match="already has component reactions"):
        split_biomass_hierarchical(model, fractions)
