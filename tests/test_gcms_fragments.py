"""GC/MS fragment pools: does a fragment carry exactly its retained carbons?

``A`` is made from the tracer ``S`` (labeled at C1-C2) by two routes: one that keeps the
carbon order and one that reverses it. Whole ``A`` is M+2 either way, so only a fragment
can tell the routes apart -- C2-C3 reads M+1 from the first and M+2 from the second --
which is what makes the expected fragment MIDs closed-form in the route split ``x``.
"""

import os

import cobra
import numpy as np
import pandas as pd
import pytest
import torch
from cobra.io.json import load_json_model, save_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.measurement_pools import (add_fragment_pools, check_mapping_csv,
                                              fragment_pools, measurement_token,
                                              pool_members)
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU

FRAGMENTS = {"A_C23": ("A", [2, 3]), "A_C12": ("A", [1, 2]), "A_all": ("A", [1, 2, 3])}


def _expected(x):
    """Closed-form MIDs, M+0..M+n, at route split *x*."""
    return {"A": np.array([0.0, 0.0, 1.0, 0.0]),
            "A_C23": np.array([0.0, x, 1.0 - x]),
            "A_C12": np.array([0.0, 1.0 - x, x]),
            "A_all": np.array([0.0, 0.0, 1.0, 0.0])}


@pytest.fixture
def fragment_model():
    model = cobra.Model("fragments")
    for met_id in ("S", "A"):
        met = cobra.Metabolite(met_id)
        met.annotation = {"atom_pattern": "abc"}
        model.add_metabolites([met])
    m = {i.id: i for i in model.metabolites}
    for rxn_id, stoich, mapping in [
            ("EX_S", {m["S"]: 1}, " -> S (abc)"),
            ("R_keep", {m["S"]: -1, m["A"]: 1}, "S (abc) -> A (abc)"),
            ("R_flip", {m["S"]: -1, m["A"]: 1}, "S (abc) -> A (cba)"),
            ("DM_A", {m["A"]: -1}, " A (abc) -> ")]:
        rxn = cobra.Reaction(rxn_id)
        rxn.lower_bound, rxn.upper_bound = 0, 1000
        model.add_reactions([rxn])
        rxn.add_metabolites(stoich)
        rxn.annotation = {"atom_mapping": mapping}
    add_fragment_pools(model, FRAGMENTS)
    return model


def _fluxes(x):
    v = {"EX_S": 1.0, "R_keep": x, "R_flip": 1.0 - x, "DM_A": 1.0}
    for frag in FRAGMENTS:
        v[f"POOL_{frag}"] = v[f"POOL_SK_{frag}"] = 1.0
    return pd.Series(v)


def _simulate(model, fluxes, tmp_path):
    """In-house EMU MIDs, M+0..M+n, for ``A`` and every fragment."""
    mfa = MFAModel(model, "test", initial_flux_distribution=fluxes)
    patterns = {m.id: m.annotation["atom_pattern"] for m in model.metabolites}
    met_ids = ["A", *FRAGMENTS]
    targets = {EMU(mfa.metabolites.get_by_id(i), range(1, len(patterns[i]) + 1))
               for i in met_ids}
    tracer = Tracer(mfa.metabolites.get_by_id("S"), 1.0, [1, 1, 0])
    experiment = Experiment(mid_data=MassIsotopomerData(), tracers=[tracer],
                            fluxes=None, enrichment=None)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        mfa.prepare([experiment], targets=targets, with_backward=False)
        with torch.no_grad():
            simulated = mfa().detach().cpu().numpy()
    finally:
        os.chdir(cwd)
    return {i: simulated[:len(patterns[i]) + 1, mfa._metabolite_hash[i]]
            for i in met_ids}


# ── Construction ──────────────────────────────────────────────────────────────

def test_fragment_pool_shape(fragment_model):
    frag = fragment_model.metabolites.get_by_id("A_C23")
    assert frag.annotation["atom_pattern"] == "ab"
    leg = fragment_model.reactions.get_by_id("POOL_A_C23")
    assert leg.annotation["atom_mapping"] == "A (abc) -> A_C23 (bc)"
    assert leg.annotation["zero_stoichiometry"] == ["A"]
    assert fragment_model.reactions.get_by_id("POOL_SK_A_C23").bounds == (1.0, 1.0)


def test_fragments_do_not_consume_their_member(fragment_model):
    S = create_stoichiometric_matrix(fragment_model)
    met_ix = {m.id: i for i, m in enumerate(fragment_model.metabolites)}
    rxn_ix = {r.id: i for i, r in enumerate(fragment_model.reactions)}
    for frag in FRAGMENTS:
        assert S[met_ix["A"], rxn_ix[f"POOL_{frag}"]] == 0.0
    v = _fluxes(0.3)[[r.id for r in fragment_model.reactions]].to_numpy()
    assert np.abs(S @ v).max() < 1e-12


def test_fragments_survive_the_json_round_trip(fragment_model, tmp_path):
    path = str(tmp_path / "fragments.json")
    save_json_model(fragment_model, path)
    assert fragment_pools(load_json_model(path)) == {
        frag: (member, carbons) for frag, (member, carbons) in FRAGMENTS.items()}


def test_a_member_is_not_measured_as_its_fragment(fragment_model):
    """A member has several fragments, so none of them may alias it."""
    assert measurement_token(fragment_model, "A") == "A"
    assert measurement_token(fragment_model, "A_C23") == "A_C23"
    assert check_mapping_csv(fragment_model, [("A232", "A_C23"), ("A", "A")]) == []
    assert set(pool_members(fragment_model)) == set(FRAGMENTS)


@pytest.mark.parametrize("carbons", [[0, 1], [3, 4], []])
def test_carbons_outside_the_member_are_rejected(fragment_model, carbons):
    with pytest.raises(ValueError, match="outside"):
        add_fragment_pools(fragment_model, {"bad": ("A", carbons)})


def test_a_partial_fragment_of_a_symmetric_member_is_rejected(fragment_model):
    fragment_model.metabolites.get_by_id("S").annotation["symmetries"] = ["abc.cba"]
    with pytest.raises(ValueError, match="symmetric"):
        add_fragment_pools(fragment_model, {"S_C12": ("S", [1, 2])})
    add_fragment_pools(fragment_model, {"S_all": ("S", [1, 2, 3])})


# ── The fragment is exactly its retained carbons ─────────────────────────────

@pytest.mark.parametrize("x", [0.2, 0.5, 0.9])
def test_emu_fragment_mid_is_the_retained_carbons(fragment_model, tmp_path, x):
    mid = _simulate(fragment_model, _fluxes(x), tmp_path)
    for met_id, expected in _expected(x).items():
        np.testing.assert_allclose(mid[met_id], expected, atol=1e-12, rtol=0)


# ── Natural-isotope correction ────────────────────────────────────────────────

def test_correction_recovers_the_tracer_label():
    from sigmaflux.utils.natural_abundance import correct_mid, correction_matrix
    label = np.array([0.4, 0.15, 0.45])
    raw = correction_matrix("C10H26NOSi2+", 2, 5) @ label
    np.testing.assert_allclose(correct_mid(raw, "C10H26NOSi2+", 2), label / label.sum(),
                               atol=1e-10, rtol=0)


def test_an_unlabeled_ion_corrects_to_pure_m0():
    """Column 0 is the natural distribution of the whole ion, backbone carbons included."""
    from sigmaflux.utils.natural_abundance import correct_mid, correction_matrix
    raw = correction_matrix("C11H26NO2Si2+", 3, 6)[:, 0]
    assert raw[1] > 0.1                                   # Si/C isotopes are not small
    np.testing.assert_allclose(correct_mid(raw, "C11H26NO2Si2+", 3), [1, 0, 0, 0],
                               atol=1e-10, rtol=0)


def test_formula_parsing():
    from sigmaflux.utils.natural_abundance import parse_formula
    assert parse_formula("C17H40NO3Si3+") == {"C": 17, "H": 40, "N": 1, "O": 3, "Si": 3}
