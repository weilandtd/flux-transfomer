"""Whole-cell measurement pools: is the pool exactly the mixture, and is S untouched?

The construction only works if two things hold at once, and each fails silently on its
own. The pool must *not* appear in its members' mass balance -- a zero coefficient COBRA
cannot even store, so it lives in an annotation -- and it must still appear in their
labeling balance, which means every walk of a reaction's participants that follows carbon
has to go through ``labeling_educts`` rather than ``rxn.reactants``. Miss the second and
the pool gets no incoming EMU edge, is classified as a network boundary, and is pinned to
pure M+0 without any error.

The x3cflux section pins the other encoding of the same zero. That engine has no zero
stoichiometry, but it drops any reaction that both requires and produces a metabolite
from that metabolite's balance -- mass and label alike -- so the tap is written
``member + carrier -> pool + member`` and the zero comes out of the reaction's shape.
"""

import cobra
import numpy as np
import pandas as pd
import pytest
import torch
from cobra.io.json import load_json_model, save_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.measurement_pools import (
    add_measurement_pools, check_mapping_csv, degenerate_pool_members,
    measurement_token, plan_measurement_pools, pool_members, pool_reaction_ids,
)
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU
from sigmaflux.expansion.utils import labeling_educts, zero_stoichiometry_educts


def _met(met_id, pattern="abc"):
    met = cobra.Metabolite(met_id)
    met.annotation = {"atom_pattern": pattern}
    return met


def _rxn(model, rxn_id, stoichiometry, atom_mapping):
    rxn = cobra.Reaction(rxn_id)
    rxn.lower_bound, rxn.upper_bound = 0, 1000
    model.add_reactions([rxn])
    rxn.add_metabolites(stoichiometry)
    rxn.annotation = {"atom_mapping": atom_mapping}
    return rxn


@pytest.fixture
def two_compartment_model():
    """``A_c`` fully labeled, ``A_m`` half labeled -- so a mixture of them is not either.

    ``S`` supplies the tracer; ``U`` is an unlabeled boundary. ``A_c`` is made from ``S``
    alone while ``A_m`` is made half from ``S`` and half from ``U``, which is what keeps
    the mixture test from being vacuous. The pools are derived from the model, as they
    are for a real organism -- nothing here names ``A`` or its members.
    """
    model = cobra.Model("two_compartment")
    model.add_metabolites([_met("S"), _met("U"), _met("A_c"), _met("A_m")])
    m = {i.id: i for i in model.metabolites}
    _rxn(model, "EX_S", {m["S"]: 1}, " -> S (abc)")
    _rxn(model, "EX_U", {m["U"]: 1}, " -> U (abc)")
    _rxn(model, "TC", {m["S"]: -1, m["A_c"]: 1}, "S (abc) -> A_c (abc)")
    _rxn(model, "TM1", {m["S"]: -1, m["A_m"]: 1}, "S (abc) -> A_m (abc)")
    _rxn(model, "TM2", {m["U"]: -1, m["A_m"]: 1}, "U (abc) -> A_m (abc)")
    _rxn(model, "DM_Ac", {m["A_c"]: -1}, " A_c (abc) -> ")
    _rxn(model, "DM_Am", {m["A_m"]: -1}, " A_m (abc) -> ")
    add_measurement_pools(model)
    return model


def _fluxes(v_c, v_m):
    return pd.Series({"EX_S": 3, "EX_U": 1, "TC": 2, "TM1": 1, "TM2": 1,
                      "DM_Ac": 2, "DM_Am": 2,
                      "POOL_A_c": v_c, "POOL_A_m": v_m, "POOL_SK_A": v_c + v_m})


def _simulate(model, fluxes, met_ids, tmp_path):
    """MIDs at *fluxes* as ``{metabolite_id: array}``, indexed M+0..M+n."""
    import os
    mfa = MFAModel(model, "test", initial_flux_distribution=fluxes)
    targets = {EMU(mfa.metabolites.get_by_id(i), [1, 2, 3]) for i in met_ids}
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
    return {i: simulated[:4, mfa._metabolite_hash[i]] for i in met_ids}


# ── The pool must not enter its members' mass balance ─────────────────────────

def test_pool_does_not_consume_its_members(two_compartment_model):
    """The mixing leg's column is empty in every member's row.

    This is the whole point of the zero coefficient: the pool observes the compartments
    it measures without drawing flux from them, so adding it cannot move the metabolic
    solution at all.
    """
    model = two_compartment_model
    S = create_stoichiometric_matrix(model)
    met_ix = {m.id: i for i, m in enumerate(model.metabolites)}
    rxn_ix = {r.id: i for i, r in enumerate(model.reactions)}
    for member in ("A_c", "A_m"):
        for other in ("A_c", "A_m"):
            assert S[met_ix[other], rxn_ix[f"POOL_{member}"]] == 0.0


def test_pool_block_is_mass_balanced(two_compartment_model):
    v = _fluxes(0.3, 0.7)
    model = two_compartment_model
    S = create_stoichiometric_matrix(model)
    assert np.abs(S @ v[[r.id for r in model.reactions]].to_numpy()).max() < 1e-12


def test_zero_stoichiometry_survives_the_json_round_trip(two_compartment_model, tmp_path):
    """COBRA drops a zero coefficient outright, which is why it lives in the annotation."""
    path = str(tmp_path / "pooled.json")
    save_json_model(two_compartment_model, path)
    reloaded = load_json_model(path)
    leg = reloaded.reactions.get_by_id("POOL_A_c")
    assert zero_stoichiometry_educts(leg) == ("A_c",)
    assert "A_c" not in {m.id for m in leg.metabolites}
    assert [m.id for m in labeling_educts(leg)] == ["A_c"]
    assert pool_members(reloaded) == {"A": ["A_c", "A_m"]}


# ── The pool must be exactly the mixture ──────────────────────────────────────

@pytest.mark.parametrize("v_c,v_m", [(0.3, 0.7), (0.5, 0.5), (0.9, 0.1)])
def test_pool_mid_is_the_exact_flux_weighted_mixture(two_compartment_model, tmp_path,
                                                     v_c, v_m):
    """``MID(pool) == v_c MID(A_c) + v_m MID(A_m)``, to machine precision.

    The tolerance is deliberately tight rather than approximate: with the drain pinning
    the mixing fluxes to sum to one, the pool's EMU row is a single linear equation with
    -1 on the diagonal, so the mixture is exact arithmetic and not a fit.
    """
    mid = _simulate(two_compartment_model, _fluxes(v_c, v_m),
                    ["A_c", "A_m", "A"], tmp_path)
    assert not np.allclose(mid["A_c"], mid["A_m"])       # otherwise vacuous
    np.testing.assert_allclose(mid["A"], v_c * mid["A_c"] + v_m * mid["A_m"],
                               atol=1e-12, rtol=0)
    assert abs(mid["A"].sum() - 1.0) < 1e-12


def test_pool_is_not_mistaken_for_a_tracer_input(two_compartment_model, tmp_path):
    """The pool has an incoming EMU edge, so it is not a network boundary.

    Its mixing leg has no *stoichiometric* educt, so a boundary check reading
    ``rxn.reactants`` classifies the pool as a network input and hands it pure M+0 --
    silently, since a pinned input is a legitimate state.
    """
    mid = _simulate(two_compartment_model, _fluxes(0.5, 0.5), ["A"], tmp_path)
    assert not np.allclose(mid["A"], [1.0, 0.0, 0.0, 0.0])


def test_a_silent_compartment_leaves_the_pool_equal_to_the_other(two_compartment_model,
                                                                 tmp_path):
    mid = _simulate(two_compartment_model, _fluxes(1.0, 0.0), ["A_c", "A"], tmp_path)
    np.testing.assert_allclose(mid["A"], mid["A_c"], atol=1e-12, rtol=0)


# ── Planning is a property of the model ───────────────────────────────────────

def test_plan_pools_every_metabolite_held_in_several_compartments():
    model = cobra.Model("t")
    model.add_metabolites([_met("pyr_c"), _met("pyr_m"), _met("glc_c")])
    assert plan_measurement_pools(model) == {"pyr": ["pyr_c", "pyr_m"]}


def test_plan_makes_no_pool_for_a_single_compartment_metabolite():
    """A one-member pool is identically its member at any flux, so it buys nothing.

    It would still cost a metabolite, a leg, a drain and a scoring exclusion, and it is
    the difference between 15 free parameters and 80 on I. orientalis.
    """
    model = cobra.Model("t")
    model.add_metabolites([_met("glc_c"), _met("glc_e")])
    assert plan_measurement_pools(model) == {}


def test_plan_excludes_the_medium_and_the_periplasm():
    """``_e`` is not in a cell extract and ``_p`` is not a real labeled space.

    Pooling ``_e`` would also mix the tracer inlet into a measured pool; the periplasm
    equilibrates with the medium through a porous outer membrane.
    """
    model = cobra.Model("t")
    model.add_metabolites([_met("fum_c"), _met("fum_e"), _met("fum_p"), _met("fum_m")])
    assert plan_measurement_pools(model) == {"fum": ["fum_c", "fum_m"]}


def test_plan_skips_members_that_carry_no_atom_pattern():
    """The yeast models hold untracked copies the atom mapping never reaches.

    ``pyr_i``/``pyr_d1``/``pyr_d2`` exist in I. orientalis but carry no label, so mixing
    them in would dilute the pool with a metabolite the expansion cannot even simulate.
    """
    model = cobra.Model("t")
    model.add_metabolites([_met("pyr_c"), _met("pyr_m"), _met("pyr_d1", pattern="")])
    assert plan_measurement_pools(model) == {"pyr": ["pyr_c", "pyr_m"]}


def test_plan_rejects_members_of_different_carbon_count():
    model = cobra.Model("t")
    model.add_metabolites([_met("x_c", "abc"), _met("x_m", "abcd")])
    with pytest.raises(ValueError, match="disagree on carbon count"):
        plan_measurement_pools(model)


def test_building_twice_is_rejected_rather_than_silently_doubling():
    model = cobra.Model("t")
    model.add_metabolites([_met("pyr_c"), _met("pyr_m")])
    add_measurement_pools(model)
    with pytest.raises(ValueError, match="already a metabolite"):
        add_measurement_pools(model)


def test_a_reaction_already_wearing_the_prefix_is_rejected():
    """``POOL_`` is the marker the whole pipeline reads, so it cannot be ambiguous.

    A metabolic reaction wearing it would be silently reclassified as observation
    machinery: skipped by knockout screening, dropped from scoring, and rewritten by the
    x3cflux converter. The mouse T-cell model already carries a metabolic ``SK_accoa_c``,
    which is why the drain lives inside the ``POOL_`` namespace rather than using the
    conventional bare ``SK_``.
    """
    model = cobra.Model("t")
    model.add_metabolites([_met("pyr_c"), _met("pyr_m")])
    _rxn(model, "POOL_something", {model.metabolites.get_by_id("pyr_c"): 1},
         " -> pyr_c (abc)")
    with pytest.raises(ValueError, match="already use the measurement-pool prefix"):
        add_measurement_pools(model)


# ── Mapping files name the token, and are only ever checked ───────────────────

def test_a_pooled_metabolite_is_measured_as_its_pool(two_compartment_model):
    assert measurement_token(two_compartment_model, "A_c") == "A"
    assert measurement_token(two_compartment_model, "S") == "S"


def test_a_mapping_naming_a_compartment_of_a_pool_is_reported(two_compartment_model):
    """Mapping files are curated by hand, so a stale row must be named, not repaired.

    Left alone it is invisible until feature selection quietly drops the token.
    """
    assert check_mapping_csv(two_compartment_model, [("Ala", "A")]) == []
    (problem,) = check_mapping_csv(two_compartment_model, [("Ala", "A_c")])
    assert "pooled into 'A'" in problem


def test_a_mapping_naming_a_metabolite_the_model_lacks_is_reported(two_compartment_model):
    (problem,) = check_mapping_csv(two_compartment_model, [("Nope", "nope_c")])
    assert "not in the model" in problem


def test_pool_reactions_are_identifiable_for_exclusion(two_compartment_model):
    assert sorted(pool_reaction_ids(two_compartment_model)) == [
        "POOL_A_c", "POOL_A_m", "POOL_SK_A"]


# ── Constant null-space directions ────────────────────────────────────────────

def test_pinned_directions_are_split_out_of_the_predicted_basis():
    """A pool's drain is pinned, so one basis direction per pool carries no information.

    Left in, ``pinv`` over theta reads numerical noise out of it and the training loss
    gives it maximum weight on a target that is identically zero once standardised -- the
    failure that made Fisher selection return zero tokens on iAF1260.
    """
    from sigmaflux.ai.utils import split_constant_directions
    rng = np.random.default_rng(0)
    Z, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    y = np.zeros((300, 5))
    y[:, :3] = rng.normal(size=(300, 3))
    y[:, 3], y[:, 4] = 0.7, -1.3            # two pinned directions
    V = y @ Z.T

    Z_var, Z_const, y_const = split_constant_directions(Z, V)
    assert Z_const.shape[1] == 2
    reconstructed = (V @ Z_var) @ Z_var.T + y_const @ Z_const.T
    np.testing.assert_allclose(reconstructed, V, atol=1e-10, rtol=0)


def test_the_split_leaves_a_targeted_identity_block_alone():
    """Rotating the whole basis is what broke the measured fluxes when pools arrived.

    A targeted basis gives each measured reaction its own coordinate. The rotation that
    finds constant directions must not touch those columns: when it did, every measured
    flux went from owning a coordinate to being spread across the basis, and on
    I. orientalis ``DNA_c`` -- a measured *input* -- fell from validation R2 +0.996 to
    -0.225 while the non-pool median went 0.994 -> 0.595. Pools only supplied the first
    pinned directions, so the rotation fired for the first time; they were not the cause.
    """
    from sigmaflux.ai.utils import split_constant_directions
    rng = np.random.default_rng(0)
    n_targets = 3
    Q, _ = np.linalg.qr(rng.normal(size=(12, 6)))
    targeted = Q[:, :n_targets] @ rng.normal(size=(n_targets, n_targets))
    free = Q[:, n_targets:]
    Z = np.hstack([targeted, free])                 # non-orthonormal, as targeted_ica ships

    y_free = rng.normal(size=(400, 3))
    y_free[:, 2] = 2.5                              # one pinned direction, in the free block
    V = rng.normal(size=(400, n_targets)) @ targeted.T + y_free @ free.T

    Z_var, Z_const, y_const = split_constant_directions(Z, V, n_protected=n_targets)

    assert Z_const.shape[1] == 1                    # the pinned direction still comes out
    np.testing.assert_allclose(Z_var[:, :n_targets], targeted, atol=1e-12, rtol=0)
    reconstructed = (V @ (Z_var @ np.linalg.inv(Z_var.T @ Z_var))) @ Z_var.T \
        + y_const @ Z_const.T
    np.testing.assert_allclose(reconstructed, V, atol=1e-9, rtol=0)


def test_the_whole_basis_rotation_is_what_destroys_the_targeted_block():
    """The same split with no protection smears every target across the basis.

    Kept as the counter-case: it is why ``n_protected`` exists, and it reproduces the
    2026-08-29 regression in three lines.
    """
    from sigmaflux.ai.utils import split_constant_directions
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 6)))
    targeted = Q[:, :3] @ rng.normal(size=(3, 3))
    Z = np.hstack([targeted, Q[:, 3:]])
    y_free = rng.normal(size=(400, 3))
    y_free[:, 2] = 2.5
    V = rng.normal(size=(400, 3)) @ targeted.T + y_free @ Q[:, 3:].T

    unprotected, _, _ = split_constant_directions(Z, V, n_protected=0)
    assert not np.allclose(unprotected[:, :3], targeted, atol=1e-8)


def test_split_is_skipped_when_there_are_too_few_samples():
    """A rank-deficient covariance means too little data, not a pinned reaction.

    Its null eigenvectors are arbitrary, so splitting there would discard directions the
    model still has to predict.
    """
    from sigmaflux.ai.utils import split_constant_directions
    rng = np.random.default_rng(0)
    Z, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    V = rng.normal(size=(3, 5)) @ Z.T
    Z_var, Z_const, _ = split_constant_directions(Z, V)
    assert Z_const.shape[1] == 0 and Z_var.shape[1] == 5


def test_a_zeroed_member_mid_is_detected_rather_than_mixed_in(two_compartment_model):
    """A compartment nothing produces yields a zeroed MID, not an error.

    ``pinv`` on an all-zero EMU row returns zeros, so the pool would weight a meaningless
    MID by its still-nonzero mixing flux -- and because M+0 is never stored, the shortfall
    hides there instead of showing up as an invalid MID. Measured on I. orientalis:
    ``ala__L_m`` in 3.3% of sampled flux states, ``adp_m`` in 0.2%.
    """
    healthy = {"A_c": np.array([0.2, 0.3, 0.4, 0.1]),
               "A_m": np.array([0.5, 0.2, 0.2, 0.1])}
    assert degenerate_pool_members(two_compartment_model, healthy) == []

    zeroed = dict(healthy, A_m=np.zeros(4))
    assert degenerate_pool_members(two_compartment_model, zeroed) == ["A_m"]


def test_only_the_pool_drain_reopens_after_sampling(two_compartment_model):
    """The drain is pinned only so the fractions land on the simplex.

    Once the fractions are drawn the drain's absolute value carries nothing, and a
    zero-width interval is unusable downstream: the ray projection in inference needs its
    origin strictly inside the bounds, and a pinned reaction has no interior.

    The mixing legs are left alone. ``[0, 1]`` already has an interior and is what a
    compartment fraction means; widening them too asserted they could carry a hundred
    times their own meaning, and handed the MFA fit a box 100x larger than the simplex.
    """
    from sigmaflux.pipeline.sample_flux import (POOL_RELAXED_UB,
                                                relax_measurement_pool_bounds)

    model = two_compartment_model
    drain = model.reactions.get_by_id("POOL_SK_A")
    leg = model.reactions.get_by_id("POOL_A_c")
    assert drain.bounds == (1.0, 1.0)          # what sampling needs
    assert leg.bounds == (0.0, 1.0)

    reopened = relax_measurement_pool_bounds(model)
    assert reopened == ["POOL_SK_A"]
    assert drain.bounds == (0.0, POOL_RELAXED_UB)
    assert leg.bounds == (0.0, 1.0)          # a fraction, and left saying so

    # The flux state drawn under the pinned bounds is still feasible afterwards, and now
    # every pool reaction has room for the projection ray.
    v = _fluxes(0.3, 0.7)
    lb = np.array([r.lower_bound for r in model.reactions])
    ub = np.array([r.upper_bound for r in model.reactions])
    x = v[[r.id for r in model.reactions]].to_numpy(float)
    assert ((x >= lb - 1e-12) & (x <= ub + 1e-12)).all()
    pool = np.array([r.id in {"POOL_A_c", "POOL_A_m", "POOL_SK_A"}
                     for r in model.reactions])
    assert np.minimum(x - lb, ub - x)[pool].min() > 0

    # Nothing outside the block moved.
    assert model.reactions.get_by_id("TC").bounds == (0, 1000)


def test_the_predictor_copy_also_reopens_the_mixing_legs(two_compartment_model):
    """A leg capped at 1 asserts a ceiling the drain no longer has.

    Once the drain is free the whole block can scale, so a leg above 1 is the same
    compartment fraction at a larger scale, not a larger fraction. The shared prepared
    model still keeps the legs at [0, 1] -- MFA reads it and its search box should stay
    the size of the simplex -- so only the predictor's copy opens them.
    """
    from sigmaflux.pipeline.sample_flux import (POOL_RELAXED_UB,
                                                relax_measurement_pool_bounds)

    model = two_compartment_model
    reopened = relax_measurement_pool_bounds(model, legs=True)
    assert set(reopened) == {"POOL_SK_A", "POOL_A_c", "POOL_A_m"}
    for rxn_id in reopened:
        assert model.reactions.get_by_id(rxn_id).bounds == (0.0, POOL_RELAXED_UB)
    assert model.reactions.get_by_id("TC").bounds == (0, 1000)


# ── The x3cflux encoding of the same zero ─────────────────────────────────────

def _pool_facts():
    """``model_facts``-shaped dict for the two-compartment model, pools included.

    Hand-built rather than derived, so the converter is tested against a fixed input:
    ``A_c`` is fed by the tracer, ``A_m`` by an unlabeled boundary, and both are tapped
    into the pool ``A``.
    """
    def met(i, pattern="abc"):
        return {"id": i, "atom_pattern": pattern, "symmetries": [], "terminal": False}

    def rxn(i, stoich, mapping, zero=()):
        return {"id": i, "lower_bound": 0.0, "upper_bound": 1000.0, "reversible": False,
                "atom_mapping": mapping, "terminal": False,
                "zero_stoichiometry": list(zero), "stoichiometry": stoich}

    return {
        "metabolites": [met(i) for i in ("S", "U", "A_c", "A_m", "A")],
        "reactions": [
            rxn("EX_S", {"S": 1.0}, " -> S (abc)"),
            rxn("EX_U", {"U": 1.0}, " -> U (abc)"),
            rxn("TC", {"S": -1.0, "A_c": 1.0}, "S (abc) -> A_c (abc)"),
            rxn("TM", {"U": -1.0, "A_m": 1.0}, "U (abc) -> A_m (abc)"),
            rxn("DM_Ac", {"A_c": -1.0}, " A_c (abc) -> "),
            rxn("DM_Am", {"A_m": -1.0}, " A_m (abc) -> "),
            rxn("POOL_A_c", {"A": 1.0}, "A_c (abc) -> A (abc)", zero=["A_c"]),
            rxn("POOL_A_m", {"A": 1.0}, "A_m (abc) -> A (abc)", zero=["A_m"]),
            rxn("POOL_SK_A", {"A": -1.0}, " A (abc) -> "),
        ],
        "measurement_pools": {
            "A": {"members": ["A_c", "A_m"], "atom_pattern": "abc",
                  "taps": {"POOL_A_c": "A_c", "POOL_A_m": "A_m"},
                  "drain": "POOL_SK_A"}},
        "input_emus": [], "input_metabolites": ["S", "U"], "convolution_nodes": [],
        "tracers": [{"metabolite": "S", "labeling": [1, 1, 0], "enrichment": 1.0,
                     "key": "S_[1, 1, 0]_1.0"}],
        "observable_metabolites": ["A"], "measured_fluxes": [], "min_rev_flux": 0.0,
    }


def _pool_simulator():
    """``(simulator, converted, facts)`` for :func:`_pool_facts`."""
    x = pytest.importorskip("x3cflux")
    from sigmaflux.x3cflux.build_config import build_configurations
    from sigmaflux.x3cflux.check_network import assert_network
    from sigmaflux.x3cflux.convert import convert

    facts = _pool_facts()
    conv = convert(facts)
    assert_network(conv.network_data, verbose=False)
    index = [(facts["tracers"][0]["key"], met, f"M+{k}")
             for met in ("A_c", "A_m", "A") for k in (1, 2, 3)]
    configs = build_configurations(facts, conv, index, [], np.zeros(len(index)),
                                   [], [], 0.01)
    return x.create_simulator_from_data(conv.network_data, configs), conv, facts


def _mids_at(simulator, conv, facts, v_c, v_m):
    """``{metabolite: MID over M+1..M+3}`` at the given compartment fractions."""
    from sigmaflux.x3cflux.compare import measurement_slots
    from sigmaflux.x3cflux.fluxmap import FluxMap

    fluxmap = FluxMap(facts, conv, simulator.parameter_space)
    order = [r["id"] for r in facts["reactions"]]
    flux = dict(zip(order, [0.0] * len(order)))
    flux.update({"EX_S": 1.0, "EX_U": 1.0, "TC": 1.0, "TM": 1.0,
                 "DM_Ac": 1.0, "DM_Am": 1.0,
                 "POOL_A_c": v_c, "POOL_A_m": v_m, "POOL_SK_A": v_c + v_m})
    v = np.array([flux[i] for i in order])
    labeling, _ = simulator.compute_measurements(fluxmap.to_x3c(v, np.zeros(len(order))))
    return {met: np.asarray(values).ravel()
            for (_, met), values in zip(measurement_slots(simulator, facts), labeling)}


def test_the_tap_does_not_perturb_the_compartment_it_observes():
    """The whole encoding rests on this: the member's MID is independent of the tap.

    x3cflux drops a reaction that both requires and produces a metabolite from that
    metabolite's balance -- mass *and* label -- so the carrier's atoms, which land on the
    member's product copy, never reach the member itself. If they did, the pool would be
    reading a compartment its own observation had contaminated.
    """
    simulator, conv, facts = _pool_simulator()
    quiet = _mids_at(simulator, conv, facts, 0.01, 0.99)
    busy = _mids_at(simulator, conv, facts, 0.99, 0.01)
    for member in ("A_c", "A_m"):
        np.testing.assert_allclose(quiet[member], busy[member], atol=1e-12, rtol=0)


@pytest.mark.parametrize("v_c,v_m", [(0.3, 0.7), (0.5, 0.5), (0.9, 0.1)])
def test_x3cflux_pool_mid_is_the_exact_flux_weighted_mixture(v_c, v_m):
    simulator, conv, facts = _pool_simulator()
    mid = _mids_at(simulator, conv, facts, v_c, v_m)
    assert not np.allclose(mid["A_c"], mid["A_m"])       # otherwise vacuous
    np.testing.assert_allclose(mid["A"], v_c * mid["A_c"] + v_m * mid["A_m"],
                               atol=1e-12, rtol=0)


def test_a_k_member_pool_costs_k_minus_one_free_parameters():
    """The pinned drain makes the taps sum to one, so one of them is determined.

    Under the previous encoding each leg carried a free exchange, so a k-member pool cost
    k parameters and nothing stopped them all being zero -- which left the pool's EMU row
    singular and killed every Ipopt start at iteration 0.
    """
    simulator, _, _ = _pool_simulator()
    free = list(simulator.parameter_space.free_parameter_names)
    taps = [n for n in free if n.startswith("POOL_A_")]
    assert len(taps) == 1, free
    assert not [n for n in free if n.startswith("POOL_SK_")]


def test_the_pool_block_survives_the_flux_round_trip():
    """Taps and drains take the generic path, so ``FluxMap`` needs no case for them."""
    from sigmaflux.x3cflux.fluxmap import FluxMap

    simulator, conv, facts = _pool_simulator()
    fluxmap = FluxMap(facts, conv, simulator.parameter_space)
    order = [r["id"] for r in facts["reactions"]]
    flux = {"EX_S": 1.0, "EX_U": 1.0, "TC": 1.0, "TM": 1.0, "DM_Ac": 1.0, "DM_Am": 1.0,
            "POOL_A_c": 0.3, "POOL_A_m": 0.7, "POOL_SK_A": 1.0}
    v = np.array([flux[i] for i in order])
    back, _ = fluxmap.from_x3c(fluxmap.to_x3c(v, np.zeros(len(order))))
    np.testing.assert_allclose(back, v, atol=1e-9, rtol=0)


def test_a_pool_costs_k_minus_one_fitted_parameters(two_compartment_model):
    """The chi2 test must not charge the fit for the pool block's scale.

    A k-member pool puts k+1 columns and one balance row into S, so it widens the null
    space by k. Only k-1 of those are identifiable: the pool's EMU balance
    ``sum_i(v_i X_i) / sum_i(v_i)`` is invariant under scaling the whole block, so the
    scale is a direction no fit can move. x3cflux pins exactly it with the drain equality
    (it reports all 36 I. orientalis drains as constant, and 36 of 72 taps as free), and
    ``split_constant_directions`` removes it for the AI -- but ``fitted_parameter_count``
    reads ``null(S)``, which is blind to bounds and constraints alike, so it charged one
    phantom parameter per pool. On I. orientalis that was 152 instead of 116, costing 36
    dof and tightening the acceptance bound for nothing.
    """
    from sigmaflux.pipeline.run_mfa_benchmark import fitted_parameter_count

    unpooled = two_compartment_model.copy()
    unpooled.remove_reactions(
        [r for r in unpooled.reactions if r.id.startswith("POOL_")], remove_orphans=True)

    pooled = MFAModel(two_compartment_model, "pooled")
    unpooled_n = fitted_parameter_count(MFAModel(unpooled, "unpooled"))
    assert fitted_parameter_count(pooled, {"A"}) == unpooled_n + 1   # observed -> k-1
    assert fitted_parameter_count(pooled) == unpooled_n + 1          # default: as observed


def test_a_pool_nothing_measures_costs_no_fitted_parameters(two_compartment_model):
    """An unobserved pool is a parameter with no matching measurement.

    Every one of its k directions is flat in the objective, so counting any of them
    shrinks dof without ever adding to the measurement count. On I. orientalis 22 of the
    36 pools are unmeasured, worth 44 of the 58 phantom directions.

    Both this and the scale correction are structural -- model plus mapping CSV, never a
    flux state. That is the point: the chi2 test is the only criterion computable when the
    true fluxes are unknown, so its parameter count must not depend on a flux solution.
    """
    from sigmaflux.pipeline.run_mfa_benchmark import fitted_parameter_count

    unpooled = two_compartment_model.copy()
    unpooled.remove_reactions(
        [r for r in unpooled.reactions if r.id.startswith("POOL_")], remove_orphans=True)

    pooled = MFAModel(two_compartment_model, "pooled")
    unpooled_n = fitted_parameter_count(MFAModel(unpooled, "unpooled"))
    assert fitted_parameter_count(pooled, observed=set()) == unpooled_n
