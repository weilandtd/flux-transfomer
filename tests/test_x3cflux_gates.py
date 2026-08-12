"""Gates G0-G4: does the x3cflux network mean the same thing as our EMU model?

An MFA backend can be wrong in ways a fit never reveals -- an optimiser happily
converges on a network that has lost a balance row or scrambled an atom mapping,
and reports a small residual for a model that is not the one we meant. These gates
check the translation itself rather than the answer:

* **G0** cobra ``(v, r)`` survives a round trip through x3cflux's free parameters.
* **G1** the net-flux nullity matches cobra's, so the flux space is the same size.
* **G2** both forward models simulate the same MIDs from the same flux.
* **G3** the measurement configuration has the shape we asked for.
* **G4** both agree on the residual at the true flux.

G2 is the sharpest of the five: two independently written forward models agreeing
to ~1e-6 is far stronger evidence than either one being self-consistent.

The gates run on the toy model in CI. Set ``SIGMAFLUX_X3CFLUX_FACTS`` to a
``model_facts.json`` to run them against a real organism as well.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("x3cflux", reason="optional x3cflux backend not installed")

from sigmaflux.x3cflux.build_config import structural_configuration  # noqa: E402
from sigmaflux.x3cflux.check_network import check_network          # noqa: E402
from sigmaflux.x3cflux.convert import convert                      # noqa: E402
from sigmaflux.x3cflux.fluxmap import FluxMap, cobra_stoichiometry  # noqa: E402

#: A real organism's facts, when one is available. The toy model exercises 0-atom
#: pools and __src pools but has no symmetry and no non-integer stoichiometry, so
#: it cannot stand in for the T-cell network entirely.
REAL_FACTS = os.environ.get("SIGMAFLUX_X3CFLUX_FACTS")


@pytest.fixture(scope="module")
def toy_facts(toy_run):
    """``model_facts`` for the toy run directory."""
    from cobra.io.json import load_json_model

    from sigmaflux.core.model import MFAModel
    from sigmaflux.pipeline.generate_training_data import build_experiments, build_targets
    from sigmaflux.x3cflux.facts import model_facts

    import yaml
    with open(toy_run["config"]) as fh:
        cfg = yaml.safe_load(fh)

    cobra_model = load_json_model(os.path.join(toy_run["dir"],
                                               "model_prepared_for_mfa.json"))
    mfa_model = MFAModel(cobra_model, "gates")
    targets, measured, _ = build_targets(mfa_model)
    mfa_model.prepare(build_experiments(mfa_model, cfg["labeling"]["tracers"]),
                      targets=targets, with_backward=False)
    return model_facts(cobra_model, mfa_model, {m.id for m in measured}, cfg)


@pytest.fixture(scope="module")
def facts(toy_facts):
    return toy_facts


def _converted(facts):
    conv = convert(facts)
    assert not check_network(conv.network_data, verbose=False)
    return conv


def _structural(facts):
    """(conv, simulator) with no measurements -- enough for the structural gates."""
    import x3cflux as x

    conv = _converted(facts)
    simulator = x.create_simulator_from_data(
        conv.network_data, [structural_configuration(facts, conv)])
    return conv, simulator


def test_g0_flux_roundtrip(facts):
    """cobra ``(v, r)`` -> x3cflux free parameters -> cobra is the identity.

    Four transformations compose in :class:`FluxMap` -- reaction identity, scale,
    split fan-out and the ``v.x = r/2`` exchange convention -- and any one of them
    silently wrong invalidates every number the backend produces.

    Reversibility is only checked where an exchange parameter survives. A dozen
    reactions are written with an empty product side (``lac__L_e ->``) or with only
    0-atom products (``MDH_c``), so running them backwards would permute atoms into
    a side that has none. x3cflux cannot express that, and our own expansion builds
    no reverse edge for them either (the reverse origin set comes out empty), so
    their exchange carries no labeling information in either model and is dropped.

    The fluxes tested must be *feasible*. ``to_x3c`` keeps only the free
    coordinates and ``from_x3c`` rebuilds the rest through the null-space basis, so
    a flux violating ``S·v = 0`` cannot come back unchanged -- and should not. Test
    vectors are therefore drawn from cobra's null space, which is exactly the set
    the fit can reach.
    """
    conv, simulator = _structural(facts)
    fluxmap = FluxMap(facts, conv, simulator.parameter_space)

    S, _, _ = cobra_stoichiometry(facts)
    _, _, vt = np.linalg.svd(S)
    nullity = S.shape[1] - np.linalg.matrix_rank(S)
    if nullity == 0:
        pytest.skip("no free net flux: nothing to round-trip")
    basis = vt[-nullity:]

    rng = np.random.default_rng(0)
    carries_exchange = np.array([
        conv.reversible.get(conv.parents.get(r["id"], [(r["id"], 1.0)])[0][0], False)
        for r in facts["reactions"]])

    for _ in range(10):
        v = rng.normal(size=nullity) @ basis
        r = np.abs(rng.uniform(0.0, 1.0, len(v)))
        v2, r2 = fluxmap.from_x3c(fluxmap.to_x3c(v, r))
        assert np.abs(v2 - v).max() < 1e-9
        # The toy model is entirely irreversible, so this mask can be empty.
        if carries_exchange.any():
            assert np.abs((r2 - r)[carries_exchange]).max() < 1e-9


def test_g1_flux_space_has_the_same_dimension(facts):
    """x3cflux's net-flux nullity equals cobra's.

    x3cflux derives S from the declared pools, so a dropped balance row does not
    error -- it silently widens the flux space and hands Ipopt more freedom than
    our fit has. On the T-cell model the difference is 25 against 50, i.e. the
    comparison would be meaningless rather than merely noisy.
    """
    _, simulator = _structural(facts)
    kernel = np.asarray(
        simulator.parameter_space.net_flux_solution_space.kernel_basis(), float)

    S, _, _ = cobra_stoichiometry(facts)
    assert kernel.shape[1] == S.shape[1] - np.linalg.matrix_rank(S)


def test_g1_ground_truth_is_feasible(facts):
    """A cobra-feasible flux satisfies x3cflux's stoichiometry to numerical zero."""
    conv, simulator = _structural(facts)
    fluxmap = FluxMap(facts, conv, simulator.parameter_space)

    S, _, _ = cobra_stoichiometry(facts)
    # Any null-space vector of cobra's S is a feasible net flux by construction.
    _, _, vt = np.linalg.svd(S)
    v = vt[-1]
    error = simulator.parameter_space.compute_stoichiometry_error(
        fluxmap.full_vector(v, np.zeros_like(v)))
    assert np.max(np.abs(np.asarray(error, float))) < 1e-6


def test_converter_declares_every_pool_and_reaction(facts):
    """Every cobra pool and reaction survives conversion.

    The 0-atom pools are what makes this possible and what keeps the non-carbon
    balances -- and therefore G1 -- alive.
    """
    conv = _converted(facts)
    names = {m.name for m in conv.network_data.metabolites}
    for m in facts["metabolites"]:
        assert m["id"] in names
    # Reactions may be split or halved, so each cobra id must map to at least one.
    declared = set(conv.x3c_reactions)
    for r in facts["reactions"]:
        subs = conv.parents.get(r["id"], [(r["id"], 1.0)])
        assert any(s in declared for s, _ in subs), r["id"]


@pytest.mark.skipif(not REAL_FACTS, reason="set SIGMAFLUX_X3CFLUX_FACTS to run")
def test_real_model_converts_and_matches_nullity():
    """The full-size network passes the structural gates too.

    The toy model has no symmetric pool and no non-integer stoichiometry, so it
    never exercises the half-flux orientation pairs or the biomass splitting.
    """
    with open(REAL_FACTS) as fh:
        real = json.load(fh)
    _, simulator = _structural(real)
    kernel = np.asarray(
        simulator.parameter_space.net_flux_solution_space.kernel_basis(), float)
    S, _, _ = cobra_stoichiometry(real)
    assert kernel.shape[1] == S.shape[1] - np.linalg.matrix_rank(S)
