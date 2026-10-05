"""Does the in-house EMU model agree with x3cflux?

An MFA backend can be wrong in ways a fit never reveals -- an optimiser happily
converges on a network that has lost a balance row or scrambled an atom mapping,
and reports a small residual for a model that is not the one we meant. This
exercises :mod:`sigmaflux.pipeline.check_mfa_model`'s four checks against the toy
model, in order, each depending on the last: the model builds in x3cflux, every
reaction delivers its full stoichiometry to
every EMU it produces, both forward models predict the same
labeling from the same flux, and a fit starts (convergence is not required -- see
``check_fit_starts``).

The toy model runs in CI. Set ``SIGMAFLUX_CHECK_CONFIGS=1`` to also sweep
``paper/configs/*.yaml`` against the real organisms -- slow, since the x3cflux
simulator build is single-threaded and takes minutes to hours depending on the
model.
"""

import os

import pytest
import yaml

pytest.importorskip("x3cflux", reason="optional x3cflux backend not installed")

from sigmaflux.pipeline._benchmark import build_mfa_context          # noqa: E402
from sigmaflux.pipeline.check_mfa_model import (                     # noqa: E402
    check_build, check_edge_weights, check_fit_starts, check_labeling,
    interior_flux, measurable_metabolites,
)
from sigmaflux.x3cflux.facts import model_facts                      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def toy_context(toy_run):
    """Everything the checks need, built the way :func:`check_model` builds it.

    The toy config has no ``sampling:`` block and no ``mapping_csv``, so it never
    goes through :func:`~sigmaflux.pipeline.check_mfa_model.prepared_model` --
    ``toy_run`` already leaves a ``model_prepared_for_mfa.json`` with
    ``atom_pattern`` annotations, which is the only preparation the toy model
    needs, and :func:`measurable_metabolites` correctly returns ``None`` (target
    everything) rather than raising.
    """
    from cobra.io.json import load_json_model

    with open(toy_run["config"]) as fh:
        cfg = yaml.safe_load(fh)

    model = load_json_model(os.path.join(toy_run["dir"], "model_prepared_for_mfa.json"))
    keep = measurable_metabolites(cfg, REPO_ROOT)
    v, r = interior_flux(model)

    mfa_model, _, _, labeling_to, labeling_index = build_mfa_context(
        model, cfg["labeling"]["tracers"], initial_flux=v, name=cfg["name"],
        with_backward=False, restrict_targets_to=keep)
    observable = keep or {met for _, met, _ in labeling_index}
    facts = model_facts(model, mfa_model, observable, cfg)

    return {"facts": facts, "mfa_model": mfa_model, "labeling_to": labeling_to,
            "labeling_index": [tuple(k) for k in labeling_index], "v": v, "r": r}


@pytest.fixture(scope="module")
def toy_check_results(toy_context):
    """All four checks, run once and shared by every test below.

    Stops at the first failure, exactly as :func:`check_model` does -- there is no
    simulator to compare against if the model does not build, and no point
    starting a fit against a network the two forward models already disagree on.
    Uses the shipped ``check_*`` functions rather than a second hand-rolled path,
    so this test and the CLI cannot drift apart.
    """
    facts = toy_context["facts"]
    results = {"build": None, "edge_weights": None, "labeling": None,
               "fit_starts": None}

    results["build"], conv, simulator, fluxmap, ours = check_build(
        facts, toy_context["mfa_model"], toy_context["labeling_to"],
        toy_context["labeling_index"], toy_context["v"], toy_context["r"])
    if not results["build"]["passed"]:
        return results

    results["edge_weights"] = check_edge_weights(toy_context["mfa_model"], conv)
    if not results["edge_weights"]["passed"]:
        return results

    results["labeling"] = check_labeling(
        facts, simulator, fluxmap, toy_context["labeling_index"], ours,
        toy_context["v"], toy_context["r"])
    if not results["labeling"]["passed"]:
        return results

    # isolate=False: a crash here should be debuggable, not swallowed to None.
    results["fit_starts"] = check_fit_starts(
        facts, conv, simulator, fluxmap, toy_context["v"], toy_context["r"],
        isolate=False)
    return results


def test_model_builds(toy_check_results):
    result = toy_check_results["build"]
    assert result["passed"], result


def test_edge_weights_match_stoichiometry(toy_check_results):
    """A reaction delivers its whole flux to every EMU it produces.

    Weights summing below the stoichiometric coefficient is flux carrying no
    label, above it is label invented from nothing -- see
    :func:`check_edge_weights`.
    """
    result = toy_check_results["edge_weights"]
    assert result is not None, "the build check must pass first"
    assert result["passed"], result
    assert result["n_producers"] > 0


def test_forward_models_agree(toy_check_results):
    result = toy_check_results["labeling"]
    assert result is not None, "the build check must pass first"
    assert result["passed"], result
    assert result["n_compared"] > 0


def test_fit_starts(toy_check_results):
    """The fit takes real steps and reports a finite residual.

    Convergence is explicitly not required: at ``max_iter=DEFAULT_FIT_ITERS``,
    Ipopt reporting ``Maximum_Iterations_Exceeded`` (status -1) is an accepted,
    expected outcome -- see :func:`check_fit_starts`.
    """
    result = toy_check_results["fit_starts"]
    assert result is not None, "the labeling check must pass first"
    assert result["passed"], result


# ── Real organisms, opt-in ─────────────────────────────────────────────────────
#
# Not a marker: this repo registers none (no [tool.pytest.ini_options], no
# pytest.ini), and gates expensive tests with importorskip + an env var
# elsewhere (e.g. tests/test_pretrained_models.py). Follow that convention.

REAL_CONFIGS = os.environ.get("SIGMAFLUX_CHECK_CONFIGS")


@pytest.mark.skipif(not REAL_CONFIGS, reason="set SIGMAFLUX_CHECK_CONFIGS=1 to run")
@pytest.mark.parametrize("config_name", [
    "e_coli_core", "e_coli", "s_cerevisiae", "i_orientalis", "r_toruloides",
    "m_musculus_t_cell",
])
def test_real_model(config_name, tmp_path):
    from sigmaflux.pipeline._paths import load_config
    from sigmaflux.pipeline.check_mfa_model import check_model

    config_path = os.path.join(REPO_ROOT, "paper", "configs", f"{config_name}.yaml")
    cfg, root = load_config(config_path)
    ok, results = check_model(cfg, root, str(tmp_path))
    assert ok, results
