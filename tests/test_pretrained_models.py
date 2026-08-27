"""
Contract tests for the shipped ``models/pretrained/*.sfx`` releases.

These are the artifacts a downstream consumer — the web interface above all —
loads and serves, with no config YAML, run directory or COBRA JSON in sight.
What they must guarantee is exactly what is tested here: the manifest is
readable without touching weights, ``load_model`` returns a working ensemble,
and ``predict()`` turns human-readable tracer names plus a MID table into
finite, deterministic fluxes of the advertised shape.

Every model in the directory is covered, so shipping a new organism gets the
same coverage for free. The suite is skipped, not failed, when the files are
unfetched git-lfs pointers.

CPU threading note: ``torch`` defaults to one thread per core, and on a
many-core node that oversubscription makes a 20-member ensemble ~100× slower
(36 s to load and 87 s for one sample, measured on a 192-core node, versus
~0.5 s each at 4 threads). The fixture pins the pool, and a serving process
must do the same.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest
import torch

from sigmaflux.ai._sfx import SCHEMA_VERSION
from sigmaflux.ai.export import inspect_model, load_model, read_stoichiometry
from sigmaflux.ai.inference import EnsembleSigmaFlux
from sigmaflux.pipeline import inspect_model as inspect_cli

from conftest import REPO_ROOT

PRETRAINED_DIR = os.path.join(REPO_ROOT, "models", "pretrained")
PRETRAINED = sorted(glob.glob(os.path.join(PRETRAINED_DIR, "*.sfx")))

#: An unfetched git-lfs pointer is a ~130-byte text stub, not a model.
_LFS_POINTER_MAX_BYTES = 1024


def _is_fetched(path):
    """Whether *path* is a real model rather than an unfetched git-lfs pointer.

    A CI runner checks out without LFS, so the files are present at full name and
    near-zero size; anything that opens one as a model gets ``BadZipFile``.
    """
    return os.path.getsize(path) > _LFS_POINTER_MAX_BYTES


#: The subset a test may actually open. Tests that take a single model go through
#: the sfx_path fixture and skip individually; the rest must filter on this.
FETCHED = [p for p in PRETRAINED if _is_fetched(p)]

pytestmark = pytest.mark.skipif(not PRETRAINED,
                                reason=f"no .sfx models in {PRETRAINED_DIR}")


@pytest.fixture(scope="module", autouse=True)
def _pinned_threads():
    """Keep the ensemble off every core on the box (see the module docstring)."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(4, before))
    yield
    torch.set_num_threads(before)


@pytest.fixture(params=PRETRAINED, ids=lambda p: os.path.basename(p))
def sfx_path(request):
    if not _is_fetched(request.param):
        pytest.skip(f"{os.path.basename(request.param)} is a git-lfs pointer; "
                    "run 'git lfs pull' to fetch the weights")
    return request.param


_LOADED = {}


@pytest.fixture
def model(sfx_path):
    """The loaded ensemble, cached across tests — loading 20 members is not free."""
    if sfx_path not in _LOADED:
        _LOADED[sfx_path] = load_model(sfx_path, device="cpu")
    return _LOADED[sfx_path]


def make_df_L(model, n_samples=2, seed=0):
    """A MID table in the shape a caller submits: display tracer names, long format.

    Built from the model's own token index, which is the only description of the
    expected inputs an .sfx carries — a web front-end has to drive it the same way.
    """
    internal_to_display = {v: k for k, v in model.manifest["tracers"].items()}
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        {"Tracer Name": internal_to_display[tracer], "Metabolite ID": met,
         "Isotopomer": iso, "Sample": s, "Fraction": float(rng.uniform(0.0, 0.3))}
        for s in range(n_samples)
        for (tracer, met, iso) in model.spec.token_index
    ])


# ── Manifest: readable without loading any weights ────────────────────────────

def test_manifest_is_readable_and_current(sfx_path):
    """inspect_model works with no torch load, and the schema is one we can read."""
    m = inspect_model(sfx_path)

    assert m["schema_version"] <= SCHEMA_VERSION, "shipped model is newer than the reader"
    assert m["name"] and m["model_version"]
    assert m["n_members"] >= 1
    assert m["precision"] in ("fp16", "fp32")
    assert m["model"]["n_fluxes"] > 0 and m["model"]["n_tokens"] > 0
    # A caller picks tracers by display name; without them the model is undrivable.
    assert m["tracers"], "manifest carries no tracer display names"


def test_reported_validation_is_healthy(sfx_path):
    """Every member's stored validation R² is release-grade.

    Guards against shipping an undertrained or mismatched artifact: the number is
    read straight from the manifest, so this costs nothing and still fails loudly
    if a bad export replaces a good one.
    """
    val = inspect_model(sfx_path).get("validation", {})
    assert val.get("included"), "release models must carry their validation block"

    medians = [r["flux_median"] for r in val["reported"]]
    assert len(medians) == inspect_model(sfx_path)["n_members"]
    assert min(medians) > 0.9, f"member flux R² median dropped to {min(medians):.4f}"


def test_inspect_cli_reads_every_model(sfx_path, capsys):
    """`sigmaflux-inspect <model>` — the no-torch path a deploy script uses."""
    assert inspect_cli.cli([sfx_path]) == 0
    out = capsys.readouterr().out
    assert inspect_model(sfx_path)["name"] in out
    assert "tracers" in out


# ── Loading and running ───────────────────────────────────────────────────────

def test_load_model_matches_its_manifest(model, sfx_path):
    """load_model returns a complete ensemble that agrees with the manifest."""
    m = inspect_model(sfx_path)

    assert isinstance(model, EnsembleSigmaFlux)
    assert len(model.members) == m["n_members"]
    assert len(model.flux_names) == m["model"]["n_fluxes"]
    assert len(model.reversibility_names) == m["model"]["n_rev"]
    assert model.tracer_names == list(m["tracers"])
    assert model.measured_flux_ids, "no measured fluxes — df_Vm input would be inert"
    assert set(model.measured_flux_ids) <= set(model.flux_names)


def test_predict_returns_finite_fluxes_of_the_advertised_shape(model):
    """The end-to-end call a front-end makes: MID table in, per-sample fluxes out."""
    n_samples = 2
    result = model.predict(make_df_L(model, n_samples))

    n_outputs = len(model.flux_names) + len(model.reversibility_names)
    assert len(result) == n_samples * n_outputs
    assert {"Sample", "Flux Name", "Value"}.issubset(result.columns)
    assert np.isfinite(result["Value"].values).all()
    assert sorted(result["Sample"].unique()) == list(range(n_samples))

    # Fluxes then reversibilities, per sample — the order a caller slices by.
    first = result[result["Sample"] == 0]
    assert list(first["Flux Name"][:len(model.flux_names)]) == model.flux_names
    rev = first["Value"].values[len(model.flux_names):]
    assert (rev >= 0).all(), "reversibility ratios must be non-negative"


def test_predict_is_deterministic(model):
    """Same input, same output — a served model may not drift between requests."""
    df_L = make_df_L(model, n_samples=1)
    first = model.predict(df_L)["Value"].values
    second = model.predict(df_L)["Value"].values
    assert np.array_equal(first, second)


def test_measured_fluxes_reach_the_prediction(model):
    """Supplying df_Vm moves the prediction: the constraint is not silently dropped.

    The measured fluxes are soft constraints reconciled against the MID data, so
    the returned values are near — not equal to — what was submitted; only the
    fact that they take effect is asserted here.
    """
    df_L = make_df_L(model, n_samples=1)
    df_Vm = pd.DataFrame({"Sample": 0, "Flux Name": model.measured_flux_ids,
                          "Value": 1.0})

    without = model.predict(df_L)["Value"].values
    with_vm = model.predict(df_L, df_Vm=df_Vm)["Value"].values
    assert np.isfinite(with_vm).all()
    assert not np.allclose(without, with_vm), "df_Vm did not reach the model"


def test_each_member_and_the_ensemble_are_mass_balanced(model, sfx_path):
    """Every member is mass-balanced, and so is what ``predict()`` reports.

    A member reconstructs ``fluxes = x_l @ Z_full.T`` and so cannot leave the null
    space. ``EnsembleSigmaFlux.predict`` combines members with an elementwise **mean**,
    a convex combination, which stays in the null space — so the reported value is a
    flux state too. It was an elementwise median before, which is not a convex
    combination and did leave it (measured ``max|S v| = 5.74`` on E. coli core LC-MS
    against a glucose uptake of 8.6); this pins the fix.
    """
    if not inspect_model(sfx_path).get("network", {}).get("included"):
        pytest.skip("release predates the packaged COBRA network; re-export to cover it")

    S, _, _ = read_stoichiometry(sfx_path)
    df_L = make_df_L(model, n_samples=1)
    n_flux = len(model.flux_names)
    for i, member in enumerate(model.members):
        flux = member.predict(df_L)["Value"].values[:n_flux]
        resid = np.abs(S @ flux).max()
        assert resid < 1e-5 * max(np.abs(flux).max(), 1.0), f"member {i}: |S@v|max={resid:.3e}"

    combined = model.predict(df_L)["Value"].values[:n_flux]
    resid = np.abs(S @ combined).max()
    assert resid < 1e-5 * max(np.abs(combined).max(), 1.0), \
        f"ensemble aggregate: |S@v|max={resid:.3e}"


@pytest.mark.skipif(len(FETCHED) < 2, reason="needs two fetched models")
def test_models_are_distinct_organisms():
    """Two releases are two models — catches a duplicated or mislabelled artifact."""
    names = [inspect_model(p)["name"] for p in FETCHED]
    assert len(set(names)) == len(names), f"duplicate model names: {names}"
