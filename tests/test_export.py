"""Tests for packaging a model into a ``.sfx`` and loading it back.

The contract an export makes is strong: an fp32 export must round-trip every
weight bit-exactly, an fp16 export must round-trip to exactly the defined
quantization, and either must load with no config YAML, run directory or COBRA
JSON in reach. These tests assert that, plus the container's failure modes.
"""

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sigmaflux.ai._sfx import (
    MANIFEST,
    MEMBER_FMT,
    SfxFormatError,
    SfxIntegrityError,
    SfxVersionError,
    check_schema,
)
from sigmaflux.ai.data import load_stoichiometric_matrix
from sigmaflux.ai.export import (
    export_model,
    inspect_model,
    load_model,
    read_cobra_model,
    read_metabolites,
    read_stoichiometry,
    validate_model,
)
from sigmaflux.ai.inference import EnsembleSigmaFlux, SigmaFlux, load_predictor

from conftest import TRACER_ID


def _toy_df_L(n_samples=2):
    """Long-format labeling frame matching the toy token skeleton."""
    rng = np.random.default_rng(11)
    records = []
    for s in range(n_samples):
        ser = rng.dirichlet(np.ones(4))
        for iso, frac in zip(["M+1", "M+2", "M+3"], ser[1:]):
            records.append({"Tracer Name": TRACER_ID, "Sample": s,
                            "Metabolite ID": "Ser", "Isotopomer": iso, "Fraction": float(frac)})
        gly = rng.dirichlet(np.ones(3))
        for iso, frac in zip(["M+1", "M+2"], gly[1:]):
            records.append({"Tracer Name": TRACER_ID, "Sample": s,
                            "Metabolite ID": "Gly", "Isotopomer": iso, "Fraction": float(frac)})
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def toy_sfx(toy_run, tmp_path_factory):
    """An fp16 export of the toy ensemble, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("sfx") / "toy-v1.0.0.sfx"
    return str(export_model(toy_run["ensemble_dir"], out, name="toy", model_version="1.0.0",
                            config=toy_run["config"], with_val_inputs=False, verbose=False))


def _export(toy_run, tmp_path, **kw):
    kw.setdefault("name", "toy")
    kw.setdefault("model_version", "1.0.0")
    kw.setdefault("config", toy_run["config"])
    kw.setdefault("with_val_inputs", False)
    kw.setdefault("verbose", False)
    src = kw.pop("source", toy_run["ensemble_dir"])
    return str(export_model(src, tmp_path / "m.sfx", **kw))


# ── Round-trip ────────────────────────────────────────────────────────────────

def test_export_roundtrip_fp32_bit_exact(toy_run, tmp_path):
    """fp32 must reproduce every tensor exactly, regenerated ones included.

    This is what makes dropping the 14 frozen tensors safe rather than merely
    plausible: they are compared here too.
    """
    path = _export(toy_run, tmp_path, precision="fp32")
    loaded = load_model(path, device="cpu")

    for i, member_path in enumerate(sorted(glob.glob(
            os.path.join(toy_run["ensemble_dir"], "model_*.pt")))):
        original = torch.load(member_path, map_location="cpu",
                              weights_only=False)["model_state_dict"]
        rebuilt = loaded.members[i]._model.state_dict()
        assert set(original) == set(rebuilt)
        differing = [k for k in original
                     if not torch.equal(rebuilt[k].cpu(), original[k].cpu())]
        assert not differing, f"member {i} differs: {differing}"


def test_export_roundtrip_fp16_quantized(toy_run, tmp_path):
    """fp16 is a *defined* lossy transform, so the comparison is still exact."""
    path = _export(toy_run, tmp_path, precision="fp16")
    loaded = load_model(path, device="cpu")
    regenerated = set(inspect_model(path)["regenerated_tensors"])

    original = torch.load(toy_run["member_paths"][0], map_location="cpu",
                          weights_only=False)["model_state_dict"]
    rebuilt = loaded.members[0]._model.state_dict()
    for key, value in original.items():
        expected = value.cpu() if key in regenerated else value.cpu().half().float()
        assert torch.equal(rebuilt[key].cpu(), expected), key


def test_predict_matches_source_ensemble_fp32(toy_run, tmp_path):
    path = _export(toy_run, tmp_path, precision="fp32")
    df_L = _toy_df_L()
    expected = EnsembleSigmaFlux(toy_run["config"], toy_run["ensemble_dir"],
                                 device="cpu").predict(df_L)
    got = load_model(path, device="cpu").predict(df_L)

    assert list(got.columns) == list(expected.columns)
    assert got["Flux Name"].tolist() == expected["Flux Name"].tolist()
    np.testing.assert_allclose(got["Value"], expected["Value"], rtol=1e-5, atol=1e-6)


def test_predict_fp16_within_tolerance(toy_run, tmp_path, toy_sfx):
    fp32 = load_model(_export(toy_run, tmp_path, precision="fp32"), device="cpu")
    fp16 = load_model(toy_sfx, device="cpu")
    df_L = _toy_df_L()
    a = fp32.predict(df_L)["Value"].to_numpy()
    b = fp16.predict(df_L)["Value"].to_numpy()
    np.testing.assert_allclose(b, a, rtol=2e-2, atol=1e-3 * max(np.abs(a).max(), 1e-9))


def test_mass_balance_after_fp16_export(toy_run, toy_sfx):
    """S @ v stays ~0: the null-space basis is kept float32 regardless of precision."""
    S, _ = load_stoichiometric_matrix(toy_run["model_json"])
    result = load_model(toy_sfx, device="cpu").predict(_toy_df_L(n_samples=1))
    flux = result.iloc[:toy_run["n_fluxes"]]["Value"].to_numpy()
    assert np.abs(S @ flux).max() < 1e-4


# ── Packaged network ──────────────────────────────────────────────────────────

def test_export_embeds_the_prepared_network(toy_sfx, toy_run):
    """The run's prepared COBRA model travels inside the export, found on its own."""
    net = inspect_model(toy_sfx)["network"]

    assert net["included"]
    assert net["source"] == "model_prepared_for_mfa.json"
    assert net["n_reactions"] == toy_run["n_fluxes"]
    assert net["covers_flux_names"] and net["reaction_order_matches"]


def test_read_stoichiometry_is_aligned_to_the_outputs(toy_sfx, toy_run):
    """S comes back in flux_names order, so it multiplies straight into a prediction."""
    S, met_ids, cobra_model = read_stoichiometry(toy_sfx)
    model = load_model(toy_sfx, device="cpu")

    assert S.shape == (len(met_ids), len(model.flux_names))
    assert met_ids == [m.id for m in cobra_model.metabolites]

    # Column i really is flux_names[i], not the network's own ordering.
    for i, name in enumerate(model.flux_names):
        rxn = cobra_model.reactions.get_by_id(name)
        for met, coeff in rxn.metabolites.items():
            assert S[met_ids.index(met.id), i] == pytest.approx(coeff)


def test_mass_balance_from_the_export_alone(toy_sfx):
    """S @ v ≈ 0 using only the packaged file — the check a consumer can now run."""
    S, _, _ = read_stoichiometry(toy_sfx)
    model = load_model(toy_sfx, device="cpu")
    flux = model.predict(_toy_df_L(n_samples=1))["Value"].to_numpy()[:len(model.flux_names)]

    assert np.abs(S @ flux).max() < 1e-4


def test_metabolite_list_covers_the_network(toy_sfx):
    """One row per network metabolite, in network order; observable means a token metabolite."""
    mets = read_metabolites(toy_sfx)
    cobra_model = read_cobra_model(toy_sfx)
    model = load_model(toy_sfx, device="cpu")

    assert list(mets.columns) == ["metabolite_id", "description", "observable"]
    assert mets["metabolite_id"].tolist() == [m.id for m in cobra_model.metabolites]
    assert set(mets.loc[mets["observable"], "metabolite_id"]) == set(model.expected_metabolites())
    assert mets["description"].tolist() == [m.name for m in cobra_model.metabolites]

    block = inspect_model(toy_sfx)["metabolites"]
    assert block == {"included": True, "n_metabolites": len(mets),
                     "n_observable": int(mets["observable"].sum())}


def test_metabolite_names_come_from_the_config_model(toy_run, tmp_path):
    """The config's model_file supplies the names, in the list and the embedded network alike."""
    renamed = json.loads(Path(toy_run["model_json"]).read_text())
    for met in renamed["metabolites"]:
        met["name"] = f"name of {met['id']}"
    model_file = tmp_path / "renamed.json"
    model_file.write_text(json.dumps(renamed))
    config = tmp_path / "config.yaml"
    config.write_text(Path(toy_run["config"]).read_text().replace(
        "models/atomistic/toy/one_carbon_annotated.json", str(model_file)))

    path = _export(toy_run, tmp_path, config=str(config))

    mets = read_metabolites(path)
    assert (mets["description"] == "name of " + mets["metabolite_id"]).all()
    assert all(m.name == f"name of {m.id}" for m in read_cobra_model(path).metabolites)
    assert inspect_model(path)["network"]["names_source"] == "renamed.json"


def test_network_can_be_omitted(toy_run, tmp_path):
    """--no-model-json leaves it out, and reading it then fails with a way forward."""
    path = _export(toy_run, tmp_path, with_network=False)

    assert inspect_model(path)["network"] == {"included": False}
    assert inspect_model(path)["metabolites"] == {"included": False}
    with pytest.raises(SfxFormatError, match="with_network=True"):
        read_cobra_model(path)
    with pytest.raises(SfxFormatError, match="no metabolite list"):
        read_metabolites(path)


def test_network_missing_a_predicted_reaction_is_refused(toy_run, tmp_path):
    """A network that cannot describe every output is worse than none.

    with_network=True says so loudly; "auto" drops it and exports anyway, because
    a stoichiometry indexed against the wrong reaction set silently produces
    nonsense mass balances.
    """
    truncated = json.loads(Path(toy_run["model_json"]).read_text())
    truncated["reactions"] = truncated["reactions"][:-1]
    partial = tmp_path / "partial_model.json"
    partial.write_text(json.dumps(truncated))

    with pytest.raises(ValueError, match="missing reactions this model predicts"):
        _export(toy_run, tmp_path, model_json=str(partial), with_network=True)

    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    path = _export(toy_run, auto_dir, model_json=str(partial))
    assert inspect_model(path)["network"] == {"included": False}


# ── "Every model is an ensemble" ──────────────────────────────────────────────

def test_single_member_export_is_an_ensemble(toy_run, tmp_path):
    path = _export(toy_run, tmp_path, source=toy_run["checkpoint"], precision="fp32")
    loaded = load_model(path, device="cpu")

    assert isinstance(loaded, EnsembleSigmaFlux)
    assert len(loaded.members) == 1
    assert inspect_model(path)["n_members"] == 1

    df_L = _toy_df_L()
    single = SigmaFlux(toy_run["config"], toy_run["checkpoint"], device="cpu").predict(df_L)
    np.testing.assert_allclose(loaded.predict(df_L)["Value"], single["Value"],
                               rtol=1e-5, atol=1e-6)


# ── Self-containment ──────────────────────────────────────────────────────────

def test_no_config_or_cobra_needed_to_load(toy_run, tmp_path, monkeypatch):
    """The export must load with the YAML and COBRA JSON moved out of reach."""
    path = _export(toy_run, tmp_path, precision="fp32")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    for original in (toy_run["config"], toy_run["model_json"]):
        shutil.copy(original, hidden / os.path.basename(original))

    monkeypatch.setattr("sigmaflux.ai.data.load_stoichiometric_matrix",
                        lambda *a, **k: pytest.fail("loading the export read the COBRA JSON"))
    monkeypatch.chdir(tmp_path)

    loaded = load_model(path, device="cpu")
    result = loaded.predict(_toy_df_L())
    assert np.isfinite(result["Value"]).all()
    # the config's human-readable tracer name survived the trip
    assert loaded.expected_tracers() == ["labeled-Ser"]


# ── Container behaviour ───────────────────────────────────────────────────────

def test_inspect_model_fields(toy_sfx):
    m = inspect_model(toy_sfx)
    assert m["format"] == "sigmaflux-model"
    assert isinstance(m["schema_version"], int)
    assert m["name"] == "toy" and m["model_version"] == "1.0.0"
    assert m["precision"] == "fp16" and m["n_members"] == 2
    for key in ("created_utc", "provenance", "model", "tracers", "tensors",
                "regenerated_tensors", "entries", "member_nbytes"):
        assert key in m, key
    assert m["provenance"]["sigmaflux_version"]
    # the tensor table must tile the member blob exactly
    offset = 0
    for spec in m["tensors"]:
        assert spec["offset"] == offset
        offset += spec["nbytes"]
    assert offset == m["member_nbytes"]


def test_manifest_readable_without_torch(toy_sfx, tmp_path):
    """Loading the container module must not drag in torch."""
    script = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "_sfx", {os.path.join(os.path.dirname(__file__), '..', 'src', 'sigmaflux', 'ai', '_sfx.py')!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        manifest = m.read_manifest({toy_sfx!r})
        assert manifest["name"] == "toy"
        assert "torch" not in sys.modules, "reading the manifest imported torch"
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_manifest_is_first_and_stored(toy_sfx):
    """Cheap inspection depends on both properties."""
    with zipfile.ZipFile(toy_sfx) as zf:
        first = zf.infolist()[0]
    assert first.filename == MANIFEST
    assert first.compress_type == zipfile.ZIP_STORED


def test_schema_version_rejects_newer():
    with pytest.raises(SfxVersionError, match="Upgrade sigmaflux"):
        check_schema({"format": "sigmaflux-model", "schema_version": 999})


def test_wrong_format_rejected():
    with pytest.raises(SfxFormatError):
        check_schema({"format": "something-else", "schema_version": 1})


def test_unknown_manifest_key_tolerated():
    """Adding an optional key must not require a schema bump."""
    m = check_schema({"format": "sigmaflux-model", "schema_version": 1, "future_key": 42})
    assert m["future_key"] == 42


def test_checksums_detect_corruption(toy_sfx, tmp_path):
    corrupt = tmp_path / "corrupt.sfx"
    with zipfile.ZipFile(toy_sfx) as zf:
        header = zf.getinfo(MEMBER_FMT.format(0)).header_offset
    raw = bytearray(open(toy_sfx, "rb").read())
    raw[header + 30 + len(MEMBER_FMT.format(0))] ^= 0xFF
    corrupt.write_bytes(raw)

    with pytest.raises(SfxIntegrityError):
        load_model(str(corrupt), device="cpu")


# ── Export-time validation ────────────────────────────────────────────────────

def test_heterogeneous_members_rejected(toy_run, tmp_path):
    """Members that were not trained together must not be packaged as an ensemble."""
    odd = tmp_path / "odd.pt"
    ckpt = torch.load(toy_run["member_paths"][1], map_location="cpu", weights_only=False)
    ckpt["flux_names"] = list(ckpt["flux_names"])[:-1] + ["RENAMED"]
    torch.save(ckpt, odd)

    with pytest.raises(ValueError, match="disagree on 'flux_names'"):
        export_model([toy_run["member_paths"][0], str(odd)], tmp_path / "bad.sfx",
                     name="toy", model_version="1.0.0", verbose=False)


def test_tracer_map_validated_against_vocab(toy_run, tmp_path):
    """A tracer name that resolves to nothing would silently zero every token."""
    import yaml
    with open(toy_run["config"]) as fh:
        cfg = yaml.safe_load(fh)
    cfg["labeling"]["tracers"][0]["tracer_id"] = "not_a_real_tracer"
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="do not resolve"):
        export_model(toy_run["ensemble_dir"], tmp_path / "bad.sfx", name="toy",
                     model_version="1.0.0", config=str(broken), verbose=False)


def test_refuses_to_overwrite(toy_run, tmp_path):
    path = _export(toy_run, tmp_path)
    with pytest.raises(FileExistsError):
        _export(toy_run, tmp_path)
    assert os.path.exists(path)


def test_rejects_bad_precision(toy_run, tmp_path):
    with pytest.raises(ValueError, match="precision"):
        _export(toy_run, tmp_path, precision="bfloat16")


# ── Integration ───────────────────────────────────────────────────────────────

def test_load_predictor_accepts_sfx(toy_sfx):
    """`--checkpoint model.sfx` must work at every existing call site."""
    predictor = load_predictor(None, toy_sfx, device="cpu")
    assert isinstance(predictor, EnsembleSigmaFlux)
    assert len(predictor.members) == 2


def test_validate_model_reproduces_r2(toy_run, tmp_path):
    """R² must be recomputable from the export alone."""
    from sigmaflux.ai.train import per_reaction_r2

    path = str(export_model(toy_run["ensemble_dir"], tmp_path / "v.sfx", name="toy",
                            model_version="1.0.0", config=toy_run["config"],
                            precision="fp32", with_val_inputs=True,
                            data_dir=toy_run["dir"], verbose=False))
    assert inspect_model(path)["validation"]["inputs_included"]

    scored = validate_model(path, device="cpu")
    assert list(scored.columns) == ["reaction", "r2"]
    assert scored["reaction"].tolist() == toy_run["flux_names"]

    per_member = validate_model(path, device="cpu", per_member=True)
    assert "r2_member_0" in per_member and "r2_member_1" in per_member


def test_validate_model_without_inputs_is_a_clear_error(toy_sfx):
    with pytest.raises(SfxFormatError, match="with_val_inputs"):
        validate_model(toy_sfx, device="cpu")


def test_cli_export_then_inspect(toy_run, tmp_path, capsys):
    from sigmaflux.pipeline import export_model as export_cli
    from sigmaflux.pipeline import inspect_model as inspect_cli

    out = tmp_path / "cli.sfx"
    assert export_cli.cli([
        "--source", toy_run["ensemble_dir"], "--out", str(out),
        "--name", "toy", "--model-version", "2.1.0",
        "--config", toy_run["config"], "--no-val-inputs", "--check",
    ]) == 0
    assert out.exists()

    assert inspect_cli.cli([str(out), "--verify"]) == 0
    printed = capsys.readouterr().out
    assert "toy  v2.1.0" in printed
    assert "all checksums and CRCs match" in printed
