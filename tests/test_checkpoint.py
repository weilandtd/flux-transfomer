"""Tests for sigmaflux.ai.checkpoint and the ensemble prediction path.

``load_eval_context``/``build_flux_decoder`` are the single point of failure for
every analysis script, and ``EnsembleSigmaFlux`` is now the default predictor
whenever a run has an ``ensemble/`` directory. Both were previously untested.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sigmaflux.ai.checkpoint import (
    ALL,
    VAL,
    build_flux_decoder,
    flux_decoder_config,
    load_eval_context,
    promote_norm_buffers,
    resolve_device,
)
from sigmaflux.ai.inference import EnsembleSigmaFlux, SigmaFlux, load_predictor

from conftest import N_SAMPLES, TRACER_ID


def _toy_df_L(n_samples=2):
    """Long-format labeling frame matching the toy token skeleton."""
    rng = np.random.default_rng(7)
    records = []
    for s in range(n_samples):
        ser = rng.dirichlet(np.ones(4))
        for iso, frac in zip(["M+1", "M+2", "M+3"], ser[1:]):
            records.append({"Tracer Name": "labeled-Ser", "Sample": s,
                            "Metabolite ID": "Ser", "Isotopomer": iso, "Fraction": float(frac)})
        gly = rng.dirichlet(np.ones(3))
        for iso, frac in zip(["M+1", "M+2"], gly[1:]):
            records.append({"Tracer Name": "labeled-Ser", "Sample": s,
                            "Metabolite ID": "Gly", "Isotopomer": iso, "Fraction": float(frac)})
    return pd.DataFrame(records)


# ── load_eval_context ─────────────────────────────────────────────────────────

def test_eval_context_roundtrip(toy_run):
    """The context reports the checkpoint's own split, names and architecture."""
    ctx = load_eval_context(toy_run["checkpoint"])

    assert list(ctx.sample_indices) == toy_run["val_indices"]
    assert len(ctx.dataset) == len(toy_run["val_indices"])
    assert ctx.flux_names == toy_run["flux_names"]
    assert ctx.reversibility_names == toy_run["flux_names"]
    assert ctx.config == flux_decoder_config(ctx.ckpt["model_config"])
    assert ctx.config.n_fluxes == toy_run["n_fluxes"]
    assert ctx.data_dir == toy_run["dir"]
    assert ctx.model_json == toy_run["model_json"]
    # labeling rows are restricted and ordered to the trained token sequence
    assert list(ctx.labeling_df.index) == list(ctx.ckpt["token_index"])


def test_sample_selection_modes(toy_run):
    """VAL, ALL and an explicit index array each select what they say."""
    assert len(load_eval_context(toy_run["checkpoint"], samples=VAL).dataset) == len(toy_run["val_indices"])
    assert len(load_eval_context(toy_run["checkpoint"], samples=ALL).dataset) == N_SAMPLES

    subset = toy_run["val_indices"][:3]
    ctx = load_eval_context(toy_run["checkpoint"], samples=subset)
    assert list(ctx.sample_indices) == list(subset)
    assert len(ctx.dataset) == 3

    with pytest.raises(ValueError, match="samples must be"):
        load_eval_context(toy_run["checkpoint"], samples="nonsense")


def test_full_batch_shapes_and_device(toy_run):
    """full_batch returns the whole split as one batch, on the requested device."""
    ctx = load_eval_context(toy_run["checkpoint"])
    (cids, lids, tids, fracs), targets = ctx.full_batch(device="cpu")

    n = len(toy_run["val_indices"])
    assert cids.shape[0] == lids.shape[0] == tids.shape[0] == fracs.shape[0] == n
    assert targets.shape == (n, 2 * toy_run["n_fluxes"])
    assert all(t.device.type == "cpu" for t in (cids, lids, tids, fracs, targets))
    # cached, so an ensemble does not re-tokenize per member
    again, _ = ctx.full_batch(device="cpu")
    assert torch.equal(again[0], cids)


def test_reuses_supplied_data(toy_run):
    """data= skips the HDF5 read and yields an identical dataset."""
    ctx = load_eval_context(toy_run["checkpoint"])
    reused = load_eval_context(
        toy_run["checkpoint"],
        data=(ctx.labeling_df, ctx.flux_df, ctx.reversibility_df),
    )
    assert torch.equal(reused.full_batch()[1], ctx.full_batch()[1])


def test_build_dataset_false_skips_dataset(toy_run):
    ctx = load_eval_context(toy_run["checkpoint"], build_dataset=False)
    assert ctx.dataset is None
    with pytest.raises(ValueError, match="build_dataset=False"):
        ctx.loader()


# ── build_flux_decoder ────────────────────────────────────────────────────────

def test_build_flux_decoder_matches_saved_weights(toy_run):
    """Reconstructed weights are elementwise the saved ones.

    This is the assertion that protects every analysis script: it proves the
    frozen embedding tables and null-space buffers rebuilt from the context are
    the ones training saved, not freshly initialised stand-ins.
    """
    ctx = load_eval_context(toy_run["checkpoint"])
    model = build_flux_decoder(ctx, device="cpu")

    saved = ctx.ckpt["model_state_dict"]
    rebuilt = model.state_dict()
    assert set(saved) == set(rebuilt)
    differing = [k for k in saved if not torch.equal(saved[k].cpu(), rebuilt[k].cpu())]
    assert not differing, f"weights differ: {differing}"
    assert not model.training


def test_ensemble_shares_one_context(toy_run):
    """Members differ in weights but share the context's basis and token map."""
    ctx = load_eval_context(toy_run["checkpoint"])
    models = [build_flux_decoder(ctx, p, device="cpu") for p in toy_run["member_paths"]]
    assert len(models) == 2

    a, b = (m.state_dict() for m in models)
    assert torch.equal(a["token_experiment_ids"], b["token_experiment_ids"])
    assert torch.equal(a["Z_full"], b["Z_full"])
    # the trained weights themselves are seed-varied
    assert any(not torch.equal(a[k], b[k]) for k in a), "members are identical"

    (cids, lids, tids, fracs), targets = ctx.full_batch(device="cpu")
    fx = models[0].fixed_idx_t
    with torch.no_grad():
        preds = [m(cids, lids, tids, fracs,
                   measured_fluxes=targets.index_select(1, fx), measured_flux_ids=fx)[0]
                 for m in models]
    assert preds[0].shape == (len(toy_run["val_indices"]), toy_run["n_fluxes"])
    assert not torch.allclose(preds[0], preds[1])


# ── helpers ───────────────────────────────────────────────────────────────────

def test_promote_norm_buffers_fills_missing_stats():
    """Older checkpoints omit the norm stats the weights already carry."""
    state = {"y_norm_mean": torch.tensor([1.0, 2.0]), "y_norm_std": torch.tensor([3.0, 4.0]),
             "rev_norm_mean": torch.tensor([5.0]), "rev_norm_std": torch.tensor([6.0])}
    original = {"y_mean": None, "y_std": None, "rev_log_mean": None, "rev_log_std": None}
    out = promote_norm_buffers(original, state)

    assert np.allclose(out["y_mean"], [1.0, 2.0])
    assert np.allclose(out["rev_log_std"], [6.0])
    assert original["y_mean"] is None, "input dict was mutated"

    kept = promote_norm_buffers({"y_mean": np.array([9.0])}, state)
    assert np.allclose(kept["y_mean"], [9.0]), "existing stats must win"


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(None).type in ("cpu", "cuda")


def test_flux_decoder_config_ignores_extra_keys(toy_run):
    """Checkpoints carry bookkeeping keys the dataclass does not accept."""
    raw = dict(torch.load(toy_run["checkpoint"], map_location="cpu",
                          weights_only=False)["model_config"])
    raw["not_a_field"] = 123
    assert flux_decoder_config(raw).n_fluxes == toy_run["n_fluxes"]


# ── ensemble prediction (the default predict path) ────────────────────────────

def test_ensemble_predict_is_median(toy_run):
    """EnsembleSigmaFlux returns the elementwise median with SigmaFlux's schema."""
    ens = EnsembleSigmaFlux(toy_run["config"], toy_run["ensemble_dir"], device="cpu")
    assert len(ens.members) == 2

    df_L = _toy_df_L(n_samples=2)
    combined = ens.predict(df_L)
    per_member = np.stack([m.predict(df_L)["Value"].to_numpy() for m in ens.members])

    assert np.allclose(combined["Value"].to_numpy(), np.median(per_member, axis=0))

    single = SigmaFlux(toy_run["config"], toy_run["member_paths"][0], device="cpu")
    assert list(combined.columns) == list(single.predict(df_L).columns)
    assert len(combined) == len(single.predict(df_L))


def test_ensemble_missing_members_raises(tmp_path, toy_run):
    with pytest.raises(FileNotFoundError, match="No model_.*pt"):
        EnsembleSigmaFlux(toy_run["config"], str(tmp_path), device="cpu")


def test_load_predictor_prefers_sibling_ensemble(toy_run):
    """A sibling ensemble/ wins over the single .pt it was pointed at."""
    predictor = load_predictor(toy_run["config"], toy_run["checkpoint"], device="cpu")
    assert isinstance(predictor, EnsembleSigmaFlux)
    assert len(predictor.members) == 2


def test_load_predictor_falls_back_to_single_model(tmp_path, toy_run):
    """With no ensemble/ next to it, the single checkpoint is used."""
    import shutil
    lone = tmp_path / "model.pt"
    shutil.copy(toy_run["checkpoint"], lone)
    predictor = load_predictor(toy_run["config"], str(lone), device="cpu")
    assert isinstance(predictor, SigmaFlux)
