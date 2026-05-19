"""
Integration tests for the AI inference pipeline using the toy one-carbon model.

A module-scoped fixture builds a complete but minimal FluxDecoder checkpoint
(random weights, correct stoichiometric structure) without running training.
Tests verify output shape, mass-balance, fixed-flux propagation, and the
predict.py CLI end-to-end.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from dataclasses import asdict

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.inference import SigmaFlux
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.utils import compute_sparse_nullspace

_TOY_MODEL_JSON = os.path.join(_REPO_ROOT, 'models', 'toy', 'one_carbon_annotated.json')
_TOY_CONFIG_YAML = os.path.join(_REPO_ROOT, 'data', 'test', 'toy_config.yaml')

# Internal tracer ID: f"{tracer_id}_{labeling}_{enrichment}" (matches toy_config.yaml)
_TRACER_ID = 'Ser_in_111_0.1'


# ── helpers ───────────────────────────────────────────────────────────────────

def make_toy_df_L(n_samples: int = 2) -> pd.DataFrame:
    """Long-format labeling DataFrame compatible with SigmaFlux.predict."""
    rng = np.random.default_rng(7)
    records = []
    for s in range(n_samples):
        # Serine: 3 carbons → M+1, M+2, M+3 (exclude M+0)
        ser_w = rng.dirichlet(np.ones(4))          # M+0..M+3
        for iso, frac in zip(['M+1', 'M+2', 'M+3'], ser_w[1:]):
            records.append({'Tracer Name': 'labeled-Ser', 'Sample': s,
                            'Metabolite ID': 'Ser', 'Isotopomer': iso,
                            'Fraction': float(frac)})
        # Glycine: 2 carbons → M+1, M+2 (exclude M+0)
        gly_w = rng.dirichlet(np.ones(3))          # M+0..M+2
        for iso, frac in zip(['M+1', 'M+2'], gly_w[1:]):
            records.append({'Tracer Name': 'labeled-Ser', 'Sample': s,
                            'Metabolite ID': 'Gly', 'Isotopomer': iso,
                            'Fraction': float(frac)})
    return pd.DataFrame(records)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def toy_checkpoint(tmp_path_factory):
    """
    Build a complete FluxDecoder checkpoint on the toy model without training.

    The model has random weights but the correct stoichiometric structure,
    null-space basis, and normalisation statistics.  Non-zero fusion_layer
    weights are set so that fixed_fluxes visibly influence predictions.
    """
    S, cobra_model = load_stoichiometric_matrix(_TOY_MODEL_JSON)
    flux_names = [r.id for r in cobra_model.reactions]   # 23 reactions
    n_fluxes   = len(flux_names)
    n_samples  = 30

    measured_fluxes = ['EX_Ser_in']
    fixed_idx = [flux_names.index(r) for r in measured_fluxes]
    free_idx  = [i for i in range(n_fluxes) if i not in set(fixed_idx)]

    # ── Synthetic labeling_df: 2 metabolites × 3 isotopomers × 1 tracer ──────
    # Use internal tracer ID (SigmaFlux maps user names to these at predict time)
    rows = [
        (_TRACER_ID, 'Ser', 'M+1'), (_TRACER_ID, 'Ser', 'M+2'), (_TRACER_ID, 'Ser', 'M+3'),
        (_TRACER_ID, 'Gly', 'M+1'), (_TRACER_ID, 'Gly', 'M+2'),
    ]
    midx = pd.MultiIndex.from_tuples(rows, names=['Tracer', 'Metabolite', 'Isotopomer'])
    rng = np.random.default_rng(42)
    # Each column sums to ≈1 per metabolite (realistic MID fractions)
    ser_fracs = rng.dirichlet(np.ones(3), n_samples).T.astype(np.float32)  # (3, n)
    gly_fracs = rng.dirichlet(np.ones(2), n_samples).T.astype(np.float32)  # (2, n)
    frac_data = np.vstack([ser_fracs, gly_fracs])                           # (5, n)
    labeling_df = pd.DataFrame(frac_data, index=midx, columns=list(range(n_samples)))

    # ── Synthetic flux_df and reversibility_df ────────────────────────────────
    flux_data = np.abs(rng.normal(2.0, 1.0, (n_fluxes, n_samples))).astype(np.float32)
    flux_df   = pd.DataFrame(flux_data, index=flux_names, columns=list(range(n_samples)))
    rev_data  = np.clip(rng.normal(0.5, 0.2, (n_fluxes, n_samples)), 0, 1).astype(np.float32)
    rev_df    = pd.DataFrame(rev_data, index=flux_names, columns=list(range(n_samples)))

    # ── Build vocabs and dataset (no bypass tokens — matches inference.py) ────
    _, base_tracer_names = create_tracer_mapping(labeling_df)
    vocabs = build_radial_vocabs(cobra_model, labeling_df, base_tracer_names)

    (dataset, norm_stats, token_experiment_ids,
     tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df, flux_df, rev_df,
        vocabs          = vocabs,
        train_indices   = list(range(n_samples)),
        normalize       = True,
        measured_fluxes = None,
    )

    # ── Null-space basis and y normalisation ──────────────────────────────────
    Z_full  = compute_sparse_nullspace(S)
    n_free  = Z_full.shape[1]
    Z_t     = torch.tensor(Z_full, dtype=torch.float32)

    flux_targets = dataset.targets[:, :n_fluxes]          # (n_samples, n_fluxes)
    y_train = flux_targets @ Z_t
    y_mean  = y_train.mean(dim=0).numpy()
    y_std   = y_train.std(dim=0).clamp_min(1e-6).numpy()

    # Fixed-flux normalisation (EX_Ser_in column of targets)
    t_fixed          = dataset.targets[:, fixed_idx]
    fixed_flux_mean  = t_fixed.mean(dim=0).numpy()
    fixed_flux_std   = t_fixed.std(dim=0).clamp_min(1e-6).numpy()

    obs_weights = np.ones(n_free, dtype=np.float32)

    null_space_data = dict(
        Z_free          = Z_full,
        free_idx        = free_idx,
        fixed_idx       = fixed_idx,
        y_mean          = y_mean,
        y_std           = y_std,
        obs_weights     = obs_weights,
        fixed_flux_mean = fixed_flux_mean,
        fixed_flux_std  = fixed_flux_std,
    )

    # ── Build FluxDecoder ─────────────────────────────────────────────────────
    cfg = FluxDecoderConfig(
        d_model     = 16,
        n_heads     = 1,
        n_layers    = 1,
        dropout     = 0.0,
        n_compounds = len(vocabs['compound_vocab']),
        n_labels    = len(vocabs['label_vocab']),
        n_tracers   = len(tracer_combo_vocab),
        d_comp      = vocabs['h_compound'] * 2,
        d_label     = vocabs['h_label']    * 2,
        d_tracer    = vocabs['h_tracer']   * 2,
        n_fluxes    = n_fluxes,
        n_rev       = n_fluxes,
        n_measured  = len(measured_fluxes),
        n_free      = n_free,
    )

    model = FluxDecoder(
        cfg,
        stoichiometric_matrix = S,
        token_experiment_ids  = token_experiment_ids,
        compound_emb_table    = vocabs['compound_emb_table'],
        label_emb_table       = vocabs['label_emb_table'],
        tracer_emb_table      = tracer_combo_emb_table,
        fixed_flux_indices    = fixed_idx,
        null_space_data       = null_space_data,
    )

    # Give fusion_layer non-trivial weights so fixed_fluxes visibly affect output
    if model.fusion_layer is not None:
        with torch.no_grad():
            model.fusion_layer.weight.fill_(0.5)
            model.fusion_gate.fill_(2.0)   # sigmoid(2) ≈ 0.88

    # ── Assemble and save checkpoint ──────────────────────────────────────────
    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'model_config':         asdict(cfg),
        'vocabs': {k: v for k, v in vocabs.items()
                   if k not in ('compound_emb_table', 'label_emb_table', 'tracer_emb_table')},
        'tracer_combo_vocab':   tracer_combo_vocab,
        'token_experiment_ids': token_experiment_ids.tolist(),
        'token_index':          list(labeling_df.index),
        'norm_stats':           norm_stats,
        'train_indices':        list(range(n_samples)),
        'val_indices':          [],
        'log':                  {'train_loss': [], 'val_loss': [], 'grad_norm': []},
        'val_correlations':     {},
        'flux_names':           flux_names,
        'reversibility_names':  flux_names,
        'measured_fluxes':      measured_fluxes,
        'null_space_data':      null_space_data,
        'data_dir':             '',
        'model_json':           _TOY_MODEL_JSON,
    }

    ckpt_path = str(tmp_path_factory.mktemp('ckpt') / 'toy_ckpt.pt')
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


# ── tests ─────────────────────────────────────────────────────────────────────

def test_predict_output_shape(toy_checkpoint):
    """Predictions have the right number of rows and finite values."""
    sf     = SigmaFlux(_TOY_CONFIG_YAML, toy_checkpoint)
    df_L   = make_toy_df_L(n_samples=2)
    result = sf.predict(df_L)

    _, cobra_model = load_stoichiometric_matrix(_TOY_MODEL_JSON)
    n_fluxes = len(cobra_model.reactions)   # 23

    assert isinstance(result, pd.DataFrame)
    assert {'Flux Name', 'Value'}.issubset(result.columns)
    # n_fluxes flux rows + n_fluxes reversibility rows, per sample × 2 samples
    assert len(result) == 2 * n_fluxes * 2
    assert result['Value'].notna().all()
    assert np.isfinite(result['Value'].values).all()


def test_mass_balance(toy_checkpoint):
    """Predicted fluxes satisfy S @ v ≈ 0 (guaranteed by NullSpaceDecoder)."""
    S, cobra_model = load_stoichiometric_matrix(_TOY_MODEL_JSON)
    flux_names = [r.id for r in cobra_model.reactions]
    n_fluxes   = len(flux_names)

    sf     = SigmaFlux(_TOY_CONFIG_YAML, toy_checkpoint)
    result = sf.predict(make_toy_df_L(n_samples=1))   # single sample → no Sample col

    # First n_fluxes rows are the flux predictions; the rest are reversibilities
    flux_vals = result.iloc[:n_fluxes]['Value'].values   # (23,)
    sv        = S @ flux_vals
    assert np.abs(sv).max() < 1e-4, f"Max |Sv| = {np.abs(sv).max():.2e}"


def test_fixed_fluxes_affect_prediction(toy_checkpoint):
    """
    Predictions change when df_Vm values differ.

    This would silently fail if fixed_fluxes were double-normalised (the
    bug fixed in inference.py) because the mangled signal would be near
    constant regardless of the user-supplied value.
    """
    sf   = SigmaFlux(_TOY_CONFIG_YAML, toy_checkpoint)
    df_L = make_toy_df_L(n_samples=1)

    df_Vm_lo = pd.DataFrame({'Sample': [0], 'Flux Name': ['EX_Ser_in'], 'Value': [0.01]})
    df_Vm_hi = pd.DataFrame({'Sample': [0], 'Flux Name': ['EX_Ser_in'], 'Value': [50.0]})

    res_lo = sf.predict(df_L, df_Vm=df_Vm_lo)['Value'].values
    res_hi = sf.predict(df_L, df_Vm=df_Vm_hi)['Value'].values

    assert not np.allclose(res_lo, res_hi, atol=1e-6), (
        "Predictions are identical for very different EX_Ser_in values — "
        "fixed_fluxes may not be reaching fusion_layer."
    )


def test_predict_cli(toy_checkpoint, tmp_path):
    """predict.py CLI runs end-to-end and produces a valid CSV."""
    labeling_csv = tmp_path / 'labeling.csv'
    out_csv      = tmp_path / 'preds.csv'
    make_toy_df_L(n_samples=1).to_csv(labeling_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            os.path.join(_REPO_ROOT, 'paper', 'pipeline', 'predict.py'),
            _TOY_CONFIG_YAML,
            toy_checkpoint,
            str(labeling_csv),
            '--out', str(out_csv),
        ],
        check=True,
        cwd=_REPO_ROOT,
    )

    result = pd.read_csv(out_csv)
    assert 'Flux Name' in result.columns
    assert 'Value'     in result.columns
    assert len(result) > 0
    assert np.isfinite(result['Value'].values).all()
