"""
Train SigmaFlux's FluxDecoder (cross-attention decoder) from generated HDF5 data.

Example:
    PYTHONPATH=src python paper/train_ai_model.py --config paper/io_config.yaml
"""

import argparse
import datetime
import os
from dataclasses import asdict
from functools import partial

import numpy as np
import scipy.linalg
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.train import evaluate_validation_correlations, train_model, var_norm_mse_loss
from sigmaflux.ai.utils import compute_sparse_nullspace


def split_indices(n_samples, val_fraction, seed):
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int((1.0 - val_fraction) * n_samples)
    return indices[:split], indices[split:]


def build_loss_function(train_dataset, flux_dim, loss_name):
    if loss_name == 'mse':
        return None  # train_model defaults to MSELoss
    targets  = train_dataset.targets
    flux_var = torch.var(targets[:, :flux_dim], dim=0).clamp_min(0.1)
    rev_var  = torch.var(targets[:, flux_dim:], dim=0).clamp_min(0.1)
    return {
        'flux':          partial(var_norm_mse_loss, precomputed_variance=flux_var),
        'reversibility': partial(var_norm_mse_loss, precomputed_variance=rev_var),
    }


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labeling_df, flux_df, reversibility_df = load_data(args.data_dir)
    if args.max_samples is not None and args.max_samples < labeling_df.shape[1]:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(labeling_df.shape[1])[:args.max_samples]
        labeling_df      = labeling_df.iloc[:, idx]
        flux_df          = flux_df.iloc[:, idx]
        reversibility_df = reversibility_df.iloc[:, idx]
        print(f"Randomly subsampled {args.max_samples} samples (seed={args.seed})")

    _, base_tracer_names = create_tracer_mapping(labeling_df)
    stoichiometric_matrix, cobra_model = load_stoichiometric_matrix(args.model_json)

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound=args.h_compound,
        h_label=args.h_label,
        h_tracer=args.h_tracer,
    )

    train_indices, val_indices = split_indices(labeling_df.shape[1], args.val_fraction, args.seed)

    # Drop constant reversibilities before building datasets — they add dead queries.
    _rv_std = reversibility_df.values.astype(np.float32).std(axis=1)
    rev_variable_idx = np.where(_rv_std > 1e-6)[0]
    reversibility_df = reversibility_df.iloc[rev_variable_idx]
    print(f"Reversibilities: {len(rev_variable_idx)} variable / {_rv_std.shape[0]} total (dropped {_rv_std.shape[0] - len(rev_variable_idx)})")

    norm_stats_input = None
    if args.normalize:
        _fv = flux_df.values.astype(np.float32)
        flux_mean_all = _fv.mean(axis=1)
        flux_std_all  = np.where(_fv.std(axis=1) < 1e-6, 1.0, _fv.std(axis=1))
        _rv = reversibility_df.values.astype(np.float32)
        rev_mean_all  = _rv.mean(axis=1)
        rev_std_all   = np.where(_rv.std(axis=1) < 1e-6, 1.0, _rv.std(axis=1))
        norm_stats_input = {
            'flux_mean': flux_mean_all.tolist(),
            'flux_std':  flux_std_all.tolist(),
            'rev_mean':  rev_mean_all.tolist(),
            'rev_std':   rev_std_all.tolist(),
        }
        print(f"Flux/rev norm stats computed from all {flux_df.shape[1]} samples")

    (train_dataset, norm_stats, token_experiment_ids,
     tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=train_indices,
        measured_fluxes=args.measured_fluxes,
        normalize=args.normalize,
        norm_stats_input=norm_stats_input,
    )
    (val_dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=val_indices,
        measured_fluxes=args.measured_fluxes,
        norm_stats_input=norm_stats,
        normalize=args.normalize,
    )

    _fmean = norm_stats.get('flux_mean')
    _fstd  = norm_stats.get('flux_std')
    flux_mean_t = torch.tensor(_fmean, dtype=torch.float32) if _fmean is not None else None
    flux_std_t  = torch.tensor(_fstd,  dtype=torch.float32) if _fstd  is not None else None

    if _fmean is not None:
        _tgt_f = train_dataset.targets[:, :flux_df.shape[0]]
        _tgt_r = train_dataset.targets[:, flux_df.shape[0]:]
        print(f"Flux targets:  mean=[{_tgt_f.mean(dim=0).min():.3f}, {_tgt_f.mean(dim=0).max():.3f}]  std=[{_tgt_f.std(dim=0).min():.3f}, {_tgt_f.std(dim=0).max():.3f}]")
        print(f"Rev  targets:  mean=[{_tgt_r.mean(dim=0).min():.3f}, {_tgt_r.mean(dim=0).max():.3f}]  std=[{_tgt_r.std(dim=0).min():.3f}, {_tgt_r.std(dim=0).max():.3f}]")
    else:
        print("Targets NOT normalized (normalize=False)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    flux_dim          = flux_df.shape[0]
    reversibility_dim = reversibility_df.shape[0]

    config = FluxDecoderConfig(
        d_model     = args.combined_dim,
        n_heads     = args.num_attn_heads,
        n_layers    = args.num_transformer_layers,
        dropout     = args.dropout,
        n_compounds = len(vocabs['compound_vocab']),
        n_labels    = len(vocabs['label_vocab']),
        n_tracers   = len(tracer_combo_vocab),
        d_comp      = vocabs['h_compound'] * 2,
        d_label     = vocabs['h_label'] * 2,
        d_tracer    = vocabs['h_tracer'] * 2,
        n_fluxes    = flux_dim,
        n_rev       = reversibility_dim,
        n_measured  = len(args.measured_fluxes) if args.measured_fluxes else 0,
    )

    fixed_flux_indices = None
    if args.measured_fluxes:
        fixed_flux_indices = [flux_df.index.get_loc(name) for name in args.measured_fluxes]

    # ── Null-space reparameterization (full null space) ─────────────────────
    # Compute the orthonormal null-space basis Z_full of the FULL stoichiometric
    # matrix S. The model predicts basis coefficients y ∈ R^38 and reconstructs
    # all 386 fluxes as V = y @ Z_full.T, guaranteeing S@V=0 by construction.
    # Measured fluxes are fed as a soft feature through fusion_layer — not hard-enforced.
    free_idx  = [i for i in range(flux_dim) if i not in set(fixed_flux_indices or [])]
    fixed_idx = fixed_flux_indices or []

    Z_full       = compute_sparse_nullspace(stoichiometric_matrix)  # (n_fluxes, n_basis=38)
    n_free_basis = Z_full.shape[1]

    print(f"Full null-space basis dimension: {n_free_basis}  (was {flux_dim - len(fixed_idx) - n_free_basis + n_free_basis} from S_free)")

    # ── Compute y normalization and observability weights from training data ──
    # y_true = v_targets @ Z_full  (project all flux targets onto null-space basis)
    # Normalize so the model predicts y_norm ≈ N(0,1) — clean gradient signal.
    # obs_weights[i] = max|Pearson r(y_i, labeling_j)| across all labeling features.
    if n_free_basis > 0:
        train_tgts      = train_dataset.targets                          # (n_train, flux+rev)
        train_tgts_flux = train_tgts[:, :flux_dim]                       # (n_train, 386)
        Z_t             = torch.tensor(Z_full, dtype=torch.float32)
        y_train         = train_tgts_flux @ Z_t                          # (n_train, 38)
        y_mean          = y_train.mean(dim=0).numpy()
        y_std           = y_train.std(dim=0).clamp_min(1e-6).numpy()
        print(f"y_true stats: mean_abs={np.abs(y_mean).mean():.2f}, "
              f"std range=[{y_std.min():.2f}, {y_std.max():.2f}]")

        # Observability: Pearson r between each y-component and each labeling feature
        L_train = torch.tensor(
            labeling_df.iloc[:, train_indices].values.T, dtype=torch.float32
        )                                                                # (n_train, n_meas)
        y_c     = y_train - y_train.mean(dim=0, keepdim=True)
        L_c     = L_train - L_train.mean(dim=0, keepdim=True)
        y_std_c = y_c.std(dim=0).clamp_min(1e-10)
        L_std_c = L_c.std(dim=0).clamp_min(1e-10)
        cov_yl  = (y_c.T @ L_c) / y_train.shape[0]                     # (38, n_meas)
        corr_yl = cov_yl / y_std_c.unsqueeze(1) / L_std_c.unsqueeze(0)
        obs_weights = corr_yl.abs().max(dim=1).values                   # (38,) max|r|
        obs_weights_np = obs_weights.numpy()
        n_obs = int((obs_weights > 0.1).sum().item())
        print(f"Observability: {n_obs}/{n_free_basis} y-components have max|r|>0.1 "
              f"; max|r| range=[{obs_weights_np.min():.3f}, {obs_weights_np.max():.3f}]")
    else:
        y_mean, y_std, obs_weights_np = None, None, None

    # Builf the null-space data dict, which the model uses to reconstruct fluxes 
    # the latent space 
    null_space_data = dict(
        Z_free          = Z_full,               # (n_fluxes, 38) — field name kept for compat
        free_idx        = free_idx,             # metadata: indices of non-measured reactions
        fixed_idx       = fixed_idx,            # metadata: indices of measured reactions
        obs_weights     = obs_weights_np,
    )

    config.n_free = n_free_basis

    model = FluxDecoder(
        config               = config,
        token_experiment_ids = token_experiment_ids,
        compound_emb_table   = vocabs['compound_emb_table'],
        label_emb_table      = vocabs['label_emb_table'],
        tracer_emb_table     = tracer_combo_emb_table,
        null_space_data      = null_space_data,
    )
    model.flux_noise_std = args.flux_noise_std

    loss_function = build_loss_function(train_dataset, flux_dim, args.loss)

    log = train_model(
        model,
        train_loader,
        val_loader        = val_loader,
        num_epochs        = args.epochs,
        learning_rate     = args.learning_rate,
        device            = device,
        gradient_clip     = args.gradient_clip,
        use_scheduler     = not args.no_scheduler,
        warmup_epochs     = args.warmup_epochs,
        met_mask_prob     = args.met_mask_prob,
        loss_function     = loss_function,
    )

    flux_names = flux_df.index.tolist()
    val_correlations = evaluate_validation_correlations(model, val_loader, device, flux_dim, flux_names=flux_names)

    model_config = asdict(config)

    # Token index: list of (Tracer, Metabolite, Isotopomer) tuples — defines
    # the full token sequence the model was trained on. Stored so inference
    # can reconstruct the token structure without needing the training HDF5 data.
    token_index = list(labeling_df.index)   # list of (Tracer, Metabolite, Isotopomer)

    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "model_config":         model_config,
        "vocabs": {k: v for k, v in vocabs.items()
                   if k not in ('compound_emb_table', 'label_emb_table', 'tracer_emb_table')},
        "tracer_combo_vocab":   tracer_combo_vocab,
        "token_experiment_ids": token_experiment_ids.tolist(),
        "token_index":          token_index,
        "norm_stats":           norm_stats,
        "train_indices":        train_indices.tolist(),
        "val_indices":          val_indices.tolist(),
        "log":                  log,
        "val_correlations":     val_correlations,
        "flux_names":           flux_names,
        "reversibility_names":  reversibility_df.index.tolist(),
        "measured_fluxes":      args.measured_fluxes,
        "null_space_data": {
            "Z_free":           Z_full,              # (n_fluxes, 38) full null-space basis
            "free_idx":         free_idx,
            "fixed_idx":        fixed_idx,
            "obs_weights":      obs_weights_np,
        },
        "data_dir":             args.data_dir,
        "model_json":           args.model_json,
    }
    torch.save(checkpoint, args.output)

    print(f"\nSaved {args.output}")
    print(f"Final train loss: {log['train_loss'][-1]:.6g}")
    print(f"Final val loss: {log['val_loss'][-1]:.6g}")
    print(
        "Validation flux R²: "
        f"mean={val_correlations['flux_mean']:.4f}, "
        f"median={val_correlations['flux_median']:.4f}, "
        f"n={val_correlations['flux_n']}"
    )
    print(
        "Validation reversibility R²: "
        f"mean={val_correlations['reversibility_mean']:.4f}, "
        f"median={val_correlations['reversibility_median']:.4f}, "
        f"n={val_correlations['reversibility_n']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SigmaFlux FluxDecoder.")
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    parser.add_argument("--data-dir", default=None, help="Directory containing HDF5 training data (overrides date-based default)")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    t = cfg.get("training", {})
    output_dir   = cli.data_dir or f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"
    output_model = os.path.join(output_dir, t.get("output_model", "transformer.pt"))

    measured_fluxes = t.get("measured_fluxes") or None

    args = argparse.Namespace(
        data_dir               = output_dir,
        model_json             = os.path.join(_REPO_ROOT, cfg["model_file"]) if not os.path.isabs(cfg["model_file"]) else cfg["model_file"],
        output                 = output_model,
        device                 = cfg.get("device", "auto"),
        seed                   = cfg.get("seed", 42),
        max_samples            = t.get("max_samples"),
        epochs                 = t.get("epochs", 100),
        batch_size             = t.get("batch_size", 64),
        learning_rate          = t.get("learning_rate", 1e-3),
        warmup_epochs          = t.get("warmup_epochs", 5),
        gradient_clip          = t.get("gradient_clip", 1.0),
        no_scheduler           = not t.get("scheduler", True),
        val_fraction           = t.get("val_fraction", 0.2),
        combined_dim           = t.get("combined_dim", 64),
        num_attn_heads         = t.get("num_attn_heads", 4),
        num_transformer_layers = t.get("num_transformer_layers", 4),
        dropout                = t.get("dropout", 0.3),
        met_mask_prob          = t.get("met_mask_prob", 0.0),
        measured_fluxes        = measured_fluxes,
        h_compound             = t.get("h_compound"),
        h_label                = t.get("h_label"),
        h_tracer               = t.get("h_tracer"),
        flux_noise_std         = t.get("flux_noise_std", 0.0),
        normalize      = t.get("normalize", True),
        loss                   = t.get("loss", "variance"),
    )
    main(args)
