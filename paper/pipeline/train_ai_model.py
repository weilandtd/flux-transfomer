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
from sigmaflux.ai.train import train_model, var_norm_mse_loss
from sigmaflux.ai.utils import compute_sparse_nullspace


def split_indices(n_samples, val_fraction, seed):
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int((1.0 - val_fraction) * n_samples)
    return indices[:split], indices[split:]


def build_loss_function(train_dataset, flux_dim, loss_name):
    if loss_name == "mse":
        return None

    targets = train_dataset.targets  # (num_samples, flux_dim + rev_dim)
    flux_variance          = torch.var(targets[:, :flux_dim], dim=0).clamp_min(1.0)
    reversibility_variance = torch.var(targets[:, flux_dim:], dim=0).clamp_min(1.0)

    return {
        "flux":          partial(var_norm_mse_loss, precomputed_variance=flux_variance),
        "reversibility": partial(var_norm_mse_loss, precomputed_variance=reversibility_variance),
    }


def r2_score(predictions, targets):
    predictions = predictions.detach().cpu()
    targets     = targets.detach().cpu()
    variable    = targets.std(dim=0) > 1e-8
    if not variable.any():
        return torch.zeros(0), 0

    predictions = predictions[:, variable]
    targets     = targets[:, variable]
    ss_res = ((predictions - targets) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2     = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    return torch.nan_to_num(r2, nan=0.0), int(variable.sum())


def evaluate_validation_correlations(model, val_loader, device, flux_dim, flux_names=None):
    model.eval()
    flux_preds, flux_tgts, rev_preds, rev_tgts = [], [], [], []

    with torch.no_grad():
        for batch_data in val_loader:
            (cids, lids, tids, fracs), targets = batch_data
            cids    = cids.to(device)
            lids    = lids.to(device)
            tids    = tids.to(device)
            fracs   = fracs.to(device)
            targets = targets.to(device)

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                fixed_idx = (model.null_decoder.fixed_idx if model.null_decoder is not None
                             else model.fba_pro.fixed_idx)
                fixed_fluxes = targets[:, fixed_idx]
            pred_flux, pred_rev = model(cids, lids, tids, fracs, None, fixed_fluxes)
            flux_preds.append(pred_flux)
            flux_tgts.append(targets[:, :flux_dim])
            rev_preds.append(pred_rev)
            rev_tgts.append(targets[:, flux_dim:])

    all_flux_preds = torch.cat(flux_preds)
    all_flux_tgts  = torch.cat(flux_tgts)
    all_rev_preds  = torch.cat(rev_preds)
    all_rev_tgts   = torch.cat(rev_tgts)

    flux_r2, n_flux = r2_score(all_flux_preds, all_flux_tgts)
    rev_r2,  n_rev  = r2_score(all_rev_preds,  all_rev_tgts)

    result = {
        "flux_mean":            float(flux_r2.mean())   if len(flux_r2) else 0.0,
        "flux_median":          float(flux_r2.median()) if len(flux_r2) else 0.0,
        "flux_n":               n_flux,
        "reversibility_mean":   float(rev_r2.mean())    if len(rev_r2) else 0.0,
        "reversibility_median": float(rev_r2.median())  if len(rev_r2) else 0.0,
        "reversibility_n":      n_rev,
    }

    # Per-reaction R² — only for variable reactions; stored alongside names if provided
    if flux_names is not None and len(flux_r2) > 0:
        variable_mask = all_flux_tgts.std(dim=0) > 1e-8
        var_names = [flux_names[i] for i, v in enumerate(variable_mask.tolist()) if v]
        result["flux_r2_per_reaction"] = flux_r2.tolist()
        result["flux_reaction_names"]  = var_names

    return result


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labeling_df, flux_df, reversibility_df = load_data(args.data_dir, max_samples=args.max_samples)
    if args.max_samples is not None:
        labeling_df      = labeling_df.iloc[:, :args.max_samples]
        flux_df          = flux_df.iloc[:, :args.max_samples]
        reversibility_df = reversibility_df.iloc[:, :args.max_samples]
        print(f"Using first {labeling_df.shape[1]} samples after --max-samples")

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

    (train_dataset, norm_stats, token_experiment_ids,
     tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=train_indices,
        normalize=True,
        measured_fluxes=args.measured_fluxes,
    )
    (val_dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=val_indices,
        normalize=True,
        measured_fluxes=args.measured_fluxes,
        norm_stats_input=norm_stats,
    )

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

    # Normalization stats for fixed_fluxes (measured exchange fluxes).
    # Used to bring v_known from mmol/gDCW/h scale to ~N(0,1) before fusion_layer,
    # preventing scale mismatch from corrupting unobservable null-space directions.
    t_fixed = train_dataset.targets[:, fixed_idx]       # (n_train, 8)
    fixed_flux_mean_np = t_fixed.mean(dim=0).numpy()
    fixed_flux_std_np  = t_fixed.std(dim=0).clamp_min(1e-6).numpy()
    print(f"Fixed-flux normalization: mean={fixed_flux_mean_np.round(2)}, "
          f"std={fixed_flux_std_np.round(2)}")

    null_space_data = dict(
        Z_free          = Z_full,               # (n_fluxes, 38) — field name kept for compat
        free_idx        = free_idx,             # metadata: indices of non-measured reactions
        fixed_idx       = fixed_idx,            # metadata: indices of measured reactions
        y_mean          = y_mean,
        y_std           = y_std,
        obs_weights     = obs_weights_np,
        fixed_flux_mean = fixed_flux_mean_np,   # for fusion_layer input normalization
        fixed_flux_std  = fixed_flux_std_np,
    )

    config.n_free = n_free_basis

    model = FluxDecoder(
        config                = config,
        stoichiometric_matrix = stoichiometric_matrix,
        token_experiment_ids  = token_experiment_ids,
        compound_emb_table    = vocabs['compound_emb_table'],
        label_emb_table       = vocabs['label_emb_table'],
        tracer_emb_table      = tracer_combo_emb_table,
        fixed_flux_indices    = fixed_flux_indices,
        null_space_data       = null_space_data,
    )
    # Noise injection magnitude for denoising autoencoder training (plain float, not a parameter)
    model.flux_noise_std = args.flux_noise_std

    loss_function = build_loss_function(train_dataset, flux_dim, args.loss)

    log = train_model(
        model,
        train_loader,
        val_loader        = val_loader,
        num_epochs        = args.epochs,
        learning_rate     = args.learning_rate,
        device            = device,
        loss_function     = loss_function,
        gradient_clip     = args.gradient_clip,
        use_scheduler     = not args.no_scheduler,
        warmup_epochs     = args.warmup_epochs,
        y_l2_reg          = args.y_l2_reg,
        met_mask_prob     = args.met_mask_prob,
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
            "y_mean":           y_mean,
            "y_std":            y_std,
            "obs_weights":      obs_weights_np,
            "fixed_flux_mean":  fixed_flux_mean_np,
            "fixed_flux_std":   fixed_flux_std_np,
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
        loss                   = t.get("loss", "variance"),
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
        y_l2_reg               = t.get("y_l2_reg", 0.0),
    )
    main(args)
