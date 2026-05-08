"""
Train SigmaFlux's labeling-to-flux transformer from generated HDF5 data.

Example:
    PYTHONPATH=src python paper/train_ai_model.py --config paper/io_config.yaml
"""

import argparse
import datetime
import os
from functools import partial

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_dataset,
)
from sigmaflux.ai.model import LabelingToFluxTransformer
from sigmaflux.ai.train import train_model, var_norm_mse_loss


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
    flux_variance = torch.var(targets[:, :flux_dim], dim=0).clamp_min(1.0)
    reversibility_variance = torch.var(targets[:, flux_dim:], dim=0).clamp_min(1.0)

    return {
        "flux": partial(var_norm_mse_loss, precomputed_variance=flux_variance),
        "reversibility": partial(var_norm_mse_loss, precomputed_variance=reversibility_variance),
    }


def r2_score(predictions, targets):
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    variable = targets.std(dim=0) > 1e-8
    if not variable.any():
        return torch.zeros(0), 0

    predictions = predictions[:, variable]
    targets = targets[:, variable]
    ss_res = ((predictions - targets) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    return torch.nan_to_num(r2, nan=0.0), int(variable.sum())


def evaluate_validation_correlations(model, val_loader, device, flux_dim):
    model.eval()
    flux_predictions = []
    flux_targets = []
    reversibility_predictions = []
    reversibility_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred_flux, pred_reversibility = model(inputs, mask=None)

            flux_predictions.append(pred_flux)
            flux_targets.append(targets[:, :flux_dim])
            reversibility_predictions.append(pred_reversibility)
            reversibility_targets.append(targets[:, flux_dim:])

    flux_r2, n_flux_r2 = r2_score(
        torch.cat(flux_predictions, dim=0),
        torch.cat(flux_targets, dim=0),
    )
    reversibility_r2, n_reversibility_r2 = r2_score(
        torch.cat(reversibility_predictions, dim=0),
        torch.cat(reversibility_targets, dim=0),
    )

    return {
        "flux_mean": float(flux_r2.mean()) if len(flux_r2) else 0.0,
        "flux_median": float(flux_r2.median()) if len(flux_r2) else 0.0,
        "flux_n": n_flux_r2,
        "reversibility_mean": float(reversibility_r2.mean()) if len(reversibility_r2) else 0.0,
        "reversibility_median": float(reversibility_r2.median()) if len(reversibility_r2) else 0.0,
        "reversibility_n": n_reversibility_r2,
    }


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
        labeling_df = labeling_df.iloc[:, :args.max_samples]
        flux_df = flux_df.iloc[:, :args.max_samples]
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

    train_dataset, norm_stats, token_experiment_ids = prepare_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=train_indices,
        normalize=True,
        measured_fluxes=args.measured_fluxes,
    )
    val_dataset, _, _ = prepare_dataset(
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
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    flux_dim = flux_df.shape[0]
    reversibility_dim = reversibility_df.shape[0]

    model_config = {
        "token_dim": vocabs['token_dim'],
        "flux_dim": flux_dim,
        "reversibility_dim": reversibility_dim,
        "combined_dim": args.combined_dim,
        "dropout": args.dropout,
        "num_attn_heads": args.num_attn_heads,
        "num_transformer_layers": args.num_transformer_layers,
    }

    model = LabelingToFluxTransformer(
        stoichiometric_matrix=stoichiometric_matrix,
        token_experiment_ids=token_experiment_ids,
        **model_config,
    )

    loss_function = build_loss_function(train_dataset, flux_dim, args.loss)

    log = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        loss_function=loss_function,
        gradient_clip=args.gradient_clip,
        use_scheduler=not args.no_scheduler,
        mask_prob=args.mask_prob,
        min_experiments=args.min_experiments,
        warmup_epochs=args.warmup_epochs,
    )

    val_correlations = evaluate_validation_correlations(
        model,
        val_loader,
        device,
        flux_dim,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "vocabs": {k: v for k, v in vocabs.items()
                   if k not in ('compound_emb_table', 'label_emb_table', 'tracer_emb_table')},
        "token_experiment_ids": token_experiment_ids.tolist(),
        "norm_stats": norm_stats,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "log": log,
        "val_correlations": val_correlations,
        "data_dir": args.data_dir,
        "model_json": args.model_json,
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
    parser = argparse.ArgumentParser(description="Train the SigmaFlux labeling-to-flux transformer.")
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    t = cfg.get("training", {})
    output_dir = f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"
    output_model = os.path.join(output_dir, t.get("output_model", "transformer.pt"))

    measured_fluxes = t.get("measured_fluxes") or None  # list of reaction IDs, or None

    args = argparse.Namespace(
        data_dir=output_dir,
        model_json=os.path.join(_REPO_ROOT, cfg["model_file"]) if not os.path.isabs(cfg["model_file"]) else cfg["model_file"],
        output=output_model,
        device=cfg.get("device", "auto"),
        seed=cfg.get("seed", 42),
        max_samples=t.get("max_samples"),
        epochs=t.get("epochs", 100),
        batch_size=t.get("batch_size", 64),
        learning_rate=t.get("learning_rate", 1e-3),
        warmup_epochs=t.get("warmup_epochs", 5),
        gradient_clip=t.get("gradient_clip", 1.0),
        no_scheduler=not t.get("scheduler", True),
        loss=t.get("loss", "variance"),
        val_fraction=t.get("val_fraction", 0.2),
        combined_dim=t.get("combined_dim", 256),
        num_attn_heads=t.get("num_attn_heads", 8),
        num_transformer_layers=t.get("num_transformer_layers", 2),
        dropout=t.get("dropout", 0.15),
        mask_prob=t.get("mask_prob", 0.2),
        min_experiments=t.get("min_experiments", 1),
        measured_fluxes=measured_fluxes,
        h_compound=t.get("h_compound"),   # None → auto from formula
        h_label=t.get("h_label"),
        h_tracer=t.get("h_tracer"),
    )
    main(args)
