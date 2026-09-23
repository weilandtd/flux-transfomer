"""
Prediction utilities for flux prediction models.

Provides functions for single model and ensemble predictions with uncertainty estimation.
"""

import torch
from tqdm import tqdm


def predict_fluxes(model, data_loader, device='cuda', verbose=True):
    """Generate flux predictions from tokenized labeling data.
    Returns a (num_samples, flux_dim + reversibility_dim) tensor.

    Args:
        model: trained FluxDecoder.
        data_loader: yields ((compound_ids, label_ids, tracer_ids, fractions), targets) batches.
        device: torch device to run inference on.
        verbose: show a tqdm progress bar.
    """
    model.to(device)
    model.eval()

    all_predictions = []

    iterator = tqdm(data_loader, desc="Predicting") if verbose else data_loader

    with torch.no_grad():
        for batch_data in iterator:
            (compound_ids, label_ids, tracer_ids, fractions), _ = batch_data
            compound_ids = compound_ids.to(device)
            label_ids    = label_ids.to(device)
            tracer_ids   = tracer_ids.to(device)
            fractions    = fractions.to(device)

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                bypass_mask  = model.token_experiment_ids == -1
                fixed_fluxes = fractions[:, bypass_mask, 0]

            flux_predictions, reversibility_predictions = model(
                compound_ids, label_ids, tracer_ids, fractions,
                measured_fluxes=fixed_fluxes,
            )
            predictions = torch.cat([flux_predictions, reversibility_predictions], dim=1)

            all_predictions.append(predictions.cpu())

    return torch.cat(all_predictions, dim=0)


def create_ensemble(num_models, config, token_experiment_ids,
                    compound_emb_table=None, label_emb_table=None,
                    tracer_emb_table=None, null_space_data=None):
    """Create num_models FluxDecoder instances sharing the same config, embedding
    tables, and null-space data. Returns a list of untrained models.

    Args:
        num_models: number of instances to create.
        config: FluxDecoderConfig shared by every instance.
        token_experiment_ids: per-token source-experiment index, -1 for bypass tokens.
        compound_emb_table: frozen compound embedding table, shared across instances.
        label_emb_table: frozen label embedding table, shared across instances.
        tracer_emb_table: frozen tracer embedding table, shared across instances.
        null_space_data: dict with Z_free, fixed_idx, and optional norm stats.
    """
    from .model import FluxDecoder

    models = []
    for i in range(num_models):
        print(f"Creating ensemble model {i+1}/{num_models}")
        model = FluxDecoder(
            config=config,
            token_experiment_ids=token_experiment_ids,
            compound_emb_table=compound_emb_table,
            label_emb_table=label_emb_table,
            tracer_emb_table=tracer_emb_table,
            null_space_data=null_space_data,
        )
        models.append(model)

    return models


def ensemble_predict(models, data_loader, device='cuda', verbose=True):
    """Run every model in the ensemble and return (mean_predictions, std_predictions,
    all_predictions) — per-sample ensemble mean, std, and the full
    (num_models, num_samples, flux_dim + reversibility_dim) stack.

    Args:
        models: trained FluxDecoder instances.
        data_loader: yields ((compound_ids, label_ids, tracer_ids, fractions), targets) batches.
        device: torch device to run inference on.
        verbose: show progress output.
    """
    all_model_predictions = []

    for i, model in enumerate(models):
        if verbose:
            print(f"Ensemble model {i+1}/{len(models)} predictions...")
        predictions = predict_fluxes(
            model, data_loader, device=device, verbose=verbose,
        )
        all_model_predictions.append(predictions)

    all_predictions  = torch.stack(all_model_predictions, dim=0)
    mean_predictions = all_predictions.mean(dim=0)
    std_predictions  = all_predictions.std(dim=0)

    return mean_predictions, std_predictions, all_predictions


def train_ensemble(models, train_loader, val_loader=None, num_epochs=100,
                  learning_rate=1e-3, device='cuda', loss_function=None,
                  gradient_clip=1.0, use_scheduler=True, verbose=True,
                  mask_prob=0.0, min_experiments=1):
    """Train each model in the ensemble independently via :func:`train_model`.
    Returns a list of training logs, one per model.

    Args:
        models: untrained FluxDecoder instances.
        train_loader: training batches.
        val_loader: validation batches, or None to skip validation.
        num_epochs: epochs to train each model for.
        learning_rate: Adam learning rate.
        device: torch device to train on.
        loss_function: callable, or dict with 'flux'/'reversibility' keys (None = MSELoss).
        gradient_clip: max gradient norm (<=0 disables clipping).
        use_scheduler: enable ReduceLROnPlateau.
        verbose: print per-model progress.
        mask_prob: passed through to :func:`train_model`.
        min_experiments: passed through to :func:`train_model`.
    """
    from .train import train_model

    logs = []

    for i, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"Training ensemble model {i+1}/{len(models)}")
        print(f"{'='*60}")

        log = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            loss_function=loss_function,
            gradient_clip=gradient_clip,
            use_scheduler=use_scheduler,
            verbose=verbose,
            mask_prob=mask_prob,
            min_experiments=min_experiments,
        )

        logs.append(log)

    return logs
