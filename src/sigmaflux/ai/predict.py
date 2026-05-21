"""
Prediction utilities for flux prediction models.

Provides functions for single model and ensemble predictions with uncertainty estimation.
"""

import torch
from tqdm import tqdm


def predict_fluxes(model, data_loader, device='cuda', verbose=True, extract_mask=False):
    """
    Generate flux predictions from tokenized labeling data.

    Args:
        model: Trained FluxDecoder model
        data_loader: DataLoader yielding IDTokenizedDataset batches
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to show progress bar
        extract_mask: Whether to automatically detect and use masks from data
            (based on zero-valued fractions)

    Returns:
        predictions: (num_samples, flux_dim + reversibility_dim) tensor
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

            padding_mask = None
            if extract_mask:
                from sigmaflux.ai.data import extract_mask_from_data
                # fractions is (batch, N, 1); extract_mask_from_data uses tokens[:, :, -1]
                valid_mask   = extract_mask_from_data(fractions, model.token_experiment_ids)
                padding_mask = ~valid_mask

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                bypass_mask  = model.token_experiment_ids == -1
                fixed_fluxes = fractions[:, bypass_mask, 0]

            flux_predictions, reversibility_predictions = model(
                compound_ids, label_ids, tracer_ids, fractions, padding_mask, fixed_fluxes
            )
            predictions = torch.cat([flux_predictions, reversibility_predictions], dim=1)

            all_predictions.append(predictions.cpu())

    return torch.cat(all_predictions, dim=0)


def create_ensemble(num_models, config, token_experiment_ids,
                    compound_emb_table=None, label_emb_table=None, tracer_emb_table=None,
                    null_space_data=None):
    """
    Create an ensemble of FluxDecoder models for uncertainty estimation.

    Args:
        num_models: Number of models in ensemble
        config: FluxDecoderConfig
        token_experiment_ids: (num_tokens,) int64 tensor
        compound_emb_table: frozen compound embedding weights
        label_emb_table: frozen label embedding weights
        tracer_emb_table: frozen tracer embedding weights
        null_space_data: null-space dict (Z_free, free_idx, fixed_idx, y_mean, y_std, ...)

    Returns:
        models: list of FluxDecoder models
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


def ensemble_predict(models, data_loader, device='cuda', verbose=True, extract_mask=False):
    """
    Generate predictions from an ensemble of models with uncertainty estimation.

    Args:
        models: List of trained FluxDecoder models
        data_loader: DataLoader with IDTokenizedDataset batches
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to show progress
        extract_mask: Whether to automatically detect and use masks from data

    Returns:
        mean_predictions: (num_samples, flux_dim + reversibility_dim) — ensemble mean
        std_predictions:  (num_samples, flux_dim + reversibility_dim) — ensemble std
        all_predictions:  (num_models, num_samples, flux_dim + reversibility_dim)
    """
    all_model_predictions = []

    for i, model in enumerate(models):
        if verbose:
            print(f"Ensemble model {i+1}/{len(models)} predictions...")
        predictions = predict_fluxes(
            model, data_loader, device=device, verbose=verbose,
            extract_mask=extract_mask,
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
    """
    Train an ensemble of FluxDecoder models independently.

    Args:
        models: List of FluxDecoder models
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on
        loss_function: Loss function to use (default: nn.MSELoss())
        gradient_clip: Max norm for gradient clipping
        use_scheduler: Whether to use learning rate scheduler
        verbose: Whether to print training progress

    Returns:
        logs: List of training logs for each model
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
