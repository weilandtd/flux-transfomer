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
        model: Trained LabelingToFluxTransformer model
        data_loader: DataLoader yielding (token_tensor, targets) batches
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to show progress bar
        extract_mask: Whether to automatically detect and use masks from data
            (based on zero-valued fractions in the token tensor)

    Returns:
        predictions: (num_samples, flux_dim + reversibility_dim) tensor
    """
    model.to(device)
    model.eval()

    all_predictions = []

    iterator = tqdm(data_loader, desc="Predicting") if verbose else data_loader

    with torch.no_grad():
        for batch_data in iterator:
            inputs, _ = batch_data
            inputs = inputs.to(device)

            mask = None
            if extract_mask:
                from sigmaflux.ai.data import extract_mask_from_data
                mask = extract_mask_from_data(inputs, model.token_experiment_ids)

            flux_predictions, reversibility_predictions = model(inputs, mask=mask)
            predictions = torch.cat([flux_predictions, reversibility_predictions], dim=1)

            all_predictions.append(predictions.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    return predictions


def create_ensemble(num_models, stoichiometric_matrix, token_dim, flux_dim,
                   reversibility_dim, token_experiment_ids, **model_kwargs):
    """
    Create an ensemble of models for uncertainty estimation.

    Args:
        num_models: Number of models in ensemble
        stoichiometric_matrix: Stoichiometric matrix for models
        token_dim: Dimension of each input token
        flux_dim: Dimension of flux output
        reversibility_dim: Dimension of reversibility output
        token_experiment_ids: (num_tokens,) tensor mapping tokens to experiments
        **model_kwargs: Additional arguments for LabelingToFluxTransformer

    Returns:
        models: List of LabelingToFluxTransformer models
    """
    from .model import LabelingToFluxTransformer

    models = []
    for i in range(num_models):
        print(f"Creating ensemble model {i+1}/{num_models}")
        model = LabelingToFluxTransformer(
            stoichiometric_matrix=stoichiometric_matrix,
            token_dim=token_dim,
            flux_dim=flux_dim,
            reversibility_dim=reversibility_dim,
            token_experiment_ids=token_experiment_ids,
            **model_kwargs,
        )
        models.append(model)

    return models


def ensemble_predict(models, data_loader, device='cuda', verbose=True, extract_mask=False):
    """
    Generate predictions from an ensemble of models with uncertainty estimation.

    Args:
        models: List of trained LabelingToFluxTransformer models
        data_loader: DataLoader with labeling experiments
        device: Device to run on ('cuda' or 'cpu')
        verbose: Whether to show progress
        extract_mask: Whether to automatically detect and use masks from data

    Returns:
        mean_predictions: (num_samples, flux_dim + reversibility_dim) - ensemble mean
        std_predictions: (num_samples, flux_dim + reversibility_dim) - ensemble std
        all_predictions: (num_models, num_samples, flux_dim + reversibility_dim)
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

    all_predictions = torch.stack(all_model_predictions, dim=0)
    mean_predictions = all_predictions.mean(dim=0)
    std_predictions = all_predictions.std(dim=0)

    return mean_predictions, std_predictions, all_predictions


def train_ensemble(models, train_loader, val_loader=None, num_epochs=100,
                  learning_rate=1e-3, device='cuda', loss_function=None,
                  gradient_clip=1.0, use_scheduler=True, verbose=True,
                  mask_prob=0.0, min_experiments=1):
    """
    Train an ensemble of models independently.

    Args:
        models: List of LabelingToFluxTransformer models
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
