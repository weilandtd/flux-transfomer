"""
Training utilities for flux prediction models.

Provides training loops, loss functions, and optimization utilities.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def var_norm_mse_loss(predictions, targets, precomputed_variance, epsilon=1e-10):
    """
    Variance-normalized MSE loss.
    
    Computes MSE per feature and normalizes by precomputed feature variance.
    This is essentially 1 - R² without mean-centering.
    
    Important: variance must be precomputed from the full training dataset,
    not computed per batch, to ensure consistent normalization.
    
    Args:
        predictions: (batch_size, num_features) predicted values
        targets: (batch_size, num_features) target values
        precomputed_variance: (num_features,) precomputed variance for each feature
        epsilon: Small constant to avoid division by zero
    
    Returns:
        loss: Scalar loss averaged across all features
    
    Example:
        >>> # Precompute variance from training data
        >>> train_variance = torch.var(train_targets, dim=0)
        >>> # Create loss function with variance bound
        >>> from functools import partial
        >>> loss_fn = partial(var_norm_mse_loss, precomputed_variance=train_variance)
        >>> # Use in training
        >>> train_model(model, train_loader, loss_function=loss_fn)
    """
    precomputed_variance = precomputed_variance.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    # MSE per feature (across batch dimension)
    mse_per_feature = torch.mean((predictions - targets) ** 2, dim=0)
    
    # Normalize MSE by precomputed variance
    normalized_loss_per_feature = mse_per_feature / (precomputed_variance + epsilon)
    
    # Average across all features
    loss = torch.mean(normalized_loss_per_feature)
    
    return loss


def train_model(model, train_loader, val_loader=None, num_epochs=100,
                learning_rate=1e-3, device='cuda', loss_function=None,
                gradient_clip=1.0, use_scheduler=True, verbose=True,
                mask_prob=0.0, min_experiments=1, warmup_epochs=0):
    """
    Train a labeling-to-flux model.

    Args:
        model: LabelingToFluxTransformer model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        loss_function: Loss function to use (default: nn.MSELoss())
        gradient_clip: Max norm for gradient clipping (0 to disable)
        use_scheduler: Whether to use learning rate scheduler
        verbose: Whether to print training progress
        mask_prob: Probability of masking experiments (0.0 = no masking, 0.3 = 30% chance)
        min_experiments: Minimum experiments to keep when masking
        warmup_epochs: Number of epochs for linear LR warmup from 1% to 100% of learning_rate
                       (0 = disabled). After warmup, ReduceLROnPlateau takes over if use_scheduler.

    Returns:
        log: Dictionary with 'train_loss', 'val_loss', and 'grad_norm' lists
    """
    model.to(device)
    
    # Verify device placement
    if verbose:
        param_device = next(model.parameters()).device
        print(f"Model is on device: {param_device}")
        if str(device) == 'cuda' and torch.cuda.is_available():
            print(f"CUDA memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Default to MSE loss if none provided
    if loss_function is None:
        loss_function = nn.MSELoss()
    
    # Handle dict of loss functions (for separate flux/reversibility losses)
    if isinstance(loss_function, dict):
        flux_loss_fn = loss_function['flux']
        reversibility_loss_fn = loss_function['reversibility']
    else:
        flux_loss_fn = loss_function
        reversibility_loss_fn = loss_function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    warmup_scheduler = None
    plateau_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
    if use_scheduler:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    if verbose:
        if isinstance(loss_function, dict):
            print(f"Using loss function: {flux_loss_fn.__class__.__name__} (separate for flux/reversibility)")
        else:
            print(f"Using loss function: {loss_function.__class__.__name__}")
        if gradient_clip > 0:
            print(f"Gradient clipping enabled: max_norm={gradient_clip}")
        if warmup_epochs > 0:
            print(f"LR warmup enabled: {warmup_epochs} epochs (1% → 100% of lr={learning_rate})")
        if use_scheduler:
            print(f"ReduceLROnPlateau scheduler enabled (after warmup)")
        if mask_prob > 0:
            print(f"Missing data augmentation enabled: mask_prob={mask_prob}, min_experiments={min_experiments}")
    
    log = {
        'train_loss': [],
        'val_loss': [] if val_loader is not None else None,
        'grad_norm': [],
    }

    train_losses = []
    val_losses = [] if val_loader is not None else None
    grad_norms = []
    
    iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)
    
    for epoch in iterator:
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Debug: Print device info for first batch of first epoch
            if epoch == 0 and batch_idx == 0 and verbose:
                print(f"First batch - Input device: {inputs.device}, Target device: {targets.device}")
                if str(device) == 'cuda':
                    print(f"CUDA memory after moving batch to GPU: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            # Apply random masking augmentation if enabled
            mask = None
            if mask_prob > 0:
                from sigmaflux.ai.data import apply_random_masking
                _labeling_and_tracer_dim = inputs.shape[1] - model.bypass_dim
                _num_experiments = _labeling_and_tracer_dim // (model.input_dim + model.num_base_tracers)
                inputs, mask = apply_random_masking(
                    inputs, mask_prob=mask_prob,
                    min_experiments=min_experiments,
                    num_base_tracers=model.num_base_tracers,
                    bypass_dim=model.bypass_dim,
                    num_experiments=_num_experiments
                )
            
            optimizer.zero_grad()
            flux_predictions, reversibility_predictions = model(inputs, mask=mask)
            
            # Split targets
            flux_targets = targets[:, :model.flux_dim]
            reversibility_targets = targets[:, model.flux_dim:]
            
            # Compute losses (use separate loss functions if provided)
            flux_loss = flux_loss_fn(flux_predictions, flux_targets)
            reversibility_loss = reversibility_loss_fn(reversibility_predictions, reversibility_targets)
            
            loss = flux_loss + reversibility_loss
            loss.backward()
            
            if gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip).item()
            else:
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
            epoch_grad_norm += grad_norm

            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_grad_norm = epoch_grad_norm / len(train_loader)
        train_losses.append(avg_epoch_loss)
        grad_norms.append(avg_epoch_grad_norm)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_epoch_loss = 0.0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_targets = val_batch
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    # No masking during validation
                    val_flux_predictions, val_reversibility_predictions = model(val_inputs, mask=None)
                    val_flux_targets = val_targets[:, :model.flux_dim]
                    val_reversibility_targets = val_targets[:, model.flux_dim:]
                    
                    # Compute losses (use separate loss functions if provided)
                    val_flux_loss = flux_loss_fn(val_flux_predictions, val_flux_targets)
                    val_reversibility_loss = reversibility_loss_fn(val_reversibility_predictions, 
                                                           val_reversibility_targets)
                    
                    val_loss = val_flux_loss + val_reversibility_loss
                    val_epoch_loss += val_loss.item()
            
            avg_val_loss = val_epoch_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Scheduler stepping: warmup takes priority, then plateau
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)

            # Update progress bar
            if verbose:
                iterator.set_postfix({'train_loss': f'{avg_epoch_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}', 'grad_norm': f'{avg_epoch_grad_norm:.4f}'})
                tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train: {avg_epoch_loss:.4f}, Val: {avg_val_loss:.4f}, GradNorm: {avg_epoch_grad_norm:.4f}")
        else:
            # Scheduler stepping: warmup takes priority, then plateau
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_epoch_loss)

            # Update progress bar
            if verbose:
                iterator.set_postfix({'train_loss': f'{avg_epoch_loss:.4f}', 'grad_norm': f'{avg_epoch_grad_norm:.4f}'})
                if (epoch + 1) % 10 == 0:
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train: {avg_epoch_loss:.4f}, GradNorm: {avg_epoch_grad_norm:.4f}")
    
    log['train_loss'] = train_losses
    if val_loader is not None:
        log['val_loss'] = val_losses
    log['grad_norm'] = grad_norms

    return log
