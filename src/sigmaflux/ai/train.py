"""
Training utilities for flux prediction models.

Provides training loops, loss functions, and optimization utilities.
"""

import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from sigmaflux.ai.data import generate_metabolite_mask


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
                warmup_epochs=0, y_l2_reg=0.0, met_mask_prob=0.0,
                best_model_path=None):
    """
    Train a FluxDecoder model.

    Args:
        model: FluxDecoder model
        train_loader: DataLoader for training data (IDTokenizedDataset)
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
        loss_function: Loss function to use (default: nn.MSELoss())
        gradient_clip: Max norm for gradient clipping (0 to disable)
        use_scheduler: Whether to use learning rate scheduler
        verbose: Whether to print training progress
        warmup_epochs: Number of epochs for linear LR warmup from 1% to 100% of
                       learning_rate (0 = disabled). After warmup, ReduceLROnPlateau
                       takes over if use_scheduler.
        y_l2_reg: L2 regularization coefficient on y_norm predictions (null-space mode
                  only). Pulls unobservable y-components toward 0 (the prior mean in
                  normalized y-space), preventing overfitting to training-data noise in
                  directions with no MID signal. Applied to train loss only (default: 0.0).

    Returns:
        log: Dictionary with 'train_loss', 'val_loss', and 'grad_norm' lists
    """
    model.to(device)

    if verbose:
        param_device = next(model.parameters()).device
        print(f"Model is on device: {param_device}")
        if str(device) == 'cuda' and torch.cuda.is_available():
            print(f"CUDA memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    if loss_function is None:
        loss_function = nn.MSELoss()

    if isinstance(loss_function, dict):
        flux_loss_fn         = loss_function['flux']
        reversibility_loss_fn = loss_function['reversibility']
    else:
        flux_loss_fn         = loss_function
        reversibility_loss_fn = loss_function

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    warmup_scheduler  = None
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
            print("ReduceLROnPlateau scheduler enabled (after warmup)")
        if met_mask_prob > 0:
            print(f"Metabolite masking enabled: met_mask_prob={met_mask_prob}")
        if y_l2_reg > 0:
            print(f"y_norm L2 regularization enabled: lambda={y_l2_reg} (prevents unobservable-direction overfitting)")

    log = {
        'train_loss': [],
        'val_loss': [] if val_loader is not None else None,
        'grad_norm': [],
    }

    train_losses = []
    val_losses   = [] if val_loader is not None else None
    grad_norms   = []

    best_val_loss   = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())

    iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

    for epoch in iterator:
        model.train()
        epoch_loss     = 0.0
        epoch_grad_norm = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            (compound_ids, label_ids, tracer_ids, fractions), targets = batch_data
            compound_ids = compound_ids.to(device)
            label_ids    = label_ids.to(device)
            tracer_ids   = tracer_ids.to(device)
            fractions    = fractions.to(device)
            targets      = targets.to(device)

            if epoch == 0 and batch_idx == 0 and verbose:
                print(f"First batch — compound_ids device: {compound_ids.device}")
                if str(device) == 'cuda':
                    print(f"CUDA memory after moving batch to GPU: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

            padding_mask = None
            if met_mask_prob > 0:
                met_valid    = generate_metabolite_mask(
                    compound_ids[0],               # compound_ids shared across batch
                    batch_size=compound_ids.shape[0],
                    device=device,
                    mask_prob=met_mask_prob,
                )
                met_padding  = ~met_valid
                padding_mask = met_padding if padding_mask is None else (padding_mask | met_padding)

            optimizer.zero_grad()
            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                fixed_idx = (model.null_decoder.fixed_idx if model.null_decoder is not None
                             else model.fba_pro.fixed_idx)
                fixed_fluxes = targets[:, fixed_idx]

            # Denoising: corrupt measured fluxes before they enter rotation_fusion
            # during training. Loss is always against clean targets.
            fixed_fluxes_input = fixed_fluxes
            if (model.training and fixed_fluxes is not None
                    and getattr(model, 'flux_noise_std', 0.0) > 0.0):
                noise = torch.randn_like(fixed_fluxes) * model.flux_noise_std * fixed_fluxes.abs()
                fixed_fluxes_input = fixed_fluxes + noise

            flux_predictions, reversibility_predictions = model(
                compound_ids, label_ids, tracer_ids, fractions, padding_mask, fixed_fluxes_input
            )

            flux_targets          = targets[:, :model.flux_dim]
            reversibility_targets = targets[:, model.flux_dim:]

            # Null-space mode: plain MSE in normalized y-space.
            # All 38 components receive gradient — unobservable ones converge toward 0
            # (the prior mean in normalized space), preventing Z_full explosion.
            if model.null_decoder is not None:
                y_pred = model.null_decoder.encode_targets(flux_predictions)
                y_true = model.null_decoder.encode_targets(flux_targets)
                flux_loss = ((y_pred - y_true) ** 2).mean()
                if y_l2_reg > 0:
                    flux_loss = flux_loss + y_l2_reg * (y_pred ** 2).mean()
            else:
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

        avg_epoch_loss     = epoch_loss / len(train_loader)
        avg_epoch_grad_norm = epoch_grad_norm / len(train_loader)
        train_losses.append(avg_epoch_loss)
        grad_norms.append(avg_epoch_grad_norm)

        if val_loader is not None:
            model.eval()
            val_epoch_loss = 0.0
            y_preds_ep, y_trues_ep = [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    (val_cids, val_lids, val_tids, val_fracs), val_targets = val_batch
                    val_cids    = val_cids.to(device)
                    val_lids    = val_lids.to(device)
                    val_tids    = val_tids.to(device)
                    val_fracs   = val_fracs.to(device)
                    val_targets = val_targets.to(device)

                    val_fixed_fluxes = None
                    if model.cfg.n_measured > 0:
                        fixed_idx = (model.null_decoder.fixed_idx if model.null_decoder is not None
                                     else model.fba_pro.fixed_idx)
                        val_fixed_fluxes = val_targets[:, fixed_idx]
                    val_flux_predictions, val_reversibility_predictions = model(
                        val_cids, val_lids, val_tids, val_fracs, None, val_fixed_fluxes
                    )
                    val_flux_targets          = val_targets[:, :model.flux_dim]
                    val_reversibility_targets = val_targets[:, model.flux_dim:]

                    if model.null_decoder is not None:
                        y_pred = model.null_decoder.encode_targets(val_flux_predictions)
                        y_true = model.null_decoder.encode_targets(val_flux_targets)
                        val_flux_loss = ((y_pred - y_true) ** 2).mean()
                        y_preds_ep.append(y_pred.cpu())
                        y_trues_ep.append(y_true.cpu())
                    else:
                        val_flux_loss = flux_loss_fn(val_flux_predictions, val_flux_targets)
                    val_reversibility_loss = reversibility_loss_fn(
                        val_reversibility_predictions, val_reversibility_targets
                    )
                    val_epoch_loss += (val_flux_loss + val_reversibility_loss).item()

            avg_val_loss = val_epoch_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss   = avg_val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                if best_model_path is not None:
                    torch.save({'model_state_dict': best_state_dict,
                                'epoch': epoch + 1,
                                'val_loss': best_val_loss},
                               best_model_path)

            # Per-epoch y-space R² (null-space mode only)
            y_r2_mean = y_r2_median = None
            if y_preds_ep:
                Y_pred = torch.cat(y_preds_ep)
                Y_true = torch.cat(y_trues_ep)
                ss_res = ((Y_pred - Y_true) ** 2).sum(dim=0)
                ss_tot = ((Y_true - Y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
                r2     = (1.0 - ss_res / ss_tot.clamp_min(1e-12)).nan_to_num(nan=0.0)
                y_r2_mean   = float(r2.mean())
                y_r2_median = float(r2.median())

            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)

            if verbose:
                if y_r2_mean is not None:
                    iterator.set_postfix({'train': f'{avg_epoch_loss:.4f}', 'val': f'{avg_val_loss:.4f}', 'y_R2': f'{y_r2_mean:.3f}'})
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}  train={avg_epoch_loss:.4f}  val={avg_val_loss:.4f}  "
                               f"y_R2_mean={y_r2_mean:.3f}  y_R2_median={y_r2_median:.3f}  grad={avg_epoch_grad_norm:.3f}")
                else:
                    iterator.set_postfix({'train_loss': f'{avg_epoch_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}', 'grad_norm': f'{avg_epoch_grad_norm:.4f}'})
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train: {avg_epoch_loss:.4f}, Val: {avg_val_loss:.4f}, GradNorm: {avg_epoch_grad_norm:.4f}")
        else:
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_epoch_loss)

            if verbose:
                iterator.set_postfix({'train_loss': f'{avg_epoch_loss:.4f}', 'grad_norm': f'{avg_epoch_grad_norm:.4f}'})
                if (epoch + 1) % 10 == 0:
                    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train: {avg_epoch_loss:.4f}, GradNorm: {avg_epoch_grad_norm:.4f}")

    log['train_loss'] = train_losses
    if val_loader is not None:
        log['val_loss'] = val_losses
    log['grad_norm'] = grad_norms

    # Restore best weights before returning so the caller saves the best model.
    if val_loader is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            tqdm.write(f"Restored best model (val_loss={best_val_loss:.6g})")

    return log
