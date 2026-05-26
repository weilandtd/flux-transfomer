"""
Training utilities for flux prediction models.
"""

import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from sigmaflux.ai.data import generate_metabolite_mask


def var_norm_mse_loss(predictions, targets, precomputed_variance, epsilon=1e-10):
    """MSE per reaction divided by its precomputed training variance (≈1 - R²)."""
    precomputed_variance = precomputed_variance.to(
        device=predictions.device, dtype=predictions.dtype
    )
    mse_per_feature = torch.mean((predictions - targets) ** 2, dim=0)
    return torch.mean(mse_per_feature / (precomputed_variance + epsilon))


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
                fixed_fluxes = targets[:, model.known_flux_encoder.fixed_idx]
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

    if flux_names is not None and len(flux_r2) > 0:
        variable_mask = all_flux_tgts.std(dim=0) > 1e-8
        var_names = [flux_names[i] for i, v in enumerate(variable_mask.tolist()) if v]
        result["flux_r2_per_reaction"] = flux_r2.tolist()
        result["flux_reaction_names"]  = var_names

    return result


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, device, met_mask_prob, gradient_clip,
                 flux_loss_fn=None, rev_loss_fn=None):
    """Run one training epoch. Returns (avg_loss, avg_grad_norm)."""
    model.train()
    if flux_loss_fn is None:
        flux_loss_fn = nn.MSELoss()
    if rev_loss_fn is None:
        rev_loss_fn = nn.MSELoss()
    total_loss      = 0.0
    total_grad_norm = 0.0

    for (compound_ids, label_ids, tracer_ids, fractions), targets in loader:
        compound_ids = compound_ids.to(device)
        label_ids    = label_ids.to(device)
        tracer_ids   = tracer_ids.to(device)
        fractions    = fractions.to(device)
        targets      = targets.to(device)

        padding_mask = None
        if met_mask_prob > 0:
            padding_mask = ~generate_metabolite_mask(
                compound_ids[0],
                batch_size=compound_ids.shape[0],
                device=device,
                mask_prob=met_mask_prob,
            )

        fixed_fluxes = None
        if model.cfg.n_measured > 0:
            fixed_fluxes = targets[:, model.known_flux_encoder.fixed_idx]
            noise_std = getattr(model, 'flux_noise_std', 0.0)
            if noise_std > 0.0:
                fixed_fluxes = fixed_fluxes + torch.randn_like(fixed_fluxes) * noise_std

        optimizer.zero_grad()
        flux_preds, rev_preds = model(
            compound_ids, label_ids, tracer_ids, fractions, padding_mask, fixed_fluxes
        )

        loss = (flux_loss_fn(flux_preds, targets[:, :model.flux_dim])
                + rev_loss_fn(rev_preds, targets[:, model.flux_dim:]))
        loss.backward()

        if gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            ).item()
        else:
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5

        optimizer.step()
        total_loss      += loss.item()
        total_grad_norm += grad_norm

    n = len(loader)
    return total_loss / n, total_grad_norm / n


def _val_epoch(model, loader, device, flux_loss_fn=None, rev_loss_fn=None):
    """Run one validation epoch. Returns avg_loss."""
    model.eval()
    if flux_loss_fn is None:
        flux_loss_fn = nn.MSELoss()
    if rev_loss_fn is None:
        rev_loss_fn = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for (cids, lids, tids, fracs), targets in loader:
            cids    = cids.to(device)
            lids    = lids.to(device)
            tids    = tids.to(device)
            fracs   = fracs.to(device)
            targets = targets.to(device)

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                fixed_fluxes = targets[:, model.known_flux_encoder.fixed_idx]

            flux_preds, rev_preds = model(cids, lids, tids, fracs, None, fixed_fluxes)
            total_loss += (
                flux_loss_fn(flux_preds, targets[:, :model.flux_dim])
                + rev_loss_fn(rev_preds, targets[:, model.flux_dim:])
            ).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader=None, num_epochs=100,
                learning_rate=1e-3, device='cuda',
                gradient_clip=1.0, use_scheduler=True, verbose=True,
                warmup_epochs=0, met_mask_prob=0.0,
                loss_function=None):
    """
    Train a FluxDecoder model.

    Args:
        model:          FluxDecoder model
        train_loader:   DataLoader for training data
        val_loader:     Optional DataLoader for validation data
        num_epochs:     Number of training epochs
        learning_rate:  Initial learning rate (Adam)
        device:         'cuda' or 'cpu'
        gradient_clip:  Max gradient norm (0 = disabled)
        use_scheduler:  ReduceLROnPlateau after warmup
        verbose:        Print progress
        warmup_epochs:  Linear LR warmup epochs (0 = disabled)
        met_mask_prob:  Per-epoch metabolite masking probability
        loss_function:  callable, or dict with 'flux'/'reversibility' keys (None = MSELoss)

    Returns:
        log: dict with 'train_loss', 'val_loss', 'grad_norm' lists
    """
    model.to(device)

    if isinstance(loss_function, dict):
        flux_loss_fn = loss_function.get('flux')
        rev_loss_fn  = loss_function.get('reversibility')
    else:
        flux_loss_fn = loss_function
        rev_loss_fn  = loss_function

    if verbose:
        name = getattr(flux_loss_fn, '__name__', type(flux_loss_fn).__name__) if flux_loss_fn else 'MSELoss'
        print(f"Loss function: {name}")

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

    train_losses, val_losses, grad_norms = [], [], []
    best_val_loss   = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())

    iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

    for epoch in iterator:
        train_loss, grad_norm = _train_epoch(
            model, train_loader, optimizer, device, met_mask_prob, gradient_clip,
            flux_loss_fn=flux_loss_fn, rev_loss_fn=rev_loss_fn,
        )
        train_losses.append(train_loss)
        grad_norms.append(grad_norm)

        if val_loader is not None:
            val_loss = _val_epoch(model, val_loader, device,
                                  flux_loss_fn=flux_loss_fn, rev_loss_fn=rev_loss_fn)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())

            scheduler_loss = val_loss
        else:
            scheduler_loss = train_loss

        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(scheduler_loss)

        if verbose:
            postfix = {'train': f'{train_loss:.4f}', 'grad': f'{grad_norm:.4f}'}
            if val_loader is not None:
                postfix['val'] = f'{val_loss:.4f}'
            iterator.set_postfix(postfix)

    if val_loader is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            tqdm.write(f"Restored best model (val_loss={best_val_loss:.6g})")

    log = {'train_loss': train_losses, 'grad_norm': grad_norms}
    if val_loader is not None:
        log['val_loss'] = val_losses
    return log
