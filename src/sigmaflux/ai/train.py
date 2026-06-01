"""
Training utilities for flux prediction models.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sigmaflux.ai.data import generate_metabolite_mask


class FluxNorm:
    """
    Per-reaction standardization of raw flux scalars.

    Computed once from the complete flux data (all reactions × all samples)
    and applied to measured_fluxes before they enter FluxDecoder as flux tokens.

    Usage:
        norm = FluxNorm(flux_df.values)           # (n_reactions, n_samples)
        v_norm = norm(flux_val, flux_idx)          # (B, n_idx) → standardized
    """

    def __init__(self, flux_data):
        data = torch.tensor(np.asarray(flux_data), dtype=torch.float32)  # (n_reactions, n_samples)
        self.mean = data.mean(dim=1)             # (n_reactions,)
        self.std  = data.std(dim=1).clamp_min(1e-6)

    def __call__(self, flux_val, flux_idx):
        """
        Args:
            flux_val:  (B, n_idx) tensor of raw flux values
            flux_idx:  (n_idx,) int tensor/list of reaction indices
        Returns:
            (B, n_idx) standardized values
        """
        device = flux_val.device
        mean = self.mean[flux_idx].to(device)
        std  = self.std[flux_idx].to(device)
        return (flux_val - mean) / std

    def to_buffers(self):
        """Return (mean, std) tensors for registering as model buffers."""
        return self.mean.clone(), self.std.clone()


class BasisSpaceLoss:
    """
    Loss wrapper that operates in null-space basis (y) space.

    Projects both flux predictions and targets through Z_full before computing
    the loss, compressing the 388-dim flux space down to 38-dim basis space.
    This avoids the gradient amplification caused by Z_full.T having rows with
    uneven norms (some reactions load heavily across many basis vectors).

    Drop-in replacement for any flux loss function — same (predictions, targets)
    call signature, no changes to the training loop required.

    Args:
        base_loss_fn: any callable (predictions, targets) -> scalar
        Z_full:       (n_fluxes, n_free) null-space basis matrix
    """

    def __init__(self, base_loss_fn, Z_full):
        self.base_loss_fn = base_loss_fn
        if not isinstance(Z_full, torch.Tensor):
            Z_full = torch.tensor(Z_full, dtype=torch.float32)
        self.Z_full = Z_full

    def __call__(self, flux_preds, targets_flux):
        Z     = self.Z_full.to(device=flux_preds.device, dtype=flux_preds.dtype)
        y_hat  = flux_preds   @ Z      # (B, n_free)
        y_true = targets_flux @ Z      # (B, n_free)
        return self.base_loss_fn(y_hat, y_true)


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

            measured_fluxes = None
            if len(model.fixed_idx) > 0:
                measured_fluxes = targets[:, model.fixed_idx]
            pred_flux, pred_rev = model(cids, lids, tids, fracs, measured_fluxes=measured_fluxes)
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
        "val_predictions":      all_flux_preds.cpu().numpy(),
        "val_actuals":          all_flux_tgts.cpu().numpy(),
    }

    if flux_names is not None and len(flux_r2) > 0:
        variable_mask = all_flux_tgts.std(dim=0) > 1e-8
        var_names = [flux_names[i] for i, v in enumerate(variable_mask.tolist()) if v]
        result["flux_r2_per_reaction"] = flux_r2.tolist()
        result["flux_reaction_names"]  = var_names

    return result


def apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std):
    """
    Apply Gaussian noise to isotopologue labeling fractions while maintaining sum=1 constraint.
    Fully vectorized using segmented operations (no loops).

    Args:
        fractions: (batch_size, num_tokens, 1) tensor of stored fractions (M+1..M+n)
        compound_ids: (num_tokens,) or (batch_size, num_tokens) tensor mapping tokens to metabolites
        tracer_ids: (num_tokens,) or (batch_size, num_tokens) tensor mapping tokens to tracers
        noise_std: scalar noise standard deviation

    Returns:
        noisy_fractions: (batch_size, num_tokens, 1) tensor with added noise, renormalized
    """
    if noise_std <= 0:
        return fractions

    B, num_tokens, _ = fractions.shape
    device = fractions.device
    dtype = fractions.dtype

    # Handle both 1D and 2D ID tensors (batch may or may not stack them)
    if compound_ids.dim() == 2:
        compound_ids = compound_ids[0]  # Use first sample (same for all)
    if tracer_ids.dim() == 2:
        tracer_ids = tracer_ids[0]  # Use first sample (same for all)

    # Create unique group ID for each token
    max_tracer = tracer_ids.max().item() + 1
    group_ids = compound_ids * max_tracer + tracer_ids  # (num_tokens,)
    num_groups = group_ids.max().item() + 1

    # Squeeze fractions
    fracs = fractions.squeeze(-1)  # (B, num_tokens)

    # Compute sum per group using segment reduction
    fracs_flat = fracs.reshape(-1)  # (B*num_tokens,)
    group_ids_expanded = group_ids.unsqueeze(0).expand(B, -1).reshape(-1)  # (B*num_tokens,)
    batch_ids_expanded = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(-1, num_tokens).reshape(-1)  # (B*num_tokens,)

    # Create 2D index for scatter operations: (batch_id, group_id)
    combined_idx = batch_ids_expanded * num_groups + group_ids_expanded

    # Sum per (batch, group) using scatter_add
    sums = torch.zeros(B * num_groups, device=device, dtype=dtype)
    sums.scatter_add_(0, combined_idx, fracs_flat)
    group_sums = sums.reshape(B, num_groups)  # (B, num_groups)

    # Implicit M+0
    m0_per_group = 1.0 - group_sums  # (B, num_groups)

    # Add noise to stored fractions and M+0
    noise_stored = torch.randn_like(fracs) * noise_std
    noise_m0 = torch.randn(B, num_groups, device=device, dtype=dtype) * noise_std

    noisy_stored = fracs + noise_stored  # (B, num_tokens)
    noisy_m0 = m0_per_group + noise_m0  # (B, num_groups)

    # Sum of noisy stored per group
    noisy_flat = noisy_stored.reshape(-1)
    sums_noisy = torch.zeros(B * num_groups, device=device, dtype=dtype)
    sums_noisy.scatter_add_(0, combined_idx, noisy_flat)
    noisy_stored_sum = sums_noisy.reshape(B, num_groups)  # (B, num_groups)

    # Total per group
    total_per_group = noisy_m0 + noisy_stored_sum  # (B, num_groups)
    total_per_group = total_per_group.clamp(min=1e-10)

    # Broadcast total back to token dimension: total_per_token[b, t] = total_per_group[b, group_ids[t]]
    total_flat = total_per_group.reshape(-1).gather(0, combined_idx)  # (B*num_tokens,)
    total_per_token = total_flat.reshape(B, num_tokens)  # (B, num_tokens)

    # Renormalize fractions
    noisy_fractions = (noisy_stored / total_per_token).clamp(0.0, 1.0).unsqueeze(-1)

    return noisy_fractions


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, device, met_mask_prob, gradient_clip,
                 flux_mask_prob=0.0, flux_loss_fn=None, rev_loss_fn=None,
                 measured_flux_loss_weight=0.0, labeling_noise_std=0.0):
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

        if labeling_noise_std > 0:
            fractions = apply_labeling_noise(fractions, compound_ids, tracer_ids, labeling_noise_std)

        labeling_mask = None
        if met_mask_prob > 0:
            labeling_mask = ~generate_metabolite_mask(
                compound_ids[0],
                batch_size=compound_ids.shape[0],
                device=device,
                mask_prob=met_mask_prob,
            )

        measured_fluxes = None
        flux_mask = None
        if len(model.fixed_idx) > 0:
            measured_fluxes = targets[:, model.fixed_idx]
            noise_std = getattr(model, 'flux_noise_std', 0.0)
            if noise_std > 0.0:
                measured_fluxes = measured_fluxes * (
                    1.0 + noise_std * torch.randn_like(measured_fluxes)
                )
            # Randomly mask flux tokens during training for robustness
            if flux_mask_prob > 0:
                B, n_fx = measured_fluxes.shape
                flux_mask = torch.rand(B, n_fx, device=device) < flux_mask_prob

        optimizer.zero_grad()
        flux_preds, rev_preds = model(
            compound_ids, label_ids, tracer_ids, fractions,
            labeling_mask=labeling_mask,
            measured_fluxes=measured_fluxes,
            flux_mask=flux_mask,
        )
        flux_loss = flux_loss_fn(flux_preds, targets[:, :model.flux_dim])
        loss = flux_loss + rev_loss_fn(rev_preds, targets[:, model.flux_dim:])
        if measured_flux_loss_weight > 0 and measured_fluxes is not None and len(model.fixed_idx) > 0:
            meas_pred = flux_preds[:, model.fixed_idx]
            meas_true = targets[:, model.fixed_idx]
            loss = loss + measured_flux_loss_weight * F.mse_loss(meas_pred, meas_true)
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

            measured_fluxes = None
            if len(model.fixed_idx) > 0:
                measured_fluxes = targets[:, model.fixed_idx]

            flux_preds, rev_preds = model(cids, lids, tids, fracs, measured_fluxes=measured_fluxes)
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
                warmup_epochs=0, met_mask_prob=0.0, flux_mask_prob=0.0,
                loss_function=None, measured_flux_loss_weight=0.0, labeling_noise_std=0.0):
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
        flux_mask_prob: Per-epoch measured flux masking probability
        labeling_noise_std: Gaussian noise std for isotopologue fractions (0 = disabled)
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
            flux_mask_prob=flux_mask_prob,
            flux_loss_fn=flux_loss_fn, rev_loss_fn=rev_loss_fn,
            measured_flux_loss_weight=measured_flux_loss_weight,
            labeling_noise_std=labeling_noise_std,
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
