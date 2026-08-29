"""
Training utilities for flux prediction models.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm



class FluxNorm:
    """Per-reaction standardization of raw flux scalars, computed once from the
    complete flux data and applied to measured_fluxes before they enter
    FluxDecoder as flux tokens."""

    def __init__(self, flux_data):
        data = torch.tensor(np.asarray(flux_data), dtype=torch.float32)  # (n_reactions, n_samples)
        self.mean = data.mean(dim=1)             # (n_reactions,)
        self.std  = data.std(dim=1).clamp_min(1e-6)

    def __call__(self, flux_val, flux_idx):
        """Standardize flux_val (B, n_idx) using the per-reaction stats at flux_idx.

        Args:
            flux_val: raw flux values to standardize.
            flux_idx: reaction indices selecting rows of mean/std.
        """
        device = flux_val.device
        mean = self.mean[flux_idx].to(device)
        std  = self.std[flux_idx].to(device)
        return (flux_val - mean) / std

    def to_buffers(self):
        """Return (mean, std) tensors for registering as model buffers."""
        return self.mean.clone(), self.std.clone()


class BasisSpaceLoss:
    """Loss wrapper that projects flux predictions and targets through Z_full into
    null-space basis coordinates before computing the loss, avoiding the gradient
    amplification caused by Z_full.T's uneven row norms. Drop-in replacement —
    same (predictions, targets) call signature as any flux loss function."""

    def __init__(self, base_loss_fn, Z_full):
        """Args:
            base_loss_fn: loss called on the projected (predictions, targets).
            Z_full: null-space basis, shape (n_fluxes, n_free).
        """
        self.base_loss_fn = base_loss_fn
        if not isinstance(Z_full, torch.Tensor):
            Z_full = torch.tensor(Z_full, dtype=torch.float32)
        self.Z_full = Z_full

    def __call__(self, flux_preds, targets_flux):
        """Args:
            flux_preds: predicted fluxes, shape (B, n_fluxes).
            targets_flux: target fluxes, shape (B, n_fluxes).
        """
        Z     = self.Z_full.to(device=flux_preds.device, dtype=flux_preds.dtype)
        y_hat  = flux_preds   @ Z      # (B, n_free)
        y_true = targets_flux @ Z      # (B, n_free)
        return self.base_loss_fn(y_hat, y_true)


def var_norm_mse_loss(predictions, targets, precomputed_variance, epsilon=1e-10):
    """MSE per reaction divided by its precomputed training variance (≈1 - R²).

    Args:
        predictions: predicted values, shape (B, n_features).
        targets: target values, shape (B, n_features).
        precomputed_variance: per-feature training variance, shape (n_features,).
        epsilon: added to the denominator to avoid division by zero.
    """
    precomputed_variance = precomputed_variance.to(
        device=predictions.device, dtype=predictions.dtype
    )
    mse_per_feature = torch.mean((predictions - targets) ** 2, dim=0)
    return torch.mean(mse_per_feature / (precomputed_variance + epsilon))


def r2_score(predictions, targets):
    """Per-column R², dropping columns where targets are constant.

    Args:
        predictions: predicted values, shape (B, n_features).
        targets: target values, shape (B, n_features).

    Returns:
        (r2, n_variable): per-column R² over the variable columns, and their count.
    """
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


def per_reaction_r2(predictions, targets, eps=1e-9):
    """Per-column R², NaN where the target column's total sum of squares is <= eps
    (i.e. constant). Unlike :func:`r2_score` (which drops constant columns), this
    keeps one entry per column so the result stays indexed alongside flux_names.

    Args:
        predictions: predicted values, shape (B, n_features).
        targets: target values, shape (B, n_features).
        eps: total-sum-of-squares threshold below which a column is constant.
    """
    predictions = np.asarray(predictions)
    targets     = np.asarray(targets)
    ss_res = ((predictions - targets) ** 2).sum(0)
    ss_tot = ((targets - targets.mean(0)) ** 2).sum(0)
    # np.where evaluates both branches, so the constant columns divide by zero
    # before being discarded; silence that rather than warn on every call.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > eps, 1.0 - ss_res / ss_tot, np.nan)


def evaluate_validation_correlations(model, val_loader, device, flux_dim, flux_names=None):
    """Run the model over val_loader and compute per-reaction R² for flux and
    reversibility outputs.

    Args:
        model: trained FluxDecoder.
        val_loader: validation batches.
        device: torch device to run inference on.
        flux_dim: number of flux outputs; splits targets into flux/reversibility.
        flux_names: reaction names, used to label the per-reaction R² breakdown.
    """
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
    """Apply Gaussian noise to isotopologue labeling fractions while preserving the
    sum=1 constraint (renormalizes after noise). Fully vectorized via segmented
    operations.

    Args:
        fractions: labeling fractions, shape (B, num_tokens, 1).
        compound_ids: token compound ids, used to group tokens into isotopologue sets.
        tracer_ids: token tracer ids, used to group tokens into isotopologue sets.
        noise_std: Gaussian noise standard deviation; <=0 is a no-op.
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

    # Add noise to stored fractions and M+0.
    #
    # Clamped at zero BEFORE the renormalisation below, not after. Clamping the
    # normalised fractions instead — and never clamping M+0 — lets a metabolite whose
    # true M+0 sits near zero take negative noise, so the stored fractions renormalise
    # to more than one and the implied M+0 goes negative. That is not a MID: an EMU
    # simulation can never produce it, so the model would be trained on labeling no
    # flux state can generate. Measured on real T-cell data, the old form left 78 of
    # 456 metabolite groups summing above one, with implied M+0 down to -0.13.
    noise_stored = torch.randn_like(fracs) * noise_std
    noise_m0 = torch.randn(B, num_groups, device=device, dtype=dtype) * noise_std

    noisy_stored = (fracs + noise_stored).clamp(min=0.0)  # (B, num_tokens)
    noisy_m0 = (m0_per_group + noise_m0).clamp(min=0.0)  # (B, num_groups)

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

    # Renormalize fractions. Both terms are already non-negative, so the quotient is
    # in [0, 1] and the group sums to one by construction — no post-hoc clamp, which
    # is what broke the constraint before.
    noisy_fractions = (noisy_stored / total_per_token).unsqueeze(-1)

    return noisy_fractions


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, device, gradient_clip,
                 flux_token_dropout=0.0, labeling_token_dropout=0.0,
                 flux_loss_fn=None, rev_loss_fn=None,
                 labeling_noise_std=0.0):
    """Run one training epoch. Returns (avg_loss, avg_grad_norm).

    Args:
        model: FluxDecoder being trained.
        loader: training batches.
        optimizer: optimizer stepped after each batch.
        device: torch device to train on.
        gradient_clip: max gradient norm; <=0 disables clipping (norm still reported).
        flux_token_dropout: per-batch probability each flux token is excluded.
        labeling_token_dropout: per-batch probability each labeling token's fraction is zeroed.
        flux_loss_fn: loss on flux outputs (None = MSELoss).
        rev_loss_fn: loss on reversibility outputs (None = MSELoss).
        labeling_noise_std: Gaussian noise std applied to fractions (0 disables).
    """
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

        # ── Flux token dropout ────────────────────────────────────────────────
        measured_fluxes = None
        flux_mask       = None
        n_fx = len(model.fixed_idx)
        if n_fx > 0:
            all_mf = targets[:, model.fixed_idx]
            noise_std = getattr(model, 'flux_noise_std', 0.0)
            if noise_std > 0.0:
                all_mf = all_mf * (1.0 + noise_std * torch.randn_like(all_mf))
            if flux_token_dropout > 0.0:
                keep = torch.bernoulli(
                    torch.full((n_fx,), 1.0 - flux_token_dropout, device=device)
                ).bool()
                if not keep.any():
                    keep[torch.randint(n_fx, (1,), device=device)] = True
                all_mf    = all_mf * keep.float()
                flux_mask = (~keep).unsqueeze(0).expand(compound_ids.shape[0], -1)
            measured_fluxes = all_mf

        if labeling_token_dropout > 0.0:
            keep = torch.bernoulli(
                torch.full(fractions.shape[:2], 1.0 - labeling_token_dropout, device=device)
            ).unsqueeze(-1)
            fractions = fractions * keep

        optimizer.zero_grad()
        flux_preds, rev_preds = model(
            compound_ids, label_ids, tracer_ids, fractions,
            measured_fluxes=measured_fluxes,
            flux_mask=flux_mask,
        )
        flux_loss = flux_loss_fn(flux_preds, targets[:, :model.flux_dim])
        loss = flux_loss + rev_loss_fn(rev_preds, targets[:, model.flux_dim:])
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
    """Run one validation epoch. Returns avg_loss.

    Args:
        model: FluxDecoder being evaluated.
        loader: validation batches.
        device: torch device to run inference on.
        flux_loss_fn: loss on flux outputs (None = MSELoss).
        rev_loss_fn: loss on reversibility outputs (None = MSELoss).
    """
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
                warmup_epochs=0, flux_token_dropout=0.0,
                labeling_token_dropout=0.0, token_information=None,
                loss_function=None, labeling_noise_std=0.0,
                early_stopping_patience=None):
    """Train a FluxDecoder model.

    Args:
        model:                   FluxDecoder to train, in place.
        train_loader:            training batches.
        val_loader:              validation batches; enables best-model restore
                                  and early stopping (None = disabled).
        num_epochs:               number of epochs to train for.
        learning_rate:            Adam learning rate.
        device:                   torch device to train on.
        gradient_clip:            max gradient norm; <=0 disables clipping.
        use_scheduler:            enable ReduceLROnPlateau on the training/val loss.
        verbose:                  print a tqdm progress bar and per-epoch summary.
        warmup_epochs:            linear LR warmup epochs before the plateau scheduler engages.
        flux_token_dropout:      Per-batch probability each flux token is excluded
        labeling_token_dropout:  Per-batch probability each labeling token's fraction is zeroed
        token_information:       (n_tokens,) tensor of scores (used for reporting only)
        loss_function:           callable, or dict with 'flux'/'reversibility' keys (None = MSELoss)
        labeling_noise_std:       Gaussian noise std applied to fractions (0 disables).
        early_stopping_patience: Stop after this many consecutive epochs without
                                  val_loss improvement (None = disabled; requires val_loader)

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
    epochs_no_improve = 0

    iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

    for epoch in iterator:
        train_loss, grad_norm = _train_epoch(
            model, train_loader, optimizer, device, gradient_clip,
            flux_token_dropout=flux_token_dropout,
            labeling_token_dropout=labeling_token_dropout,
            flux_loss_fn=flux_loss_fn, rev_loss_fn=rev_loss_fn,
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
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

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

        if (early_stopping_patience is not None and val_loader is not None
                and epochs_no_improve >= early_stopping_patience):
            if verbose:
                tqdm.write(f"Early stopping at epoch {epoch + 1}: no val_loss "
                           f"improvement for {epochs_no_improve} epochs "
                           f"(best val_loss={best_val_loss:.6g})")
            break

    if val_loader is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            tqdm.write(f"Restored best model (val_loss={best_val_loss:.6g})")

    log = {'train_loss': train_losses, 'grad_norm': grad_norms}
    if val_loader is not None:
        log['val_loss'] = val_losses
    return log
