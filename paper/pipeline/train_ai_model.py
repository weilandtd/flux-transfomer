"""
Train SigmaFlux's FluxDecoder (cross-attention decoder) from generated HDF5 data.

Example:
    PYTHONPATH=src python paper/train_ai_model.py --config paper/io_config.yaml
"""

import argparse
import datetime
import math
import os
from dataclasses import asdict
from functools import partial

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
)
from sigmaflux.ai.utils import compute_token_scores, load_fisher_token_scores
from sigmaflux.ai.diagnostics import diagnose_signal_bottleneck
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.train import (
    BasisSpaceLoss, FluxNorm, reduce_compartment_tokens,
    evaluate_validation_correlations, train_model, var_norm_mse_loss,
)
from sigmaflux.ai.utils import compute_sparse_nullspace, drop_constraint_rows


def split_indices(n_samples, val_fraction, seed):
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int((1.0 - val_fraction) * n_samples)
    return indices[:split], indices[split:]


class MixedFluxLoss:
    """Combines basis-space and flux-space losses."""

    def __init__(self, basis_loss_fn, flux_loss_fn, flux_weight):
        """
        Args:
            basis_loss_fn: loss function in basis space
            flux_loss_fn:  loss function in flux space
            flux_weight:   weight for flux-space loss (0-1); basis weight = (1 - flux_weight)
        """
        self.basis_loss_fn = basis_loss_fn
        self.flux_loss_fn = flux_loss_fn
        self.flux_weight = flux_weight
        self.basis_weight = 1.0 - flux_weight

    def __call__(self, pred, targets):
        basis_loss = self.basis_loss_fn(pred, targets)
        flux_loss = self.flux_loss_fn(pred, targets)
        return self.basis_weight * basis_loss + self.flux_weight * flux_loss


def build_loss_function(train_dataset, flux_dim, loss_name,
                        flux_loss_space='flux', Z_full=None, y_train=None, mixed_flux_weight=0.1,
                        variance_min_cutoff=0.1):
    """
    Build flux and reversibility loss functions.

    Args:
        train_dataset:   training dataset (used to compute per-feature variance)
        flux_dim:        number of flux targets
        loss_name:       'mse' or 'variance' (variance-normalized MSE)
        flux_loss_space: 'flux', 'basis', or 'mixed' — train in reconstructed flux, null-space, or combination
        Z_full:          (n_fluxes, n_free) null-space basis; required when flux_loss_space in ['basis', 'mixed']
        y_train:         (n_train, n_free) pre-projected basis targets; used to compute
                         y-space variance when loss_name='variance' and flux_loss_space='basis'
        mixed_flux_weight: weight for flux-space loss when flux_loss_space='mixed'
        variance_min_cutoff: min variance clamp (default 0.1); prevents division by zero and extreme weighting
    """
    targets = train_dataset.targets

    rev_var = torch.var(targets[:, flux_dim:], dim=0).clamp_min(variance_min_cutoff)
    rev_loss_fn = (nn.MSELoss() if loss_name == 'mse'
                   else partial(var_norm_mse_loss, precomputed_variance=rev_var))

    if loss_name == 'mse':
        base_flux_loss = nn.MSELoss()
    else:
        flux_var = torch.var(targets[:, :flux_dim], dim=0).clamp_min(variance_min_cutoff)
        base_flux_loss = partial(var_norm_mse_loss, precomputed_variance=flux_var)

    if flux_loss_space == 'basis':
        assert Z_full is not None, "Z_full required for flux_loss_space='basis'"
        if loss_name == 'variance' and y_train is not None:
            basis_var = torch.var(y_train, dim=0).clamp_min(variance_min_cutoff)
            basis_loss = partial(var_norm_mse_loss, precomputed_variance=basis_var)
        else:
            basis_loss = base_flux_loss
        flux_loss_fn = BasisSpaceLoss(basis_loss, Z_full)
    elif flux_loss_space == 'mixed':
        assert Z_full is not None, "Z_full required for flux_loss_space='mixed'"
        if loss_name == 'variance' and y_train is not None:
            basis_var = torch.var(y_train, dim=0).clamp_min(variance_min_cutoff)
            basis_loss = partial(var_norm_mse_loss, precomputed_variance=basis_var)
        else:
            basis_loss = base_flux_loss
        basis_loss_fn = BasisSpaceLoss(basis_loss, Z_full)
        flux_loss_fn = MixedFluxLoss(basis_loss_fn, base_flux_loss, mixed_flux_weight)
    else:  # flux_loss_space == 'flux'
        flux_loss_fn = base_flux_loss

    return {'flux': flux_loss_fn, 'reversibility': rev_loss_fn}


def _print_token_selection(token_information, labeling_df, fixed_idx, method,
                           features_csv=None, n_original=None):
    """Print a summary of the token scoring method and pool size."""
    n_fx  = len(fixed_idx) if fixed_idx else 0
    n_all = labeling_df.shape[0]

    if method == 'fisher' and features_csv:
        method_label = f"Fisher/D-optimal (pre-computed: {features_csv})"
        pool_str = f"{n_fx} flux tokens + {n_all} selected labeling tokens (dropout applied during training)"
        if n_original is not None and n_original != n_all:
            pool_str += f"  [filtered from {n_original}]"
    elif token_information is not None:
        method_label = method
        pool_str = f"{n_fx} flux tokens + {n_all} labeling tokens (all used; dropout applied during training)"
    else:
        method_label = "all tokens (no scoring)"
        pool_str = f"{n_fx} flux tokens + {n_all} labeling tokens (all used; dropout applied during training)"

    print(f"\n=== Token Selection ===")
    print(f"Method:  {method_label}")
    print(f"Pool:    {pool_str}")


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labeling_df, flux_df, reversibility_df = load_data(args.data_dir)
    if args.max_samples is not None and args.max_samples < labeling_df.shape[1]:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(labeling_df.shape[1])[:args.max_samples]
        labeling_df      = labeling_df.iloc[:, idx]
        flux_df          = flux_df.iloc[:, idx]
        reversibility_df = reversibility_df.iloc[:, idx]
        print(f"Randomly subsampled {args.max_samples} samples (seed={args.seed})")

    if getattr(args, 'exclude_extracellular', False):
        n_before = labeling_df.shape[0]
        mask = ~labeling_df.index.get_level_values('Metabolite').str.endswith('_e')
        labeling_df = labeling_df.loc[mask]
        print(f"Excluded extracellular tokens: {n_before} → {labeling_df.shape[0]}")

    if getattr(args, 'compartment_mix', False):
        n_before = labeling_df.shape[0]
        labeling_df = reduce_compartment_tokens(labeling_df, seed=args.seed)
        print(f"Compartment mixing: reduced tokens {n_before} → {labeling_df.shape[0]}")

    n_tokens_all = labeling_df.shape[0]
    if getattr(args, 'token_selection_method', 'pca') == 'fisher':
        if not os.path.isfile(args.features_csv):
            raise FileNotFoundError(
                f"token_selection_method='fisher' requires features.csv at "
                f"{args.features_csv}. Run extract_informative_features.py first."
            )
        _, _fisher_mask = load_fisher_token_scores(args.features_csv, labeling_df)
        labeling_df = labeling_df[_fisher_mask.numpy()]
        print(f"Fisher selection: {n_tokens_all} → {labeling_df.shape[0]} labeling tokens")

    _, base_tracer_names = create_tracer_mapping(labeling_df)
    stoichiometric_matrix, cobra_model = load_stoichiometric_matrix(args.model_json)
    if args.skip_constraints:
        metabolite_ids = [m.id for m in cobra_model.metabolites]
        reaction_ids   = [r.id for r in cobra_model.reactions]
        stoichiometric_matrix = drop_constraint_rows(
            stoichiometric_matrix, metabolite_ids, reaction_ids, args.skip_constraints
        )

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound=args.h_compound,
        h_label=args.h_label,
        h_tracer=args.h_tracer,
    )

    train_indices, val_indices = split_indices(labeling_df.shape[1], args.val_fraction, args.seed)

    # Drop constant reversibilities before building datasets — they add dead queries.
    _rv_std = reversibility_df.values.astype(np.float32).std(axis=1)
    rev_variable_idx = np.where(_rv_std > 1e-6)[0]
    reversibility_df = reversibility_df.iloc[rev_variable_idx]
    print(f"Reversibilities: {len(rev_variable_idx)} variable / {_rv_std.shape[0]} total (dropped {_rv_std.shape[0] - len(rev_variable_idx)})")

    (train_dataset, norm_stats, token_experiment_ids,
     tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=train_indices,
        measured_fluxes=args.measured_fluxes,
    )
    (val_dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        train_indices=val_indices,
        measured_fluxes=args.measured_fluxes,
        norm_stats_input=norm_stats,
    )

    _tgt_f = train_dataset.targets[:, :flux_df.shape[0]]
    _tgt_r = train_dataset.targets[:, flux_df.shape[0]:]
    print(f"Flux targets:  mean=[{_tgt_f.mean(dim=0).min():.3f}, {_tgt_f.mean(dim=0).max():.3f}]  std=[{_tgt_f.std(dim=0).min():.3f}, {_tgt_f.std(dim=0).max():.3f}]")
    print(f"Rev  targets (raw): mean=[{_tgt_r.mean(dim=0).min():.3f}, {_tgt_r.mean(dim=0).max():.3f}]  median_overall={_tgt_r.median():.3f}  max={_tgt_r.max():.1f}")

    # ── Reversibility log-space normalization ──────────────────────────────────
    # Reversibilities span ~4 orders of magnitude (median ≈ 5, max ≈ 5000).
    # Log1p + per-reaction z-score maps them to ~N(0,1), matching the basis-
    # coefficient normalization already applied to flux predictions.
    flux_dim_for_rev = flux_df.shape[0]
    if flux_dim_for_rev < train_dataset.targets.shape[1]:
        log1p_rev_train = torch.log1p(train_dataset.targets[:, flux_dim_for_rev:])
        rev_log_mean    = log1p_rev_train.mean(dim=0)                        # (n_rev,)
        rev_log_std     = log1p_rev_train.std(dim=0).clamp_min(1e-6)         # (n_rev,)
        train_dataset.targets[:, flux_dim_for_rev:] = (
            (log1p_rev_train - rev_log_mean) / rev_log_std
        )
        log1p_rev_val = torch.log1p(val_dataset.targets[:, flux_dim_for_rev:])
        val_dataset.targets[:, flux_dim_for_rev:] = (
            (log1p_rev_val - rev_log_mean) / rev_log_std
        )
        rev_log_mean_np = rev_log_mean.numpy()
        rev_log_std_np  = rev_log_std.numpy()
        print(f"Rev  targets (log1p-norm): mean_abs={rev_log_mean_np.mean():.3f}  std_range=[{rev_log_std_np.min():.3f}, {rev_log_std_np.max():.3f}]")
    else:
        rev_log_mean_np = rev_log_std_np = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    flux_dim          = flux_df.shape[0]
    reversibility_dim = reversibility_df.shape[0]

    n_tracers           = len(base_tracer_names)
    n_labeling_per_tracer = labeling_df.shape[0] // max(n_tracers, 1)
    n_bypass            = len(args.measured_fluxes) if args.measured_fluxes else 0
    print(f"\n=== Input / Output Summary ===")
    print(f"Samples:       train={len(train_dataset):>6,}  val={len(val_dataset):>6,}")
    print(f"Input tokens:  labeling={labeling_df.shape[0]:>5,}"
          f"  ({n_labeling_per_tracer} per tracer × {n_tracers} tracers)"
          f"  bypass_flux={n_bypass}")
    print(f"Targets:       flux={flux_dim}  reversibility={reversibility_dim}")
    print()

    # Reaction radial embedding for flux tokens
    h_reaction = args.h_reaction
    if h_reaction is None:
        h_reaction = max(1, math.ceil(math.log2(flux_dim)))
    reaction_emb_table = radial_embedding_table(flux_dim, h_reaction)
    print(f"  Reactions: {flux_dim} → H={h_reaction}, emb_dim={2*h_reaction}")

    # Token-type radial embedding (frozen; 2 types: 0=labeling, 1=flux)
    h_type = getattr(args, 'h_type', 2)
    type_emb_table = radial_embedding_table(2, h_type)
    print(f"  Token type: 2 → H={h_type}, emb_dim={2*h_type}")

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
        d_reaction  = 2 * h_reaction,
        d_type      = 2 * h_type,
        n_fluxes    = flux_dim,
        n_rev       = reversibility_dim,
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

    Z_full       = compute_sparse_nullspace(stoichiometric_matrix)  # varimax-rotated, orthonormal (n_fluxes, k)
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
        obs_weights = corr_yl.abs().max(dim=1).values                   # (38,) from labeling

        token_sel_method = getattr(args, 'token_selection_method', 'pca')
        if token_sel_method == 'fisher':
            token_information = None  # selection already applied via labeling_df filter above
        else:
            token_information = compute_token_scores(
                L_c, method=token_sel_method,
                n_components=getattr(args, 'n_pca_components', 50),
                y_c=y_c, y_std_c=y_std_c,
            )

        # Extend observability to include measured fluxes — basis modes that are
        # directly constrained by known fluxes are observable even if labeling alone
        # cannot determine them.
        if fixed_idx:
            F_train  = train_tgts_flux[:, fixed_idx]                     # (n_train, n_fixed)
            F_c      = F_train - F_train.mean(dim=0, keepdim=True)
            F_std_c  = F_c.std(dim=0).clamp_min(1e-10)
            cov_yF   = (y_c.T @ F_c) / y_train.shape[0]                 # (38, n_fixed)
            corr_yF  = cov_yF / y_std_c.unsqueeze(1) / F_std_c.unsqueeze(0)
            obs_from_flux   = corr_yF.abs().max(dim=1).values            # (38,)
            obs_weights_all = torch.max(obs_weights, obs_from_flux)      # (38,) combined
        else:
            obs_weights_all = obs_weights

        obs_weights_np = obs_weights_all.numpy()
        n_obs_lab  = int((obs_weights > 0.1).sum().item())
        n_obs_all  = int((obs_weights_all > 0.1).sum().item())
        print(f"Observability (labeling only): {n_obs_lab}/{n_free_basis} max|r|>0.1")
        print(f"Observability (labeling+flux): {n_obs_all}/{n_free_basis} max|r|>0.1 "
              f"; max|r| range=[{obs_weights_np.min():.3f}, {obs_weights_np.max():.3f}]")
    else:
        y_mean, y_std, obs_weights_np = None, None, None
        token_information = None

    # Builf the null-space data dict, which the model uses to reconstruct fluxes
    # the latent space
    null_space_data = dict(
        Z_free           = Z_full,               # (n_fluxes, 38) — field name kept for compat
        free_idx         = free_idx,             # metadata: indices of non-measured reactions
        fixed_idx        = fixed_idx,            # metadata: indices of measured reactions
        obs_weights      = obs_weights_np,
        y_mean           = y_mean,               # basis-space mean for denormalization
        y_std            = y_std,                # basis-space std for denormalization
        rev_variable_idx = rev_variable_idx.tolist(),  # indices of variable-reversibility reactions
        rev_log_mean     = rev_log_mean_np,      # per-reaction log1p mean for rev denormalization
        rev_log_std      = rev_log_std_np,       # per-reaction log1p std for rev denormalization
    )

    config.n_free = n_free_basis

    # Per-reaction flux normalization — computed from full dataset (all samples)
    flux_norm = FluxNorm(flux_df.values)  # (n_reactions, n_samples)
    flux_norm_stats = {
        'mean': flux_norm.mean.numpy(),
        'std':  flux_norm.std.numpy(),
    }

    model = FluxDecoder(
        config               = config,
        token_experiment_ids = token_experiment_ids,
        compound_emb_table   = vocabs['compound_emb_table'],
        label_emb_table      = vocabs['label_emb_table'],
        tracer_emb_table     = tracer_combo_emb_table,
        reaction_emb_table   = reaction_emb_table,
        type_emb_table       = type_emb_table,
        null_space_data      = null_space_data,
        flux_norm_stats      = flux_norm_stats,
    )
    model.flux_noise_std = args.flux_noise_std
    model.labeling_noise_std = args.labeling_noise_std

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model Size ===")
    print(f"Total parameters:     {total_params:>10,}")
    print(f"Trainable parameters: {trainable_params:>10,}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30s} {n:>10,}")
    print()

    y_train_for_loss = y_train if n_free_basis > 0 else None
    loss_function = build_loss_function(
        train_dataset, flux_dim, args.loss,
        flux_loss_space = args.flux_loss_space,
        Z_full          = Z_full if args.flux_loss_space in ['basis', 'mixed'] else None,
        y_train         = y_train_for_loss,
        mixed_flux_weight = args.mixed_flux_weight,
        variance_min_cutoff = args.variance_min_cutoff,
    )

    _print_token_selection(
        token_information, labeling_df, fixed_idx,
        method=getattr(args, 'token_selection_method', 'pca'),
        features_csv=getattr(args, 'features_csv', None),
        n_original=n_tokens_all,
    )

    log = train_model(
        model,
        train_loader,
        val_loader                 = val_loader,
        num_epochs                 = args.epochs,
        learning_rate              = args.learning_rate,
        device                     = device,
        gradient_clip              = args.gradient_clip,
        use_scheduler              = not args.no_scheduler,
        warmup_epochs              = args.warmup_epochs,
        flux_token_dropout         = args.flux_token_dropout,
        labeling_token_dropout     = getattr(args, 'labeling_token_dropout', 0.0),
        token_information          = token_information,
        labeling_noise_std         = args.labeling_noise_std,
        loss_function              = loss_function,
        measured_flux_loss_weight  = getattr(args, 'measured_flux_loss_weight', 0.0),
    )

    flux_names = flux_df.index.tolist()
    val_correlations = evaluate_validation_correlations(model, val_loader, device, flux_dim, flux_names=flux_names)

    # === Diagnostic: Reaction Learning Asymmetry Analysis ===
    flux_r2_per_reaction = val_correlations.get('flux_r2_per_reaction', [])
    if flux_r2_per_reaction:
        flux_r2 = torch.tensor(flux_r2_per_reaction, dtype=torch.float32)
        n_reactions_with_r2 = len(flux_r2)

        high_r2_mask = flux_r2 > 0.5
        med_r2_mask = (0.2 <= flux_r2) & (flux_r2 <= 0.5)
        low_r2_mask = flux_r2 < 0.2

        print(f"\n=== Reaction Learning Asymmetry ===")
        print(f"High R² (>0.5): {high_r2_mask.sum().item()} reactions, median={flux_r2[high_r2_mask].median():.4f}")
        print(f"Med R² (0.2-0.5): {med_r2_mask.sum().item()} reactions, median={flux_r2[med_r2_mask].median():.4f}")
        print(f"Low R² (<0.2): {low_r2_mask.sum().item()} reactions, median={flux_r2[low_r2_mask].median():.4f}")
        print(f"Overall: {(flux_r2 > 0).sum().item()} learn, {(flux_r2 <= 0).sum().item()} mislearn (R²<0)")

        # Check if high-R² reactions are in measured flux list
        measured_set = set(args.measured_fluxes) if args.measured_fluxes else set()
        if measured_set and n_reactions_with_r2 <= len(flux_names):
            high_r2_names = [flux_names[i] for i in range(n_reactions_with_r2) if high_r2_mask[i].item()]
            high_in_measured = sum(1 for n in high_r2_names if n in measured_set)
            print(f"High-R² reactions in measured_fluxes: {high_in_measured}/{len(high_r2_names)} ({100*high_in_measured/max(1, len(high_r2_names)):.1f}%)")

            # Show which high-R² reactions are measured vs unmeasured
            high_measured = [n for n in high_r2_names if n in measured_set]
            high_unmeasured = [n for n in high_r2_names if n not in measured_set]
            if high_unmeasured[:5]:
                print(f"  High-R² unmeasured (examples): {', '.join(high_unmeasured[:5])}")

        # Show reactions stuck at floor (0.13±0.005)
        floor_mask = torch.abs(flux_r2 - 0.13) < 0.005
        if floor_mask.sum() > 0:
            print(f"Reactions stuck at R²≈0.13 floor: {floor_mask.sum().item()}")

    # === Z Basis Conditioning and Sparsity ===
    try:
        Z_np = Z_full
        Z_t = torch.tensor(Z_np, dtype=torch.float32)

        # Condition number
        U, S, Vt = torch.linalg.svd(Z_t, full_matrices=False)
        condition = S[0] / S[-1]
        rank_t = torch.linalg.matrix_rank(Z_t).item()

        # Sparsity: fraction of elements with small magnitude
        Z_sparse_frac = np.mean(np.abs(Z_np) < 0.01)  # % near-zero
        Z_nnz = np.count_nonzero(Z_np) / Z_np.size  # % non-zero

        # Check per-column concentration (max abs per column)
        col_max_abs = np.max(np.abs(Z_np), axis=0)
        col_concentration = np.mean(col_max_abs ** 2)  # "energy concentration"

        print(f"\n=== Z Basis Properties ===")
        print(f"Shape: {Z_np.shape}, Rank: {rank_t}/38")
        print(f"Condition number: {condition:.2e}")
        print(f"Sparsity: {100*Z_sparse_frac:.1f}% near-zero, {100*Z_nnz:.1f}% non-zero")
        print(f"Singular values range: [{S[-1]:.2e}, {S[0]:.2e}]")
        print(f"Column max abs: mean={col_max_abs.mean():.4f}, std={col_max_abs.std():.4f}")

        residual = np.linalg.norm(stoichiometric_matrix @ Z_np, 'fro')
        print(f"Null-space check: ||S @ Z|| = {residual:.2e}")

    except Exception as e:
        print(f"Warning: Could not compute Z properties: {e}")

    # === Labeling Signal & Attention Analysis ===
    try:
        flux_r2_per_reaction = val_correlations.get('flux_r2_per_reaction', [])
        if flux_r2_per_reaction:
            flux_r2_tensor = torch.tensor(flux_r2_per_reaction, dtype=torch.float32)
            attn_df = diagnose_signal_bottleneck(model, val_loader, device, flux_names, flux_r2_tensor, labeling_df)
    except Exception as e:
        print(f"Warning: Could not run attention diagnostics: {e}")

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
            "Z_free":            Z_full,
            "free_idx":          free_idx,
            "fixed_idx":         fixed_idx,
            "obs_weights":       obs_weights_np,
            "rev_variable_idx":  rev_variable_idx.tolist(),
            "y_mean":            y_mean,
            "y_std":             y_std,
            "rev_log_mean":      rev_log_mean_np,
            "rev_log_std":       rev_log_std_np,
            "skip_constraints":  args.skip_constraints,
        },
        "token_information":      token_information.numpy() if token_information is not None else None,
        "token_selection_method": getattr(args, 'token_selection_method', 'pca'),
        "h_reaction":           h_reaction,
        "h_type":               h_type,
        "flux_norm_stats":      flux_norm_stats,
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
    f = cfg.get("features", {})
    output_dir   = cli.data_dir or f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"
    output_model = os.path.join(output_dir, t.get("output_model", "transformer.pt"))

    measured_fluxes = t.get("measured_fluxes") or None

    _prepared = os.path.join(output_dir, "model_prepared_for_mfa.json")
    _default  = os.path.join(_REPO_ROOT, cfg["model_file"]) if not os.path.isabs(cfg["model_file"]) else cfg["model_file"]

    args = argparse.Namespace(
        data_dir               = output_dir,
        model_json             = _prepared if os.path.isfile(_prepared) else _default,
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
        val_fraction           = t.get("val_fraction", 0.2),
        combined_dim           = t.get("combined_dim", 64),
        num_attn_heads         = t.get("num_attn_heads", 4),
        num_transformer_layers = t.get("num_transformer_layers", 4),
        dropout                = t.get("dropout", 0.3),
        flux_token_dropout     = t.get("flux_token_dropout", 0.0),
        labeling_token_dropout = f.get("token_dropout", 0.0),
        measured_fluxes        = measured_fluxes,
        h_compound             = t.get("h_compound"),
        h_label                = t.get("h_label"),
        h_tracer               = t.get("h_tracer"),
        h_reaction             = t.get("h_reaction"),
        h_type                 = t.get("h_type", 2),
        flux_noise_std         = t.get("flux_noise_std", 0.0),
        labeling_noise_std     = t.get("labeling_noise_std", 0.0),
        loss                      = t.get("loss", "variance"),
        flux_loss_space           = t.get("flux_loss_space", "flux"),
        mixed_flux_weight         = t.get("mixed_flux_weight", 0.1),
        variance_min_cutoff       = t.get("variance_min_cutoff", 0.1),
        measured_flux_loss_weight = t.get("measured_flux_loss_weight", 0.0),
        skip_constraints          = t.get("skip_constraints") or [],
        compartment_mix           = t.get("compartment_mix", False),
        exclude_extracellular     = t.get("exclude_extracellular", False),
        # Feature selection (from features: config section)
        token_selection_method = f.get("token_selection_method", "pca"),
        n_pca_components       = f.get("n_pca_components", 50),
        features_csv           = os.path.join(output_dir, f.get("output", "features.csv")),
    )
    main(args)
