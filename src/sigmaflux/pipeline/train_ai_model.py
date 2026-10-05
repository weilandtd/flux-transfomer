"""
Train SigmaFlux's FluxDecoder (cross-attention decoder) from generated HDF5 data.

Example:
    sigmaflux-train --config paper/io_config.yaml
"""

import argparse
import datetime
import math
import os
from pathlib import Path
from dataclasses import asdict
from functools import partial

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sigmaflux.pipeline._paths import (
    add_root_argument,
    default_output_dir,
    load_config,
    resolve_path,
)
from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
)
from sigmaflux.ai.features import TOKEN_SELECTION_METHODS
from sigmaflux.ai.utils import load_fisher_token_scores
from sigmaflux.ai.diagnostics import diagnose_signal_bottleneck
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.train import (
    BasisSpaceLoss, FluxNorm,
    evaluate_validation_correlations, train_model, var_norm_mse_loss,
)
from sigmaflux.ai.utils import (
    compute_sparse_nullspace, compute_targeted_nullspace,
    compute_ica_nullspace, compute_targeted_ica_nullspace,
    split_constant_directions,
)


def share_basis(Z_full, basis_file):
    """Return the null-space basis every ensemble member must use.

    Writes *Z_full* to *basis_file* the first time and loads it thereafter (ICA bases
    aren't bit-reproducible across BLAS thread counts, so members must share one).

    Args:
        Z_full: null-space basis computed by this run.
        basis_file: path to share the basis across ensemble members; None disables
            sharing and returns Z_full unchanged.
    """
    if not basis_file:
        return Z_full
    if os.path.exists(basis_file):
        loaded = np.load(basis_file)['Z_full']
        if loaded.shape != Z_full.shape:
            raise ValueError(
                f"basis file {basis_file} holds a {loaded.shape} basis but this run "
                f"computed {Z_full.shape}; it belongs to a different model"
            )
        print(f"Loaded shared null-space basis from {basis_file} {loaded.shape}")
        return loaded
    parent = os.path.dirname(os.path.abspath(basis_file))
    os.makedirs(parent, exist_ok=True)
    np.savez_compressed(basis_file, Z_full=Z_full)
    print(f"Saved shared null-space basis to {basis_file} {Z_full.shape}")
    return Z_full


def split_indices(n_samples, val_fraction, seed):
    """Return (train_indices, val_indices) from a shuffled split.

    Args:
        n_samples: total number of samples to split.
        val_fraction: fraction of samples held out for validation.
        seed: RNG seed for the shuffle.
    """
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int((1.0 - val_fraction) * n_samples)
    return indices[:split], indices[split:]


def shuffle_targets(flux_df, reversibility_df, seed):
    """Null-model control: permute (flux, reversibility) samples, independent of labeling.

    ``flux_df`` and ``reversibility_df`` are permuted by the *same* draw, so each target
    stays the real, internally consistent pair a sampled flux state and its reversibility
    trace always are -- S.v = 0 is untouched. What breaks is only the association with a
    labeling sample: metabolite M+i measurements are now paired with an unrelated flux
    state, so a model trained on this pairing can carry no real metabolite-to-flux
    attribution. If one still shows attribution structure on this data, that structure is
    an artifact of the pipeline, not of the biology.

    Args:
        flux_df: (reactions, samples).
        reversibility_df: (reactions, samples), same sample axis as flux_df.
        seed: RNG seed, shared across an ensemble so every member sees one broken pairing.

    Returns (flux_df, reversibility_df), permuted and with sample columns renumbered
    0..n-1 to match what :func:`sigmaflux.ai.data.load_data` itself produces.
    """
    perm = np.random.default_rng(seed).permutation(flux_df.shape[1])
    flux_df = flux_df.iloc[:, perm]
    flux_df.columns = range(flux_df.shape[1])
    reversibility_df = reversibility_df.iloc[:, perm]
    reversibility_df.columns = range(reversibility_df.shape[1])
    return flux_df, reversibility_df


class MixedFluxLoss:
    """Combines basis-space and flux-space losses."""

    def __init__(self, basis_loss_fn, flux_loss_fn, flux_weight):
        """Args:
            basis_loss_fn: loss computed in null-space basis coordinates.
            flux_loss_fn: loss computed in reconstructed flux space.
            flux_weight: weight on flux_loss_fn, in [0, 1]; basis weight is 1 - flux_weight.
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
    """Build flux and reversibility loss functions.

    Args:
        train_dataset: training dataset holding the target tensors.
        flux_dim: number of flux outputs (targets beyond this are reversibilities).
        loss_name: 'mse' or 'variance'-normalized MSE.
        flux_loss_space: 'flux' (reconstructed), 'basis' (null-space), or 'mixed'.
        Z_full: null-space basis; required when flux_loss_space is 'basis' or 'mixed'.
        y_train: basis-space training targets, used for variance normalization.
        mixed_flux_weight: flux-space weight when flux_loss_space is 'mixed'.
        variance_min_cutoff: floor on per-output variance used for normalization.
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


def _print_token_selection(labeling_df, fixed_idx, method, features_csv, n_original=None):
    """Print a summary of the token selection method and pool size.

    Args:
        labeling_df: labeling (MID) token table, used for the pool size.
        fixed_idx: indices of the fixed (measured) flux tokens.
        method: token selection method name ('fisher', 'fisher_fast', or 'all').
        features_csv: pre-computed Fisher significance features file.
        n_original: token count before any filtering, for reporting.
    """
    n_fx  = len(fixed_idx) if fixed_idx else 0
    n_all = labeling_df.shape[0]
    per_tracer = labeling_df.index.get_level_values('Tracer').value_counts()

    pool_str = f"{n_fx} flux tokens + {n_all} selected labeling tokens (dropout applied during training)"
    if n_original is not None and n_original != n_all:
        pool_str += f"  [filtered from {n_original}]"
    pool_str += "\n         " + ", ".join(f"{t}: {n}" for t, n in per_tracer.items())

    print(f"\n=== Token Selection ===")
    print(f"Method:  {method} (pre-computed: {features_csv})")
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

    if getattr(args, "shuffle_targets", False):
        # shuffle_seed, not args.seed: an ensemble member's args.seed is base_seed + i
        # (see the ensemble loop in cli()), but every member of a null ensemble must see
        # the same broken pairing, or cross-seed agreement tests nothing.
        _shuffle_seed = getattr(args, "shuffle_seed", args.seed)
        flux_df, reversibility_df = shuffle_targets(flux_df, reversibility_df, _shuffle_seed)
        print(f"--shuffle-targets: permuted {flux_df.shape[1]} flux/reversibility samples "
              f"relative to labeling (seed={_shuffle_seed}) -- null-model control")

    if args.max_samples is not None and args.max_samples < labeling_df.shape[1]:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(labeling_df.shape[1])[:args.max_samples]
        labeling_df      = labeling_df.iloc[:, idx]
        flux_df          = flux_df.iloc[:, idx]
        reversibility_df = reversibility_df.iloc[:, idx]
        print(f"Randomly subsampled {args.max_samples} samples (seed={args.seed})")

    n_tokens_all = labeling_df.shape[0]
    if not os.path.isfile(args.features_csv):
        raise FileNotFoundError(
            f"features.csv not found at {args.features_csv}. Run sigmaflux-select-features first."
        )
    _, _fisher_mask = load_fisher_token_scores(args.features_csv, labeling_df)
    labeling_df = labeling_df[_fisher_mask.numpy()]
    print(f"Fisher selection: {n_tokens_all} → {labeling_df.shape[0]} labeling tokens")

    _, base_tracer_names = create_tracer_mapping(labeling_df)

    # Load stoichiometric matrix from the MFA-prepared model in the data directory.
    # The prepared model has BIOMASS/skip-constraint reactions removed and its
    # reaction ordering matches flux_df exactly — the annotated model_file does not.
    import os as _os
    _prepared = _os.path.join(args.data_dir, 'model_prepared_for_mfa.json')
    stoichiometric_matrix, cobra_model = load_stoichiometric_matrix(_prepared)

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound=args.h_compound,
        h_label=args.h_label,
        h_tracer=args.h_tracer,
    )

    if getattr(args, 'split_file', None):
        import json as _json
        _sp = _json.load(open(args.split_file))
        train_indices = np.asarray(_sp['train_indices'], dtype=int)
        val_indices   = np.asarray(_sp['val_indices'], dtype=int)
        print(f"Loaded split from {args.split_file}: train={len(train_indices)} val={len(val_indices)}")
    else:
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
        normalize_mids=args.normalize_mids,
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

    # Per-tracer selection means the tracers no longer carry equal token counts, so this
    # is counted rather than divided out.
    per_tracer = labeling_df.index.get_level_values('Tracer').value_counts()
    n_bypass   = len(args.measured_fluxes) if args.measured_fluxes else 0
    print(f"\n=== Input / Output Summary ===")
    print(f"Samples:       train={len(train_dataset):>6,}  val={len(val_dataset):>6,}")
    print(f"Input tokens:  labeling={labeling_df.shape[0]:>5,}"
          f"  ({', '.join(f'{t}: {n}' for t, n in per_tracer.items())})"
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

    # Leading columns of a targeted basis are the identity block, one per measured
    # reaction; split_constant_directions must not rotate them away.
    n_targets = 0
    use_targeted = (
        args.null_space_type == 'targeted'
        and args.measured_fluxes
    )
    if use_targeted:
        rxn_names = list(flux_df.index)
        Z_full, basis_status = compute_targeted_nullspace(
            stoichiometric_matrix,
            reaction_names=rxn_names,
            target_reactions=args.measured_fluxes,
        )
        if not basis_status['separable']:
            # No silent substitution. Swapping in a Varimax basis leaves the measured
            # fluxes without a dedicated coordinate while the run still calls itself
            # 'targeted', which is how the T cell trained for months against a basis
            # nobody asked for. Redundant targets are now dropped individually, so
            # reaching here means something the caller has to decide about.
            raise RuntimeError(
                f"targeted basis could not be built: {basis_status['fallback_reason']}. "
                f"Choose a different null_space_type explicitly, or fix measured_fluxes."
            )
        else:
            n_t = n_targets = len(basis_status['targets_kept'])
            print(f"Targeted basis: {n_t} measured reaction DOF(s) at cols 0–{n_t - 1}, "
                  f"cond(S_D)={basis_status['cond_SD']:.2e}")
            if basis_status['targets_dropped']:
                print(f"  {len(basis_status['targets_dropped'])} redundant target(s) share a "
                      f"DOF with a more connected one: "
                      f"{', '.join(basis_status['targets_dropped'])}")
    elif args.null_space_type == 'ica_informed':
        flux_matrix = flux_df.values.T   # (n_samples, n_reactions)
        Z_full, basis_status = compute_ica_nullspace(
            stoichiometric_matrix, flux_matrix, **args.ica_nullspace_params
        )
        if basis_status['fallback']:
            print(f"ICA-informed basis: FastICA failed; fell back to Varimax (d={Z_full.shape[1]})")
    elif args.null_space_type == 'targeted_ica':
        rxn_names = list(flux_df.index)
        flux_matrix = flux_df.values.T
        if not args.measured_fluxes:
            # targeted_ica without targets is a contradiction, not a default to paper over.
            raise ValueError(
                "null_space_type 'targeted_ica' needs training.measured_fluxes. "
                "Set null_space_type: ica_informed if that is what you meant."
            )
        Z_full, basis_status = compute_targeted_ica_nullspace(
            stoichiometric_matrix, flux_matrix,
            reaction_names=rxn_names,
            target_reactions=args.measured_fluxes,
            **args.ica_nullspace_params,
        )
        n_t = n_targets = len(basis_status['targets_kept'])
        print(f"Targeted-ICA basis: {n_t} identity DOF(s), "
              f"{basis_status['n_ica_free']} ICA free modes, "
              f"{basis_status['n_complement_free']} complement "
              f"(free rank {basis_status['n_active_free']}/{Z_full.shape[1] - n_t})")
    elif args.null_space_type == 'boundary':
        from sigmaflux.ai.boundary_basis import compute_boundary_basis
        boundary_solver = getattr(args, 'boundary_solver', None)
        Z_full, basis_status = compute_boundary_basis(
            cobra_model,
            solver=boundary_solver,
            tol=1e-8,
        )
        print(f"Boundary basis: {basis_status['n_throughput']} throughput + "
              f"{basis_status['n_cycle']} cycle columns "
              f"(solver={basis_status['solver_used']!r})")
        infeasible = sum(
            1 for m in basis_status['throughput_meta']
            if not m['forward_feasible']
        )
        if infeasible:
            print(f"  {infeasible} coordinate(s) forward-infeasible (flagged, not dropped)")
    else:
        print(f"Null-space basis: Varimax (null_space_type={args.null_space_type!r})")
        Z_full = compute_sparse_nullspace(stoichiometric_matrix)

    # ── Share one basis across an ensemble ────────────────────────────────────
    # ICA-based bases are not bit-reproducible: FastICA's fixed point depends on
    # the covariance's last bits, which change with the BLAS thread count, so
    # members trained on different nodes can land on different (equally valid)
    # rotations of the same subspace. Predictions are unaffected — fluxes are
    # physical — but any per-member comparison in basis coordinates is not, so
    # members must share one basis, exactly as they share one train/val split.
    Z_full = share_basis(Z_full, getattr(args, 'basis_file', None))

    # Directions no sample moves along -- a measurement pool's pinned drain, an
    # FVA-fixed route -- still have to be representable for v = y @ Z.T to reconstruct a
    # real flux vector, but there is nothing to predict along them. Left in, each one
    # takes maximum loss weight (variance_min_cutoff) on a target that is identically
    # zero once standardised, and inflates every basis-space metric.
    _V_all = flux_df.loc[[r for r in flux_df.index]].to_numpy(np.float64).T
    Z_full, Z_const, y_const = split_constant_directions(Z_full, _V_all,
                                                        n_protected=n_targets)
    flux_offset = (y_const @ Z_const.T).astype(np.float32)
    if Z_const.shape[1]:
        print(f"Constant basis directions (pinned reactions): {Z_const.shape[1]} held "
              f"fixed, predicting {Z_full.shape[1]} of "
              f"{Z_full.shape[1] + Z_const.shape[1]}")

    flux_names = flux_df.index.tolist()

    # Stored in the checkpoint so the predictor can clamp its reconstruction into the
    # box; the affine map gives S v = 0 but nothing keeps a state inside the bounds.
    #
    # The predictor's copy of the bounds reopens the pool mixing legs, which the shared
    # prepared model leaves at [0, 1] so the MFA fit keeps a simplex-sized search box.
    # Sampling drew the fractions with the drain pinned, so it is unaffected either way;
    # what the legs must not do here is assert a ceiling of 1 on a block whose drain is
    # already free to scale. On I. orientalis this frees 72 of the 144 facets that
    # otherwise cap the polytope's inscribed radius.
    from sigmaflux.pipeline.sample_flux import relax_measurement_pool_bounds
    _reopened = relax_measurement_pool_bounds(cobra_model, legs=True)
    if _reopened:
        print(f"Reopened {len(_reopened)} measurement-pool reaction(s) for prediction; "
              f"model_prepared_for_mfa.json keeps the legs at [0, 1]")
    _bounds = {r.id: (r.lower_bound, r.upper_bound) for r in cobra_model.reactions}
    _missing = [n for n in flux_names if n not in _bounds]
    if _missing:
        raise ValueError(
            f"{len(_missing)} predicted reaction(s) are absent from the prepared model, "
            f"so their bounds cannot be stored: {_missing[:5]}"
        )
    flux_bounds = {
        "lb": np.array([_bounds[n][0] for n in flux_names], dtype=np.float64),
        "ub": np.array([_bounds[n][1] for n in flux_names], dtype=np.float64),
    }

    n_free_basis = Z_full.shape[1]
    config.n_free = n_free_basis

    print(f"Full null-space basis dimension: {n_free_basis} "
          f"(head width {config.n_free})")

    # Z_proj: recovers true null-space coordinates  y = v @ Z_proj  (v = y @ Z_full.T).
    # For orthonormal Z (Varimax fallback): G = I → Z_proj = Z_full (no change).
    # For column-normalised targeted Z: G is a Gram matrix; bounded condition number.
    _G = Z_full.T @ Z_full
    print(f"Basis Gram cond: {np.linalg.cond(_G):.2e}")
    if np.allclose(_G, np.eye(_G.shape[0]), atol=1e-6):
        Z_proj = Z_full
    else:
        Z_proj = Z_full @ np.linalg.solve(_G, np.eye(_G.shape[0]))

    # ── Compute y normalization and observability weights from training data ──
    # y_true = v_targets @ Z_proj  (proper coordinates for the null-space basis)
    # Normalize so the model predicts y_norm ≈ N(0,1) — clean gradient signal.
    # obs_weights[i] = max|Pearson r(y_i, labeling_j)| across all labeling features.
    if n_free_basis > 0:
        train_tgts      = train_dataset.targets                          # (n_train, flux+rev)
        train_tgts_flux = train_tgts[:, :flux_dim]                       # (n_train, n_rxn)
        Z_proj_t        = torch.tensor(Z_proj, dtype=torch.float32)
        y_train         = train_tgts_flux @ Z_proj_t                     # (n_train, k)
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

        token_information = None  # selection already applied via labeling_df filter above

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

    # Build the null-space data dict, which the model uses to reconstruct fluxes
    # in the latent space
    null_space_data = dict(
        Z_free           = Z_full,               # (n_fluxes, 38) — field name kept for compat
        flux_offset      = flux_offset,          # constant null-space offset for pinned reactions
        free_idx         = free_idx,             # metadata: indices of non-measured reactions
        fixed_idx        = fixed_idx,            # metadata: indices of measured reactions
        obs_weights      = obs_weights_np,
        y_mean           = y_mean,               # basis-space mean for denormalization
        y_std            = y_std,                # basis-space std for denormalization
        rev_variable_idx = rev_variable_idx.tolist(),  # indices of variable-reversibility reactions
        rev_log_mean     = rev_log_mean_np,      # per-reaction log1p mean for rev denormalization
        rev_log_std      = rev_log_std_np,       # per-reaction log1p std for rev denormalization
    )
    null_space_data.update(flux_bounds=flux_bounds)

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
        Z_full          = Z_proj if args.flux_loss_space in ['basis', 'mixed'] else None,
        y_train         = y_train_for_loss,
        mixed_flux_weight = args.mixed_flux_weight,
        variance_min_cutoff = args.variance_min_cutoff,
    )

    _print_token_selection(
        labeling_df, fixed_idx,
        method=args.token_selection_method,
        features_csv=args.features_csv,
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
        early_stopping_patience    = getattr(args, 'early_stopping_patience', None),
    )

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
        # The prepared model's own reaction bounds, aligned to flux_names. Inference
        # projects every predicted flux state back inside them (see
        # SigmaFlux._project_into_bounds); storing them here is what lets it do that
        # without the COBRA model or a solver. Read off the prepared model as-is -- no
        # FVA is re-run, these are the bounds sampling and training already saw.
        "flux_bounds":          flux_bounds,
        "reversibility_names":  reversibility_df.index.tolist(),
        "measured_fluxes":      args.measured_fluxes,
        "null_space_data": {
            "Z_free":            Z_full,
            "flux_offset":       flux_offset,
            "free_idx":          free_idx,
            "fixed_idx":         fixed_idx,
            "obs_weights":       obs_weights_np,
            "rev_variable_idx":  rev_variable_idx.tolist(),
            "y_mean":            y_mean,
            "y_std":             y_std,
            "rev_log_mean":      rev_log_mean_np,
            "rev_log_std":       rev_log_std_np,
            # The predictor clamps into these, so they travel with the checkpoint.
            "flux_bounds":       null_space_data["flux_bounds"],
        },
        "token_information":      token_information.numpy() if token_information is not None else None,
        "token_selection_method": args.token_selection_method,
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


def build_parser():
    parser = argparse.ArgumentParser(description="Train the SigmaFlux FluxDecoder.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", default=None, help="Directory containing HDF5 training data (overrides date-based default)")
    parser.add_argument("--normalize-mids", action="store_true",
                        help="per-token z-score the labeling (MID) fractions to match flux input scale")
    parser.add_argument("--output", default=None, help="Override output checkpoint path (.pt)")
    parser.add_argument("--split-file", default=None, help="JSON with explicit train_indices/val_indices (e.g. KO-aware split)")
    parser.add_argument("--basis-file", default=None,
                        help="NPZ holding the shared null-space basis. Loaded if it exists, "
                             "else written after computing it. Ensemble members MUST share one: "
                             "ICA bases are not bit-reproducible across BLAS thread counts.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--ensemble-size", type=int, default=None,
                        help="Number of models to train (override training.ensemble_size). >1 = ensemble.")
    parser.add_argument("--shuffle-targets", action="store_true",
                        help="Null-model control: permute flux/reversibility samples relative to "
                             "labeling samples before training, seeded by --seed. Each target "
                             "stays a real, internally consistent (flux, reversibility) pair -- "
                             "S.v = 0 still holds -- it is just paired with the wrong MIDs. Any "
                             "learned metabolite-to-flux attribution must come out flat under this.")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    cli = build_parser().parse_args(argv)
    cfg, root = load_config(cli.config, cli.root)

    t = cfg.get("training", {})
    f = cfg.get("features", {})
    output_dir   = default_output_dir(cfg, cli.data_dir)
    output_model = cli.output or os.path.join(output_dir, t.get("output_model", "transformer.pt"))

    measured_fluxes = t.get("measured_fluxes") or None

    _prepared = os.path.join(output_dir, "model_prepared_for_mfa.json")
    _default  = resolve_path(cfg["model_file"], root)

    _seed = cli.seed if cli.seed is not None else cfg.get("seed", 42)
    args = argparse.Namespace(
        data_dir               = output_dir,
        model_json             = _prepared if os.path.isfile(_prepared) else _default,
        output                 = output_model,
        normalize_mids         = cli.normalize_mids or t.get("normalize_mids", False),
        split_file             = cli.split_file,
        basis_file             = cli.basis_file,
        device                 = cfg.get("device", "auto"),
        seed                   = _seed,
        max_samples            = t.get("max_samples"),
        epochs                 = t.get("epochs", 100),
        early_stopping_patience = t.get("early_stopping_patience"),
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
        labeling_token_dropout = t.get("labeling_token_dropout", 0.0),
        measured_fluxes        = measured_fluxes,
        null_space_type        = t.get("null_space_type", "targeted"),   # targeted | varimax | ica_informed | targeted_ica
        ica_nullspace_params   = t.get("ica_nullspace", {}),
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
        shuffle_targets           = cli.shuffle_targets,
        # Fixed here, NOT re-read as args.seed inside main(): the ensemble loop below
        # sets member.seed = base_seed + i per member, and a null ensemble's whole point
        # is that every member trains on the identical broken pairing. Using the
        # per-member seed would give each one its own unrelated shuffle instead, and
        # cross-seed agreement would test nothing.
        shuffle_seed              = _seed,
        # Feature selection (from features: config section)
        token_selection_method = f.get("token_selection_method", "fisher"),
        features_csv           = os.path.join(output_dir, f.get("output", "features.csv")),
    )
    if args.token_selection_method not in TOKEN_SELECTION_METHODS:
        raise ValueError(
            f"features.token_selection_method must be one of {sorted(TOKEN_SELECTION_METHODS)}, "
            f"got {args.token_selection_method!r}"
        )

    # ── Ensemble: N models (distinct seeds) sharing ONE split and ONE basis ──
    # ensemble_size > 1 -> members saved to <data_dir>/ensemble/model_{i}.pt.
    # Both the split and the null-space basis are shared (from --split-file /
    # --basis-file, or captured from member 0): the split so per-reaction
    # confidence is measured on a common val set, the basis because an ICA basis
    # is not bit-reproducible across BLAS thread counts and members would
    # otherwise use different coordinates.
    ensemble_size = cli.ensemble_size if cli.ensemble_size is not None else t.get("ensemble_size", 1)
    if ensemble_size and ensemble_size > 1:
        import copy, json as _json, torch as _torch
        ens_dir = os.path.join(output_dir, "ensemble")
        os.makedirs(ens_dir, exist_ok=True)
        base_seed, shared_split = args.seed, args.split_file
        # One basis for the whole ensemble: member 0 writes it, the rest load it.
        shared_basis = args.basis_file or os.path.join(ens_dir, "basis.npz")
        print(f"\n=== Ensemble training: {ensemble_size} members (base seed {base_seed}) -> {ens_dir}/ ===")
        for i in range(ensemble_size):
            member = copy.copy(args)
            member.seed = base_seed + i
            member.output = os.path.join(ens_dir, f"model_{i}.pt")
            member.split_file = shared_split
            member.basis_file = shared_basis
            print(f"\n----- ensemble member {i+1}/{ensemble_size}  seed={member.seed}  -> {member.output} -----")
            main(member)
            if shared_split is None:      # capture member 0's split so all members share it
                _ck = _torch.load(member.output, map_location="cpu", weights_only=False)
                shared_split = os.path.join(ens_dir, "split.json")
                _json.dump({"train_indices": list(_ck["train_indices"]),
                            "val_indices":  list(_ck["val_indices"])}, open(shared_split, "w"))
                print(f"shared split captured -> {shared_split}")
        print(f"\n=== Ensemble done: {ensemble_size} models in {ens_dir}/ ===")
        return None
    return main(args)


if __name__ == "__main__":
    cli()
