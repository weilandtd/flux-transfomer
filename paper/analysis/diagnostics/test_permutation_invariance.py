"""
Test permutation invariance of FluxDecoder.

Shuffles the token order (MID inputs) and checks that flux predictions
are identical up to floating-point tolerance. A well-built model with no
positional encoding should be exactly invariant; any difference reveals
positional information leaking in.

Usage:
    python paper/test_permutation_invariance.py path/to/checkpoint.pt
    python paper/test_permutation_invariance.py ...pt --n-samples 64 --n-shuffles 5
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--n-samples',  type=int,   default=32,
                        help='Samples to test (default 32)')
    parser.add_argument('--n-shuffles', type=int,   default=5,
                        help='Random permutations per sample (default 5)')
    parser.add_argument('--atol',       type=float, default=5e-4,
                        help='Absolute tolerance for PASS (default 5e-4; GPU float32 rounding with ~500 tokens accumulates ~1e-4)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict   = ckpt['model_config']
    nsd        = ckpt['null_space_data']
    data_dir   = ckpt['data_dir']
    model_json = ckpt.get('model_json')

    # ── Rebuild data + vocabs ─────────────────────────────────────────────────
    labeling_df, flux_df, reversibility_df = load_data(
        data_dir, max_samples=args.n_samples * 2
    )
    _, base_tracer_names = create_tracer_mapping(labeling_df)
    _, cobra_model       = load_stoichiometric_matrix(model_json)

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound = cfg_dict.get('d_comp',   32) // 2,
        h_label    = cfg_dict.get('d_label',  10) // 2,
        h_tracer   = cfg_dict.get('d_tracer',  2) // 2,
    )

    val_indices = np.arange(min(args.n_samples, labeling_df.shape[1]))
    (val_dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df, flux_df, reversibility_df,
        vocabs            = vocabs,
        train_indices     = val_indices,
        normalize         = True,
        measured_fluxes   = ckpt.get('measured_fluxes'),
        norm_stats_input  = ckpt.get('norm_stats'),
    )

    loader = DataLoader(val_dataset, batch_size=args.n_samples, shuffle=False)
    batch  = next(iter(loader))
    (cids, lids, tids, fracs), targets = batch

    cids   = cids.to(device)     # (B, N)
    lids   = lids.to(device)     # (B, N)
    tids   = tids.to(device)     # (B, N)
    fracs  = fracs.to(device)    # (B, N, 1) or (B, N)
    N      = cids.shape[1]

    fixed_fluxes = None
    if cfg_dict.get('n_measured', 0) > 0:
        fixed_fluxes = targets[:, nsd.get('fixed_idx', [])].to(device)

    # ── Reconstruct model ─────────────────────────────────────────────────────
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__
                               if k in cfg_dict})
    token_ids_tensor = ckpt['model_state_dict']['token_experiment_ids']
    model = FluxDecoder(
        cfg,
        stoichiometric_matrix = None,
        token_experiment_ids  = token_ids_tensor,
        null_space_data       = nsd,
        fixed_flux_indices    = nsd.get('fixed_idx', []),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Tokens per sample: {N}')
    print(f'Testing {args.n_shuffles} shuffles × {cids.shape[0]} samples\n')

    # ── Baseline prediction ───────────────────────────────────────────────────
    with torch.no_grad():
        fluxes_orig, _ = model(cids, lids, tids, fracs,
                               padding_mask=None, fixed_fluxes=fixed_fluxes)

    # ── Shuffle tests ─────────────────────────────────────────────────────────
    all_pass = True
    max_abs_global = 0.0

    for i in range(args.n_shuffles):
        perm = torch.randperm(N, device=device)

        fracs_s = fracs[:, perm] if fracs.dim() == 2 else fracs[:, perm, :]

        with torch.no_grad():
            fluxes_shuf, _ = model(
                cids[:, perm], lids[:, perm], tids[:, perm], fracs_s,
                padding_mask=None, fixed_fluxes=fixed_fluxes,
            )

        diff     = (fluxes_orig - fluxes_shuf).abs()
        max_abs  = diff.max().item()
        mean_abs = diff.mean().item()
        max_rel  = (diff / fluxes_orig.abs().clamp(min=1e-6)).max().item()
        passed   = max_abs < args.atol

        max_abs_global = max(max_abs_global, max_abs)
        all_pass = all_pass and passed

        status = 'PASS' if passed else 'FAIL'
        print(f'  Shuffle {i+1}: [{status}]  '
              f'max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  max_rel={max_rel:.2e}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 55)
    if all_pass:
        print(f'  RESULT: PERMUTATION INVARIANT  (all {args.n_shuffles} shuffles passed)')
        print(f'  Worst-case max |diff|: {max_abs_global:.2e}  (tol={args.atol:.0e})')
        print(f'  Differences are pure floating-point rounding — expected.')
    else:
        print(f'  RESULT: NOT INVARIANT  — positional info is leaking in.')
        print(f'  Worst-case max |diff|: {max_abs_global:.2e}  (tol={args.atol:.0e})')
        print(f'  Check: positional encoding added? padding mask shuffled correctly?')
    print('=' * 55)


if __name__ == '__main__':
    main()
