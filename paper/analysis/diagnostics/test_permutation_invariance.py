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

import numpy as np
import torch

from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device


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

    device = resolve_device()
    print(f'Device: {device}')

    # ── Load checkpoint, data and model ───────────────────────────────────────
    # The leading n_samples columns, not the saved val split — invariance is a
    # structural property, so which samples are used does not matter.
    probe = load_eval_context(args.checkpoint, build_dataset=False,
                              max_samples=args.n_samples * 2)
    val_indices = np.arange(min(args.n_samples, probe.labeling_df.shape[1]))
    ctx = load_eval_context(
        probe.ckpt, samples=val_indices,
        data=(probe.labeling_df, probe.flux_df, probe.reversibility_df),
    )

    (cids, lids, tids, fracs), targets = ctx.full_batch(device=device)
    N = cids.shape[1]

    model = build_flux_decoder(ctx, device=device)

    measured_flux_ids = model.fixed_idx_t.to(device)
    measured_fluxes = (targets.index_select(1, measured_flux_ids)
                       if len(model.fixed_idx) > 0 else None)

    print(f'Model: {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Tokens per sample: {N}')
    print(f'Testing {args.n_shuffles} shuffles × {cids.shape[0]} samples\n')

    # ── Baseline prediction ───────────────────────────────────────────────────
    with torch.no_grad():
        fluxes_orig, _ = model(cids, lids, tids, fracs,
                               measured_fluxes=measured_fluxes,
                               measured_flux_ids=measured_flux_ids)

    # ── Shuffle tests ─────────────────────────────────────────────────────────
    all_pass = True
    max_abs_global = 0.0

    for i in range(args.n_shuffles):
        perm = torch.randperm(N, device=device)

        fracs_s = fracs[:, perm] if fracs.dim() == 2 else fracs[:, perm, :]

        with torch.no_grad():
            fluxes_shuf, _ = model(
                cids[:, perm], lids[:, perm], tids[:, perm], fracs_s,
                measured_fluxes=measured_fluxes,
                measured_flux_ids=measured_flux_ids,
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
