"""
Test model robustness to missing metabolites.

Step 1: Baseline inference (all metabolites present) → find fluxes with R² > threshold.
Step 2: For each masking rate, randomly mask N metabolites and measure how many
        of those baseline-recoverable fluxes are still recovered.

Reports:
  - Baseline recoverable fluxes (R² > --r2-threshold)
  - Per masking scenario: which metabolites were masked, recovered vs lost fluxes
  - Summary table across rates

Usage:
    python paper/analysis/test_missing_metabolites.py I_orientalis_20260508/model/imisor_transformer.pt

    python paper/analysis/test_missing_metabolites.py ...pt \\
        --n-samples 500 --rates 0.1 0.2 0.3 0.5 0.7 --seeds 5 --r2-threshold 0.3
"""

import argparse

import numpy as np
import torch

from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device
from sigmaflux.ai.model import MISSING_LABEL_VALUE
from sigmaflux.ai.train import per_reaction_r2


def run_inference(model, loader, device, missing_fn=None):
    """Run model over all batches, return (preds, targets) as numpy arrays.

    missing_fn, when given, marks a subset of labeling tokens "not measured" the same
    way real missing data is represented -- MISSING_LABEL_VALUE in `fractions` -- not by
    hiding them from attention (the model isn't trained with the latter; see
    MISSING_LABEL_VALUE in ai/model.py).
    """
    all_preds, all_targets = [], []
    model.eval()
    measured_flux_ids = model.fixed_idx_t.to(device)
    with torch.no_grad():
        for batch in loader:
            (cids, lids, tids, fracs), targets = batch
            cids  = cids.to(device)
            lids  = lids.to(device)
            tids  = tids.to(device)
            fracs = fracs.to(device)
            B     = cids.shape[0]

            if missing_fn is not None:
                missing = missing_fn(cids[0], B, device)          # (B, N_lab) bool
                fracs = fracs.clone()
                fracs[missing] = MISSING_LABEL_VALUE

            measured_fluxes = None
            if len(model.fixed_idx) > 0:
                measured_fluxes = targets.to(device).index_select(1, measured_flux_ids)

            fluxes, _ = model(cids, lids, tids, fracs,
                              measured_fluxes=measured_fluxes,
                              measured_flux_ids=measured_flux_ids)
            all_preds.append(fluxes.cpu().numpy())
            all_targets.append(targets[:, :model.flux_dim].numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def make_missing_fn(masked_met_set, token_labels):
    """Return a missing_fn flagging tokens belonging to masked_met_set as unmeasured."""
    token_mets = [lbl.split(' ')[0] for lbl in token_labels]
    missing_indices = [j for j, m in enumerate(token_mets) if m in masked_met_set]

    def fn(compound_ids_template, batch_size, device):
        N       = len(token_labels)
        missing = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        if missing_indices:
            missing[:, missing_indices] = True
        return missing

    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--n-samples',    type=int,   default=500)
    parser.add_argument('--rates',        type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.5, 0.7],
                        help='Fraction of metabolites to mask (default: 0.1 0.2 0.3 0.5 0.7)')
    parser.add_argument('--seeds',        type=int,   default=3,
                        help='Random seeds per rate (default: 3)')
    parser.add_argument('--r2-threshold', type=float, default=0.3,
                        help='R² threshold for "recoverable" (default: 0.3)')
    parser.add_argument('--out',          default=None,
                        help='Save report to text file')
    args = parser.parse_args()

    device = resolve_device()

    lines = []
    def out(msg=''):
        print(msg)
        lines.append(msg)

    # ── Load checkpoint, data and model ───────────────────────────────────────
    # The leading n_samples columns, not the saved val split — this measures
    # degradation against its own baseline, so the sample choice is arbitrary.
    probe = load_eval_context(args.checkpoint, build_dataset=False,
                              max_samples=args.n_samples * 2)
    val_indices = np.arange(min(args.n_samples, probe.labeling_df.shape[1]))
    ctx = load_eval_context(
        probe.ckpt, samples=val_indices,
        data=(probe.labeling_df, probe.flux_df, probe.reversibility_df),
    )

    data_dir    = ctx.data_dir
    flux_names  = ctx.flux_names
    vocabs      = ctx.vocabs
    val_dataset = ctx.dataset
    loader      = ctx.loader(batch_size=64)

    # ── Build token labels + metabolite list ──────────────────────────────────
    (cids0, lids0, tids0, _), _ = next(iter(ctx.loader(batch_size=1)))
    cids0 = cids0[0].numpy()
    lids0 = lids0[0].numpy()

    compound_vocab = vocabs['compound_vocab']
    label_vocab    = vocabs['label_vocab']

    def strip_compartment(mid):
        for s in ('_c','_m','_e','_x','_n','_r','_g','_l'):
            if mid.endswith(s): return mid[:-2]
        return mid

    token_labels = []
    for c, l in zip(cids0, lids0):
        met = strip_compartment(compound_vocab[c]) if c < len(compound_vocab) else str(c)
        iso = label_vocab[l] if l < len(label_vocab) else str(l)
        token_labels.append(f"{met} {iso}")

    unique_mets = []
    seen = set()
    for lbl in token_labels:
        m = lbl.split(' ')[0]
        if m not in seen:
            unique_mets.append(m)
            seen.add(m)
    n_mets = len(unique_mets)

    # ── Reconstruct model ─────────────────────────────────────────────────────
    model = build_flux_decoder(ctx, device=device)

    out(f"Model : {args.checkpoint}")
    out(f"Data  : {data_dir}  ({len(val_dataset)} samples)")
    out(f"Tokens: {len(token_labels)}  |  Unique metabolites: {n_mets}")
    out(f"R² threshold: {args.r2_threshold}")
    out()

    # ── Step 1: Baseline (all metabolites) ────────────────────────────────────
    preds_base, targets_base = run_inference(model, loader, device)
    r2_base = np.nan_to_num(per_reaction_r2(preds_base, targets_base))

    recoverable_idx  = np.where(r2_base >= args.r2_threshold)[0]
    recoverable_names = [flux_names[i] if i < len(flux_names) else str(i)
                         for i in recoverable_idx]
    n_rec = len(recoverable_idx)

    out('═' * 65)
    out(f'BASELINE — all {n_mets} metabolites present')
    out('═' * 65)
    out(f'Recoverable fluxes (R² ≥ {args.r2_threshold}): {n_rec} / {len(flux_names)}')
    out()
    out(f'  {"Reaction":<30}  {"R²":>6}')
    out(f'  ' + '─' * 40)
    for i in recoverable_idx[np.argsort(-r2_base[recoverable_idx])]:
        name = flux_names[i] if i < len(flux_names) else str(i)
        out(f'  {name:<30}  {r2_base[i]:>6.3f}')
    out()

    if n_rec == 0:
        out('No recoverable fluxes found. Exiting.')
        return

    # ── Step 2: Masking experiments ───────────────────────────────────────────
    # Summary table: rate → list of (seed, n_recovered)
    summary = {}

    for rate in args.rates:
        n_mask = max(1, round(rate * n_mets))
        summary[rate] = []

        out('═' * 65)
        out(f'MASKING RATE: {rate:.0%}  ({n_mask}/{n_mets} metabolites masked per run)')
        out('═' * 65)

        for seed in range(args.seeds):
            rng          = np.random.default_rng(seed)
            masked_mets  = set(rng.choice(unique_mets, size=n_mask, replace=False).tolist())
            missing_fn   = make_missing_fn(masked_mets, token_labels)

            preds_m, targets_m = run_inference(model, loader, device,
                                               missing_fn=missing_fn)
            r2_m = np.nan_to_num(per_reaction_r2(preds_m, targets_m))

            # Among the baseline-recoverable set, which are still recovered?
            r2_rec_masked   = r2_m[recoverable_idx]
            still_recovered = (r2_rec_masked >= args.r2_threshold)
            n_still         = still_recovered.sum()
            summary[rate].append(n_still)

            out(f'\n  Seed {seed+1}  —  masked: {", ".join(sorted(masked_mets))}')
            out(f'  Recovered: {n_still}/{n_rec}  ({100*n_still/n_rec:.1f}%)')

            lost_mask = ~still_recovered
            if lost_mask.any():
                out(f'  Not recovered (R² dropped below {args.r2_threshold}):')
                for j in np.where(lost_mask)[0]:
                    flux_i    = recoverable_idx[j]
                    name      = flux_names[flux_i] if flux_i < len(flux_names) else str(flux_i)
                    r2_before = r2_base[flux_i]
                    r2_after  = r2_m[flux_i]
                    out(f'    {name:<30}  {r2_before:.3f} → {r2_after:.3f}')
            else:
                out('  All recoverable fluxes still recovered.')

        out()

    # ── Summary table ─────────────────────────────────────────────────────────
    out('═' * 65)
    out('SUMMARY')
    out('═' * 65)
    out(f'  {"Rate":>6}  {"Masked":>7}  {"Recovered (mean±std)":>25}  {"Min":>6}')
    out('  ' + '─' * 55)
    for rate in args.rates:
        vals   = np.array(summary[rate])
        n_mask = max(1, round(rate * n_mets))
        mean_r = vals.mean() / n_rec * 100
        std_r  = vals.std()  / n_rec * 100
        min_r  = vals.min()  / n_rec * 100
        out(f'  {rate:>6.0%}  {n_mask:>7d}  '
            f'{mean_r:>10.1f}% ± {std_r:>5.1f}%         {min_r:>5.1f}%')
    out()

    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(lines))
        print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
