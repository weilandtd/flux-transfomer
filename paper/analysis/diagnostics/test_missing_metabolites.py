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


def r2_per_flux(preds, targets):
    """R² score per flux dimension. Returns (n_fluxes,) array."""
    ss_res = ((preds - targets) ** 2).sum(axis=0)
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    return np.where(ss_tot > 1e-8, 1.0 - ss_res / ss_tot, 0.0)


def run_inference(model, loader, device, nsd, padding_mask_fn=None):
    """Run model over all batches, return (preds, targets) as numpy arrays."""
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (cids, lids, tids, fracs), targets = batch
            cids  = cids.to(device)
            lids  = lids.to(device)
            tids  = tids.to(device)
            fracs = fracs.to(device)
            B     = cids.shape[0]

            pmask = None
            if padding_mask_fn is not None:
                pmask = padding_mask_fn(cids[0], B, device)

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                fixed_idx    = nsd.get('fixed_idx', [])
                fixed_fluxes = targets[:, fixed_idx].to(device)

            fluxes, _ = model(cids, lids, tids, fracs,
                              padding_mask=pmask, fixed_fluxes=fixed_fluxes)
            all_preds.append(fluxes.cpu().numpy())
            all_targets.append(targets[:, :model.flux_dim].numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def make_mask_fn(masked_met_set, token_labels):
    """Return a padding_mask_fn that masks tokens belonging to masked_met_set."""
    token_mets = [lbl.split(' ')[0] for lbl in token_labels]
    mask_indices = [j for j, m in enumerate(token_mets) if m in masked_met_set]

    def fn(compound_ids_template, batch_size, device):
        N    = len(token_labels)
        mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        if mask_indices:
            mask[:, mask_indices] = True
        return mask

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lines = []
    def out(msg=''):
        print(msg)
        lines.append(msg)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict   = ckpt['model_config']
    nsd        = ckpt['null_space_data']
    data_dir   = ckpt['data_dir']
    model_json = ckpt.get('model_json')
    flux_names = ckpt.get('flux_names', [])

    # ── Load data + vocabs ────────────────────────────────────────────────────
    labeling_df, flux_df, reversibility_df = load_data(
        data_dir, max_samples=args.n_samples * 2
    )
    _, base_tracer_names = create_tracer_mapping(labeling_df)
    _, cobra_model       = load_stoichiometric_matrix(model_json)

    vocabs = build_radial_vocabs(
        cobra_model, labeling_df, base_tracer_names,
        h_compound = cfg_dict.get('d_comp', 32) // 2,
        h_label    = cfg_dict.get('d_label', 10) // 2,
        h_tracer   = cfg_dict.get('d_tracer',  2) // 2,
    )

    val_indices = np.arange(min(args.n_samples, labeling_df.shape[1]))
    (val_dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df, flux_df, reversibility_df,
        vocabs           = vocabs,
        train_indices    = val_indices,
        normalize        = True,
        measured_fluxes  = ckpt.get('measured_fluxes'),
        norm_stats_input = ckpt.get('norm_stats'),
    )
    loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ── Build token labels + metabolite list ──────────────────────────────────
    first_batch = next(iter(DataLoader(val_dataset, batch_size=1)))
    (cids0, lids0, tids0, _), _ = first_batch
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
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__
                               if k in cfg_dict})
    token_ids = ckpt['model_state_dict']['token_experiment_ids']
    model = FluxDecoder(cfg, stoichiometric_matrix=None,
                        token_experiment_ids=token_ids, null_space_data=nsd,
                        fixed_flux_indices=nsd.get('fixed_idx', []))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    out(f"Model : {args.checkpoint}")
    out(f"Data  : {data_dir}  ({len(val_dataset)} samples)")
    out(f"Tokens: {len(token_labels)}  |  Unique metabolites: {n_mets}")
    out(f"R² threshold: {args.r2_threshold}")
    out()

    # ── Step 1: Baseline (all metabolites) ────────────────────────────────────
    preds_base, targets_base = run_inference(model, loader, device, nsd)
    r2_base = r2_per_flux(preds_base, targets_base)

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
            mask_fn      = make_mask_fn(masked_mets, token_labels)

            preds_m, targets_m = run_inference(model, loader, device, nsd,
                                               padding_mask_fn=mask_fn)
            r2_m = r2_per_flux(preds_m, targets_m)

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
