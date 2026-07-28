"""
Ablation grid: validation R2 as a function of how much input is removed.

Removes a fraction of the LABELING metabolites (MIDs) along the rows, and removes
measured fluxes along the columns. Each cell reports the NUMBER of reactions whose
per-reaction validation R2 exceeds 0.9 (averaged over random-removal seeds), and
saves a heatmap. The identifiable set (full R2 > 0.2) is only used for the printed
sanity baseline.

MID removal   = zero the fraction tokens of the removed metabolites (treat as unmeasured).
Flux removal  = drop those bypass tokens (pass a subset to measured_flux_ids).

Flux axis is modular via --flux-mode:
  random  (default) — drop a random {0,10,20,50,70}% of the measured fluxes.
  biomass           — cumulatively drop the biomass-precursor fluxes (0..5:
                      PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c), keeping the
                      exchange fluxes. Tests whether the model leans on the
                      biomass fluxes specifically rather than fluxes in general.

R2 is measured on the exact held-out validation split saved in the checkpoint
(ck['val_indices']) — the same samples train_ai_model.py reports R2 on — NOT on
the first N samples of the (stratified) data file. So the 0%/0% cell reproduces
the training-reported validation median.

Usage:
    python paper/analysis/ablation_grid.py --checkpoint .../imisor_transformer.pt --seeds 5
    python paper/analysis/ablation_grid.py --checkpoint .../imisor_transformer.pt --flux-mode biomass
"""
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device
from sigmaflux.ai.train import per_reaction_r2

matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 12})

COLORS = {
    'bar': '#4C72B0',
}

strip_c = lambda m: m[:-2] if len(m) > 2 and m[-2] == '_' else m
LEVELS = [0, 25, 50, 75]                                # % removed (MID axis; flux axis in 'random' mode)

# Biomass-precursor measured fluxes, in the order they are removed cumulatively
# in --flux-mode biomass (level n removes the first n of these).
BIOMASS_FLUXES = ['PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c', 'CARB_c']
BIOMASS_LEVELS = [0, 1, 2, 3, 4, 5]                         # # biomass fluxes removed




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--max-val', type=int, default=None,
                    help='cap on held-out val samples used (random subset; default = all val_indices)')
    ap.add_argument('--seeds', type=int, default=10,
                    help='# random MID/flux-removal trials per cell (also the points behind the barplot error bars)')
    ap.add_argument('--threshold', type=float, default=0.9,
                    help='each cell counts reactions with per-reaction R2 > this value')
    ap.add_argument('--flux-mode', choices=['random', 'biomass'], default='random',
                    help="how the flux axis removes measured fluxes: 'random' = random fraction "
                         "({0,10,20,50,70}%%); 'biomass' = cumulatively drop biomass-precursor fluxes "
                         "(0..5), keeping the exchange fluxes")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    flux_names = ck.get('flux_names', [])

    # Evaluate on the EXACT held-out validation split (positional columns into the
    # full, original-order data load — matching how train_ai_model.py built val_dataset).
    val_idx = np.asarray(ck['val_indices'], dtype=int)
    if args.max_val is not None and args.max_val < len(val_idx):
        val_idx = np.sort(np.random.default_rng(0).choice(val_idx, size=args.max_val, replace=False))

    device = resolve_device()
    ctx = load_eval_context(ck, samples=val_idx)
    model = build_flux_decoder(ctx, device=device)
    print(f"device: {device}")

    (cids, lids, tids, fracs), targets = ctx.full_batch(device=device)
    gt = targets[:, :len(flux_names)].cpu().numpy()
    fixed_idx = np.array(model.fixed_idx)                    # reaction indices of the known fluxes
    cvocab = ctx.vocabs['compound_vocab']
    tok_met = np.array([strip_c(cvocab[int(c)]) if int(c) < len(cvocab) else str(int(c)) for c in cids[0].cpu().numpy()])
    mets = sorted(set(tok_met))
    print(f"held-out val samples={gt.shape[0]} | reactions={len(flux_names)} | metabolites={len(mets)} | known fluxes={len(fixed_idx)}")

    def predict(keep_mid_mask, keep_flux_ids):
        f = fracs.clone()
        f[:, torch.as_tensor(~keep_mid_mask, device=device)] = 0.0
        if len(keep_flux_ids) > 0:
            ids = torch.as_tensor(np.asarray(keep_flux_ids), dtype=torch.long, device=device)
            mf = targets.index_select(1, ids)
        else:
            mf, ids = None, None
        with torch.no_grad():
            out = model(cids, lids, tids, f, measured_fluxes=mf, measured_flux_ids=ids, return_attn=False)
        return per_reaction_r2(out[0].cpu().numpy(), gt)

    # identifiable reactions defined by the FULL model
    full_r2 = predict(np.ones(len(tok_met), bool), fixed_idx)
    ident = full_r2 > 0.2
    print(f"identifiable reactions (full R2>0.2): {int(ident.sum())}/{len(flux_names)}")
    # sanity: baseline counts on the held-out split
    print(f"  baseline #reactions R2>{args.threshold}: {int(np.sum(full_r2 > args.threshold))}/{len(flux_names)}\n")

    # ── flux-axis setup (modular: random fraction vs cumulative biomass removal) ──
    name_by_idx = {int(idx): flux_names[int(idx)] for idx in fixed_idx}
    if args.flux_mode == 'biomass':
        biomass_order = [int(idx) for idx in fixed_idx if name_by_idx[int(idx)] in BIOMASS_FLUXES]
        biomass_order.sort(key=lambda idx: BIOMASS_FLUXES.index(name_by_idx[idx]))
        flux_levels = BIOMASS_LEVELS
        flux_axis_label = "biomass fluxes removed"
        flux_ticklabels = [str(n) for n in BIOMASS_LEVELS]
        print(f"flux-mode=biomass | removal order: "
              f"{[name_by_idx[i] for i in biomass_order]} | always-kept: "
              f"{[name_by_idx[int(i)] for i in fixed_idx if int(i) not in biomass_order]}\n")
    else:
        flux_levels = LEVELS
        flux_axis_label = "known fluxes removed"
        flux_ticklabels = [f"{c}%" for c in LEVELS]

    def keep_flux_for(level, rng):
        """Reaction indices of the measured fluxes that remain at this flux-axis level."""
        if args.flux_mode == 'biomass':
            drop = set(biomass_order[:level])
            return np.array([int(i) for i in fixed_idx if int(i) not in drop], int)
        n_keep = len(fixed_idx) - int(round(level / 100 * len(fixed_idx)))
        return np.sort(rng.choice(fixed_idx, size=n_keep, replace=False)) if n_keep else np.array([], int)

    cube = np.zeros((len(LEVELS), len(flux_levels), args.seeds))   # per-seed counts (for error bars)
    for i, mid_rm in enumerate(LEVELS):
        for j, flx_lv in enumerate(flux_levels):
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * i + 7 * j + s)
                n_mid_drop = int(round(mid_rm / 100 * len(mets)))
                drop_mets = set(rng.choice(mets, size=n_mid_drop, replace=False)) if n_mid_drop else set()
                keep_mid = ~np.isin(tok_met, list(drop_mets))
                keep_flux = keep_flux_for(flx_lv, rng)
                r2 = predict(keep_mid, keep_flux)
                cube[i, j, s] = int(np.sum(r2 > args.threshold))  # # reactions with R2 > threshold
    grid = cube.mean(axis=2)                                       # mean over seeds (heatmap)

    # ── report ──────────────────────────────────────────────────────────────────
    print(f"#reactions with R2>{args.threshold}  (rows = % MIDs removed, cols = {flux_axis_label})")
    print("            flux:  " + "  ".join(f"{c:>6s}" for c in flux_ticklabels))
    for i, mid_rm in enumerate(LEVELS):
        print(f"  MID {mid_rm:3d}% removed:  " + "  ".join(f"{grid[i, j]:6.1f}" for j in range(len(flux_levels))))

    fig, ax = plt.subplots(figsize=(6.2, 5))
    vmax = grid.max() if grid.max() > 0 else 1
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(flux_levels))); ax.set_xticklabels(flux_ticklabels)
    ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels([f"{c}%" for c in LEVELS])
    ax.set_xlabel(flux_axis_label); ax.set_ylabel("MID metabolites removed")
    for i in range(len(LEVELS)):
        for j in range(len(flux_levels)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha='center', va='center',
                    color='white' if grid[i, j] < 0.5 * vmax else 'black', fontsize=8)
    ax.set_title(f"# reactions with R² > {args.threshold} vs input removal  (of {len(flux_names)} reactions)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"# reactions R² > {args.threshold}")
    fig.tight_layout()
    thr_tag = f"{args.threshold:g}".replace('.', 'p')
    out = args.out or args.checkpoint.replace('.pt', f'_ablation_grid_{args.flux_mode}_r2{thr_tag}.svg')
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nsaved -> {out} (+ .png)")

    # ── MID-usage barplot (all measured fluxes kept; vary how much MID is USED) ────
    # First flux column = 0% fluxes removed. MID removed {0,25,50,75}% -> MID used {100,75,50,25}%.
    # One row per (MID level, seed); error bars = 25-75th percentile (IQR) across seeds.
    # LEVELS = MID% removed (ascending) -> MID% used is already descending (100 -> 25)
    long = pd.DataFrame([
        {'mid_used': f"{100 - lv}%", 'n_reactions': cube[i, 0, s]}
        for i, lv in enumerate(LEVELS) for s in range(args.seeds)
    ])
    order = [f"{100 - lv}%" for lv in LEVELS]

    bfig, bax = plt.subplots(figsize=(3.5, 3.5))
    sns.barplot(data=long, x='mid_used', y='n_reactions', order=order,
                color=COLORS['bar'], width=0.5, estimator='mean',
                errorbar=('pi', 50), err_kws={'linewidth': 1.2}, capsize=0.25, ax=bax)
    # value labels (mean) placed just above each bar's upper IQR whisker
    _grp = long.groupby('mid_used')['n_reactions']
    _mean, _q75 = _grp.mean(), _grp.quantile(0.75)
    _pad = long['n_reactions'].max() * 0.02
    for k, lab in enumerate(order):
        bax.text(k, _q75[lab] + _pad, f"{_mean[lab]:.0f}", ha='center', va='bottom', fontsize=11)
    bax.set_xlabel("MID information used")
    bax.set_ylabel(f"Number of reactions\n(R² > {args.threshold})")
    bax.set_ylim(0, long['n_reactions'].max() * 1.18)
    sns.despine(ax=bax)
    bout = out.replace('.svg', '_midbar.svg')
    bfig.savefig(bout, bbox_inches='tight')
    bfig.savefig(bout.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    print(f"saved -> {bout} (+ .png)")


if __name__ == '__main__':
    main()
