"""
KO vs non-KO prediction-shift significance.

For each KO reaction R, compare the model's PREDICTED flux for R in:
  - KO-of-R validation samples (R was knocked out)   vs
  - non-KO (WT) validation samples (R active, normal condition)
Mann-Whitney U test on the raw predictions -> per-reaction p-value.
Bars: mean |predicted|/WT_std[R] per group (dimensionless, comparable across reactions).
Two panels split by the reaction's normal (WT) per-reaction R²: >0.5 vs <=0.5.
Expectation: well-predicted (R²>0.5) reactions show a larger, more significant KO shift.

Usage: python paper/analysis/ko_significance.py --checkpoint <ko.pt> --run-dir <ko run>
"""
import argparse, os, numpy as np, pandas as pd, torch
from scipy.stats import mannwhitneyu
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'non-KO': '#4C72B0', 'KO': '#DD8452'}


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'


def r2(pred, true):
    ss = ((true - pred) ** 2).sum(); st = ((true - true.mean()) ** 2).sum()
    return 1 - ss / st if st > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--r2-cut', type=float, default=0.5)
    args = ap.parse_args()

    dev = resolve_device()
    ctx = load_eval_context(args.checkpoint)
    m = build_flux_decoder(ctx, device=dev)

    flux_names = ctx.flux_names; name2idx = {n: i for i, n in enumerate(flux_names)}
    val_idx = ctx.sample_indices
    labels = pd.read_csv(os.path.join(args.run_dir, 'ko_labels.csv'), index_col=0).iloc[val_idx].reset_index(drop=True)

    (c, l, t, fr), tg = ctx.full_batch(device=dev)
    fid = m.fixed_idx_t.to(dev)
    with torch.no_grad():
        vp = m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid), measured_flux_ids=fid, return_attn=False)[0].cpu().numpy()
    gt = tg[:, :len(flux_names)].cpu().numpy()

    ko_rxns = [r for r in labels['ko_reaction'].unique() if r != 'WT']
    wt = labels['ko_reaction'].values == 'WT'
    rows, long = [], []
    for r in ko_rxns:
        j = name2idx[r]; koR = labels['ko_reaction'].values == r
        pk, pn = vp[koR, j], vp[wt, j]                       # predicted flux: KO vs non-KO(WT)
        wt_std = gt[wt, j].std() or 1.0
        p = mannwhitneyu(np.abs(pk), np.abs(pn), alternative='two-sided').pvalue
        R2 = r2(vp[wt, j], gt[wt, j])                        # normal (WT) predictability of R
        rows.append({'reaction': r, 'n_KO': koR.sum(), 'n_nonKO': wt.sum(), 'R2_wt': R2,
                     'mean_nonKO_rel': np.abs(pn).mean()/wt_std, 'mean_KO_rel': np.abs(pk).mean()/wt_std,
                     'p': p, 'sig': stars(p)})
        for v in pn: long.append({'reaction': r, 'cond': 'non-KO', 'rel': abs(v)/wt_std, 'R2': R2})
        for v in pk: long.append({'reaction': r, 'cond': 'KO', 'rel': abs(v)/wt_std, 'R2': R2})
    res = pd.DataFrame(rows).sort_values('R2_wt', ascending=False)
    L = pd.DataFrame(long)
    res.to_csv(os.path.join(args.run_dir, 'ko_significance.csv'), index=False)
    print(res.to_string(index=False))
    hi = res[res.R2_wt > args.r2_cut]; lo = res[res.R2_wt <= args.r2_cut]
    print(f"\nsignificant (p<0.05): R2>{args.r2_cut}: {(hi.p<0.05).mean():.2f} ({len(hi)} rxns) | "
          f"R2<={args.r2_cut}: {(lo.p<0.05).mean():.2f} ({len(lo)} rxns)")

    # ── two-panel grouped bar plot ───────────────────────────────────────────
    for tag, sub in [('r2hi', res[res.R2_wt > args.r2_cut]), ('r2lo', res[res.R2_wt <= args.r2_cut])]:
        order = list(sub['reaction'])
        if not order: continue
        d = L[L.reaction.isin(order)]
        fig, ax = plt.subplots(figsize=(max(3.5, 0.5 * len(order) + 1.5), 3.5))
        sns.barplot(data=d, x='reaction', y='rel', hue='cond', order=order,
                    hue_order=['non-KO', 'KO'], palette=COLORS, errorbar='se', ax=ax)
        ymax = d.groupby(['reaction', 'cond'])['rel'].mean().max()
        for k, rxn in enumerate(order):
            s = sub[sub.reaction == rxn]['sig'].iloc[0]
            ax.text(k, ymax * 1.05, s, ha='center', va='bottom', fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel('|predicted flux| / WT std')
        ax.set_xticklabels(order, rotation=60, ha='right', fontsize=9)
        ax.legend(title='', frameon=False, fontsize=10)
        sns.despine(ax=ax)
        out = os.path.join(args.run_dir, f'ko_significance_{tag}.svg')
        fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
        print(f"saved -> {out}")


if __name__ == '__main__':
    main()
