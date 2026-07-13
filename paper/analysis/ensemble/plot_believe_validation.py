"""
Intuitive experimental validation of the *believe* rule.

The believe rule labels each reaction HIGH/LOW from ensemble spread alone — no ground
truth. Here we check it against the independent experimental MFA reference (Shen et al.):
does the rule pick exactly the reactions that agree with MFA?

We reuse the established benchmark comparison (absolute fluxes, pooled over growth
conditions, tiny fluxes dropped) from analysis_predicted_fluxes.py, but colour every point
by the believe decision instead of the growth rate.

Figures (SVG, plotting-guideline styled):
  believe_vs_mfa_panels.svg   1x3 (C,N,P): |SigmaFlux| vs |MFA|, HIGH vs LOW, R2 for HIGH
  believe_vs_mfa_pooled.svg   single pooled panel, HIGH and LOW R2 annotated
  believe_summary.svg         stacked HIGH/LOW counts per condition (B,C,N,P)

Usage:
  python paper/analysis/ensemble/plot_believe_validation.py \
    --run paper/.results/i_orientalis_20260620 \
    --believe-dir paper/.results/i_orientalis_ko_20260628/believe \
    --out-dir paper/.results/i_orientalis_ko_20260628/believe
"""
import argparse, glob, os, sys, pathlib, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark'))
import analysis_predicted_fluxes as A          # trusted _ref_by_gc / _pred_by_gc / _r2 / helpers

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'HIGH': '#4C72B0', 'LOW': '#DD8452', 'diag': '#888888'}
CONDITIONS = ['C', 'N', 'P']                    # nutrient conditions with an MFA reference


def build_points(run: pathlib.Path, believe_dir: str, org: dict) -> pd.DataFrame:
    """One row per (condition, reaction, growth-condition): |MFA|, |SigmaFlux|, believe label."""
    ref_path = A._REPO_ROOT / org['data'] / org['reference']
    parts = []
    for cond in CONDITIONS:
        hits = sorted(glob.glob(str(run / 'predictions' / f"predictions_{org['prefix']}_{cond}_*.csv")))
        bel_f = os.path.join(believe_dir, f'believe_{cond}.csv')
        if not hits or not os.path.exists(bel_f):
            print(f"  {cond}: missing predictions or believe csv — skipping"); continue
        pred, ref = A._pred_by_gc(hits[-1]), A._ref_by_gc(ref_path, cond)
        label = pd.read_csv(bel_f).set_index('reaction')['believe'].to_dict()
        for gc in sorted(set(pred.columns) & set(ref.columns)):
            for r in pred.index.intersection(ref.index):
                parts.append({'condition': cond, 'reaction': r, 'gc': gc,
                              'ref': abs(ref.loc[r, gc]), 'pred': abs(pred.loc[r, gc]),
                              'believe': label.get(r, 'LOW')})
    df = pd.DataFrame(parts)
    return df[(df['ref'] > A.FLUX_MIN_ABS) & (df['pred'] > A.FLUX_MIN_ABS)]


def _scatter(ax, sub, lo, hi, annotate_low=False):
    diag = np.array([lo, hi])
    for lab in ('LOW', 'HIGH'):                 # HIGH drawn last (on top)
        s = sub[sub.believe == lab]
        ax.scatter(s['ref'], s['pred'], s=12, alpha=0.55, edgecolor='none',
                   color=COLORS[lab], label=lab, zorder=2 if lab == 'LOW' else 3)
    ax.plot(diag, diag, ls='--', lw=1.0, color=COLORS['diag'], zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    hi_r2 = A._r2(sub[sub.believe == 'HIGH']['pred'].values, sub[sub.believe == 'HIGH']['ref'].values)
    txt = f"HIGH $R^2$={hi_r2:.2f}"
    if annotate_low:
        lo_r2 = A._r2(sub[sub.believe == 'LOW']['pred'].values, sub[sub.believe == 'LOW']['ref'].values)
        txt += f"\nLOW $R^2$={lo_r2:.0f}"
    ax.text(0.96, 0.05, txt, transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc', boxstyle='round,pad=0.25'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=pathlib.Path, required=True)
    ap.add_argument('--believe-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    org = A._resolve_organism(args.run.resolve())
    df = build_points(args.run.resolve(), args.believe_dir, org)
    lo, hi = A.FLUX_MIN_ABS, df[['ref', 'pred']].values.max() * 2.0

    # ── Figure A: 1x3 per condition ──
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df.condition == cond]
        _scatter(ax, sub, lo, hi)
        ax.text(0.05, 0.95, cond, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
        ax.set_xlabel('Reference MFA\n[mmol gDW⁻¹ h⁻¹]'); sns.despine(ax=ax)
    axes[0].set_ylabel('SigmaFlux\n[mmol gDW⁻¹ h⁻¹]')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0, 0.86))
    fig.savefig(os.path.join(args.out_dir, 'believe_vs_mfa_panels.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, 'believe_vs_mfa_panels.png'), dpi=200, bbox_inches='tight')

    # ── Figure B: single pooled panel ──
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _scatter(ax, df, lo, hi, annotate_low=True)
    ax.set_xlabel('Reference MFA [mmol gDW⁻¹ h⁻¹]'); ax.set_ylabel('SigmaFlux [mmol gDW⁻¹ h⁻¹]')
    ax.legend(frameon=False, fontsize=9, loc='upper left'); sns.despine(ax=ax)
    fig.savefig(os.path.join(args.out_dir, 'believe_vs_mfa_pooled.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, 'believe_vs_mfa_pooled.png'), dpi=200, bbox_inches='tight')

    # ── Figure C: HIGH/LOW counts per condition (B,C,N,P) ──
    rows = []
    for f in sorted(glob.glob(os.path.join(args.believe_dir, 'believe_*.csv'))):
        cond = os.path.basename(f).split('_')[1].split('.')[0]
        b = pd.read_csv(f); n = len(b)
        rows.append({'condition': cond, 'HIGH': int((b.believe == 'HIGH').sum()),
                     'LOW': int((b.believe == 'LOW').sum()), 'total': n})
    cnt = pd.DataFrame(rows).sort_values('condition')
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.bar(cnt.condition, cnt.HIGH, color=COLORS['HIGH'], label='HIGH', width=0.62)
    ax.bar(cnt.condition, cnt.LOW, bottom=cnt.HIGH, color=COLORS['LOW'], label='LOW', width=0.62)
    for _, r in cnt.reset_index(drop=True).iterrows():
        ax.text(r.condition, r.HIGH / 2, f"{r.HIGH}", ha='center', va='center', color='white', fontsize=10)
    ax.set_ylabel('Number of reactions'); ax.set_xlabel('Condition')
    ax.legend(frameon=False, fontsize=9, loc='upper right'); sns.despine(ax=ax)
    fig.savefig(os.path.join(args.out_dir, 'believe_summary.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, 'believe_summary.png'), dpi=200, bbox_inches='tight')

    # console summary
    print("believe vs independent MFA (established metric: |flux|, pooled over growth conditions):")
    for cond in CONDITIONS:
        s = df[df.condition == cond]; hi = s[s.believe == 'HIGH']; lw = s[s.believe == 'LOW']
        print(f"  {cond}: HIGH R2={A._r2(hi.pred.values, hi.ref.values):.3f} (n={len(hi)})  "
              f"LOW R2={A._r2(lw.pred.values, lw.ref.values):.1f} (n={len(lw)})")
    print(f"\nsaved -> {args.out_dir}/believe_vs_mfa_panels.svg, believe_vs_mfa_pooled.svg, believe_summary.svg")


if __name__ == '__main__':
    main()
