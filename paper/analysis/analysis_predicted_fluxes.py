"""
Plot SigmaFlux predicted fluxes vs the Shen et al. MFA reference, as a 1×3 panel
(one nutrient condition per panel: C, N, P), colored by growth rate.

Everything is derived from a single training-run directory:
  <run_dir>/predictions/predictions_<Prefix>_<C|N|P>_*.csv     inference output
  <run_dir>/output/identifiability.csv                         Both/MFA/SF/None per reaction
  data/<Organism>/Shen_et_al_flux_data_*.xlsx                  reference MFA fluxes
                                                               (+ BIOMASS row → growth rate D)

Only reactions identifiable by BOTH MFA and SigmaFlux are plotted; predictions are RAW
(no calibration). Fluxes are compared in absolute value on log-log axes.

Usage:
    python paper/analysis/analysis_predicted_fluxes.py paper/.results/i_orientalis_20260614
"""

import argparse
import glob
import os
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Per-condition base colors → a dark:<color>_r colormap shades points by growth rate.
CONDITION_COLORS = {
    'C': '#f3c47f',   # golden (MFA-like)
    'N': '#7fb3d5',   # light blue
    'P': '#2c9c8f',   # teal / green
}
COLORS = {
    'reference': '#888888',   # identity line
    'zero':      '#cccccc',   # annotation box edge
}
CONDITIONS = ['C', 'N', 'P']

# Reactions with |flux| below this are excluded from the log-log plot.
FLUX_MIN_ABS = 1e-6

ORGANISMS = {
    'i_orientalis': dict(prefix='I_orientalis', data='data/I_orientalis',
                         reference='Shen_et_al_flux_data_IO.xlsx'),
    's_cerevisiae': dict(prefix='S_cerevisiae', data='data/S_cervesiae',
                         reference='Shen_et_al_flux_data_SC.xlsx'),
}


def _resolve_organism(run_dir: pathlib.Path) -> dict:
    name = run_dir.name
    for key, info in ORGANISMS.items():
        if name.startswith(key):
            return info
    raise ValueError(f"Cannot detect organism from '{name}'. Known: {list(ORGANISMS)}")


def _ref_by_gc(ref_path: pathlib.Path, condition: str) -> pd.DataFrame:
    """Reference MFA fluxes for one condition, wide DataFrame indexed by reaction id,
    columns = growth-condition int. Matches '<cond><n>_mfa' and bare '<cond>_mfa' (→ 1)."""
    ref = pd.read_excel(ref_path).set_index('id')
    pat = re.compile(rf'^{re.escape(condition)}(\d*)_mfa$')
    cols, gcs = [], []
    for c in ref.columns:
        m = pat.match(str(c))
        if m:
            cols.append(c)
            gcs.append(int(m.group(1)) if m.group(1) else 1)
    out = ref[cols].copy()
    out.columns = gcs
    return out


def _pred_by_gc(pred_path: str) -> pd.DataFrame:
    """Predicted flux per (growth condition, reaction), mean over bio-reps. Wide:
    index = reaction id, columns = growth-condition int."""
    df = pd.read_csv(pred_path)
    df = df[df['Type'] == 'flux']
    return (df.groupby(['GrowthCondition', 'Flux Name'])['Value']
              .mean().unstack('GrowthCondition'))


def _r2(pred: np.ndarray, ref: np.ndarray) -> float:
    if len(pred) < 2:
        return float('nan')
    ss_res = ((pred - ref) ** 2).sum()
    ss_tot = ((ref - ref.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')


def build_plot_df(run_dir: pathlib.Path, org: dict, both_rxns: set) -> pd.DataFrame:
    """Tidy DataFrame of |pred| vs |ref| per (condition, growth condition, reaction),
    restricted to Both-identifiable reactions and finite positive values."""
    ref_path = _REPO_ROOT / org['data'] / org['reference']
    parts = []

    for cond in CONDITIONS:
        hits = sorted(glob.glob(str(run_dir / 'predictions' /
                                    f"predictions_{org['prefix']}_{cond}_*.csv")))
        if not hits:
            print(f"  condition {cond}: no predictions CSV — skipping")
            continue
        pred = _pred_by_gc(hits[-1])
        ref  = _ref_by_gc(ref_path, cond)
        # Growth rate D = the biomass reaction flux per growth condition (h⁻¹).
        rates = (ref.loc['BIOMASS'].abs().round(2).to_dict()
                 if 'BIOMASS' in ref.index else {})

        for gc in sorted(set(pred.columns) & set(ref.columns)):
            shared = pred.index.intersection(ref.index)
            shared = [r for r in shared if r in both_rxns]
            if not shared:
                continue
            parts.append(pd.DataFrame({
                'condition': cond,
                'gc':        gc,
                'D':         rates.get(gc, float(gc)),
                'reaction':  shared,
                'ref':       np.abs(ref.loc[shared, gc].values),
                'pred':      np.abs(pred.loc[shared, gc].values),
            }))

    df = pd.concat(parts, ignore_index=True)
    return df[(df['ref'] > FLUX_MIN_ABS) & (df['pred'] > FLUX_MIN_ABS)]


def plot_panels(df: pd.DataFrame, out_path: pathlib.Path):
    lim_lo = FLUX_MIN_ABS
    lim_hi = df[['ref', 'pred']].values.max() * 2.0
    diag = np.array([lim_lo, lim_hi])

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df['condition'] == cond]
        if not sub.empty:
            cmap = sns.color_palette(f"dark:{CONDITION_COLORS[cond]}_r", as_cmap=True)
            sns.scatterplot(data=sub, x='ref', y='pred', hue='D', ax=ax,
                            palette=cmap, s=18, edgecolor='none', legend='full')
            ax.legend(title='D [h⁻¹]', frameon=False, fontsize=7,
                      title_fontsize=7, loc='upper left')
            r2 = _r2(sub['pred'].values, sub['ref'].values)
            ax.text(0.97, 0.04, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
                    fontsize=9, va='bottom', ha='right',
                    bbox=dict(facecolor='white', alpha=0.85,
                              edgecolor=COLORS['zero'], boxstyle='round,pad=0.25'))

        ax.plot(diag, diag, color=COLORS['reference'], lw=1.0, ls='--', zorder=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect('equal')
        ax.text(0.04, 0.95, cond, transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='top', ha='left',
                color=CONDITION_COLORS[cond])
        ax.set_xlabel('Reference MFA\n[mmol gDW⁻¹ h⁻¹]')
        ax.set_ylabel('')
        sns.despine(ax=ax)

    axes[0].set_ylabel('SigmaFlux\n[mmol gDW⁻¹ h⁻¹]')

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"\nPlot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=pathlib.Path,
                        help='Training run directory (contains predictions/ and output/)')
    parser.add_argument('--out', type=pathlib.Path, default=None,
                        help='Output SVG (default: <run_dir>/output/predicted_fluxes_panels.svg)')
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    org = _resolve_organism(run_dir)

    id_path = run_dir / 'output' / 'identifiability.csv'
    id_df = pd.read_csv(id_path)
    both_rxns = set(id_df.loc[id_df['identifiable_by'] == 'Both', 'reaction'])
    print(f"Organism: {org['prefix']}  |  Both-identifiable reactions: {len(both_rxns)}")

    df = build_plot_df(run_dir, org, both_rxns)
    print("\nPer-condition points (reaction × growth condition):")
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if not sub.empty:
            print(f"  {cond}: {len(sub):4d} points, {sub['reaction'].nunique()} reactions, "
                  f"R²={_r2(sub['pred'].values, sub['ref'].values):.3f}")

    out_path = args.out or (run_dir / 'output' / 'predicted_fluxes_panels.svg')
    plot_panels(df, out_path)


if __name__ == '__main__':
    main()
