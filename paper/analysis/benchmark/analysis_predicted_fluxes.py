"""
Plot SigmaFlux predicted fluxes vs the Shen et al. MFA reference, as a 1×3 panel
(one nutrient condition per panel: C, N, P), colored by growth rate.

Everything is derived from a single training-run directory:
  <run_dir>/predictions/predictions_<Prefix>_<C|N|P>_*.csv     inference output
  <run_dir>/output/identifiability.csv                         per-reaction benchmark R²
  data/<Organism>/Shen_et_al_flux_data_*.xlsx                  reference MFA fluxes
                                                               (+ BIOMASS row → growth rate D)

Only reactions **MFA** resolves on held-out simulations (mfa_identifiable, R² > 0.8) are
plotted. The x axis is the reference MFA flux, so where MFA cannot determine a flux the
reference value is itself unreliable and agreement with it means nothing — restricting to
MFA-identifiable reactions is what makes the comparison valid. This set is a strict subset
of the SigmaFlux-identifiable one (219 of 270 in I. orientalis, 193 of 230 in
S. cerevisiae), so it is also the harder, unbiased choice: it is not selected on SigmaFlux's
own performance.

Predictions are RAW (no calibration). Fluxes are compared in absolute value on log-log axes.

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
from scipy.stats import norm

mpl.use('Agg')
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

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


def _r2_log(pred: np.ndarray, ref: np.ndarray) -> float:
    """R² between log10 fluxes — the space the panels are actually drawn in.

    These fluxes span ~8 orders of magnitude, so a linear R² is set almost entirely by
    the largest reactions and a handful of over-predictions drive it far negative
    (S. cerevisiae N and P reached −4.6 and −9.3). Scoring log10 values matches the
    log-log axes and weights a two-fold error the same wherever it occurs. Callers
    filter to |flux| > FLUX_MIN_ABS, so both arrays are strictly positive; any
    non-positive entry is dropped rather than producing a NaN.
    """
    m = (pred > 0) & (ref > 0)
    if m.sum() < 2:
        return float('nan')
    lp, lr = np.log10(pred[m]), np.log10(ref[m])
    ss_res = ((lp - lr) ** 2).sum()
    ss_tot = ((lr - lr.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')


def worst_per_reaction(df: pd.DataFrame, top: int = None) -> pd.DataFrame:
    """One row per reaction: the point where it deviates most from the reference.

    Called per panel, so the maximum is over that panel's growth conditions: each reaction
    contributes exactly one candidate, at the dilution rate where it fails worst. That avoids
    both a hand-picked dilution rate and the five-rings-per-reaction clutter of marking every
    growth condition.
    """
    m = df.assign(log_fold=np.log10(df['pred'] / df['ref']))
    idx = m.log_fold.abs().groupby(m.reaction).idxmax()
    worst = m.loc[idx]
    worst = worst.reindex(worst.log_fold.abs().sort_values(ascending=False).index)
    return worst.head(top) if top else worst


def _label_deviations(fig, ax, sub: pd.DataFrame, mark: pd.DataFrame, max_labels: int):
    """Number the largest deviations and list the names in one block.

    Text placed next to each point cannot be made reliable here: the points pile into the
    same corner, and `savefig(bbox_inches='tight')` re-lays out the figure *after* any
    display-space collision check, so validated positions are not the ones written to file.
    Numbered rings plus a single legend block sidesteps both problems — one digit per point
    collides with almost nothing, and the block sits in the empty half of a log-log panel.

    Candidates within MERGE_PX of an already-numbered point are skipped rather than given
    their own number, so co-located coupled reactions do not consume the whole list.
    """
    MERGE_PX = 13.0
    picked, xys = [], []
    for r in mark.itertuples():
        if len(picked) >= max_labels:
            break
        xy = ax.transData.transform((r.ref, r.pred))
        if any(np.hypot(xy[0] - q[0], xy[1] - q[1]) < MERGE_PX for q in xys):
            continue
        xys.append(xy)
        picked.append(r)

    for i, r in enumerate(picked, start=1):
        ax.scatter(r.ref, r.pred, s=46, facecolor='none', edgecolor='black',
                   linewidth=0.9, zorder=4)
        ax.annotate(str(i), xy=(r.ref, r.pred), xytext=(0, 0),
                    textcoords='offset points', fontsize=5.5, fontweight='bold',
                    ha='center', va='center', zorder=5)

    if picked:
        # Lower-right: a log-log panel of an over/under-prediction cloud leaves that corner
        # empty, and the R2 box is anchored below it.
        ax.text(0.97, 0.13, "\n".join(f"{i} {r.reaction}" for i, r in enumerate(picked, 1)),
                transform=ax.transAxes, fontsize=6, va='bottom', ha='right', zorder=5,
                linespacing=1.35)
    return len(picked)


def build_plot_df(run_dir: pathlib.Path, org: dict, panel_rxns: set) -> pd.DataFrame:
    """Tidy DataFrame of |pred| vs |ref| per (condition, growth condition, reaction),
    restricted to SigmaFlux-identifiable reactions and finite positive values."""
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
            shared = list(pred.index.intersection(ref.index))
            if panel_rxns is not None:                      # None → no filter (all shared reactions)
                shared = [r for r in shared if r in panel_rxns]
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


def plot_panels(df: pd.DataFrame, out_path: pathlib.Path, max_labels: int = 5):
    lim_lo = FLUX_MIN_ABS
    lim_hi = df[['ref', 'pred']].values.max() * 2.0
    diag = np.array([lim_lo, lim_hi])

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.7))
    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df['condition'] == cond]
        if not sub.empty:
            cmap = sns.color_palette(f"dark:{CONDITION_COLORS[cond]}_r", as_cmap=True)
            sns.scatterplot(data=sub, x='ref', y='pred', hue='D', ax=ax,
                            palette=cmap, s=18, edgecolor='none', legend='full')
            # One horizontal row above the axes. Inside the panel it sat on top of the
            # condition letter and the top-left data points.
            ax.legend(title='D [h⁻¹]', frameon=False, fontsize=6.5, title_fontsize=7,
                      loc='lower left', bbox_to_anchor=(0.0, 1.01),
                      ncol=int(sub['D'].nunique()), handletextpad=0.2,
                      columnspacing=0.6, borderpad=0.0)
            _label_deviations(fig, ax, sub, worst_per_reaction(sub), max_labels)
            r2 = _r2_log(sub['pred'].values, sub['ref'].values)
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


def plot_qq(df: pd.DataFrame, out_path: pathlib.Path, n_points: int = 400):
    """Normal Q-Q plot of the log10 pred/ref residuals, one curve per condition.

    Unlike R², this needs no ground truth — only the two flux vectors being compared —
    so the same diagnostic that flags outlying fits in the simulation benchmark runs
    unchanged on the real 13C data. A curve on the diagonal means the disagreement with
    the reference MFA is a well-behaved spread; departures at the ends are the
    individual reactions driving any headline R², and are readable off the plot.
    """
    probs = (np.arange(n_points) + 0.5) / n_points
    theo  = norm.ppf(probs)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if len(sub) < 10:
            continue
        r = np.log10(sub['pred'].values / sub['ref'].values)
        # Standardise by the bulk (IQR) so the diagonal is the reference for every
        # condition regardless of how wide its own spread is.
        q1, q3 = np.percentile(r, [25, 75])
        scale = (q3 - q1) / 1.349
        if scale <= 0:
            continue
        ax.plot(theo, np.quantile((r - np.median(r)) / scale, probs),
                color=CONDITION_COLORS[cond], lw=1.8, label=cond)

    lim = float(np.abs(theo).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=COLORS['reference'], lw=0.8, ls='--', zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('Normal quantile')
    ax.set_ylabel('Standardized\nlog₁₀(SigmaFlux / MFA)')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    sns.despine(ax=ax)

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Q-Q plot saved → {out_path}")


def _one_figure(run_dir, org, panel_rxns, out_name, args):
    """Build the dataframe for one reaction set and write its panel + Q-Q figures."""
    df = build_plot_df(run_dir, org, panel_rxns)
    print("Per-condition points (reaction × growth condition):")
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if sub.empty:
            continue
        print(f"  {cond}: {len(sub):4d} points, {sub['reaction'].nunique()} reactions, "
              f"log-R²={_r2_log(sub['pred'].values, sub['ref'].values):.3f}")
        print("       top deviations (each reaction at its worst D): " + ", ".join(
            f"{r.reaction} {10 ** r.log_fold:.0f}x @D={r.D:.2f}" if r.log_fold > 0
            else f"{r.reaction} 1/{10 ** -r.log_fold:.0f}x @D={r.D:.2f}"
            for r in worst_per_reaction(sub, args.max_labels).itertuples()))

    out_path = args.out or (run_dir / 'output' / out_name)
    plot_panels(df, out_path, args.max_labels)
    plot_qq(df, out_path.with_name(out_path.stem + '_qq.svg'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=pathlib.Path,
                        help='Training run directory (contains predictions/ and output/)')
    parser.add_argument('--out', type=pathlib.Path, default=None,
                        help='Output SVG (default: <run_dir>/output/predicted_fluxes_panels[_all].svg)')
    parser.add_argument('--select', default='both', choices=['both', 'mfa', 'sf', 'sf_only'],
                        help="Which identifiable set to plot. 'mfa' is the valid comparison "
                             "(the reference is an MFA fit). 'sf_only' is the complement — "
                             "reactions SigmaFlux resolves and MFA does not. 'both' (default) "
                             "writes each as its own figure.")
    parser.add_argument('--max-labels', type=int, default=5,
                        help='Worst outliers to name per panel (default 5)')
    parser.add_argument('--all-reactions', action='store_true',
                        help='Plot all shared reactions instead of filtering to SigmaFlux-identifiable '
                             '(skips identifiability.csv). Used to view the SigmaFlux experimental '
                             'predictions standalone.')
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    org = _resolve_organism(run_dir)

    id_path = run_dir / 'output' / 'identifiability.csv'
    if args.all_reactions or not id_path.exists():
        why = "--all-reactions" if args.all_reactions else f"{id_path.name} not found"
        print(f"Organism: {org['prefix']}  |  plotting ALL shared reactions "
              f"({why}; no identifiability filter)")
        _one_figure(run_dir, org, None, 'predicted_fluxes_panels_all.svg', args)
        return

    id_df = pd.read_csv(id_path)
    for col in ('mfa_identifiable', 'sf_identifiable'):
        if col not in id_df.columns:
            raise SystemExit(f"{id_path} lacks '{col}' — rerun plot_test_performance.py")

    mfa = id_df.mfa_identifiable
    sets = {
        # The valid comparison: the x axis is a reference MFA fit, so restrict to reactions
        # MFA can actually determine.
        'mfa': ("MFA-identifiable (reference is trustworthy)",
                set(id_df.loc[mfa, 'reaction']), 'predicted_fluxes_panels.svg'),
        # The complement, as its own figure: what SigmaFlux predicts where MFA cannot. Here
        # disagreement with the reference is expected and is NOT evidence SigmaFlux is wrong.
        'sf_only': ("SigmaFlux-only (MFA cannot determine these — reference unreliable)",
                    set(id_df.loc[id_df.sf_identifiable & ~mfa, 'reaction']),
                    'predicted_fluxes_sf_only.svg'),
        'sf': ("SigmaFlux-identifiable", set(id_df.loc[id_df.sf_identifiable, 'reaction']),
               'predicted_fluxes_panels_sf.svg'),
    }
    wanted = ['mfa', 'sf_only'] if args.select == 'both' else [args.select]
    for key in wanted:
        desc, rxns, name = sets[key]
        print(f"\n########## {org['prefix']}  |  {desc}: {len(rxns)} reactions")
        if not rxns:
            print("  (empty set — skipping)")
            continue
        _one_figure(run_dir, org, rxns, name, args)


if __name__ == '__main__':
    main()
