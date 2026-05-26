"""
Analyse predicted fluxes against a reference (e.g. Shen et al. MFA data).

Reads the predictions CSV from run_inference.py and a reference xlsx, then:
  - Optionally applies null-space correction using known experimental fluxes
  - Filters reactions by model validation R² at one or more thresholds
  - Prints Pearson, R², RMSE, mean|Δ| for each threshold
  - Saves a correlation plot (pred vs ref)

Usage:
    python paper/analysis/analysis_predicted_fluxes.py \\
        --predictions paper/output/predictions_I_orientalis_C_*.csv \\
        --checkpoint  I_orientalis_20260522/imisor_transformer.pt \\
        --reference   data/I_orientalis/Shen_et_al_flux_data_IO.xlsx \\
        --config      paper/configs/i_orientalis.yaml \\
        --extflux     data/I_orientalis/I_orientalis_ExtFlux.xlsx \\
        --condition   C


"""
# write the command in 1 line for bash, so that it can be copy-pasted easily
# python paper/analysis/analysis_predicted_fluxes.py --predictions paper/output/predictions_I_orientalis_C_*.csv --checkpoint I_orientalis_20260522/imisor_transformer.pt --reference data/I_orientalis/Shen_et_al_flux_data_IO.xlsx --config paper/configs/i_orientalis.yaml --extflux data/I_orientalis/I_orientalis_ExtFlux.xlsx --condition C --no-correction

# write command now for SC 
# python paper/analysis/analysis_predicted_fluxes.py --predictions paper/output/predictions_S_cerevisiae_C_*.csv --checkpoint /scratch/gpfs/MONA/Toki/MFA/sigmaflux/S_cerevisiae_20260523/sc_transformer.pt --reference data/S_cervesiae/Shen_et_al_flux_data_SC.xlsx --config paper/configs/s_cerevisiae.yaml --extflux data/S_cervesiae/S_cerevisiae_ExtFlux.xlsx --condition C --no-correction


import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_r2(checkpoint_path):
    """Load per-reaction validation R² from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    vc = ckpt.get('val_correlations', {})
    return dict(zip(
        vc.get('flux_reaction_names', []),
        vc.get('flux_r2_per_reaction', []),
    ))


def load_reference(ref_path, condition, config_path):
    """Load reference MFA fluxes with sign-convention correction."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ref = pd.read_excel(ref_path)
    media_cfg = cfg.get('sampling', {}).get('media', {})
    uptake_rxns = {rxn for rxn, b in media_cfg.items() if b.get('ub', 1.0) <= 0}

    ref_idx = ref.set_index('id')
    ref_cols = [c for c in ref.columns if c.startswith(condition) and c.endswith('_mfa')]
    ref_vals = ref_idx[ref_cols].copy()
    flip_mask = ref_vals.index.isin(uptake_rxns)
    ref_vals.loc[flip_mask] *= -1
    return ref_vals.mean(axis=1)


def nullspace_correction(pred_df, checkpoint_path, extflux_path, config_path, condition,
                         reg_lambda=None):
    """
    Correct predicted fluxes by anchoring to known experimental fluxes in y-space.

    Uses Tikhonov-regularized correction to prevent explosion when measured fluxes
    have small null-space participation (||Z_row|| ≈ 0.01):

        Δy = A.T @ inv(A @ A.T + λI) @ residual    (regularized)
        v_corrected = v_pred + Z @ Δy

    λ controls the trade-off: λ=0 forces exact match (unstable), λ→∞ means no
    correction. Default: λ = mean(singular_values(A)²) — balances signal vs noise.

    Preserves Sv = 0 since Z is the null-space basis.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    Z = np.array(ckpt['null_space_data']['Z_free'], dtype=np.float64)  # (n_rxn, n_basis)
    flux_names = ckpt['flux_names']
    measured_fluxes = ckpt['measured_fluxes']
    name_to_idx = {n: i for i, n in enumerate(flux_names)}

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    negate = set(cfg.get('training', {}).get('negate_fluxes', []))

    # Load experimental measured flux values per condition
    df_exp = pd.read_excel(extflux_path)
    df_exp = df_exp[df_exp['Condition'].str.startswith(condition)].copy()
    df_exp['GC'] = df_exp['Condition'].str.extract(r'(\d+)').astype(int)

    # A = Z[measured_indices, :] — the null-space rows for known fluxes
    measured_idx = [name_to_idx[n] for n in measured_fluxes]
    A = Z[measured_idx, :]  # (6, 38)

    # Tikhonov regularization: λ = mean(σ²) by default
    sv = np.linalg.svd(A, compute_uv=False)
    if reg_lambda is None:
        reg_lambda = float(np.mean(sv ** 2))
    AAt_reg_inv = np.linalg.inv(A @ A.T + reg_lambda * np.eye(A.shape[0]))

    print(f"  A singular values: {np.array2string(sv, precision=4, separator=', ')}")
    print(f"  Regularization λ = {reg_lambda:.2e}  (cond(A@A.T) = {np.linalg.cond(A @ A.T):.1e})")

    corrected = pred_df.copy()
    flux_rows = corrected['Type'] == 'flux'
    n_corrected = 0
    max_dv = 0.0

    for gc in corrected['GrowthCondition'].unique():
        exp_row = df_exp[df_exp['GC'] == gc]
        if exp_row.empty:
            continue
        exp_row = exp_row.iloc[0]

        # Build experimental target vector for the 6 measured fluxes
        v_exp = np.array([
            -exp_row[n] if n in negate else exp_row[n]
            for n in measured_fluxes
        ], dtype=np.float64)

        for biorep in corrected[corrected['GrowthCondition'] == gc]['BioRep'].unique():
            mask = flux_rows & (corrected['GrowthCondition'] == gc) & (corrected['BioRep'] == biorep)
            sub = corrected.loc[mask].set_index('Flux Name')

            # Current predicted flux vector (ordered by flux_names)
            v_pred = np.array([sub.loc[n, 'Value'] if n in sub.index else 0.0
                               for n in flux_names], dtype=np.float64)

            # Residual: what the measured fluxes should be minus what they are
            residual = v_exp - v_pred[measured_idx]

            # Tikhonov-regularized correction in y-space
            dy = A.T @ AAt_reg_inv @ residual  # (38,)
            dv = Z @ dy                         # (391,)

            max_dv = max(max_dv, np.abs(dv).max())

            # Apply correction
            v_corrected = v_pred + dv
            for n in flux_names:
                if n in sub.index:
                    corrected.loc[mask & (corrected['Flux Name'] == n), 'Value'] = v_corrected[name_to_idx[n]]
            n_corrected += 1

    print(f"Null-space correction applied to {n_corrected} samples "
          f"({len(measured_fluxes)} anchors, {Z.shape[1]} y-dims, "
          f"max|Δv|={max_dv:.4f})")
    return corrected


def compute_metrics(pred, ref):
    """Return dict of Pearson, R², RMSE, mean|Δ| for aligned arrays."""
    n = len(pred)
    if n < 2:
        return {'n': n, 'pearson': float('nan'), 'r2': float('nan'),
                'rmse': float('nan'), 'mean_abs_diff': float('nan')}
    pr, _ = pearsonr(pred, ref)
    ss_res = ((pred - ref) ** 2).sum()
    ss_tot = ((ref - ref.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    rmse = np.sqrt(((pred - ref) ** 2).mean())
    mad = np.abs(pred - ref).mean()
    return {'n': n, 'pearson': pr, 'r2': r2, 'rmse': rmse, 'mean_abs_diff': mad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='CSV from run_inference.py')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint (.pt)')
    parser.add_argument('--reference', required=True, help='Reference flux xlsx (e.g. Shen et al.)')
    parser.add_argument('--config', required=True, help='YAML config used for training')
    parser.add_argument('--condition', default='C', help='Condition prefix in reference (default: C)')
    parser.add_argument('--thresholds', default='0.0,0.3,0.5,0.7',
                        help='Comma-separated R² thresholds (default: 0.0,0.3,0.5,0.7,0.9)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: auto from predictions)')
    parser.add_argument('--extflux', default=None,
                        help='ExtFlux xlsx with experimental measured fluxes (enables null-space correction)')
    parser.add_argument('--no-correction', action='store_true',
                        help='Skip null-space correction even if --extflux is provided')
    parser.add_argument('--reg-lambda', type=float, default=None,
                        help='Tikhonov regularization strength (default: auto = mean(σ²))')
    parser.add_argument('--max-flux', type=float, default=None,
                        help='Exclude reactions where |pred| or |ref| exceeds this value (default: no filter)')
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    thresholds = [float(t) for t in args.thresholds.split(',')]

    # ── Load data ────────────────────────────────────────────────────────────
    combined = pd.read_csv(args.predictions)
    # divide by 1.5
    # combined['Value'] /= 1.5
    model_r2 = load_model_r2(args.checkpoint)
    ref_mean = load_reference(args.reference, args.condition, args.config)

    # Load measured (known) flux names from checkpoint for later highlighting
    _ckpt_tmp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    measured_flux_names = set(_ckpt_tmp.get('measured_fluxes', []))
    del _ckpt_tmp

    # ── Null-space correction ────────────────────────────────────────────────
    if args.extflux and not args.no_correction:
        combined = nullspace_correction(
            combined, args.checkpoint, args.extflux, args.config, args.condition,
            reg_lambda=args.reg_lambda)

    # Average predictions across GCs and bio-replicates
    pred_mean = (
        combined[combined['Type'] == 'flux']
        .groupby('Flux Name')['Value']
        .mean()
    )

    # ── Known fluxes — load experimental ref from ExtFlux (not Shen et al) ─────
    n_k_all = np.array([n for n in measured_flux_names if n in pred_mean])
    p_k_raw = np.array([pred_mean[n] for n in n_k_all])
    r_k_all = np.full(len(n_k_all), np.nan)
    if args.extflux:
        df_exp_k = pd.read_excel(os.path.join(_REPO_ROOT, args.extflux))
        df_exp_k = df_exp_k[df_exp_k['Condition'].str.startswith(args.condition)].copy()
        with open(os.path.join(_REPO_ROOT, args.config)) as _f2:
            _cfg2 = yaml.safe_load(_f2)
        negate_k = set(_cfg2.get('training', {}).get('negate_fluxes', []))
        for i, nm in enumerate(n_k_all):
            if nm in df_exp_k.columns:
                val = df_exp_k[nm].mean()
                r_k_all[i] = -val if nm in negate_k else val
    valid_k = ~np.isnan(r_k_all)
    n_k_all = n_k_all[valid_k]
    p_k_raw = p_k_raw[valid_k]
    r_k_all = r_k_all[valid_k]

    # ── Scale correction: use only glucose + biomass components for slope ───────
    # Exclude other EX_ secretion products (EX_etoh_e, EX_ac_e, etc.) — they are
    # noisier and less fundamental than glucose uptake + growth components.
    slope_mask = np.array([
        n == 'EX_glc__D_e' or not n.startswith('EX_')
        for n in n_k_all
    ])
    n_k_slope = n_k_all[slope_mask]
    p_k_slope = p_k_raw[slope_mask]
    r_k_slope = r_k_all[slope_mask]

    # ── Scale correction: divide all predictions by the known-flux slope ──────
    # The known fluxes fit y ≈ slope·x with intercept ≈ 0 (through origin),
    # meaning every predicted flux is over-scaled by the same factor. One
    # division corrects the entire network with no free parameters.
    scale_factor = 1.0
    if len(r_k_slope) >= 2:
        sk_pre, ik_pre, *_ = linregress(r_k_slope, p_k_slope)
        if abs(sk_pre) > 0.1:
            scale_factor = sk_pre
            print(f"\nScale correction: dividing all predictions by {scale_factor:.4f} "
                  f"(slope from {len(r_k_slope)} fluxes: glucose + biomass components, "
                  f"intercept={ik_pre:.4f})")
            pred_mean = pred_mean / scale_factor

    p_k_all = p_k_raw / scale_factor

    print(f"\n── Known fluxes vs ExtFlux (after scale correction, n={len(n_k_all)}) ──")
    print(f"{'Reaction':<25}  {'Pred':>9}  {'Exp':>9}  {'|Δ|':>9}  {'used_for_slope':>14}")
    for rn, pv, rv in zip(n_k_all, p_k_all, r_k_all):
        used = '*' if rn in set(n_k_slope) else ''
        print(f"  {rn:<23}  {pv:>9.4f}  {rv:>9.4f}  {abs(pv-rv):>9.4f}  {used:>14}")

    # ── Build aligned arrays (after scale correction) ────────────────────────
    shared = pred_mean.index.intersection(ref_mean.index)
    p = pred_mean[shared].values
    r = ref_mean[shared].values
    rxn_r2 = np.array([model_r2.get(n, float('nan')) for n in shared])

    # Optional: drop reactions where |pred| or |ref| exceeds --max-flux
    if args.max_flux is not None:
        flux_ok = (np.abs(p) <= args.max_flux) & (np.abs(r) <= args.max_flux)
        n_dropped = (~flux_ok).sum()
        if n_dropped:
            print(f"max-flux filter ({args.max_flux}): dropped {n_dropped} reactions "
                  f"({', '.join(shared[~flux_ok].tolist()[:5])}{'...' if n_dropped > 5 else ''})")
        shared, p, r, rxn_r2 = shared[flux_ok], p[flux_ok], r[flux_ok], rxn_r2[flux_ok]

    # ── Metrics per threshold ────────────────────────────────────────────────
    print(f"\nPredictions: {args.predictions}")
    print(f"Reference:   {args.reference}  condition={args.condition}")
    print(f"Scale factor applied: {scale_factor:.4f}")
    print(f"Shared reactions: {len(shared)}")
    print()
    print(f"{'Threshold':>10}  {'n':>5}  {'Pearson':>8}  {'R²':>8}  {'RMSE':>8}  {'mean|Δ|':>8}")
    print("─" * 58)

    for thr in thresholds:
        mask = rxn_r2 >= thr
        if mask.sum() < 2:
            print(f"{'≥'+str(thr):>10}  {int(mask.sum()):>5}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}")
            continue
        m = compute_metrics(p[mask], r[mask])
        print(f"{'≥'+str(thr):>10}  {m['n']:>5}  {m['pearson']:>+8.4f}  {m['r2']:>+8.4f}  "
              f"{m['rmse']:>8.4f}  {m['mean_abs_diff']:>8.4f}")

    # ── Per-reaction table (best threshold) ──────────────────────────────────
    # Pick the threshold that gives the highest R² against the reference.
    # Uses reference only for display selection, not for model tuning.
    best_thr = max(
        (t for t in thresholds if (rxn_r2 >= t).sum() >= 2),
        key=lambda t: compute_metrics(p[rxn_r2 >= t], r[rxn_r2 >= t])['r2'],
    )
    best_thr = 0.3
    mask = rxn_r2 >= best_thr
    abs_diff = np.abs(p - r)

    print(f"\n── Per-reaction (R²_mdl ≥ {best_thr}, n={int(mask.sum())}) ──")
    print(f"{'Reaction':<30}  {'Pred':>9}  {'Ref':>9}  {'|Δ|':>9}  {'R²(mdl)':>8}")
    print(f"{'─'*30}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    for i in np.argsort(-abs_diff):
        if mask[i]:
            print(f"{shared[i]:<30}  {p[i]:>9.4f}  {r[i]:>9.4f}  "
                  f"{abs_diff[i]:>9.4f}  {rxn_r2[i]:>8.4f}")

    # ── Correlation plot — single panel, R²_mdl ≥ best_thr ──────────────────
    pf, rf = p[mask], r[mask]
    nf = shared[mask]
    adf = abs_diff[mask]
    m_filt = compute_metrics(pf, rf)

    # Linear regression fit (all filtered reactions)
    slope, intercept, *_ = linregress(rf, pf)
    # Axis limits include known fluxes (from ExtFlux) even if outside filtered set
    all_vals = np.concatenate([pf, rf, p_k_all, r_k_all])
    lim2 = max(np.abs(all_vals).max() * 1.1, 1.0)
    fit_x = np.array([-lim2, lim2])

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(rf, pf, s=30, alpha=0.60, color='#3A7DBF', zorder=3,
               edgecolors='white', linewidths=0.4, label='All reactions')

    ax.plot(fit_x, fit_x, color='#888888', lw=1.3, ls='--', alpha=0.7,
            label='y = x  (ideal)', zorder=2)
    ax.plot(fit_x, slope * fit_x + intercept, color='#E05C3A', lw=1.8,
            label=f'Fit:  y = {slope:.2f}x + {intercept:.2f}', zorder=4)

    ax.axhline(0, color='#cccccc', lw=0.8, zorder=1)
    ax.axvline(0, color='#cccccc', lw=0.8, zorder=1)
    ax.set_xlim(-lim2, lim2)
    ax.set_ylim(-lim2, lim2)

    # Annotate top outliers
    top_f = list(np.argsort(-adf)[:8])
    for i in top_f:
        ax.annotate(nf[i], xy=(rf[i], pf[i]), fontsize=7.5, color='#C0392B',
                    ha='left', va='bottom', xytext=(6, 4),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color='#E8A89C', lw=0.7))
    ax.scatter(rf[top_f], pf[top_f], s=40, color='#C0392B', zorder=5,
               edgecolors='white', linewidths=0.5, alpha=0.9)

    ax.set_xlabel('Reference MFA  [mmol gDW⁻¹ h⁻¹]')
    ax.set_ylabel('SigmaFlux prediction  [mmol gDW⁻¹ h⁻¹]')
    ax.set_title(
        f'SigmaFlux vs. Reference MFA  '
        f'(R²_mdl ≥ {best_thr},  n = {m_filt["n"]})',
        pad=10,
    )

    stats_text = (
        f"Pearson  $r$ = {m_filt['pearson']:+.3f}\n"
        f"$R^2$ = {m_filt['r2']:+.3f}\n"
        f"RMSE = {m_filt['rmse']:.3f}\n"
        f"mean|Δ| = {m_filt['mean_abs_diff']:.3f}"
    )
    ax.text(0.04, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, va='top', linespacing=1.6,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                      boxstyle='round,pad=0.4'))

    ax.legend(loc='lower right', framealpha=0.85, edgecolor='#cccccc')
    ax.grid(True, alpha=0.15, linestyle='-')
    ax.set_aspect('equal')

    fig.tight_layout()
    out = args.out or args.predictions.replace('.csv', '_analysis.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f"\nPlot saved → {out}")


if __name__ == '__main__':
    main()
