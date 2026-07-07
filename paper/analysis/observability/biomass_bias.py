"""
Is the model's accuracy a "biomass bias"? — i.e. are the well-predicted reactions
just the ones stoichiometrically determined by the measured (biomass-precursor +
exchange) fluxes, rather than resolved by isotope labeling?

For each reaction we measure how well a simple LINEAR model on the measured fluxes
predicts that reaction's flux across the training flux samples (train/test split).
High linear-R2 => reaction is determined by the measured fluxes (no labeling needed).
We then cross this with the transformer's per-reaction validation R2.

Usage:
    python paper/analysis/biomass_bias.py \
        --checkpoint paper/.results/i_orientalis_20260620/imisor_transformer.pt
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sigmaflux.ai.data import load_data

BIOMASS_PREC = ['PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c', 'CARB_c']


def lin_r2(X, y):
    """Out-of-sample R2 of OLS y~X, 50/50 split."""
    n = len(y); h = n // 2
    Xtr = np.c_[np.ones(h), X[:h]]; Xte = np.c_[np.ones(n - h), X[h:]]
    coef, *_ = np.linalg.lstsq(Xtr, y[:h], rcond=None)
    pred = Xte @ coef
    ss_res = ((y[h:] - pred) ** 2).sum(); ss_tot = ((y[h:] - y[h:].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    flux_names = list(ck['flux_names'])
    measured = list(ck['measured_fluxes'])
    bm = [m for m in measured if m in BIOMASS_PREC]
    ex = [m for m in measured if m not in BIOMASS_PREC]
    print(f"measured fluxes ({len(measured)}): biomass-precursors={bm}  exchanges={ex}")

    _, flux_df, _ = load_data(ck['data_dir'])
    F = flux_df.reindex(flux_names)                          # (n_rxn, n_samples), reaction order = model
    Fv = F.values.astype(float)

    def r2_set(pred_names):
        X = flux_df.reindex(pred_names).values.T.astype(float)   # (samples, n_pred)
        return np.array([lin_r2(X, Fv[k]) for k in range(len(flux_names))])

    r2_all = r2_set(measured)
    r2_bm  = r2_set(bm) if bm else np.full(len(flux_names), np.nan)
    r2_ex  = r2_set(ex) if ex else np.full(len(flux_names), np.nan)

    model_r2 = np.asarray(ck['val_correlations']['flux_r2_per_reaction'], float)
    mnames = list(ck['val_correlations']['flux_reaction_names'])
    # align model_r2 to flux_names order
    midx = {n: i for i, n in enumerate(mnames)}
    model_r2 = np.array([model_r2[midx[n]] if n in midx else np.nan for n in flux_names])

    def frac(a, thr): return f"{int(np.nansum(a > thr))}/{int(np.isfinite(a).sum())}"
    print(f"\nLinear determinability of each reaction from measured fluxes (out-of-sample R2):")
    print(f"  from ALL 10 measured : median={np.nanmedian(r2_all):.2f}  R2>0.9: {frac(r2_all,0.9)}  R2>0.5: {frac(r2_all,0.5)}")
    print(f"  from 5 biomass-prec  : median={np.nanmedian(r2_bm):.2f}  R2>0.9: {frac(r2_bm,0.9)}  R2>0.5: {frac(r2_bm,0.5)}")
    print(f"  from 5 exchanges     : median={np.nanmedian(r2_ex):.2f}  R2>0.9: {frac(r2_ex,0.9)}  R2>0.5: {frac(r2_ex,0.5)}")

    # cross: of the reactions the MODEL predicts well, how many are already
    # linearly determined by measured fluxes?
    ok = np.isfinite(model_r2) & np.isfinite(r2_all)
    mwell = ok & (model_r2 > 0.9)
    print(f"\nCross with model per-reaction R2 ({int(ok.sum())} reactions):")
    print(f"  model R2>0.9: {int(mwell.sum())} reactions")
    print(f"    of these, linearly determined by measured fluxes (R2>0.9): "
          f"{int((mwell & (r2_all>0.9)).sum())}  ({100*np.mean(r2_all[mwell]>0.9):.0f}%)")
    hard = ok & (r2_all < 0.5)      # NOT determined by measured fluxes — where labeling must do the work
    print(f"  reactions NOT determined by measured fluxes (linear R2<0.5): {int(hard.sum())}")
    print(f"    model's median R2 on those: {np.nanmedian(model_r2[hard]):.2f}  (model R2>0.9 on {frac(model_r2[hard],0.9)})")
    from scipy.stats import spearmanr
    print(f"  Spearman(model R2, measured-linear R2) = {spearmanr(model_r2[ok], r2_all[ok])[0]:.2f}")

    # plot
    out = args.out or args.checkpoint.replace('.pt', '_biomass_bias.svg')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r2_all[ok].clip(-0.5, 1), model_r2[ok].clip(-0.5, 1), s=10, alpha=0.4, color='#2c7fb8')
    ax.plot([-0.5, 1], [-0.5, 1], 'k--', lw=1)
    ax.set_xlabel('linear R² from measured fluxes\n(= determined by biomass/exchange)')
    ax.set_ylabel('transformer per-reaction R²')
    ax.set_title('Is model accuracy a biomass/measured-flux bias?', fontsize=10)
    ax.set_xlim(-0.5, 1.02); ax.set_ylim(-0.5, 1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nsaved -> {out} (+ .png)")


if __name__ == '__main__':
    main()
