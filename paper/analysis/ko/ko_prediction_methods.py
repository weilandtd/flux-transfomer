"""KO-flux prediction: SigmaFlux (measures the KO) vs FBApro (predicts it from WT).

Two 4-bar figures on the saved KO run, using the trained SigmaFlux model. Four methods
per knocked-out reaction j:
  SF|KO           SigmaFlux run on the KO samples (their MIDs + measured exchanges)
  SF|WT           SigmaFlux run on WT samples (its WT estimate)
  SF+FBApro       FBApro projection of SigmaFlux's WT prediction with x[j]=0 (unbounded)
  SF+FBApro+var   same, variance-weighted (1/var) so small fluxes don't flip

Figure 1: |predicted flux of the KO'd reaction| / WT-std, averaged over KO reactions.
          FBApro bars are 0 by construction; SF|KO should be << SF|WT.
Figure 2: R2 over the FULL flux vector vs the true KO flux. SigmaFlux should stand ~0.98;
          FBApro should fall. Also reports how many reactions FBApro flips the wrong way.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device
from sigmaflux.core.fbapro import fbapro

mpl.rcParams.update({"font.family": "Arial", "font.size": 12})
COLORS = ["#4C72B0", "#DD8452", "#B0B0B0", "#000000"]
ORDER = ["SF|KO", "SF|WT", "SF+FBApro", "SF+FBApro+var"]


def r2_pooled(true, pred):
    true = np.concatenate([np.ravel(t) for t in true])
    pred = np.concatenate([np.ravel(p) for p in pred])
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - np.sum((true - pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def r2_median_per_reaction(true_list, pred_list):
    """Median over reactions of the per-reaction R2 across the pooled KO samples --
    SigmaFlux's own val metric (~0.98), the 'median score' the comparison targets."""
    T = np.concatenate(true_list, 0)
    P = np.concatenate(pred_list, 0)                    # (N_ko_samples, n_rxn)
    r2s = []
    for i in range(T.shape[1]):
        sst = np.sum((T[:, i] - T[:, i].mean()) ** 2)
        if sst > 1e-9:
            r2s.append(1 - np.sum((T[:, i] - P[:, i]) ** 2) / sst)
    return float(np.median(r2s))


def sign_flips(v_pred, v_ref, ignore_j, thresh):
    """# reactions whose predicted flux runs opposite the WT reference (skip near-zero & j)."""
    mask = np.abs(v_ref) > thresh
    mask[ignore_j] = False
    return int(np.sum(np.sign(v_pred[mask]) * np.sign(v_ref[mask]) < 0))


def bar_plot(values, ylabel, out, ylim=None, fmt="{:.2f}"):
    df = pd.DataFrame({"method": ORDER, "value": values})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.barplot(data=df, x="method", y="value", order=ORDER, palette=COLORS, width=0.5, ax=ax)
    lo, hi = ylim if ylim else (0, max(values) * 1.18)
    pad = 0.03 * (hi - lo)
    for k, v in enumerate(values):
        ty = min(max(v, lo + pad), hi - pad)           # keep the label inside the view
        ax.text(k, ty + (pad if v >= 0 else -pad), fmt.format(v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(lo, hi)
    ax.set_xticklabels(ORDER, rotation=20, ha="right", fontsize=10)
    sns.despine(ax=ax)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="paper/.results/i_orientalis_ko_20260628")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--flip-thresh", type=float, default=1e-6)
    args = ap.parse_args()
    ckpt = args.checkpoint or os.path.join(args.run_dir, "imisor_transformer_ko.pt")
    out_dir = os.path.join(args.run_dir, "ko_methods")
    os.makedirs(out_dir, exist_ok=True)

    dev = resolve_device()
    ctx = load_eval_context(ckpt)                       # VAL split (held-out)
    m = build_flux_decoder(ctx, device=dev)
    flux_names = list(ctx.flux_names)
    name2idx = {n: i for i, n in enumerate(flux_names)}
    S = np.asarray(ctx.stoichiometric_matrix, dtype=np.float64)   # aligned to flux_names
    labels = pd.read_csv(os.path.join(args.run_dir, "ko_labels.csv"), index_col=0) \
        .iloc[ctx.sample_indices].reset_index(drop=True)

    (c, l, t, fr), tg = ctx.full_batch(device=dev)
    fid = m.fixed_idx_t.to(dev)
    with torch.no_grad():
        vp = m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid),
               measured_flux_ids=fid, return_attn=False)[0].cpu().numpy()   # (n_val, n_rxn)
    gt = tg[:, :len(flux_names)].cpu().numpy()

    wt = labels.ko_reaction.values == "WT"
    v_ref = vp[wt].mean(0)                              # SigmaFlux's WT flux prediction
    w = 1.0 / (vp[wt].var(0) + 1e-8)                    # inverse-variance weights
    ko_rxns = [r for r in labels.ko_reaction.unique() if r != "WT" and r in name2idx]

    # accumulators
    f1 = {k: [] for k in ORDER}                         # Fig1 per-reaction |flux|/wt_std
    true_acc = {k: [] for k in ORDER}                   # Fig2 pooled R2 inputs
    pred_acc = {k: [] for k in ORDER}
    flips = {"SF+FBApro": 0, "SF+FBApro+var": 0}
    rows = []

    for r in ko_rxns:
        j = name2idx[r]
        koR = labels.ko_reaction.values == r
        if koR.sum() < 5:
            continue
        wt_std = gt[wt, j].std() or 1.0
        v3 = fbapro(S, v_ref, fixed=[j], fixed_values=[0.0])              # SF+FBApro
        v4 = fbapro(S, v_ref, fixed=[j], fixed_values=[0.0], weights=w)   # SF+FBApro+var
        assert abs(v3[j]) < 1e-8 and abs(v4[j]) < 1e-8                    # KO enforced

        gt_ko = gt[koR, :]
        # Figure 1: |predicted flux of j| / wt_std
        f1["SF|KO"].append(np.abs(vp[koR, j]).mean() / wt_std)
        f1["SF|WT"].append(np.abs(vp[wt, j]).mean() / wt_std)
        f1["SF+FBApro"].append(abs(v3[j]) / wt_std)
        f1["SF+FBApro+var"].append(abs(v4[j]) / wt_std)
        # Figure 2: SF|KO measures the KO (per-sample vs KO truth); SF+FBApro predict it.
        # SF|WT is handled separately below (SigmaFlux's own WT accuracy, per-sample vs WT truth).
        true_acc["SF|KO"].append(gt_ko);      pred_acc["SF|KO"].append(vp[koR, :])
        true_acc["SF+FBApro"].append(gt_ko);  pred_acc["SF+FBApro"].append(np.tile(v3, (koR.sum(), 1)))
        true_acc["SF+FBApro+var"].append(gt_ko); pred_acc["SF+FBApro+var"].append(np.tile(v4, (koR.sum(), 1)))
        # sign flips (vs WT reference direction)
        fl3 = sign_flips(v3, v_ref, j, args.flip_thresh)
        fl4 = sign_flips(v4, v_ref, j, args.flip_thresh)
        flips["SF+FBApro"] += fl3
        flips["SF+FBApro+var"] += fl4
        rows.append(dict(reaction=r, n_ko=int(koR.sum()), wt_std=wt_std,
                         f1_sf_ko=f1["SF|KO"][-1], f1_sf_wt=f1["SF|WT"][-1],
                         flips_fba=fl3, flips_fbaw=fl4))
        print(f"  {r:12s} SF|KO={f1['SF|KO'][-1]:.3f} SF|WT={f1['SF|WT'][-1]:.3f} "
              f"flips: fba={fl3} fbaw={fl4}", flush=True)

    # aggregate bars. SF|WT = SigmaFlux's own WT accuracy (per-sample WT prediction vs true
    # WT flux) -- the "naive SF", expected ~0.98 like SF|KO. The other three are vs KO truth.
    true_acc["SF|WT"] = [gt[wt]]
    pred_acc["SF|WT"] = [vp[wt]]
    fig1 = [float(np.mean(f1[k])) for k in ORDER]
    fig2_med = [r2_median_per_reaction(true_acc[k], pred_acc[k]) for k in ORDER]   # headline
    fig2_pooled = [r2_pooled(true_acc[k], pred_acc[k]) for k in ORDER]             # context

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ko_methods_results.csv"), index=False)
    pd.DataFrame({"method": ORDER, "fig1_flux": fig1,
                  "fig2_r2_median": fig2_med, "fig2_r2_pooled": fig2_pooled,
                  "sign_flips": [np.nan, np.nan, flips["SF+FBApro"], flips["SF+FBApro+var"]]}) \
        .to_csv(os.path.join(out_dir, "ko_methods_summary.csv"), index=False)

    bar_plot(fig1, "|predicted flux (KO rxn)| / WT std",
             os.path.join(out_dir, "fig_ko_reaction_flux.svg"))
    bar_plot(fig2_med, "median per-reaction R² vs true KO flux",
             os.path.join(out_dir, "fig_full_r2.svg"), ylim=(-1.5, 1.05))

    print(f"\n===== {len(rows)} KO reactions =====", flush=True)
    print("Figure 1 |flux|/WTstd :", {k: round(v, 3) for k, v in zip(ORDER, fig1)}, flush=True)
    print("Figure 2 median-per-reaction R2:", {k: round(v, 3) for k, v in zip(ORDER, fig2_med)}, flush=True)
    print("        (full-vector pooled R2):", {k: round(v, 3) for k, v in zip(ORDER, fig2_pooled)}, flush=True)
    print("Sign flips (wrong direction): FBApro=%d  FBApro+var=%d"
          % (flips["SF+FBApro"], flips["SF+FBApro+var"]), flush=True)
    print(f"wrote figures + CSVs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
