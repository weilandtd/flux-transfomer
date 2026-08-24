"""KO-flux prediction: SigmaFlux (measures the KO) vs FBApro (predicts it from WT).

Two 4-bar figures on the saved KO run, using the trained SigmaFlux model. Four methods
per knocked-out reaction j:
  SF|KO           SigmaFlux run on the KO samples (their MIDs + measured exchanges)
  SF|WT           SigmaFlux run on WT samples (its WT estimate)
  SF+FBApro       FBApro CLEANUP of SigmaFlux's per-sample KO prediction, UNWEIGHTED
                  (plain L2): project vp onto {Sx=0, x[j]=0}. Same anchor as the +var
                  bar but no weights -- the correction spills onto small confident
                  fluxes and wrecks R2, showing the weighting is what makes it work.
  SF+FBApro+var   Same cleanup, inverse-variance weighted (FBAproFixed/Partial anchored
                  to SF). Keeps SF's R2 while forcing the KO'd reaction to exactly zero.

Figure 1: |predicted flux of the KO'd reaction|, averaged over KO reactions, then
          normalised so SF|WT = 1. SF|KO is a fraction < 1 (how much of the WT flux
          SigmaFlux still assigns to the KO'd reaction -- lower = better KO detection);
          FBApro bars are 0 by construction (they pin the KO'd reaction to exactly 0).
Figure 2: R2 over the FULL flux vector vs the true KO flux. SigmaFlux should stand ~0.98;
          FBApro should fall. Also reports how many reactions FBApro flips the wrong way.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.linalg import null_space
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
    """(flips, slots): reactions whose predicted flux runs opposite ``v_ref`` (sign
    disagreement), and the number of reactions actually checked (|v_ref| > thresh, minus
    the KO'd reaction j). Divide flips/slots for the wrong-direction rate."""
    mask = np.abs(v_ref) > thresh
    mask[ignore_j] = False
    flips = int(np.sum(np.sign(v_pred[mask]) * np.sign(v_ref[mask]) < 0))
    return flips, int(mask.sum())


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
    ap.add_argument("--ko-list", default=None, help="comma-separated KO reactions to restrict to")
    ap.add_argument("--tag", default="", help="output subdir suffix (ko_methods<tag>)")
    ap.add_argument("--data-dir", default=None,
                    help="override the checkpoint's data dir (uses ALL samples, zero-shot)")
    args = ap.parse_args()
    ckpt = args.checkpoint or os.path.join(args.run_dir, "imisor_transformer_ko.pt")
    out_dir = os.path.join(args.run_dir, "ko_methods" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    keep = set(args.ko_list.split(",")) if args.ko_list else None

    dev = resolve_device()
    ctx = (load_eval_context(ckpt, data_dir=args.data_dir, samples="all")
           if args.data_dir else load_eval_context(ckpt))   # VAL split, or ALL samples zero-shot
    m = build_flux_decoder(ctx, device=dev)
    flux_names = list(ctx.flux_names)
    name2idx = {n: i for i, n in enumerate(flux_names)}
    S = np.asarray(ctx.stoichiometric_matrix, dtype=np.float64)   # aligned to flux_names
    Z = null_space(S)                                             # null-space basis, computed once
    labels_all = pd.read_csv(os.path.join(args.run_dir, "ko_labels.csv"), index_col=0)
    labels = (labels_all.reset_index(drop=True) if ctx.sample_indices is None
              else labels_all.iloc[ctx.sample_indices].reset_index(drop=True))

    (c, l, t, fr), tg = ctx.full_batch(device=dev)
    fid = m.fixed_idx_t.to(dev)
    with torch.no_grad():
        vp = m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid),
               measured_flux_ids=fid, return_attn=False)[0].cpu().numpy()   # (n_val, n_rxn)
    gt = tg[:, :len(flux_names)].cpu().numpy()

    wt = labels.ko_reaction.values == "WT"
    v_ref = vp[wt].mean(0)                              # SigmaFlux's WT flux prediction
    w = 1.0 / (vp[wt].var(0) + 1e-8)                    # inverse-variance weights
    ko_rxns = [r for r in labels.ko_reaction.unique() if r != "WT" and r in name2idx
               and (keep is None or r in keep)]

    # accumulators
    f1 = {k: [] for k in ORDER}                         # Fig1 per-reaction |flux|/wt_std
    true_acc = {k: [] for k in ORDER}                   # Fig2 pooled R2 inputs
    pred_acc = {k: [] for k in ORDER}
    flips = {"SF+FBApro": 0, "SF+FBApro+var": 0}         # vs WT direction
    flips_true = {"SF+FBApro": 0, "SF+FBApro+var": 0}    # vs the true KO direction
    slots_wt = 0                                          # reaction-slots checked (vs WT)
    slots_true = 0                                        # reaction-slots checked (vs true)
    rows = []

    for r in ko_rxns:
        j = name2idx[r]
        koR = labels.ko_reaction.values == r
        if koR.sum() < 5:
            continue
        vfixed = np.full(len(flux_names), np.nan)
        vfixed[j] = 0.0
        v3 = fbapro(vp[koR], Z=Z, vfixed=vfixed)                # SF+FBApro: SF-anchored, unweighted
        v4 = fbapro(vp[koR], Z=Z, vfixed=vfixed, weights=w)     # SF+FBApro+var: SF-anchored, +var
        assert np.abs(v3[:, j]).max() < 1e-8 and np.abs(v4[:, j]).max() < 1e-8   # KO enforced

        gt_ko = gt[koR, :]
        # Figure 1: |predicted flux of the KO'd reaction j| (normalised to SF|WT below)
        f1["SF|KO"].append(np.abs(vp[koR, j]).mean())
        f1["SF|WT"].append(np.abs(vp[wt, j]).mean())
        f1["SF+FBApro"].append(np.abs(v3[:, j]).mean())
        f1["SF+FBApro+var"].append(np.abs(v4[:, j]).mean())
        # Figure 2: SF|KO measures the KO (per-sample vs KO truth). SF+FBApro and
        # SF+FBApro+var both clean up SF's own per-sample KO prediction (vs KO truth),
        # differing only by the inverse-variance weighting.
        # SF|WT is handled separately below (SigmaFlux's own WT accuracy, per-sample vs WT truth).
        true_acc["SF|KO"].append(gt_ko);      pred_acc["SF|KO"].append(vp[koR, :])
        true_acc["SF+FBApro"].append(gt_ko);  pred_acc["SF+FBApro"].append(v3)
        true_acc["SF+FBApro+var"].append(gt_ko); pred_acc["SF+FBApro+var"].append(v4)
        # sign flips on each block-mean prediction, vs WT and vs the true KO flux
        gt_dir = gt_ko.mean(0)
        fl3, s3 = sign_flips(v3.mean(0), v_ref, j, args.flip_thresh)
        fl4, _ = sign_flips(v4.mean(0), v_ref, j, args.flip_thresh)
        fl3t, s3t = sign_flips(v3.mean(0), gt_dir, j, args.flip_thresh)
        fl4t, _ = sign_flips(v4.mean(0), gt_dir, j, args.flip_thresh)
        flips["SF+FBApro"] += fl3
        flips["SF+FBApro+var"] += fl4
        flips_true["SF+FBApro"] += fl3t
        flips_true["SF+FBApro+var"] += fl4t
        slots_wt += s3
        slots_true += s3t
        rows.append(dict(reaction=r, n_ko=int(koR.sum()),
                         f1_sf_ko=f1["SF|KO"][-1], f1_sf_wt=f1["SF|WT"][-1],
                         flips_fba=fl3, flips_fbaw=fl4,
                         flips_fba_true=fl3t, flips_fbaw_true=fl4t, slots=s3t))
        print(f"  {r:12s} SF|KO={f1['SF|KO'][-1]:.3f} SF|WT={f1['SF|WT'][-1]:.3f} "
              f"flips: fba={fl3} fbaw={fl4}", flush=True)

    # aggregate bars. SF|WT = SigmaFlux's own WT accuracy (per-sample WT prediction vs true
    # WT flux) -- the "naive SF", expected ~0.98 like SF|KO. The other three are vs KO truth.
    true_acc["SF|WT"] = [gt[wt]]
    pred_acc["SF|WT"] = [vp[wt]]
    # Figure 1: mean |flux of the KO'd reaction| over KOs, normalised so SF|WT = 1.
    fig1_raw = [float(np.mean(f1[k])) for k in ORDER]
    wt_scale = fig1_raw[ORDER.index("SF|WT")] or 1.0
    fig1 = [v / wt_scale for v in fig1_raw]
    fig2_med = [r2_median_per_reaction(true_acc[k], pred_acc[k]) for k in ORDER]   # headline
    fig2_pooled = [r2_pooled(true_acc[k], pred_acc[k]) for k in ORDER]             # context

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ko_methods_results.csv"), index=False)
    pd.DataFrame({"method": ORDER, "fig1_flux": fig1,
                  "fig2_r2_median": fig2_med, "fig2_r2_pooled": fig2_pooled,
                  "sign_flips": [np.nan, np.nan, flips["SF+FBApro"], flips["SF+FBApro+var"]]}) \
        .to_csv(os.path.join(out_dir, "ko_methods_summary.csv"), index=False)

    bar_plot(fig1, "flux of KO'd reaction (rel. to SF|WT)",
             os.path.join(out_dir, "fig_ko_reaction_flux.svg"), ylim=(0, 1.15))
    bar_plot(fig2_med, "median per-reaction R² vs true KO flux",
             os.path.join(out_dir, "fig_full_r2.svg"), ylim=(-1.5, 1.05))

    print(f"\n===== {len(rows)} KO reactions =====", flush=True)
    print("Figure 1 flux(KO rxn), SF|WT=1:", {k: round(v, 3) for k, v in zip(ORDER, fig1)}, flush=True)
    print("Figure 2 median-per-reaction R2:", {k: round(v, 3) for k, v in zip(ORDER, fig2_med)}, flush=True)
    print("        (full-vector pooled R2):", {k: round(v, 3) for k, v in zip(ORDER, fig2_pooled)}, flush=True)
    print("Sign flips vs WT dir  (per block-mean, summed over %d KOs, /%d slots):" % (len(rows), slots_wt), flush=True)
    print("    FBApro=%d/%d (%.1f%%)  FBApro+var=%d/%d (%.1f%%)"
          % (flips["SF+FBApro"], slots_wt, 100 * flips["SF+FBApro"] / max(slots_wt, 1),
             flips["SF+FBApro+var"], slots_wt, 100 * flips["SF+FBApro+var"] / max(slots_wt, 1)), flush=True)
    print("Sign flips vs TRUE KO (the honest wrong-direction count, /%d slots):" % slots_true, flush=True)
    print("    FBApro=%d/%d (%.1f%%)  FBApro+var=%d/%d (%.1f%%)"
          % (flips_true["SF+FBApro"], slots_true, 100 * flips_true["SF+FBApro"] / max(slots_true, 1),
             flips_true["SF+FBApro+var"], slots_true, 100 * flips_true["SF+FBApro+var"] / max(slots_true, 1)), flush=True)
    print(f"wrote figures + CSVs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
