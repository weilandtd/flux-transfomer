"""Do an arm's predicted flux states reproduce the measured labeling?

Flux-space R2 is not the yardstick the pipeline is judged on: MIDs depend on flux
*ratios*, not on distance, so two states that differ in flux space can agree in labeling
space and vice versa. This runs the EMU forward pass on each arm's predictions and scores
them against the simulated truth for the same samples.

It also reports how many forward passes return something that is not a MID at all. An
infeasible flux state does not merely predict labeling badly -- measured on I. orientalis
it returned a largest "fraction" of 9.9e+39 -- so the feasibility guarantee and the
labeling agreement are two halves of the same question.

Usage:
    python paper/analysis/diagnostics/emu_agreement.py --config paper/configs/e_coli_core.yaml \\
        --n-benchmarks 100 paper/.results/e_coli_core_gctl_20260831 ...
"""

import argparse
import os

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.pipeline._benchmark import (
    build_mfa_context,
    forward_labeling,
    load_benchmark_inputs,
    pin_threads,
    select_samples,
)
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.pipeline.run_ai_benchmark import build_df_L, build_df_Vm


def score_arm(run_dir, cfg, root, args, shared):
    """Ensemble-mean MID RMSE for one arm, plus how often the pass returns a non-MID."""
    cobra_model, training_labeling_df, training_flux_df, sel = shared
    checkpoint = next((os.path.join(run_dir, n) for n in sorted(os.listdir(run_dir))
                       if n.endswith(".pt")), None)
    sf = load_predictor(args.config, checkpoint, data_dir=run_dir)
    flux_names = list(sf.flux_names)
    inv_tracer = {v: k for k, v in sf._tracer_name_to_id.items()}

    df_L = build_df_L(training_labeling_df, sel, args.noise_std, args.seed, inv_tracer)
    df_Vm = build_df_Vm(training_flux_df, cfg.get("training", {}).get("measured_fluxes", []),
                        sel, args.noise_std, args.flux_noise_std, training_labeling_df,
                        args.seed)
    # The ensemble mean is the reported prediction, and it is a convex combination. On a
    # projection arm the box is a convex set every member's state lies in, so the mean lies
    # in it too; on an affine arm nothing makes that true and the mean can sit outside.
    res = sf.predict(df_L, df_Vm=df_Vm)
    res["_pos"] = res.groupby("Sample", sort=False).cumcount()
    flux = (res[res["_pos"] < len(flux_names)]
            .pivot(index="Sample", columns="Flux Name", values="Value"))
    rev = (res[res["_pos"] >= len(flux_names)]
           .pivot(index="Sample", columns="Flux Name", values="Value"))

    mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"],
        training_flux_df.iloc[:, int(sel[0])], "emu_agreement", with_backward=False)
    fl = flux.reindex(columns=reaction_ids).to_numpy(float)
    rv = rev.reindex(columns=reaction_ids).to_numpy(float)

    truth = training_labeling_df.iloc[:, sel]
    rmse, invalid, worst = [], 0, 0.0
    for i in range(len(sel)):
        lab = np.asarray(forward_labeling(mfa_model, np.nan_to_num(fl[i], nan=0.0),
                                          np.nan_to_num(rv[i], nan=1.0), labeling_to),
                         dtype=float)
        pred = pd.Series(lab, index=labeling_index).reindex(truth.index)
        ok = pred.notna().to_numpy() & np.isfinite(pred.to_numpy())
        # A fraction above 1 is not a MID; the pass returned something else entirely.
        worst = max(worst, float(np.nanmax(np.abs(pred.to_numpy()[ok])) if ok.any() else 0))
        if not ok.all() or np.nanmax(np.abs(pred.to_numpy()[ok])) > 1.0 + 1e-6:
            invalid += 1
        d = pred.to_numpy()[ok] - truth.iloc[:, i].to_numpy()[ok]
        rmse.append(float(np.sqrt((d ** 2).mean())))
    return {"rmse_median": float(np.median(rmse)), "rmse_mean": float(np.mean(rmse)),
            "rmse_p90": float(np.percentile(rmse, 90)),
            "invalid": invalid / len(sel), "worst": worst}


def main(args):
    pin_threads()
    cfg, root = load_config(args.config, args.root)
    first = resolve_path(args.run_dirs[0], root)
    cobra_model, _, lab_df, flux_df, _ = load_benchmark_inputs(first, cfg, root)
    sel, _ = select_samples(args.split, None, flux_df.shape[1], cfg, 0, args.n_benchmarks)
    shared = (cobra_model, lab_df, flux_df, sel)
    print(f"{len(sel)} samples, EMU forward pass on the ensemble-mean prediction\n")
    print(f"{'arm':<30} {'MID RMSE':>10} {'mean':>8} {'p90':>8} "
          f"{'non-MID':>8} {'max fraction':>13}")
    for rd in args.run_dirs:
        s = score_arm(resolve_path(rd, root), cfg, root, args, shared)
        print(f"{os.path.basename(rd):<30} {s['rmse_median']:>10.4f} {s['rmse_mean']:>8.4f} "
              f"{s['rmse_p90']:>8.4f} {s['invalid']:>7.1%} {s['worst']:>13.4g}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--config", required=True)
    p.add_argument("--n-benchmarks", type=int, default=100)
    p.add_argument("--split", default="val")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--flux-noise-std", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    add_root_argument(p)
    raise SystemExit(main(p.parse_args()))
