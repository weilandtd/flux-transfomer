"""
CPU hours for 1000 samples, 13CFLUX with a single start against SigmaFlux and a dummy for
INCA, as a bar graph on a log axis. CPU hours are wall time times the cores a run used.

  13CFLUX    per-sample seconds of the MFA benchmark's samples that ran a single start, on
             one core each: median, with the 10th and 90th percentiles as bounds
  SigmaFlux  measured by time_sigmaflux.py (<run-dir>/timing/sigmaflux_predict.json) times
             the threads it ran on, scaled to 1000 samples; bounds are the fastest and slowest
             repeat, and model loading is added once
  INCA       not measured: a hatched dummy at INCA_SLOWDOWN times 13CFLUX

Writes ``runtime_<key>.svg`` and the table behind it as ``runtime_<key>.csv``.

Usage:
    python paper/scripts/plot_runtime.py --models i_orientalis
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _figure_common import (
    COLORS, LABEL, METHOD_COLORS, add_common_args, apply_style, run_dir_for, save_figure,
    selected_models)

apply_style()

N_SAMPLES = 1000
QUANTILES = (0.1, 0.5, 0.9)
#: Assumed INCA cost relative to 13CFLUX, until INCA is measured.
INCA_SLOWDOWN = 1000


def mfa_hours(run):
    """(low, median, high) CPU hours for N_SAMPLES samples of one start each, one core each."""
    conv = pd.concat([pd.read_csv(f) for f in sorted((run / "mfa_benchmark").glob(
        "pred_*_convergence.csv"))], ignore_index=True)
    one = conv["reached_optimum"].astype(bool) & (conv["starts_used"] == 1)
    return conv.loc[one, "seconds"].quantile(QUANTILES).to_numpy() * N_SAMPLES / 3600


def sigmaflux_hours(run):
    """(low, median, high) CPU hours for N_SAMPLES samples, model loading included once."""
    rec = json.loads((run / "timing" / "sigmaflux_predict.json").read_text())
    per = np.array(rec["seconds"]) / rec["n_samples"] * N_SAMPLES
    wall = np.array([per.min(), np.median(per), per.max()]) + rec["load_seconds"]
    return wall * rec["threads"] / 3600


def plot_model(model, args):
    run = run_dir_for(model, args.run_suffix)
    mfa = mfa_hours(run)
    tbl = pd.DataFrame([("SigmaFlux", *sigmaflux_hours(run)), ("13CFLUX", *mfa),
                        ("INCA (dummy)", *(INCA_SLOWDOWN * mfa))],
                       columns=["method", "cpu_hours_lo", "cpu_hours", "cpu_hours_hi"])
    tbl.to_csv(args.out_dir / f"runtime_{model['key']}.csv", index=False)
    print(tbl.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    x = np.arange(len(tbl))
    # Bars start at the axis floor, not at 0: on a log axis a bar from 0 runs off the canvas
    # under the clip, and the hidden geometry makes the SVG huge for vector editors.
    floor = tbl["cpu_hours_lo"].min() / 2
    bars = ax.bar(x, tbl["cpu_hours"] - floor, bottom=floor, width=0.6,
                  color=[METHOD_COLORS["ai"], METHOD_COLORS["mfa"], "none"],
                  yerr=[tbl["cpu_hours"] - tbl["cpu_hours_lo"],
                        tbl["cpu_hours_hi"] - tbl["cpu_hours"]],
                  error_kw=dict(elinewidth=0.8, capsize=3))
    bars[-1].set(edgecolor=COLORS["reference"], hatch="///")
    for xi, (value, top) in enumerate(zip(tbl["cpu_hours"], tbl["cpu_hours_hi"])):
        ax.annotate(f"{value:.2g} CPU h" if value < 1 else f"{value:,.0f} CPU h", (xi, top),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, [LABEL["ai"], r"$^{13}$CFLUX", "INCA\n(dummy)"])
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_ylabel(f"CPU hours for {N_SAMPLES:,} data sets")
    sns.despine(ax=ax)
    save_figure(fig, args.out_dir / f"runtime_{model['key']}.svg")


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in selected_models(args):
        plot_model(model, args)


if __name__ == "__main__":
    main()
