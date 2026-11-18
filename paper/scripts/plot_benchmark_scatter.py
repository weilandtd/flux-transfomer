"""
Ground-truth vs predicted flux for one reaction, SigmaFlux and MFA simulation benchmarks
overlaid in a single panel.

Writes ``benchmark_scatter_<model>_<reaction>.svg``. Each point is one benchmark sample;
the legend carries each method's R² over the samples it returned a prediction for.

The default is pyruvate kinase, ``R7`` (PEP --> Pyr + ATP) in the E. coli core network.

Usage:
    python paper/scripts/plot_benchmark_scatter.py
    python paper/scripts/plot_benchmark_scatter.py --model s_cerevisiae --reaction HEX1 --label HEX
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _figure_common import (
    COLORS, LABEL, METHOD_COLORS, MODELS_BY_KEY, apply_style, axis_floor, load_benchmark,
    load_data, run_dir_for, save_figure)

apply_style()

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

UNITS = r"[mmol gDW$^{-1}$ h$^{-1}$]"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="e_coli_core", choices=list(MODELS_BY_KEY))
    p.add_argument("--reaction", default="R7")
    p.add_argument("--label", default="PYK", help="Reaction name on the axis labels")
    p.add_argument("--max", type=float, default=20.0, help="Upper limit of both axes")
    p.add_argument("--log", action="store_true",
                   help="Log-log axes; points at or below zero are dropped (R² still uses all)")
    p.add_argument("--out-dir", type=pathlib.Path, default=_REPO_ROOT / "paper" / "figures")
    p.add_argument("--run-suffix", default=None,
                   help="Replace the date suffix of the model's run directory")
    return p.parse_args()


def r2(gt, pred):
    return 1.0 - ((pred - gt) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()


def load_pairs(run, reaction):
    """Tidy (method, gt, pred) frame for one reaction across both benchmarks."""
    _, flux_df, _ = load_data(str(run))
    flux_df.columns = range(flux_df.shape[1])
    if reaction not in flux_df.index:
        raise SystemExit(f"{reaction} not in {run.name}")
    frames = []
    for method in ("mfa", "ai"):
        df, _ = load_benchmark(run / f"{method}_benchmark", flux_df.loc[[reaction]])
        df["method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_scatter(df, label, hi, path, log=False):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lo = axis_floor(df["gt"], df["pred"]) if log else 0.0
    ax.plot([lo, hi], [lo, hi], color=COLORS["reference"], lw=0.8, ls="--", zorder=0)

    for method in ("mfa", "ai"):
        grp = df[df["method"] == method]
        score = r2(grp["gt"].to_numpy(), grp["pred"].to_numpy())
        print(f"  {LABEL[method]:10s} n={len(grp)}  R²={score:.3f}")
        if log:
            shown = grp[(grp["gt"] > 0) & (grp["pred"] > 0)]
            if len(shown) < len(grp):
                print(f"    {len(grp) - len(shown)} point(s) at or below zero not drawn")
            grp = shown
        sns.scatterplot(data=grp, x="gt", y="pred", ax=ax, color=METHOD_COLORS[method],
                        s=8, alpha=0.5, linewidth=0,
                        label=f"{LABEL[method]} ($R^2$ = {score:.2f})")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    if not log:
        ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.set_xlabel(f"True {label} flux\n" + UNITS)
    ax.set_ylabel(f"Predicted {label} flux\n" + UNITS)
    ax.legend(frameon=False, fontsize=9, loc="lower right", markerscale=2)
    sns.despine(ax=ax)
    save_figure(fig, path)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run = run_dir_for(MODELS_BY_KEY[args.model], args.run_suffix)
    df = load_pairs(run, args.reaction)
    suffix = "_log" if args.log else ""
    plot_scatter(df, args.label, args.max,
                 out_dir / f"benchmark_scatter_{args.model}_{args.reaction}{suffix}.svg",
                 log=args.log)


if __name__ == "__main__":
    main()
