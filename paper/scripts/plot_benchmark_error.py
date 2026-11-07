"""
Per-reaction NRMSE of each method against the ground truth of the simulation benchmark,
one pair of bars per species.

  benchmark_mean_nrmse.svg   mean over reactions of each reaction's NRMSE
  benchmark_nrmse_q90.svg    90th percentile over reactions, the large per-reaction
                             failures that drive the mean
  benchmark_nrmse.csv        the values behind both figures, with reaction counts

A reaction's NRMSE is its RMSE over the test samples divided by its root-mean-square flux
over the whole population (``benchmark_metrics``, the table behind the NRMSE violins), so
every reaction is on its own scale and no single near-zero sample can dominate. The mean
over reactions weighs a reaction a method gets badly wrong in full; the 90th percentile
shows those failures directly.

Usage:
    python paper/scripts/plot_benchmark_error.py
    python paper/scripts/plot_benchmark_error.py --models s_cerevisiae,i_orientalis
"""

import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    METHOD_COLORS, ORGANISM_SHORT, add_common_args, apply_style, benchmark_metrics,
    save_figure, selected_models)

apply_style()

METHOD_LABEL = {"ai": "Flux transformer", "mfa": "Traditional MFA"}

COLORS = {
    METHOD_LABEL["ai"]:  METHOD_COLORS["ai"],    # teal
    METHOD_LABEL["mfa"]: METHOD_COLORS["mfa"],   # amber
}

STATISTICS = {"mean_nrmse": "Mean NRMSE", "nrmse_q90": "90th percentile\nNRMSE"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    return p.parse_args()


def summarize(model, run_suffix=None):
    """Mean and 90th-percentile per-reaction NRMSE of each method, or None when the model
    has no benchmark."""
    table = benchmark_metrics(model, run_suffix)
    if table is None:
        return None
    return pd.DataFrame([{
        "species": ORGANISM_SHORT.get(model["key"], model["label"]),
        "method": label,
        "mean_nrmse": table[f"nrmse_{method}"].mean(),
        "nrmse_q90": table[f"nrmse_{method}"].quantile(0.9),
        "n_reactions": len(table),
    } for method, label in METHOD_LABEL.items()])


def plot_bars(summary, statistic, order, path):
    """Paired bars of *statistic* per species, in *order*."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), gridspec_kw=dict(
        left=0.3, right=0.97, bottom=0.3, top=0.9))
    sns.barplot(data=summary, x="species", y=statistic, hue="method", order=order,
                hue_order=list(METHOD_LABEL.values()), palette=COLORS, saturation=1,
                ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_xlabel("")
    ax.set_ylabel(STATISTICS[statistic])
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
              ncol=2, handlelength=1, columnspacing=1, title=None)
    sns.despine(ax=ax)
    save_figure(fig, path, tight=False)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = [summarize(m, args.run_suffix) for m in selected_models(args)]
    summary = [s for s in summary if s is not None]
    if not summary:
        raise SystemExit("No model had benchmark predictions — nothing written.")
    summary = pd.concat(summary, ignore_index=True)

    print("\n── Per-reaction NRMSE ──")
    print(summary.to_string(index=False, float_format="%.4g"))
    path = out_dir / "benchmark_nrmse.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")

    # Species from the flux transformer's lowest mean NRMSE to its highest.
    order = (summary[summary["method"] == METHOD_LABEL["ai"]]
             .sort_values("mean_nrmse")["species"].tolist())
    plot_bars(summary, "mean_nrmse", order, out_dir / "benchmark_mean_nrmse.svg")
    plot_bars(summary, "nrmse_q90", order, out_dir / "benchmark_nrmse_q90.svg")


if __name__ == "__main__":
    main()
