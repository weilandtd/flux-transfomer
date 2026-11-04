"""
Geometric mean relative error of each method against the ground truth of the simulation
benchmark, one pair of bars per species.

  benchmark_gmean_error.svg   geometric mean of |pred - gt| / |gt| per (species, method)
  benchmark_gmean_error.csv   the values behind the bars, with point counts

Each method is scored on every test sample it returned a prediction for -- the same
``load_benchmark`` pairs as the per-reaction R² histograms, measurement-pool reactions
dropped. Points with |gt| at or below FLUX_MIN_ABS have no relative error and are
dropped; an error below FLUX_MIN_ABS is raised to it, since an exact match has no
logarithm.

Usage:
    python paper/scripts/plot_benchmark_error.py
    python paper/scripts/plot_benchmark_error.py --models s_cerevisiae,i_orientalis
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    FLUX_MIN_ABS, METHOD_COLORS, ORGANISM_SHORT, add_common_args, apply_style,
    drop_pool_reactions, load_benchmark, load_data, run_dir_for, save_figure,
    selected_models)

apply_style()

METHOD_LABEL = {"ai": "Flux transformer", "mfa": "Traditional MFA"}

COLORS = {
    METHOD_LABEL["ai"]:  METHOD_COLORS["ai"],    # teal
    METHOD_LABEL["mfa"]: METHOD_COLORS["mfa"],   # amber
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    return p.parse_args()


def rel_errors(model, run_suffix=None):
    """Per-(sample, reaction) |pred - gt| / |gt| of each method, or None when the model has
    no benchmark."""
    run = run_dir_for(model, run_suffix)
    if not all(list(run.glob(f"{m}_benchmark/pred_*_flux.h5")) for m in METHOD_LABEL):
        print(f"  {model['label']}: skipped — no benchmark predictions")
        return None
    _, flux_df, _ = load_data(str(run))
    flux_df.columns = range(flux_df.shape[1])
    flux_df = drop_pool_reactions(flux_df, run)

    frames = []
    for method, label in METHOD_LABEL.items():
        df, _ = load_benchmark(run / f"{method}_benchmark", flux_df)
        keep = df["gt"].abs() > FLUX_MIN_ABS
        df = df[keep]
        error = (df["pred"] - df["gt"]).abs() / df["gt"].abs()
        print(f"  {model['label']:20s} {label:16s} {len(df)} point(s) over "
              f"{df['sample'].nunique()} samples, {int((~keep).sum())} with |gt| at or "
              f"below {FLUX_MIN_ABS:g} dropped, {int((error < FLUX_MIN_ABS).sum())} "
              f"errors raised to it")
        frames.append(pd.DataFrame({
            "species": ORGANISM_SHORT.get(model["key"], model["label"]),
            "method": label,
            "rel_error": error.clip(lower=FLUX_MIN_ABS),
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    points = [rel_errors(m, args.run_suffix) for m in selected_models(args)]
    points = [p for p in points if p is not None]
    if not points:
        raise SystemExit("No model had benchmark predictions — nothing written.")
    points = pd.concat(points, ignore_index=True)

    summary = (points.groupby(["species", "method"], sort=False)["rel_error"]
               .agg(gmean_rel_error=lambda e: 10 ** np.log10(e).mean(), n_points="size")
               .reset_index())
    print("\n── Geometric mean |pred - gt| / |gt| ──")
    print(summary.to_string(index=False, float_format="%.4g"))
    path = out_dir / "benchmark_gmean_error.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")

    # Species from the flux transformer's lowest error to its highest.
    order = (summary[summary["method"] == METHOD_LABEL["ai"]]
             .sort_values("gmean_rel_error")["species"].tolist())
    fig, ax = plt.subplots(figsize=(3.5, 3.5), gridspec_kw=dict(
        left=0.3, right=0.97, bottom=0.3, top=0.97))
    sns.barplot(data=summary, x="species", y="gmean_rel_error", hue="method",
                order=order, hue_order=list(METHOD_LABEL.values()),
                palette=COLORS, saturation=1, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_xlabel("")
    ax.set_ylabel("Geometric mean\nrelative error")
    ax.legend(frameon=False, fontsize=9, loc="upper left", title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "benchmark_gmean_error.svg", tight=False)


if __name__ == "__main__":
    main()
