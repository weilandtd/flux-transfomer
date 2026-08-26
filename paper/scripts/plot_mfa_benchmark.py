"""
MFA-vs-SigmaFlux simulation benchmark — per-reaction R² distributions and Venn diagrams.

Per model:
  mfa_benchmark_r2_<key>.svg               R² against the cumulative fraction of reactions
  mfa_benchmark_venn_<key>.svg             2-set Venn of reactions each method resolves
  mfa_benchmark_r2_per_reaction_<key>.csv  the R² table behind both plots

Across all models:
  mfa_benchmark_identifiable_sets.csv      Venn set sizes for every model

R² curves plot both methods on the same axis (SigmaFlux vs MFA, ``METHOD_COLORS``);
the x axis is the fraction of reactions at or above a given R², sorted descending.
A model whose benchmark data is missing is skipped with a warning rather than failing
the run, so the figure set fills itself in as runs land.

Every R² here (``per_flux_r2`` in plot_test_performance.py) is computed in plain linear
space over the benchmark HDF5s directly -- not read from output/identifiability.csv --
so this script has one source of truth and does not depend on plot_test_performance.py
having been run first against the same run directory.

Usage:
    python paper/scripts/plot_mfa_benchmark.py
    python paper/scripts/plot_mfa_benchmark.py --models s_cerevisiae,i_orientalis
    python paper/scripts/plot_mfa_benchmark.py --run-suffix 20260807
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    COLORS, LABEL, METHOD_COLORS, R2_FLOOR, R2_THRESHOLD, add_common_args,
    apply_style, benchmark_metrics, save_figure, selected_models)

apply_style()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_common_args(p).parse_args()


def plot_r2_quantiles(model, tbl, out_dir):
    """R² against the cumulative fraction of reactions, one line per method."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.axhline(R2_THRESHOLD, color=COLORS["reference"], lw=0.8, ls="--", zorder=0)

    frames = []
    for method in ("mfa", "ai"):
        r2 = np.sort(tbl[f"r2_{method}"].to_numpy())[::-1]
        frames.append(pd.DataFrame({
            # Sorted descending, so x is the fraction of reactions at or above this R².
            "fraction": (np.arange(len(r2)) + 1) / len(r2),
            "r2": np.clip(r2, R2_FLOOR, None),
            "method": method,
        }))
    df = pd.concat(frames, ignore_index=True)

    for method, grp in df.groupby("method", sort=False):
        sns.lineplot(data=grp, x="fraction", y="r2", ax=ax, label=LABEL[method],
                     color=METHOD_COLORS[method], lw=1.8,
                     errorbar=None, estimator=None)

    ax.set_xlim(0, 1)
    ax.set_ylim(R2_FLOOR, 1.0)
    ax.set_yticks([R2_FLOOR, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["≤−0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("Fraction of reactions")
    # Mathtext, not a literal Unicode superscript: Arial has no glyph for U+207B and
    # silently falls back to a "Last Resort" placeholder box in the saved SVG.
    ax.set_ylabel(r"Flux $R^2$")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_benchmark_r2_{model['key']}.svg")


def set_counts(model, tbl):
    """The Venn partition of reactions each method resolves (R² >= R2_THRESHOLD).

    Counted on the side where a method *works*, so ``mfa_only`` / ``sf_only`` / ``both``
    drop straight into a Venn diagram. ``neither`` is outside both circles and is
    reported because the three circle numbers alone do not say how many reactions exist.
    """
    mfa_ok, sf_ok = tbl["r2_mfa"] >= R2_THRESHOLD, tbl["r2_ai"] >= R2_THRESHOLD
    return {
        "model": model["label"],
        "n_reactions": len(tbl),
        "mfa_only": int((mfa_ok & ~sf_ok).sum()),
        "sf_only": int((~mfa_ok & sf_ok).sum()),
        "both": int((mfa_ok & sf_ok).sum()),
        "neither": int((~mfa_ok & ~sf_ok).sum()),
        "mfa_total": int(mfa_ok.sum()),
        "sf_total": int(sf_ok.sum()),
    }


def report_sets(records, out_dir):
    """Print the Venn numbers and write them next to the figures."""
    print(f"\n── Venn sets: reactions resolved at R² >= {R2_THRESHOLD} ──")
    for r in records:
        print(f"\n{r['model']}   (n = {r['n_reactions']} reactions)")
        print(f"   MFA only     : {r['mfa_only']:4d}")
        print(f"   SigmaFlux only: {r['sf_only']:4d}")
        print(f"   Both         : {r['both']:4d}")
        print(f"   Neither      : {r['neither']:4d}   (outside both circles)")
        print(f"   circle totals — MFA {r['mfa_total']}, SigmaFlux {r['sf_total']}")
    path = out_dir / "mfa_benchmark_identifiable_sets.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\nSaved: {path}")


def plot_venn(model, counts, out_dir):
    """2-set Venn of reactions each method resolves (R² >= R2_THRESHOLD), drawn as an
    oval (short figure height + aspect='auto' -- venn2 otherwise forces a circular
    aspect) with no MFA/SigmaFlux set-name annotation, just the bold region counts.

    ``counts`` is ``set_counts()``'s output for this model, reused rather than
    recomputed. MFA-only and SigmaFlux-only regions use the same colours as the
    R² panels (``METHOD_COLORS``); the overlap is grey (``COLORS["reference"]``),
    matching the threshold/identity lines used elsewhere in this figure set.
    """
    from matplotlib_venn import venn2

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    v = venn2(subsets=(counts["mfa_only"], counts["sf_only"], counts["both"]),
              set_labels=("", ""), ax=ax,
              set_colors=(METHOD_COLORS["mfa"], METHOD_COLORS["ai"]))
    ax.set_aspect("auto")   # squash the (otherwise forced-circular) venn into an oval
    overlap = v.get_patch_by_id("11")
    if overlap is not None:
        overlap.set_color(COLORS["reference"])
    for patch in (v.get_patch_by_id("10"), v.get_patch_by_id("01"), overlap):
        if patch is not None:
            patch.set_edgecolor("none")
            patch.set_alpha(0.85)
    for label in v.subset_labels:
        if label is not None:
            label.set_fontsize(16)
            label.set_fontweight("bold")
    ax.set_title(f"{model['label']}\n(n = {counts['n_reactions']} reactions, "
                 f"{counts['neither']} neither)", fontsize=8)
    save_figure(fig, out_dir / f"mfa_benchmark_venn_{model['key']}.svg")


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = selected_models(args)

    print("── Per-reaction R² ──")
    records = []
    for model in models:
        tbl = benchmark_metrics(model, args.run_suffix)
        if tbl is None:
            continue
        tbl.to_csv(out_dir / f"mfa_benchmark_r2_per_reaction_{model['key']}.csv")
        plot_r2_quantiles(model, tbl, out_dir)
        counts = set_counts(model, tbl)
        records.append(counts)
        plot_venn(model, counts, out_dir)

    if records:
        report_sets(records, out_dir)
    else:
        print("No model had benchmark data — nothing written.")


if __name__ == "__main__":
    main()
