"""
MFA-vs-SigmaFlux simulation benchmark — per-reaction R²/NRMSE distributions and Venn
diagrams.

Per model:
  mfa_benchmark_r2_<key>.svg               R² against the cumulative fraction of reactions
  mfa_benchmark_nrmse_hist_<key>.svg       Per-reaction NRMSE distribution, log-x histogram
  mfa_benchmark_venn_<key>.svg             2-set Venn of reactions each method resolves
  mfa_benchmark_r2_per_reaction_<key>.csv  the R²/NRMSE table behind all three plots

Across all models:
  mfa_benchmark_identifiable_sets.csv      Venn set sizes for every model
  mfa_benchmark_nrmse_violin.svg           NRMSE distribution, all models side by side

R² curves and the NRMSE histogram/violin all plot both methods on the same axis
(SigmaFlux vs MFA, ``METHOD_COLORS``); the R² x axis is the fraction of reactions at or
above a given R², sorted descending. A model whose benchmark data is missing is skipped
with a warning rather than failing the run, so the figure set fills itself in as runs land.

Every R²/NRMSE here (``per_flux_r2``/``per_flux_nrmse`` in plot_test_performance.py) is
computed in plain linear space over the benchmark HDF5s directly -- not read from
output/identifiability.csv -- so this script has one source of truth and does not depend
on plot_test_performance.py having been run first against the same run directory. NRMSE
z-scores each reaction's GT to mean 0/std 1 across samples, applies that same transform
to pred, and takes the RMSE of the two -- equivalently RMSE(pred,gt) / std(gt), and
therefore exactly sqrt(1 - R2): a deliberate reparametrisation of the R² panel, not
independent information, so the histogram and violin reads as "how far below the R²
threshold, in an absolute, cross-reaction-comparable unit" rather than a second metric.
(Not the same NRMSE as ``plot_uncertainty_benchmark.py``, which normalises by each
reaction's own GT *range* instead -- the two scripts answer different questions and
intentionally use different denominators.)
Both metrics share the same chi2-accepted ("statistically acceptable") sample set -- see
``_figure_common.benchmark_metrics`` -- so the R² and NRMSE panels for one model describe
the identical reaction population.

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
    COLORS, FLUX_MIN_ABS, LABEL, METHOD_COLORS, R2_FLOOR, R2_THRESHOLD, add_common_args,
    apply_style, benchmark_metrics, save_figure, selected_models)

apply_style()

METHOD_ORDER = ("ai", "mfa")
METHOD_PALETTE = {LABEL[k]: METHOD_COLORS[k] for k in METHOD_ORDER}


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


def _nrmse_long(model, tbl):
    """Tidy (Model, Method, nrmse) frame from tbl's nrmse_mfa/nrmse_ai columns.

    Floored at FLUX_MIN_ABS -- reused here purely as a tiny positive epsilon (NRMSE is
    dimensionless, not a flux), same idiom as R2_FLOOR for R2 -- so the log-x histogram
    and the log10-transformed violin never see an exact zero.
    """
    long = (tbl[["nrmse_mfa", "nrmse_ai"]]
            .rename(columns={"nrmse_mfa": "mfa", "nrmse_ai": "ai"})
            .melt(var_name="method", value_name="nrmse"))
    long["nrmse"] = np.clip(long["nrmse"].to_numpy(), FLUX_MIN_ABS, None)
    long["Method"] = long["method"].map(LABEL)
    long["Model"] = model["label"]
    return long[["Model", "Method", "nrmse"]]


def plot_nrmse_histogram(model, long, out_dir):
    """Per-reaction flux NRMSE, SigmaFlux vs MFA, one model, log-x histogram."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(data=long, x="nrmse", hue="Method", hue_order=[LABEL["ai"], LABEL["mfa"]],
                palette=METHOD_PALETTE, log_scale=True, element="step", alpha=0.55, ax=ax)
    ax.set_xlabel("Per-reaction flux NRMSE")
    ax.set_ylabel("Number of reactions")
    sns.move_legend(ax, "upper right", frameon=False, title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_benchmark_nrmse_hist_{model['key']}.svg")


def plot_nrmse_violin(df, model_order, out_dir):
    """Cross-model NRMSE violins: 4 models x 2 methods = 8 violins, log-y.

    Width is the one explicit, user-approved exception to the 3.5x3.5 default
    (3.5 * 1.8 = 6.3in); height is unchanged.

    Drawn on log10(nrmse), not on nrmse with ax.set_yscale('log') after the fact:
    seaborn's KDE is fit in whatever coordinates it's given, so fitting on raw NRMSE and
    rescaling the axis afterwards would still smooth with a linear-space bandwidth and
    distort the shape wherever the (roughly log-normal) density isn't uniform on that
    scale. Fitting on log10(nrmse) directly makes the displayed shape the actual density
    in the space it's drawn in; ticks are relabelled as 10^n.
    """
    df = df.copy()
    df["log10_nrmse"] = np.log10(df["nrmse"])
    fig, ax = plt.subplots(figsize=(3.5 * 1.8, 3.5))
    sns.violinplot(data=df, x="Model", y="log10_nrmse", hue="Method", order=model_order,
                   hue_order=[LABEL["ai"], LABEL["mfa"]], palette=METHOD_PALETTE,
                   cut=0, linewidth=0.8, ax=ax)
    lo, hi = np.floor(df["log10_nrmse"].min()), np.ceil(df["log10_nrmse"].max())
    ax.set_yticks(np.arange(lo, hi + 1))
    ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in np.arange(lo, hi + 1)])
    ax.set_xlabel("")
    ax.set_ylabel("Per-reaction flux NRMSE")
    sns.move_legend(ax, "upper right", frameon=False, title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "mfa_benchmark_nrmse_violin.svg")


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

    print("── Per-reaction R²/NRMSE ──")
    records, nrmse_frames, model_order = [], [], []
    for model in models:
        tbl = benchmark_metrics(model, args.run_suffix)
        if tbl is None:
            continue
        tbl.to_csv(out_dir / f"mfa_benchmark_r2_per_reaction_{model['key']}.csv")
        plot_r2_quantiles(model, tbl, out_dir)
        long = _nrmse_long(model, tbl)
        plot_nrmse_histogram(model, long, out_dir)
        nrmse_frames.append(long)
        model_order.append(model["label"])
        counts = set_counts(model, tbl)
        records.append(counts)
        plot_venn(model, counts, out_dir)

    if nrmse_frames:
        plot_nrmse_violin(pd.concat(nrmse_frames, ignore_index=True), model_order, out_dir)

    if records:
        report_sets(records, out_dir)
    else:
        print("No model had benchmark data — nothing written.")


if __name__ == "__main__":
    main()
