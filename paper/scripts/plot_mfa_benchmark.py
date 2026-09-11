"""
MFA-vs-SigmaFlux simulation benchmark — per-reaction R²/NRMSE distributions and Venn
diagrams.

Per model:
  mfa_benchmark_r2_<key>.svg               R² against the cumulative fraction of reactions
  mfa_benchmark_venn_<key>.svg             2-set Venn of reactions each method resolves
  mfa_benchmark_r2_per_reaction_<key>.csv  the R²/RMSE/NRMSE table behind both plots

Across all models:
  mfa_benchmark_identifiable_sets.csv      Venn set sizes for every model
  mfa_benchmark_nrmse_violin.svg           NRMSE distribution, all models side by side
  mfa_benchmark_nrmse_paired_tests.csv     Per-organism Wilcoxon signed-rank test, AI vs
                                            MFA per-reaction NRMSE, paired by reaction

R² curves and the NRMSE violin both plot the two methods on the same axis (SigmaFlux vs
MFA, ``METHOD_COLORS``); the R² x axis is the fraction of reactions at or above a given
R², sorted descending. A model whose benchmark data is missing is skipped with a warning
rather than failing the run, so the figure set fills itself in as runs land.

The violin reports NRMSE, RMSE over each reaction's root-mean-square ground-truth flux,
because raw RMSE carries the reaction's units: the median across organisms spans two and
a half orders of magnitude on flux magnitude alone, so four models cannot be read side by
side in it. The absolute RMSE stays in the per-reaction CSV. NRMSE = 1 is marked on the
violin: above it a reaction's error is as large as the flux it carries.

Every metric here (``per_flux_r2``/``per_flux_rmse``/``per_flux_nrmse`` in
plot_test_performance.py) is computed in plain linear space over the benchmark HDF5s
directly -- not read from output/identifiability.csv -- so this script has one source of
truth and does not depend on plot_test_performance.py having been run first against the
same run directory. They share the same chi2-accepted ("statistically acceptable") sample
set -- see ``_figure_common.benchmark_metrics`` -- so the R² and NRMSE panels for one
model describe the identical reaction population.

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
import matplotlib.transforms as mtransforms
from scipy import stats

from _figure_common import (
    COLORS, LABEL, METHOD_COLORS, R2_FLOOR, R2_THRESHOLD, add_common_args, apply_style,
    benchmark_metrics, save_figure, selected_models)

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
    """Tidy (Model, Method, nrmse) frame from tbl's nrmse_mfa/nrmse_ai columns."""
    long = (tbl[["nrmse_mfa", "nrmse_ai"]]
            .rename(columns={"nrmse_mfa": "mfa", "nrmse_ai": "ai"})
            .melt(var_name="method", value_name="nrmse"))
    long["Method"] = long["method"].map(LABEL)
    long["Model"] = model["label"]
    return long[["Model", "Method", "nrmse"]]


def plot_nrmse_violin(df, model_order, out_dir):
    """Cross-model NRMSE violins: 4 models x 2 methods = 8 violins, log-y.

    Width is the one explicit, user-approved exception to the 3.5x3.5 default
    (3.5 * 1.8 = 6.3in); height is unchanged.

    Normalising removes the *units* that made the absolute-RMSE violin incomparable
    across organisms, but not the dynamic range -- the per-reaction population still
    spans several decades -- so the log axis stays.

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
    # Above this line a reaction's RMSE is as large as the flux it typically carries.
    ax.axhline(0.0, color=COLORS["reference"], lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel(r"Per-reaction flux NRMSE (RMSE / rms$(v)$)")
    sns.move_legend(ax, "upper right", frameon=False, title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "mfa_benchmark_nrmse_violin.svg")


def paired_nrmse_test(model, tbl):
    """Wilcoxon signed-rank test, AI vs MFA per-reaction NRMSE, paired by reaction.

    Run on the same quantity the violin draws: the signed-rank statistic ranks the
    absolute differences, and normalising per reaction reorders them, so a test on
    absolute RMSE would not be the test of what is plotted.

    ``tbl``'s nrmse_mfa/nrmse_ai columns already share one reaction index --
    ``benchmark_metrics``'s trailing ``dropna()`` keeps only reactions both methods
    have a value for -- so reaction r's MFA and AI NRMSE are a genuine matched pair,
    not an independent-groups comparison. Returns None when there are too few reactions
    for scipy's own practical floor, or every pair ties exactly.
    """
    mfa, ai = tbl["nrmse_mfa"].to_numpy(), tbl["nrmse_ai"].to_numpy()
    diff = ai - mfa
    diff = diff[np.isfinite(diff)]
    if len(diff) < 6 or np.allclose(diff, 0):
        return None
    stat, p = stats.wilcoxon(diff)
    return {
        "model": model["label"],
        "n_reactions": len(diff),
        "median_nrmse_mfa": float(np.median(mfa)),
        "median_nrmse_ai": float(np.median(ai)),
        "median_rmse_mfa": float(tbl["rmse_mfa"].median()),
        "median_rmse_ai": float(tbl["rmse_ai"].median()),
        "lower_nrmse": LABEL["ai"] if np.median(diff) < 0 else LABEL["mfa"],
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
    }


def report_paired_tests(records, out_dir):
    """Print the per-organism paired NRMSE test and write it next to the figures."""
    print(f"\n── Paired NRMSE test (Wilcoxon signed-rank, {LABEL['ai']} vs {LABEL['mfa']}, "
          f"per reaction) ──")
    for r in records:
        print(f"\n{r['model']}   (n = {r['n_reactions']} reactions)")
        print(f"   median NRMSE — MFA {r['median_nrmse_mfa']:.4g}, "
              f"SigmaFlux {r['median_nrmse_ai']:.4g}")
        print(f"   lower NRMSE: {r['lower_nrmse']}"
              f"   (p = {r['wilcoxon_p']:.3g}, W = {r['wilcoxon_stat']:.1f})")
    path = out_dir / "mfa_benchmark_nrmse_paired_tests.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\nSaved: {path}")


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


def _venn_union_bbox(venn, region_ids):
    """Display-coordinate bounding box of every drawn Venn patch."""
    boxes = [v.get_path().transformed(v.get_transform()).get_extents()
             for v in (venn.get_patch_by_id(i) for i in region_ids) if v is not None]
    return mtransforms.Bbox.union(boxes)


def _move_label_out(ax, label, centre, bounds, pad=20.0):
    """Redraw one region count outside the diagram, with a leader line back to it.

    Placed along the ray from the diagram's centre through the label, at the point that
    ray leaves ``bounds`` plus ``pad`` points, so the two crescents' labels move apart
    rather than crossing.
    """
    anchor = label.get_position()
    direction = ax.transData.transform(anchor) - centre
    norm = np.hypot(*direction)
    direction = direction / norm if norm > 1.0 else np.array([0.0, -1.0])
    # How far along the ray the diagram's own bounding box ends.
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.nanmin(np.abs(np.where(
            direction == 0, np.inf,
            np.array([bounds.width, bounds.height]) / 2.0 / direction)))
    outside = ax.transData.inverted().transform(centre + direction * (reach + pad))

    label.set_visible(False)
    ax.annotate(label.get_text(), xy=anchor, xytext=outside, textcoords="data",
                fontsize=label.get_fontsize(), fontweight=label.get_fontweight(),
                color=label.get_color(), ha="center", va="center", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=COLORS["reference"],
                                shrinkA=3, shrinkB=1))


def _relocate_overflowing_labels(fig, ax, venn, region_ids):
    """Move every region count whose glyphs do not fit inside their own region outside it.

    Measured, not estimated: the rendered text box is tested corner by corner against
    the region's own path, so this fires exactly when the number overflows.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = _venn_union_bbox(venn, region_ids)
    centre = np.array([bounds.x0 + bounds.width / 2, bounds.y0 + bounds.height / 2])
    for region in region_ids:
        label, patch = venn.get_label_by_id(region), venn.get_patch_by_id(region)
        if label is None or patch is None or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        path = patch.get_path().transformed(patch.get_transform())
        if not path.contains_points(box.corners()).all():
            _move_label_out(ax, label, centre, bounds)


def plot_venn(model, counts, out_dir):
    """2-set Venn of reactions each method resolves (R² >= R2_THRESHOLD), drawn as an
    oval (short figure height + aspect='auto' -- venn2 otherwise forces a circular
    aspect) with no MFA/SigmaFlux set-name annotation, just the bold region counts.

    ``counts`` is ``set_counts()``'s output for this model, reused rather than
    recomputed. MFA-only and SigmaFlux-only regions use the same colours as the
    R² panels (``METHOD_COLORS``); the overlap is grey (``COLORS["reference"]``),
    matching the threshold/identity lines used elsewhere in this figure set.

    An empty region carries no label at all -- a "0" printed in a sliver reads as a
    region that exists but is small -- and a count too big for the region it belongs to
    moves outside on a leader line rather than spilling over its neighbours.
    ``n_reactions`` and ``neither`` live in mfa_benchmark_identifiable_sets.csv and the
    printed report, so nothing is lost by having no title here.
    """
    from matplotlib_venn import venn2

    regions = ("10", "01", "11")
    fig, ax = plt.subplots(figsize=(3.2 / 1.2, 2.0))
    v = venn2(subsets=(counts["mfa_only"], counts["sf_only"], counts["both"]),
              set_labels=("", ""), ax=ax,
              set_colors=(METHOD_COLORS["mfa"], METHOD_COLORS["ai"]),
              subset_label_formatter=lambda n: f"{int(n)}" if n else "")
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
    _relocate_overflowing_labels(fig, ax, v, regions)
    save_figure(fig, out_dir / f"mfa_benchmark_venn_{model['key']}.svg")


def drop_superseded(out_dir):
    """Remove the absolute-RMSE outputs this script no longer writes.

    A rename that leaves the old file in place leaves two figures in paper/figures/
    claiming to be the same panel, one of them silently stale.
    """
    stale = [out_dir / "mfa_benchmark_rmse_violin.svg",
             out_dir / "mfa_benchmark_rmse_paired_tests.csv",
             *out_dir.glob("mfa_benchmark_rmse_hist_*.svg")]
    for path in stale:
        if path.exists():
            path.unlink()
            print(f"Removed superseded: {path}")


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_superseded(out_dir)

    models = selected_models(args)

    print("── Per-reaction R²/NRMSE ──")
    records, nrmse_frames, model_order, paired_records = [], [], [], []
    for model in models:
        tbl = benchmark_metrics(model, args.run_suffix)
        if tbl is None:
            continue
        tbl.to_csv(out_dir / f"mfa_benchmark_r2_per_reaction_{model['key']}.csv")
        plot_r2_quantiles(model, tbl, out_dir)
        nrmse_frames.append(_nrmse_long(model, tbl))
        model_order.append(model["label"])
        counts = set_counts(model, tbl)
        records.append(counts)
        plot_venn(model, counts, out_dir)
        paired = paired_nrmse_test(model, tbl)
        if paired is not None:
            paired_records.append(paired)

    if nrmse_frames:
        plot_nrmse_violin(pd.concat(nrmse_frames, ignore_index=True), model_order, out_dir)

    if paired_records:
        report_paired_tests(paired_records, out_dir)

    if records:
        report_sets(records, out_dir)
    else:
        print("No model had benchmark data — nothing written.")


if __name__ == "__main__":
    main()
