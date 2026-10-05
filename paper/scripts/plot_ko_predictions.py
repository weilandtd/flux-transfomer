"""
SigmaFlux on the Long & Antoniewicz (2019) E. coli knockout panel, with and without FBApro.

  ko_r2.svg              SigmaFlux R² against the published MFA fit, one bar per
                         strain, darker where the deletion removes a core reaction
  ko_flux_<gene>.svg     one narrow panel per gene experiment.knockouts pins: its
                         reaction's flux in wild type and knockout, from the published MFA,
                         SigmaFlux and SigmaFlux + FBApro (knockout only), with intervals
  ko_labeling.svg        R² of the labeling the SF and SF + FBApro flux states imply
                         against the measured MIDs, over all strains (from
                         ko_labeling.py; a state whose EMU solve fails is left out)
  ko_labeling_<gene>.svg the same in wild type and each pinned gene's knockout, laid out
                         like ko_flux_<gene>.svg
                         (n/a where the EMU solve fails)
  ko_r2.csv, ko_flux.csv, ko_labeling.csv
                         the numbers behind them
  ko_flux_tests.csv      the within-group tests behind the flux annotations

Within each group every prediction is tested against the published MFA value with a
two-sided z-test on the difference, each side's standard error read off its 95% interval
as (hi - lo) / 3.92. A pinned FBApro zero carries no uncertainty of its own, so its test
rests on the MFA interval alone. Stars: * p < 0.05, ** p < 0.01, *** p < 0.001.

Reads the two sets of sigmaflux-infer output the GC/MS E. coli core run carries:
``predictions/`` (plain) and ``predictions_fbapro/`` (``--fbapro``, which projects every
member draw onto S.v = 0 with the config's experiment.knockouts pinned to zero). Run
both first:

    bash paper/submit/submit_inference.sh paper/.results/e_coli_core_gc_<date>
    FBAPRO=1 bash paper/submit/submit_inference.sh paper/.results/e_coli_core_gc_<date>

R² is scored per strain on the reactions SigmaFlux predicts with R² > 0.9 in its own
simulation benchmark (``<run>/ai_benchmark``), with the same linear, signed definition as
plot_experimental_predictions.py. Intervals are the ones inference reports; the published
ones are the fit's 95% bounds.

Usage:
    python paper/scripts/plot_ko_predictions.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from _figure_common import (UNBOUNDED_FLUX, _REPO_ROOT, apply_style, drop_pool_reactions,
                            load_benchmark, load_data, per_flux_r2, reference_path,
                            resolve_organism, save_figure)
from plot_experimental_predictions import _r2_linear
from sigmaflux.utils import (pair_fluxes, read_predicted_fluxes, read_reference_fluxes,
                             reference_orientation)

RUN = _REPO_ROOT / "paper/.results/e_coli_core_gc_20260924"
CONDITION = "B"
BENCHMARK_R2 = 0.9
#: Prediction directory per method.
METHODS = {"sigmaflux": "predictions", "fbapro": "predictions_fbapro"}

COLORS = {
    "published": "#CCCCCC",
    "sigmaflux": "#839E9E",
    "unpinned":  "#D3DEDE",
    "fbapro":    "#315E5D",
    "reference": "#AAAAAA",
    "error":     "#333333",
}
#: Legend label per flux source, in plotting order.
SOURCES = {"published": "MFA", "sigmaflux": "SF", "fbapro": "SF + FBApro"}
WILD_TYPE_GC = 1

apply_style()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=pathlib.Path, default=RUN)
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=_REPO_ROOT / "paper" / "figures")
    return p.parse_args()


def benchmark_reactions(run):
    """Reactions SigmaFlux predicts with R² above BENCHMARK_R2 on simulated data."""
    _, flux_df, _ = load_data(str(run))
    paired, _ = load_benchmark(run / "ai_benchmark", drop_pool_reactions(flux_df, run))
    r2 = per_flux_r2(paired)
    return set(r2.index[r2 > BENCHMARK_R2])


def predictions(run, method):
    """``{'value', 'lo', 'hi'}`` predicted frames from the method's latest CSV."""
    path = sorted((run / METHODS[method]).glob(f"predictions_*_{CONDITION}_*.csv"))[-1]
    return {key: read_predicted_fluxes(path, col)
            for key, col in (("value", "Value"), ("lo", "Value_lo"), ("hi", "Value_hi"))}


def paired(run, org, reactions=None):
    """Signed pred-vs-published frame for both methods, zero fluxes kept."""
    path = reference_path(org)
    ref = {key: read_reference_fluxes(path, CONDITION, suffix, unbounded=UNBOUNDED_FLUX)
           for key, suffix in (("value", "_mfa"), ("lo", "_mfaLB"), ("hi", "_mfaUB"))}
    orientation = reference_orientation(org.get("media"))
    return pd.concat([pair_fluxes(predictions(run, m), ref, condition=CONDITION,
                                  orientation=orientation, reactions=reactions,
                                  min_abs=-np.inf).assign(method=m)
                      for m in METHODS], ignore_index=True)


def strain_names(org):
    index = pd.read_csv(_REPO_ROOT / org["data"] / "strain_index.csv")
    return {int(c[len(CONDITION):]): s for c, s in zip(index["condition"], index["strain"])}


def plot_r2(r2, pinned, out_dir):
    """SigmaFlux R² bar per strain; *pinned* is the set of growth conditions with knockouts.

    A strain whose deletion removes a core reaction is drawn in the SF color, the rest in
    a paler tint.
    """
    r2 = r2.assign(removed=r2["gc"].isin(pinned)).sort_values("gc")
    fig, ax = plt.subplots(figsize=(3.5, 6))
    sns.barplot(data=r2, y="strain", x="r2", hue="removed", orient="h", dodge=False,
                palette={True: COLORS["sigmaflux"], False: COLORS["unpinned"]}, ax=ax)
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8)
    ax.set_ylabel("")
    ax.set_xlabel("R² vs published MFA")
    for label in ax.get_yticklabels():
        label.set(fontstyle="italic")
    handles, _ = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], ["Core reaction deleted", "No core reaction deleted"],
                       frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.1))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "ko_r2.svg", extra_artists=[legend])


def z_test(a, b):
    """Two-sided p-value that *a* and *b* (rows with value/lo/hi, 95% intervals) differ."""
    se = np.hypot((a["hi"] - a["lo"]) / (2 * 1.96), (b["hi"] - b["lo"]) / (2 * 1.96))
    return 2 * norm.sf(abs(a["value"] - b["value"]) / se) if se > 0 else np.nan


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    return next((mark for cut, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*"))
                 if p < cut), "ns")


def flux_tests(flux):
    """Every prediction against the published MFA value of the same gene and group."""
    rows = []
    for (gene, wild_type), g in flux.groupby(["ko", "wild_type"], sort=False):
        g = g.set_index("source")
        for source in [s for s in SOURCES if s != "published" and s in g.index]:
            p = z_test(g.loc["published"], g.loc[source])
            rows.append({"ko": gene, "wild_type": wild_type, "source": source,
                         "p": p, "stars": stars(p)})
    return pd.DataFrame(rows)


def plot_ko_flux(flux, tests, gene, out_dir):
    """*gene*'s reaction flux in wild type and knockout, one bar per source."""
    groups = {True: "WT", False: "KO"}
    width = 0.8 / len(SOURCES)
    fig, ax = plt.subplots(figsize=(1.5, 3.5))
    top = np.nanmax(flux[["value", "hi"]].to_numpy())
    step = 0.08 * top
    for g, wild_type in enumerate(groups):
        d = flux[flux["wild_type"] == wild_type].set_index("source")
        present = [src for src in SOURCES if src in d.index]
        x = {src: g + (i - (len(present) - 1) / 2) * width for i, src in enumerate(present)}
        for source in present:
            row = d.loc[source]
            ax.bar(x[source], row["value"], width=width, color=COLORS[source],
                   label=SOURCES[source] if not wild_type else None,
                   yerr=[[row["value"] - row["lo"]], [row["hi"] - row["value"]]],
                   error_kw=dict(ecolor=COLORS["error"], capsize=2, linewidth=0.8))
            # A pinned FBApro flux has no bar to see; print its value instead.
            if source == "fbapro" and row["value"] == 0.0:
                ax.text(x[source], step / 4, "0", ha="center", va="bottom", fontsize=12,
                        color=COLORS[source])
        # One bracket per test, from MFA to the prediction, stacked above the group.
        y = np.nanmax(d[["value", "hi"]].to_numpy()) + step
        group_tests = tests[tests["wild_type"] == wild_type].set_index("source")
        for source in [s for s in present if s in group_tests.index]:
            x0, x1 = x["published"], x[source]
            ax.plot([x0, x0, x1, x1], [y, y + step / 3, y + step / 3, y],
                    color=COLORS["error"], linewidth=0.8)
            ax.text((x0 + x1) / 2, y + step / 3, group_tests.loc[source, "stars"],
                    ha="center", va="bottom", fontsize=9)
            y += 1.5 * step
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, zorder=0)
    ax.set_xticks(np.arange(len(groups)), list(groups.values()))
    ax.set_ylabel("Flux (mmol/gDW/h)")
    name = ax.text(0.5, 1.02, gene, transform=ax.transAxes, ha="center", va="bottom",
                   fontstyle="italic")
    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"ko_flux_{gene}.svg", extra_artists=[name, legend])


def labeling_r2(run):
    """Labeling R² per strain and method, from ko_labeling.py's token table.

    A state whose EMU solve failed has no labeling and gets NaN.
    """
    tokens = pd.read_csv(run / "ko_labeling" / "tokens.csv")
    expected = pd.MultiIndex.from_frame(tokens[["gc", "method"]].drop_duplicates())
    valid = tokens.dropna(subset=["sim"])
    r2 = (valid.groupby(["gc", "method"])
               .apply(lambda d: _r2_linear(d["sim"], d["measured"]), include_groups=False))
    return r2.reindex(expected).unstack("method")


def plot_labeling_summary(r2, out_dir):
    """Labeling R² over the strains, SF and SF + FBApro side by side.

    Bars are the median over strains with the interquartile range; dots are strains.
    """
    fig, ax = plt.subplots(figsize=(1.5, 3.5))
    for i, method in enumerate(METHODS):
        values = r2[method].dropna()
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        ax.bar(i, median, width=0.7, color=COLORS[method], label=SOURCES[method],
               yerr=[[median - q1], [q3 - median]],
               error_kw=dict(ecolor=COLORS["error"], capsize=2, linewidth=0.8))
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(values))
        ax.scatter(i + jitter, values, s=6, color=COLORS["error"], zorder=3,
                   linewidths=0)
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(METHODS) - 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Labeling R² vs measured")
    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "ko_labeling.svg", extra_artists=[legend])


def plot_ko_labeling(r2, gc, gene, out_dir):
    """Labeling R² in wild type and *gene*'s knockout, one bar per method."""
    groups = {WILD_TYPE_GC: "WT", gc: "KO"}
    width = 0.8 / len(METHODS)
    fig, ax = plt.subplots(figsize=(1.5, 3.5))
    for g, strain in enumerate(groups):
        # FBApro pins nothing in wild type.
        present = ["sigmaflux"] if strain == WILD_TYPE_GC else list(METHODS)
        for i, method in enumerate(present):
            x = g + (i - (len(present) - 1) / 2) * width
            value = r2.loc[strain, method]
            ax.bar(x, 0.0 if np.isnan(value) else value, width=width,
                   color=COLORS[method],
                   label=SOURCES[method] if strain != WILD_TYPE_GC else None)
            if np.isnan(value):
                # No solvable labeling for this flux state.
                ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=12,
                        color=COLORS[method])
    ax.set_xticks(np.arange(len(groups)), list(groups.values()))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Labeling R² vs measured")
    name = ax.text(0.5, 1.02, gene, transform=ax.transAxes, ha="center", va="bottom",
                   fontstyle="italic")
    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"ko_labeling_{gene}.svg", extra_artists=[name, legend])


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    org = resolve_organism(args.run)
    strains = strain_names(org)

    reactions = benchmark_reactions(args.run)
    print(f"{len(reactions)} reactions with benchmark R² > {BENCHMARK_R2}")
    scored = paired(args.run, org, reactions)
    scored = scored[scored["method"] == "sigmaflux"]
    r2 = (scored.groupby("gc")
                .apply(lambda g: pd.Series({"r2": _r2_linear(g["pred"], g["ref"]),
                                            "n": len(g)}), include_groups=False)
                .reset_index()
                .assign(strain=lambda d: d["gc"].map(strains)))
    r2.to_csv(out_dir / "ko_r2.csv", index=False)
    print(r2.round(3).to_string(index=False))
    knockouts = org["cfg"]["experiment"]["knockouts"]
    plot_r2(r2, {int(c[len(CONDITION):]) for c in knockouts}, out_dir)

    every = paired(args.run, org)
    rows = []
    for cond, rxns in knockouts.items():
        gc = int(cond[len(CONDITION):])
        for rxn in rxns:
            for strain_gc, wild_type in ((gc, False), (WILD_TYPE_GC, True)):
                hits = every[(every["gc"] == strain_gc) & (every["reaction"] == rxn)]
                if wild_type:
                    hits = hits[hits["method"] == "sigmaflux"]   # FBApro pins nothing in WT
                common = {"ko": strains[gc], "reaction": rxn, "wild_type": wild_type}
                ref = hits.iloc[0]
                rows.append({**common, "source": "published", "value": ref["ref"],
                             "lo": ref["ref_lo"], "hi": ref["ref_hi"]})
                for _, h in hits.iterrows():
                    rows.append({**common, "source": h["method"], "value": h["pred"],
                                 "lo": h["pred_lo"], "hi": h["pred_hi"]})
    flux = pd.DataFrame(rows)
    flux.to_csv(out_dir / "ko_flux.csv", index=False)
    print(flux.pivot_table(index="ko", columns=["wild_type", "source"], values="value",
                           sort=False).round(3).to_string())
    tests = flux_tests(flux)
    tests.to_csv(out_dir / "ko_flux_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))
    for gene, gene_flux in flux.groupby("ko", sort=False):
        plot_ko_flux(gene_flux, tests[tests["ko"] == gene], gene, out_dir)

    labeling = labeling_r2(args.run)
    labeling.assign(strain=labeling.index.map(strains)).to_csv(out_dir / "ko_labeling.csv")
    print(labeling.round(3).to_string())
    plot_labeling_summary(labeling, out_dir)
    for cond in knockouts:
        gc = int(cond[len(CONDITION):])
        plot_ko_labeling(labeling, gc, strains[gc], out_dir)


if __name__ == "__main__":
    main()
