"""
SigmaFlux on the Long & Antoniewicz (2019) E. coli knockout panel, with and without FBApro.

  ko_r2.svg              SigmaFlux R² against the published MFA fit, one bar per
                         strain whose gene is in the network (GENE_REACTIONS), sorted by
                         R², dark where the deletion knocks down a core reaction, light
                         where an isozyme takes over
  ko_flux_scatter.svg    each deleted gene's reaction (GENE_REACTIONS): its SF flux in
                         wild type against in the knockout on a linear axis broken at the
                         same place on x and y, with intervals,
                         dark for a reaction knockdown (belongs on y = 0), light for an
                         isozyme (belongs near y = x)
  ko_labeling_<gene>.svg predicted against measured isotopologue fraction in the
                         knockouts of LABELING_GENES, SF and SF + FBApro
  ko_labeling.svg        R² of the labeling the SF and SF + FBApro flux states imply
                         against the measured MIDs, over all strains (from
                         ko_labeling.py; a state whose EMU solve fails is left out)
  ko_labeling_r2.svg     the same labeling R² for each reaction-knockdown strain, SF and
                         SF + FBApro side by side, sorted by SF R²; a failed EMU solve
                         is marked n/a
  ko_r2.csv, ko_flux.csv, ko_labeling.csv
                         the numbers behind them

Reads the two sets of sigmaflux-infer output the GC/MS E. coli core run carries:
``predictions/`` (plain) and ``predictions_fbapro/`` (``--fbapro``, which projects every
member draw onto S.v = 0 with the config's experiment.knockouts pinned to zero). Run
both first:

    bash paper/submit/submit_inference.sh paper/.results/e_coli_core_gc_<date>
    FBAPRO=1 bash paper/submit/submit_inference.sh paper/.results/e_coli_core_gc_<date>

R² is scored per strain on the reactions SigmaFlux predicts with R² > 0.9 in its own
simulation benchmark (``<run>/ai_benchmark``), with the same linear, signed definition as
plot_experimental.py. Intervals are the ones inference reports; the published
ones are the fit's 95% bounds.

Usage:
    python paper/scripts/plot_ko_predictions.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from _figure_common import (METHOD_COLORS, UNBOUNDED_FLUX, _REPO_ROOT, apply_style,
                            drop_pool_reactions, load_benchmark, load_data, per_flux_r2,
                            r2_linear, reference_path, resolve_organism, save_figure)
from sigmaflux.utils import (pair_fluxes, read_predicted_fluxes, read_reference_fluxes,
                             reference_orientation)

RUN = _REPO_ROOT / "paper/.results/e_coli_core_gc_20260924"
CONDITION = "B"
BENCHMARK_R2 = 0.9
#: Prediction directory per method.
METHODS = {"sigmaflux": "predictions", "fbapro": "predictions_fbapro"}

COLORS = {
    "knockdown": METHOD_COLORS["ai"],   # flux transformer teal
    "isozyme":   "#A9CFCB",             # light teal
    "sigmaflux": METHOD_COLORS["ai"],
    "fbapro":    "#A9CFCB",
    "reference": "#AAAAAA",
    "error":     "#333333",
}
#: Legend label per R² bar category, in legend order.
CATEGORIES = {"knockdown": "Reaction knockdown",
              "isozyme": "Alternative isozyme",
              "wild_type": "Wild type"}
#: Legend label per prediction method.
SOURCES = {"sigmaflux": "FT", "fbapro": "FT + FBApro"}
WILD_TYPE_GC = 1
UNITS = r"[mmol gDW$^{-1}$ h$^{-1}$]"
#: R² bar plot size and margins (figure fractions); the scatter matches its axes height.
R2_FIGSIZE = (3.5, 6)
R2_MARGINS = dict(left=0.2, right=0.95, bottom=0.3, top=0.98)
#: Scatter margins in inches: room for tick and axis labels on the left and bottom.
SCATTER_MARGINS_IN = dict(left=0.62, right=0.1, bottom=0.58, top=0.1)
#: Knockouts SF alone leaves clearly above zero; each gets a labeling scatter.
LABELING_GENES = ("tpiA", "gnd", "zwf")
# Label offset (points) of each knockdown in the flux scatter, clear of its neighbours.
KNOCKDOWN_LABEL_OFFSET = {"pgl": (4, 2), "gnd": (-14, 4), "zwf": (4, -6),
                          "edd": (-16, 3), "eda": (4, -6)}
#: Gene-reaction rules of the e_coli_core (BiGG) reactions behind each core reaction id;
#: R8 lumps G6PDH (zwf) with the lactonase (pgl). pgm (G6P <-> G1P) and glk (lumped into
#: gluc_up) have no reaction of their own in the network, so neither appears.
GPR = {
    "R1": "pgi", "R2": "pfkA or pfkB", "R2x": "fbp or glpX", "R3": "fbaA or fbaB",
    "R4": "tpiA", "R8": "zwf and pgl", "R9": "gnd", "R10": "rpe or sgcE",
    "R11": "rpiA or rpiB", "R12": "tktA or tktB", "R13": "talA or talB",
    "R14": "tktA or tktB", "R15": "edd", "R16": "eda",
}
#: Reactions each gene takes part in.
GENE_REACTIONS = {}
for _rxn, _rule in GPR.items():
    for _gene in _rule.split():
        if _gene not in ("and", "or"):
            GENE_REACTIONS.setdefault(_gene, []).append(_rxn)


def is_knockdown(gene):
    """Whether deleting *gene* alone switches off a reaction it takes part in."""
    genes = {g: g != gene for g in GENE_REACTIONS}
    return any(not eval(GPR[rxn], {}, genes) for rxn in GENE_REACTIONS[gene])

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


def plot_r2(r2, knockdown, out_dir):
    """SigmaFlux R² bar per strain in the network, sorted by R²; *knockdown* is the set of
    growth conditions whose deletion knocks down a core reaction."""
    r2 = r2[(r2["gc"] == WILD_TYPE_GC) | r2["strain"].isin(GENE_REACTIONS)]
    category = np.where(r2["gc"] == WILD_TYPE_GC, "wild_type",
                        np.where(r2["gc"].isin(knockdown), "knockdown", "isozyme"))
    r2 = r2.assign(category=category).sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=R2_FIGSIZE, gridspec_kw=R2_MARGINS)
    sns.barplot(data=r2, y="strain", x="r2", hue="category", hue_order=list(CATEGORIES),
                orient="h", dodge=False, saturation=1,
                palette={"knockdown": COLORS["knockdown"], "isozyme": COLORS["isozyme"],
                         "wild_type": COLORS["reference"]}, ax=ax)
    ax.axvline(0.0, color=COLORS["reference"], linewidth=0.8)
    ax.set_ylabel("")
    ax.set_xlabel("Flux transformer vs.\n" r"published MFA $R^2$")
    for label in ax.get_yticklabels():
        label.set(fontstyle="italic")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, list(CATEGORIES.values()), frameon=False, fontsize=9,
              loc="upper left", bbox_to_anchor=(-0.2, -0.19))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "ko_r2.svg", tight=False)


def ko_flux(run, strains, knockdown):
    """SF flux of each deleted gene's reaction in wild type and in its knockout."""
    pred = predictions(run, "sigmaflux")
    gc_of = {strain: gc for gc, strain in strains.items()}
    rows = []
    # GPR also names partner genes outside the deletion panel (fbaA, glpX, rpiA).
    for gene, rxns in GENE_REACTIONS.items():
        if gene not in gc_of:
            continue
        gc = gc_of[gene]
        for rxn in rxns:
            row = {"gene": gene, "reaction": rxn, "knockdown": gc in knockdown}
            for prefix, strain_gc in (("wt", WILD_TYPE_GC), ("ko", gc)):
                for key, suffix in (("value", ""), ("lo", "_lo"), ("hi", "_hi")):
                    row[prefix + suffix] = pred[key].loc[rxn, strain_gc]
            rows.append(row)
    return pd.DataFrame(rows)


def axis_segments(values, pad=0.05):
    """Two linear axis segments split at the largest gap in *values*, each padded."""
    v = np.sort(np.append(values, 0.0))
    k = np.argmax(np.diff(v))
    low, high = (v[0], v[k]), (v[k + 1], v[-1])
    return [(a - pad * (b - a), b + pad * (b - a)) for a, b in (low, high)]


def plot_ko_scatter(flux, out_dir):
    """Wild-type against knockout SF flux of each deleted gene's reaction, on a linear axis
    broken at the same place on x and y, with y = x (no effect) and y = 0 (reaction gone);
    hue is the knockout category."""
    low, high = axis_segments(flux[["wt", "ko", "wt_lo", "wt_hi", "ko_lo", "ko_hi"]]
                              .to_numpy().ravel())
    # One set of ticks per segment, shared by x and y.
    ticks = [[t for t in MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*seg) if seg[0] <= t <= seg[1]]
             for seg in (low, high)]
    ratios = [2, 1]
    # Square axes block as tall as the R² bar plot's axes.
    side = R2_FIGSIZE[1] * (R2_MARGINS["top"] - R2_MARGINS["bottom"])
    m = SCATTER_MARGINS_IN
    width, height = m["left"] + side + m["right"], m["bottom"] + side + m["top"]
    fig, axes = plt.subplots(2, 2, figsize=(width, height), gridspec_kw=dict(
        width_ratios=ratios, height_ratios=ratios[::-1], wspace=0.06, hspace=0.06,
        left=m["left"] / width, right=1 - m["right"] / width,
        bottom=m["bottom"] / height, top=1 - m["top"] / height))
    for row, ylim in ((0, high), (1, low)):
        for col, xlim in ((0, low), (1, high)):
            ax = axes[row, col]
            ax.plot([low[0], high[1]], [low[0], high[1]], color=COLORS["reference"],
                    linewidth=0.8, linestyle="--", zorder=0)
            ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle="--",
                       zorder=0)
            for category in ("knockdown", "isozyme"):
                d = flux[flux["knockdown"] == (category == "knockdown")]
                ax.errorbar(d["wt"], d["ko"],
                            xerr=[d["wt"] - d["wt_lo"], d["wt_hi"] - d["wt"]],
                            yerr=[d["ko"] - d["ko_lo"], d["ko_hi"] - d["ko"]],
                            fmt="o", markersize=5, color=COLORS[category],
                            ecolor=COLORS[category], elinewidth=0.8, capsize=0,
                            label=CATEGORIES[category],
                            zorder=3 if category == "knockdown" else 2)
            for _, r in flux[flux["knockdown"]].iterrows():
                if xlim[0] <= r["wt"] <= xlim[1] and ylim[0] <= r["ko"] <= ylim[1]:
                    ax.annotate(r["gene"], (r["wt"], r["ko"]),
                                xytext=KNOCKDOWN_LABEL_OFFSET.get(r["gene"], (4, 4)),
                                textcoords="offset points", fontsize=7, style="italic",
                                color=COLORS["knockdown"], zorder=4)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks(ticks[col])
            ax.set_yticks(ticks[1 - row])
            sns.despine(ax=ax, left=col == 1, bottom=row == 0)
            if col == 1:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row == 0:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
    # Slanted marks where each axis is cut.
    mark = dict(color="black", linewidth=0.8, clip_on=False)
    for ax, xs in ((axes[1, 0], (1, 1)), (axes[1, 1], (0, 0))):
        ax.plot([xs[0] - 0.02, xs[1] + 0.02], [-0.03, 0.03], transform=ax.transAxes, **mark)
    for ax, ys in ((axes[1, 0], 1), (axes[0, 0], 0)):
        ax.plot([-0.03, 0.03], [ys - 0.02, ys + 0.02], transform=ax.transAxes, **mark)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, frameon=False, fontsize=7, loc="upper left")
    fig.supxlabel("Wild-type flux " + UNITS, x=(m["left"] + side / 2) / width, y=0.0,
                  fontsize=12)
    fig.supylabel("Knockout flux " + UNITS, x=0.0, y=(m["bottom"] + side / 2) / height,
                  fontsize=12)
    save_figure(fig, out_dir / "ko_flux_scatter.svg", tight=False)


def labeling_r2(tokens):
    """Labeling R² per strain and method, from ko_labeling.py's token table.

    A state whose EMU solve failed has no labeling and gets NaN.
    """
    expected = pd.MultiIndex.from_frame(tokens[["gc", "method"]].drop_duplicates())
    valid = tokens.dropna(subset=["sim"])
    r2 = (valid.groupby(["gc", "method"])
               .apply(lambda d: r2_linear(d["sim"], d["measured"]), include_groups=False))
    return r2.reindex(expected).unstack("method")


def plot_ko_labeling(tokens, r2, gc, gene, out_dir):
    """Predicted against measured isotopologue fraction in *gene*'s knockout, with and
    without FBApro."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), gridspec_kw=dict(
        left=0.22, right=0.97, bottom=0.17, top=0.97))
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linewidth=0.8, linestyle="--",
            zorder=0)
    for method in ("fbapro", "sigmaflux"):
        d = tokens[(tokens["gc"] == gc) & (tokens["method"] == method)].dropna(subset=["sim"])
        ax.scatter(d["measured"], d["sim"], s=12, color=COLORS[method], linewidths=0,
                   label=f"{SOURCES[method]} ($R^2$ = {r2.loc[gc, method]:.2f})", zorder=2)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8,
                       loc="upper left", title=gene, title_fontsize=9, alignment="left")
    legend.get_title().set_fontstyle("italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured isotopologue fraction")
    ax.set_ylabel("Predicted isotopologue fraction")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"ko_labeling_{gene}.svg", tight=False)


def plot_labeling_knockdowns(r2, strains, knockdown, out_dir):
    """Labeling R² of each knockdown strain, SF and SF + FBApro side by side, sorted by
    SF R²; a state whose EMU solve failed is marked n/a."""
    r2 = (r2.loc[[gc for gc in r2.index if gc in knockdown]]
            .sort_values("sigmaflux", ascending=False))
    order = list(r2.index.map(strains))
    long = (r2.rename(index=strains).rename_axis("strain").reset_index()
              .melt(id_vars="strain", value_vars=list(METHODS), var_name="method",
                    value_name="r2"))
    fig, ax = plt.subplots(figsize=(7, 2.8), gridspec_kw=dict(
        left=0.12, right=0.8, bottom=0.14, top=0.96))
    sns.barplot(data=long, x="strain", y="r2", hue="method", hue_order=list(METHODS),
                order=order, saturation=1, palette={m: COLORS[m] for m in METHODS},
                ax=ax)
    # seaborn splits each group's 0.8 width evenly between the methods.
    width = 0.8 / len(METHODS)
    for x, strain in enumerate(order):
        for i, method in enumerate(METHODS):
            if long[(long["strain"] == strain) & (long["method"] == method)]["r2"].isna().all():
                ax.text(x + (i - (len(METHODS) - 1) / 2) * width, 0.02, "n/a",
                        ha="center", fontsize=7, color=COLORS["error"])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Predicted vs. measured\n" r"labeling $R^2$")
    for label in ax.get_xticklabels():
        label.set(fontstyle="italic")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [SOURCES[m] for m in METHODS], frameon=False, fontsize=9,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "ko_labeling_r2.svg", tight=False)


def plot_labeling_summary(r2, out_dir):
    """Labeling R² over the strains, SF and SF + FBApro side by side.

    Bars are the median over strains with the interquartile range; dots are strains.
    """
    fig, ax = plt.subplots(figsize=(3.0, 3.5), gridspec_kw=dict(
        left=0.22, right=0.5, bottom=0.05, top=0.97))
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
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "ko_labeling.svg", tight=False)


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
                .apply(lambda g: pd.Series({"r2": r2_linear(g["pred"], g["ref"]),
                                            "n": len(g)}), include_groups=False)
                .reset_index()
                .assign(strain=lambda d: d["gc"].map(strains)))
    r2.to_csv(out_dir / "ko_r2.csv", index=False)
    print(r2.round(3).to_string(index=False))
    gc_of = {strain: gc for gc, strain in strains.items()}
    knockdown = {gc_of[gene] for gene in GENE_REACTIONS
                 if gene in gc_of and is_knockdown(gene)}
    plot_r2(r2, knockdown, out_dir)

    flux = ko_flux(args.run, strains, knockdown)
    flux.to_csv(out_dir / "ko_flux.csv", index=False)
    print(flux.round(3).to_string(index=False))
    plot_ko_scatter(flux, out_dir)

    tokens = pd.read_csv(args.run / "ko_labeling" / "tokens.csv")
    labeling = labeling_r2(tokens)
    labeling.assign(strain=labeling.index.map(strains)).to_csv(out_dir / "ko_labeling.csv")
    print(labeling.round(3).to_string())
    plot_labeling_summary(labeling, out_dir)
    plot_labeling_knockdowns(labeling, strains, knockdown, out_dir)
    for gene in LABELING_GENES:
        plot_ko_labeling(tokens, labeling, gc_of[gene], gene, out_dir)


if __name__ == "__main__":
    main()
