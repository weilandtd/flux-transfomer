"""
Labeling against the flux ratio it reports, for relations the experiments bear out across species.
1,2-13C-glucose keeps carbons 1 and 2 together through glycolysis and loses carbon 1 in the
oxPPP, so Pyr M+2 falls and Glyc3P M+1 rises as the oxPPP takes glucose from glycolysis. Each
turn of the TCA cycle splits the M+2 acetyl unit, turning malate M+2 into M+1, so Mal M+1 / M+2
rises with the O2 taken up per glucose; as a ratio of one metabolite's isotopologues it cancels
the enrichment and the dilution by unlabeled carbon (glutamine in the T cell) that offset single
fractions between species. The relations were chosen by screening measured tokens, and ratios of
tokens the signed sensitivity maps set against each other, against every branch split and every
flux per glucose, keeping those whose measured values follow the ratio SigmaFlux predicts for the
conditions with the sign the simulated samples give, across species and within each yeast.

Each panel is one token, or a ratio of two isotopologues, against one ratio, a flux over the sum of the fluxes it is compared to,
each taken in its forward direction (clipped at 0). The points are the experimental conditions,
the T cell's in front, the measured fraction (bars: its replicate standard deviation) at the ratio
SigmaFlux predicts for that condition, framed by the axes; a split never runs past 1. Fractions
are divided by the tracer enrichment (1,2-13C- and U-13C-glucose are 50% for the yeasts,
U-13C-glucose 50% for E. coli), so organisms share one axis. A point is hollow when a reaction of
its ratio is predicted with validation R² <= ``MIN_R2`` (``r2_ai`` of
mfa_benchmark_r2_per_reaction_<key>.csv), so its predicted ratio is not trusted. A panel with at
least ``MIN_FIT`` trusted points draws a least-squares line through them, pooled over organisms,
with its 95% bootstrap band, and their Spearman ρ.

In ``paper/figures/``:
  sensitivity_splits.svg   the panels, titled by tracer, the token under the y axis
  sensitivity_splits.csv   model, tracer, token, ratio, spearman over the simulated samples, and
                           per condition (empty for the simulated rows) the measured fraction or
                           isotopologue ratio, its standard deviation, the predicted ratio and
                           whether it is trusted, single fractions divided by the enrichment
"""
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _figure_common import MODELS, add_common_args, apply_style, run_dir_for, save_figure
from cross_attention import RESULTS_DIR, results_path
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from plot_labeling_vs_split import COLORS, MARKERS, MIN_FIT, MIN_R2, enrichment, metabolite
from plot_sensitivity_graph import display
from scipy.stats import spearmanr

from sigmaflux.ai.data import load_data
from sigmaflux.utils.read_reference_flux import read_predicted_fluxes

YEASTS_TCELL = ("s_cerevisiae", "i_orientalis", "m_musculus_t_cell")
# Model -> (numerator, reactions summed into the denominator).
OXPPP = {"e_coli_core": ("R8", ("gluc_up",)),
         **dict.fromkeys(YEASTS_TCELL, ("G6PDH2i_c", ("HEX1_c",)))}
O2 = {"e_coli_core": ("o2_in", ("gluc_up",)),
      **dict.fromkeys(YEASTS_TCELL, ("O2t_c_e", ("HEX1_c",)))}
# (tracer labeling pattern, token metabolite, isotopologue or (numerator, denominator)
# isotopologues, ratio, model -> reactions); a ratio whose denominator holds its numerator is a
# split.
PANELS = [
    ("[1, 1, 0, 0, 0, 0]", "pyr", "M+2", "oxPPP / glucose uptake", OXPPP),
    ("[1, 1, 0, 0, 0, 0]", "glyc3p", "M+1", "oxPPP / glucose uptake", OXPPP),
    ("[1, 1, 0, 0, 0, 0]", "mal", ("M+1", "M+2"), "O₂ uptake / glucose uptake", O2),
]
TRACER_NAMES = {"[1, 1, 0, 0, 0, 0]": "1,2-¹³C-glucose", "[1, 1, 1, 1, 1, 1]": "U-¹³C-glucose",
                "[1, 1, 1, 1, 1]": "U-¹³C-glutamine"}
# The letter is N-limited for the yeasts and naive for the T cell, one marker for both.
CONDITION_NAMES = {"B": "batch", "C": "C-limited", "N": "N-limited / naive", "P": "P-limited",
                   "A": "activated"}
COLUMNS = 3
PANEL_SIZE = 2.8    # inches per panel
LEGEND_WIDTH = 2.6  # inches right of the panels
PAD = 0.3           # margin around the experimental points, as a share of their span
MIN_SPAN = 0.1      # narrowest data range an axis frames, so points all at 0 stay readable
Y_TICKS = {"glyc3p": [0, 0.1, 0.2]}   # token -> fixed y ticks, the axis spanning them


def ratio(flux, numerator, denominator):
    """numerator / sum(denominator) per sample (or condition), each flux clipped at 0."""
    top, *rest = (np.clip(np.asarray(flux.loc[r], dtype=float), 0, None)
                  for r in (numerator, *denominator))
    with np.errstate(invalid="ignore", divide="ignore"):
        return top / sum(rest)


def frame(lo, hi, top=np.inf):
    """Limits around [lo, hi], at least ``MIN_SPAN`` wide, with ``PAD`` of the span on each side,
    within [0, ``top``]."""
    hi = max(hi, lo + MIN_SPAN)
    pad = PAD * (hi - lo)
    return max(lo - pad, 0), min(hi + pad, top)


def model_token(key, pattern, met, iso):
    """The model's (tracer, metabolite, isotopologue) token for ``met`` on the tracer with
    ``pattern``, or None when the model has none."""
    with np.load(results_path(RESULTS_DIR, key, "experimental")) as f:
        tokens = [tuple(t) for t in f["token_index"]]
    return next((t for t in tokens if pattern in t[0] and metabolite(t[1]) == met
                 and t[2] == iso), None)


def predictions(model, conditions):
    """Condition -> predicted fluxes (reactions x gc) from the latest predictions file."""
    out = {}
    for cond in conditions:
        hits = sorted(glob.glob(str(run_dir_for(model) / "predictions" /
                                    f"predictions_*_{cond}_*.csv")))
        if hits:
            out[cond] = read_predicted_fluxes(hits[-1])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    measured = pd.read_csv(args.out_dir / "experimental_mid_tokens.csv").assign(
        met=lambda t: t["Metabolite"].map(metabolite))
    used = {k for *_, reactions in PANELS for k in reactions}
    data, predicted, r2 = {}, {}, {}
    for model in (m for m in MODELS if m["key"] in used):
        key = model["key"]
        data[key] = load_data(str(run_dir_for(model)))[:2]
        predicted[key] = predictions(model, measured.loc[measured["key"] == key,
                                                         "condition"].unique())
        r2[key] = pd.read_csv(args.out_dir / f"mfa_benchmark_r2_per_reaction_{key}.csv"
                              ).set_index("reaction")["r2_ai"]
    labels = {m["key"]: m["label"] for m in MODELS}

    rows = (len(PANELS) + COLUMNS - 1) // COLUMNS
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(COLUMNS * PANEL_SIZE + LEGEND_WIDTH, rows * PANEL_SIZE),
                             squeeze=False, constrained_layout=True)
    table, conditions = [], set()
    for ax, (pattern, met, iso, name, reactions) in zip(axes.flat, PANELS):
        points = []
        label = (f"{display(met)} {iso[0]} / {iso[1]}" if isinstance(iso, tuple)
                 else f"{display(met)} {iso}")
        # Drawn in order, the T cell last and so in front.
        for layer, (key, (numerator, denominator)) in enumerate(reactions.items()):
            isos = iso if isinstance(iso, tuple) else (iso,)
            tokens = [model_token(key, pattern, met, i) for i in isos]
            if None in tokens:
                continue
            labeling, flux = data[key]
            # A ratio of isotopologues cancels the enrichment; one fraction is divided by it.
            scale = 1 if len(isos) > 1 else enrichment(tokens[0][0])
            x = ratio(flux, numerator, denominator)
            parts = [labeling.loc[t].to_numpy() for t in tokens]
            with np.errstate(invalid="ignore", divide="ignore"):
                y = (parts[0] / parts[1] if len(parts) > 1 else parts[0]) / scale
            ok = np.isfinite(x) & np.isfinite(y)
            rho = spearmanr(x[ok], y[ok])[0]
            base = {"model": key, "tracer": TRACER_NAMES[pattern], "token": label,
                    "ratio": name, "spearman": rho}
            table.append(base)
            trusted = all(r2[key].get(r, np.nan) > MIN_R2 for r in (numerator, *denominator))
            hits = measured[(measured["key"] == key) & (measured["Tracer"] == tokens[0][0])
                            & (measured["met"] == met) & measured["Isotopomer"].isin(isos)]
            for (cond, gc), h in hits.groupby(["condition", "gc"]):
                h = h.set_index("Isotopomer")
                h["measured_std"] = h["measured_std"].fillna(0.0)
                if cond not in predicted[key] or not set(isos) <= set(h.index):
                    continue
                value, std = h.loc[isos[0], "measured"], h.loc[isos[0], "measured_std"]
                if len(isos) > 1:
                    below, below_std = h.loc[isos[1], "measured"], h.loc[isos[1], "measured_std"]
                    if below <= 0:
                        continue
                    value, std = value / below, value / below * np.hypot(
                        std / value if value else 0, below_std / below)
                value, std = value / scale, std / scale
                at = float(ratio(predicted[key][cond][[gc]], numerator, denominator)[0])
                ax.errorbar(at, value, yerr=std, fmt="none", ecolor=COLORS[key], elinewidth=0.8,
                            zorder=10 + 2 * layer)
                ax.scatter(at, value, marker=MARKERS[cond], s=36, zorder=11 + 2 * layer,
                           facecolor=COLORS[key] if trusted else "none",
                           edgecolor="black" if trusted else COLORS[key], linewidth=0.6)
                conditions.add(cond)
                points.append((at, value, std, trusted))
                table.append(base | {"condition": cond, "gc": gc, "measured": value,
                                     "measured_std": std, "predicted_ratio": at,
                                     "trusted": trusted})
        xs, ys, stds, trusted = map(np.array, zip(*points))
        fit = trusted.astype(bool)
        if fit.sum() >= MIN_FIT:
            sns.regplot(x=xs[fit], y=ys[fit], scatter=False, ci=95, seed=0, color="0.35",
                        line_kws={"linewidth": 1}, ax=ax)
            ax.text(0.5, 1.02, f"ρ {spearmanr(xs[fit], ys[fit])[0]:+.2f} (n = {fit.sum()})",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        split = all(n in d for n, d in reactions.values())
        ax.set_xlim(*frame(xs.min(), xs.max(), 1 if split else np.inf))
        if met in Y_TICKS:
            ax.set_ylim(Y_TICKS[met][0], Y_TICKS[met][-1])
            ax.set_yticks(Y_TICKS[met])
        else:
            ax.set_ylim(*frame((ys - stds).min(), (ys + stds).max()))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel(f"{'Isotopologue ratio' if isinstance(iso, tuple) else 'Normalized enrichment'}"
                      f"\n({label})", fontsize=9)
        ax.set_title(TRACER_NAMES[pattern], fontsize=9, pad=16)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(PANELS):]:
        ax.axis("off")

    # Right of the panels: a column of species, a column of conditions, each under a bold
    # header; the legend fills column by column, so the shorter column is padded.
    columns = [[Line2D([], [], linestyle="", label="Species")]
               + [Line2D([], [], marker="o", linestyle="", color=COLORS[k], label=labels[k])
                  for k in COLORS if k in data],
               [Line2D([], [], linestyle="", label="Condition")]
               + [Line2D([], [], marker=MARKERS[c], linestyle="", markerfacecolor="white",
                         markeredgecolor="black", label=CONDITION_NAMES[c])
                  for c in MARKERS if c in conditions]]
    height = max(map(len, columns))
    columns = [c + [Line2D([], [], linestyle="", label=" ")] * (height - len(c)) for c in columns]
    legend = fig.legend(handles=[h for c in columns for h in c], loc="outside right center",
                        ncol=len(columns), frameon=False, fontsize=8, columnspacing=1.0)
    for text in legend.get_texts():
        if text.get_text() in ("Species", "Condition"):
            text.set_fontweight("bold")
    save_figure(fig, args.out_dir / "sensitivity_splits.svg")
    csv = args.out_dir / "sensitivity_splits.csv"
    pd.DataFrame(table).to_csv(csv, index=False)
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
