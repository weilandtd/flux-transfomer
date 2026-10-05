"""
One heatmap of which labeling tokens inform which fluxes, and which way, every tracer's tokens side
by side, from the total Sobol index and its sign (sobol_naive.py).

A cell is the token's total Sobol index for the flux: the share of the flux's variance over the
model's simulated validation samples that goes when the token alone is randomized, with every
interaction it takes part in (``ST`` from sobol_sensitivity.py). Unlike the attention contributions of plot_cross_attention.py it is 0 for a token that is the same
in every sample, however strongly attended. Every token that varies is scored, measured in the
experiments or not, so a sensitive token is an informative one. Rows are the flux columns
(identical fluxes merged, a merged column read at its first reaction) some token informs, with an
index reaching ``MIN_INDEX``, below which it is not resolved from zero; columns are the tokens of
every tracer reaching it for some flux, with a strip above marking the tracer in the graph's fills
(``TRACER_STYLES``). Each row is divided by its largest value drawn, so a cell reads as the token's
share of the flux's strongest. A cell is red when a larger token raises the flux and blue when it
lowers it, the sign of the standardized rank regression coefficient (SRRC), and 0 when the
direction is not resolved (``resolved_sign`` of plot_sensitivity_graph.py). With
``--show-measured-fluxes``, a narrow map on the right gives each measured-flux input on the same
row scale.

Fluxes and tokens are clustered separately, by Ward trees on the signed profiles scaled to unit
length, so tokens pushing the same fluxes opposite ways part,
which the row scaling leaves unchanged, each tree ordering its axis and cut into the number of bands,
from 2 to ``MAX_BANDS``, with the highest mean silhouette on cosine distance among cuts whose bands
all hold at least ``MIN_BAND`` members, so the bands follow the data without splitting off single outliers. A
strip left of the rows colors each flux by its subsystem, in the graph's colors
(``PATHWAY_FAMILIES``), listed under their family in the legend; a token band, lettered underneath,
lists its tokens in a table below the map, a column per tracer. The table lists every flux and every scored token, drawn or not,
with the unscaled index.

In ``paper/figures/``:
  sensitivity_heatmap_<key>.svg   the heatmap (``--model``, the T cell by default)
  sensitivity_heatmap_<key>.csv   input (a tracer's token or a measured flux),
                                              measured, column, reactions, ST and its 95%
                                              bootstrap half-width, sign (0: not resolved),
                                              SRRC and its 95% half-width, PRCC, mu, mu_star,
                                              the column's rank R², token and flux block
                                              (-1 when not drawn)
  sensitivity_heatmap_<key>_bands.csv   the silhouette of every cut tried, per axis
"""
import argparse
import pathlib
import re
import string
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _figure_common import add_common_args, apply_style, save_figure, tracer_label
from cross_attention import RESULTS_DIR, results_path
from matplotlib.patches import Rectangle
from plot_cross_attention import NAME_FONT, flux_columns, reaction_pathways
from plot_sensitivity_graph import (
    KEY,
    MIN_INDEX,
    PATHWAY_COLORS,
    PATHWAY_FAMILIES,
    total_indices,
    tracer_style,
)
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

MAX_BANDS = 30   # most bands a tree may be cut into (and at most half its leaves)
MIN_BAND = 2     # fewest members a band may hold
HEIGHT = 8.0       # inches of heatmap; cells are square, so the width follows the columns
STRIP = 0.12       # inches of the tracer strip above the columns and the subsystem strip left
SIDE_GAP = 0.3     # inches between the token map and the measured-flux map
CBAR = (0.12, 2.0)  # inches, width and HEIGHT of each colorbar
KEY_LINE = 1.4      # line height of the token-band table, in font sizes
SIGNED_CMAP = "RdBu_r"      # blue lowers the flux, red raises it
BAND_GAP = 3.0              # points of white between bands


def tree_bands(profiles):
    """Leaf order of a Ward tree on the unit-length profiles, the cut with the highest mean cosine
    silhouette among every cut from 2 to ``MAX_BANDS`` bands (at most half the leaves) whose bands
    hold ``MIN_BAND`` or more (any cut when none does), the bands numbered along that order, and
    the silhouette of every cut tried (bands, silhouette, smallest band)."""
    distance = squareform(pdist(profiles, "cosine"))
    unit = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
    tree = linkage(unit, method="ward", optimal_ordering=True)
    order = leaves_list(tree)
    cuts = {}
    for n in range(2, min(MAX_BANDS, len(profiles) // 2) + 1):
        cut = fcluster(tree, n, criterion="maxclust")
        cuts.setdefault(cut.max(), cut)
    curve = [(n, silhouette_score(distance, c, metric="precomputed"), np.bincount(c)[1:].min())
             for n, c in cuts.items()]
    eligible = [row for row in curve if row[2] >= MIN_BAND] or curve
    cut = cuts[max(eligible, key=lambda row: row[1])[0]]
    renumber = {c: k for k, c in enumerate(dict.fromkeys(cut[order]))}
    return order, np.array([renumber[c] for c in cut]), curve


def band_letter(i):
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'."""
    return string.ascii_uppercase[i] if i < 26 else band_letter(i // 26 - 1) + band_letter(i % 26)


def cluster_blocks(s1):
    """Flux and token band labels and orders of the signed indices ``s1``, and each tree's
    silhouette curve by axis; the order holds only the fluxes some token informs, the rest have
    band -1."""
    informed = np.flatnonzero(np.abs(s1).max(1) >= MIN_INDEX)
    order, bands, flux_curve = tree_bands(s1[informed])
    flux_block = np.full(len(s1), -1)
    flux_block[informed] = bands
    row_order = informed[order]
    col_order, token_block, token_curve = tree_bands(s1[informed].T)
    return flux_block, token_block, row_order, col_order, {"flux": flux_curve,
                                                           "token": token_curve}


def signed_mesh(ax, values):
    """``values`` (already scaled to -1..1) on the diverging map."""
    return ax.pcolormesh(values, cmap=SIGNED_CMAP, vmin=-1, vmax=1, edgecolors="face",
                         linewidth=0.3)


def plot(s1, side, side_names, tracers, flux_block, token_block, r_order, c_order, row_pathways,
         col_names):
    """Token map of the signed indices ``s1`` with its tracer strip and subsystem strip, the
    measured-flux map to its right (left out when ``side`` is None), one colorbar and the
    subsystem legend."""
    cell = HEIGHT / len(r_order)
    width = len(c_order) * cell
    side_w = 0 if side is None else side.shape[1] * cell
    fig_w, fig_h = width + SIDE_GAP + side_w + 1.6, HEIGHT + STRIP + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, width / fig_w, HEIGHT / fig_h])
    strip = fig.add_axes([0, (HEIGHT + 0.03) / fig_h, width / fig_w, STRIP / fig_h])
    rows = fig.add_axes([-(STRIP + 0.03) / fig_w, 0, STRIP / fig_w, HEIGHT / fig_h])

    # Each row divided by its largest value drawn.
    scale = np.abs(s1).max(1) if side is None else np.maximum(np.abs(s1).max(1),
                                                               np.abs(side).max(1))
    scale[scale == 0] = 1
    s1, side = s1 / scale[:, None], None if side is None else side / scale[:, None]
    main = signed_mesh(ax, s1[np.ix_(r_order, c_order)])
    ax.set_ylim(len(r_order), 0)
    panels = [ax, strip, rows]
    if side is not None:
        side_ax = fig.add_axes([(width + SIDE_GAP) / fig_w, 0, side_w / fig_w, HEIGHT / fig_h])
        panels.append(side_ax)
        signed_mesh(side_ax, side[r_order])
        side_ax.set_ylim(len(r_order), 0)
        side_ax.set_xticks(np.arange(len(side_names)) + 0.5, side_names, fontsize=NAME_FONT,
                           rotation=90)
        side_ax.set_yticks([])
        side_ax.set_title("measured fluxes", fontsize=NAME_FONT)

    for x, t in enumerate(tracers[c_order]):
        strip.add_patch(Rectangle((x, 0), 1, 1, linewidth=0.3, **tracer_style(t)))
    strip.set_xlim(0, len(c_order))
    strip.set_ylim(0, 1)
    for y, p in enumerate(np.asarray(row_pathways)[r_order]):
        rows.add_patch(Rectangle((0, y), 1, 1, facecolor=PATHWAY_COLORS[p], linewidth=0))
    rows.set_xlim(0, 1)
    rows.set_ylim(len(r_order), 0)

    rows_sorted, cols_sorted = flux_block[r_order], token_block[c_order]
    for b in np.flatnonzero(np.diff(rows_sorted)) + 1:
        for panel in (ax, rows, *panels[3:]):
            panel.axhline(b, color="white", lw=BAND_GAP)
    for b in np.flatnonzero(np.diff(cols_sorted)) + 1:
        ax.axvline(b, color="white", lw=BAND_GAP)
    letters = {k: band_letter(i) for i, k in enumerate(np.unique(cols_sorted))}
    for k, letter in letters.items():
        x = np.flatnonzero(cols_sorted == k).mean() + 0.5
        ax.text(x, len(r_order) + 1, letter, ha="center", va="top", fontsize=NAME_FONT)
    # The key as a table below the map: a row per token band, a column per tracer.
    tracer_cols = list(dict.fromkeys(tracer_label(t).split(" (")[0] for t in tracers))
    chars = int(width * 72 * 0.96 / len(tracer_cols) / (0.56 * NAME_FONT))
    # A token keeps its words together; lines break between tokens only.
    cells = [[textwrap.fill(col_names[k].get(t, "").replace(" ", "\u00a0").replace(",\u00a0", ", "),
                            chars) for t in tracer_cols] for k in letters]
    lines = [1] + [max(c.count("\n") + 1 for c in row) for row in cells]
    line_h = KEY_LINE * NAME_FONT / 72
    key_h = sum(lines) * line_h
    key_ax = fig.add_axes([0, -(0.3 + key_h) / fig_h, width / fig_w, key_h / fig_h])
    key_ax.axis("off")
    table = key_ax.table(cellText=cells, rowLabels=list(letters.values()), colLabels=tracer_cols,
                         loc="upper left", cellLoc="left", colLoc="left", edges="horizontal",
                         bbox=[0.04, 0, 0.96, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(NAME_FONT)
    for (r, c), cell in table.get_celld().items():
        cell.set_height(lines[r] / sum(lines))
        cell.set_linewidth(0.4)
        cell.get_text().set_va("center")
        if r == 0:
            cell.get_text().set_fontweight("bold")

    for a in panels:
        a.tick_params(length=0)
        for spine in a.spines.values():
            spine.set_visible(False)
    for a in (ax, strip, rows):
        a.set_xticks([])
        a.set_yticks([])
    handles = [Rectangle((0, 0), 1, 1, linewidth=0.5, label=tracer_label(t), **tracer_style(t))
               for t in dict.fromkeys(tracers)]
    strip.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.2), ncol=3,
                 frameon=False, fontsize=NAME_FONT)
    cax = fig.add_axes([(width + SIDE_GAP + side_w + 0.5) / fig_w, (HEIGHT - CBAR[1]) / fig_h,
                        CBAR[0] / fig_w, CBAR[1] / fig_h])
    bar = fig.colorbar(main, cax=cax)
    bar.solids.set_rasterized(False)
    bar.ax.tick_params(labelsize=NAME_FONT)
    bar.set_label("normalized sensitivity,\nlowers | raises", fontsize=NAME_FONT)

    # Subsystems present, under their family, below the colorbar.
    present = {row_pathways[r] for r in r_order}
    handles = []
    for family, colors in PATHWAY_FAMILIES.items():
        shown = [(p, c) for p, c in colors.items() if p in present]
        if shown:
            handles.append(Rectangle((0, 0), 1, 1, visible=False, label=family))
            handles += [Rectangle((0, 0), 1, 1, facecolor=c, linewidth=0, label=p)
                        for p, c in shown]
    legend = fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=NAME_FONT,
                        bbox_to_anchor=((width + SIDE_GAP + side_w + 0.3) / fig_w,
                                        (HEIGHT - CBAR[1] - 0.4) / fig_h))
    for text, handle in zip(legend.get_texts(), handles):
        if not handle.get_visible():
            text.set_fontweight("bold")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where cross_attention.py and sobol_sensitivity.py wrote their .npz")
    parser.add_argument("--model", default=KEY, help=f"Model key (default {KEY})")
    parser.add_argument("--sobol", default="_naive",
                        help="Suffix of the Sobol result to read (default _naive, sobol_naive.py)")
    parser.add_argument("--show-measured-fluxes", action="store_true",
                        help="Draw the measured-flux map to the right of the tokens")
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = args.model
    with np.load(results_path(args.results_dir, model, "experimental")) as f:
        d = dict(f)
    with np.load(args.results_dir / f"{model}_sobol{args.sobol}.npz") as f:
        sobol = dict(f)
    pathways = reaction_pathways(model)
    _, groups = flux_columns(d, pathways)
    drawn, tracers, _, labels, s1, side, side_names, sign, side_sign = total_indices(d, sobol,
                                                                                      groups)
    side_names = [re.sub(r"^EX_|_e$", "", n) for n in side_names]

    s1, side = s1 * sign, side * side_sign
    flux_block, token_block, r_order, c_order, curves = cluster_blocks(s1)
    chosen = {"flux": flux_block.max() + 1, "token": token_block.max() + 1}
    bands = pd.DataFrame([{"axis": axis, "bands": n, "silhouette": sil, "smallest": small,
                           "chosen": n == chosen[axis]}
                          for axis, curve in curves.items() for n, sil, small in curve])
    bands.to_csv(args.out_dir / f"sensitivity_heatmap_{model}_bands.csv", index=False)
    for axis, curve in curves.items():
        best = max(sil for n, sil, _ in curve if n == chosen[axis])
        edge = " -- the largest cut tried; raise MAX_BANDS" if chosen[axis] == curve[-1][0] else ""
        print(f"{axis} bands: {chosen[axis]} (silhouette {best:.2f}, cuts {curve[0][0]}-"
              f"{curve[-1][0]}){edge}")
    col_names = {k: {tracer_label(t).split(" (")[0]: ", ".join(
        labels[i] for i in np.flatnonzero((token_block == k) & (tracers == t)))
        for t in dict.fromkeys(tracers) if ((token_block == k) & (tracers == t)).any()}
        for k in np.unique(token_block)}
    fig = plot(s1, side if args.show_measured_fluxes else None, side_names, tracers, flux_block,
               token_block, r_order, c_order, [g["pathway"] for g in groups], col_names)
    save_figure(fig, args.out_dir / f"sensitivity_heatmap_{model}.svg")

    # The table covers every scored token, drawn or not.
    tokens, tracers, measured, labels, *_, sign, side_sign = total_indices(d, sobol, groups, None)
    table = {key: np.hstack(total_indices(d, sobol, groups, None, key)[4:6])
             for key in ("ST", "ST_conf", "SRRC", "SRRC_conf", "PRCC", "mu", "mu_star")}
    rank_r2 = sobol["rank_R2"][[g["members"][0] for g in groups]]
    block = dict(zip(drawn, token_block))
    inputs = [f"{label}, {tracer_label(t)}" for label, t in zip(labels, tracers)] + side_names
    rows = [pd.DataFrame({"input": inputs, "measured": np.r_[measured, np.ones(len(side_names), bool)],
                          "column": g["label"], "reactions": ";".join(g["names"]),
                          "ST": table["ST"][i], "ST_conf": table["ST_conf"][i],
                          "sign": np.r_[sign[i], side_sign[i]],
                          **{key: table[key][i] for key in ("SRRC", "SRRC_conf", "PRCC", "mu",
                                                            "mu_star")},
                          "rank_R2": rank_r2[i],
                          "token_block": [block.get(t, -1) for t in tokens] + [-1] * len(side_names),
                          "flux_block": fb})
            for i, (g, fb) in enumerate(zip(groups, flux_block))]
    csv = args.out_dir / f"sensitivity_heatmap_{model}.csv"
    pd.concat(rows).to_csv(csv, index=False)
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
