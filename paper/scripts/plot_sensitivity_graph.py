"""
Which labeling token informs each flux most, and which way: one bipartite graph of every tracer's
tokens against the fluxes, drawn from the total Sobol index and its sign (sobol_naive.py).

An edge's weight is the token's total Sobol index for the flux: the share of the flux's variance
of the model's prediction that goes when the token alone is randomized, with every interaction it
takes part in. The cross-attention decomposition cannot give this: four
self-attention layers mix every encoder position with the whole input first, so an attended
position is not an input, and a token that is the same in every sample still carries a large
contribution. Every token that varies across the samples is scored, measured in the experiments
or not, so a sensitive token is an informative one. As in the heatmap, only the flux columns
(identical fluxes merged, a merged column read at its first reaction) and tokens with an index
reaching ``MIN_INDEX``, below which it is not resolved from zero, are kept, and each flux is divided
by its strongest token; a flux is linked to the tokens reaching ``THRESHOLD`` of it, so a flux
several tokens inform joins them. A spring layout pulled by the edge weights puts every flux next
to the tokens it reads; tokens closer than ``TOKEN_GAP`` of its width are then pushed apart and each
flux moves with its tokens, so crowded tokens separate.

An edge is red when a larger token raises the flux and blue when it lowers it, the sign of the
standardized rank regression coefficient (SRRC); grey when the direction is not resolved: SRRC's
95% interval holds 0, the mean radial elementary effect mu disagrees with it, or |mu| / mu_star is
below ``MONOTONE``, the effect switching sign across the inputs.

Tokens are filled by tracer (``TRACER_STYLES``: black, white, striped), sized by their weighted
degree and all named; those in ``HIGHLIGHTS``, whose fraction follows a flux split for an
explainable reason, are set in bold. Hubs, whose weighted degree reaches ``--hub-min-strength``,
are listed in the hubs table.
Fluxes are colored by the model's own subsystem annotation (a merged flux column by its most
common one), shades of one hue per family of related pathways (``PATHWAY_FAMILIES``), with
transport, pseudo, unassigned and unannotated reactions in grey; the legend lists the tracers and
the pathways under their family.

In ``paper/figures/``:
  sensitivity_graph_<key>.svg           the graph (``--model``, the T cell by default)
  sensitivity_graph_<key>_edges.csv     tracer, token, measured, flux, pathway, ST, weight
                                        (of the flux's strongest), sign (0: not resolved), SRRC,
                                        PRCC, mu, mu_star
  sensitivity_graph_<key>_hubs.csv      rank, tracer, token, measured, strength, degree
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _figure_common import add_common_args, apply_style, save_figure, tracer_label
from cross_attention import RESULTS_DIR, results_path
from plot_cross_attention import (
    METABOLITES,
    flux_columns,
    measured_tokens,
    metabolite_name,
    pathway_of,
    reaction_pathways,
)

from sigmaflux.viz import bipartite_graph, hubs, plot_bipartite_graph

KEY = "m_musculus_t_cell"   # the default model
# Tokens whose fraction follows a flux split across the model's simulated samples for a reason
# the carbon routes explain, per tracer labeling pattern.
HIGHLIGHTS = {
    "[1, 1, 0, 0, 0, 0]": [
        "Pyr M+2",    # PPP vs glycolysis: the PPP drops carbon 1
        "Ser M+2",    # serine made from 3PG vs taken up
        "Cit M+6",    # carbon leaving the TCA cycle through malic enzyme
        "Glu M+2",    # glucose acetyl-CoA vs glutamine at aKG
    ],
    "[1, 1, 1, 1, 1]": [
        "Orotate M+4",    # carbon re-entering the TCA cycle through pyruvate carboxylase
        "αKG M+5",        # glutamine vs glucose carbon at aKG
    ],
}
COLORS = {"token": "#4d4d4d", "unannotated": "#bbbbbb"}
SIGN_COLORS = {1: "#b2182b", -1: "#2166ac", 0: "#4d4d4d"}   # raises, lowers, not resolved
UNANNOTATED = ["Transport", "Pseudoreaction", "Unassigned", "Other", "ATP maintenance",
               "Exchange", "Detoxification"]
# The model's pathways in hand-made families, one hue each, so related pathways read alike and
# the legend lists them under their family.
PATHWAY_FAMILIES = {
    "Glycolysis branch": {
        "Glycolysis / Gluconeogenesis": "#b2182b",
        "Lactate production": "#e7797a",
        "Pyruvate metabolism": "#d6604d",            # yeast fermentation branch
        "Glycerol secretion": "#fbb4ae",
        "Carbohydrate biosynthesis": "#fdd0a2",
        "Starch and sucrose metabolism": "#fee0b6",
        "Pentose phosphate pathway": "#f4a259",
        "Entner-Doudoroff pathway": "#e08214",
        "Acetate metabolism": "#fdae6b",
        "Methylglyoxal metabolism": "#fcbba1",
        "Glyc3P-DHAP shuttle": "#fb9a99",
    },
    "TCA cycle & respiration": {
        "Citric acid cycle": "#08306b",
        "Oxidative phosphorylation": "#2166ac",
        "Acetyl-CoA synthesis": "#4393c3",           # PDH entry and ACLY citrate export
        "Anaplerotic reaction": "#74add1",           # pyruvate carboxylase
        "Anaplerotic reactions": "#74add1",          # E. coli's name for it
        "Glyoxylate shunt": "#a6cee3",
        "Transhydrogenase": "#92c5de",
        "Glyoxylate metabolism": "#a6cee3",          # here cytosolic MDH and FUM
    },
    "Nucleotides & one-carbon": {
        "Purine metabolism": "#3f007d",
        "Pyrimidine metabolism": "#6a51a3",
        "Folate metabolism": "#9e9ac8",
        "One-carbon metabolism": "#9e9ac8",
        "Glycine and serine metabolism": "#bcbddc",
        "Glycine, serine and threonine metabolism": "#bcbddc",   # the yeasts' name for it
    },
    "Amino acids": {
        "Glutamate metabolism": "#00564d",
        "Glutamine metabolism": "#00897b",
        "Aspartate metabolism": "#26a69a",
        "Asparagine biosynthesis": "#4db6ac",
        "Alanine metabolism": "#80cbc4",
        "Arginine and proline metabolism": "#2e7d6f",
        "Alanine, aspartate and glutamate metabolism": "#26a69a",  # the yeasts' name for it
        "Cysteine and methionine metabolism": "#b2dfdb",
        "Valine, leucine and isoleucine metabolism": "#004d40",
        "Tryptophan metabolism": "#66bb6a",
        "Phenylalanine, tyrosine and tryptophan metabolism": "#a5d6a7",
        "Histidine metabolism": "#1b5e20",
        "Nitrogen metabolism": "#c8e6c9",
        "Lysine metabolism": "#43a047",
        "Sulfur metabolism": "#e0f2f1",
        "Amino acid metabolism": "#26a69a",
    },
    "Biomass & lipids": {
        "Biomass": "#6d4c41",
        "Lipid biosynthesis": "#a1887f",
        "Ergosterol biosynthesis": "#d7ccc8",
        "Terpenoid backbone biosynthesis": "#bcaaa4",
        "Fatty acid biosynthesis": "#8d6e63",
        "Fatty acid degradation": "#a1887f",
        "Fatty acid metabolism": "#795548",
        "Inositol phosphate metabolism": "#efebe9",
        "Glycerophospholipid metabolism": "#d7b5a6",
        "Amino sugar and nucleotide sugar metabolism": "#5d4037",   # cell wall and glycan
    },
    "Unannotated": dict.fromkeys(UNANNOTATED, COLORS["unannotated"]),
}
PATHWAY_COLORS = {p: c for family in PATHWAY_FAMILIES.values() for p, c in family.items()}
assert not set(PATHWAY_FAMILIES) & set(PATHWAY_COLORS), "a family header would hide a pathway"
FIG_SIZE = (7, 7)
MIN_INDEX = 0.02         # total index below which a token is not resolved from zero
THRESHOLD = 0.8          # fraction of a flux's strongest token that makes an edge
MONOTONE = 0.6           # |mu| / mu_star from which a direction holds (80% of the effect one way)
SIGNED = ("SRRC", "PRCC", "mu")   # result keys that carry a sign
HUB_MIN_STRENGTH = 5.0   # weighted degree that makes a token a hub
FLUX_SATELLITE_SIZE = 24.0    # marker area of a flux
TOKEN_GAP = 0.13         # closest two tokens may sit, as a share of the layout's width
# Token fill per tracer labeling pattern, whatever its enrichment, shared with the heatmap's
# tracer strip; the tracers are drawn in this order.
TRACER_STYLES = {
    "[1, 1, 0, 0, 0, 0]": {"facecolor": "black", "edgecolor": "black"},
    "[1, 1, 1, 1, 1, 1]": {"facecolor": "white", "edgecolor": "black"},
    "[0, 0, 1, 0, 0, 0]": {"facecolor": "#999999", "edgecolor": "black"},
    "[1, 1, 1, 1, 1]": {"facecolor": "white", "edgecolor": "black", "hatch": "//////"},
}


def pattern(tracer):
    """``'glc__D_e_[1, 1, 0, 0, 0, 0]_0.5'`` -> ``'[1, 1, 0, 0, 0, 0]'``."""
    return tracer[tracer.index("["):tracer.rindex("]") + 1]


def tracer_style(tracer):
    return TRACER_STYLES[pattern(tracer)]


def display(met_id):
    name = metabolite_name(met_id)
    return METABOLITES.get(name.lower(), (name,))[0]


def family_legend(pathways, tracers):
    """Legend entries for ``pathways``: the ``tracers``' token fills, the edge colors, then every
    family with a pathway present as a header over its pathways."""
    entries = {tracer_label(t): tracer_style(t) for t in tracers}
    entries |= {"A larger token": None, "raises the flux": SIGN_COLORS[1],
                "lowers the flux": SIGN_COLORS[-1], "direction not resolved": SIGN_COLORS[0]}
    for family, colors in PATHWAY_FAMILIES.items():
        present = {p: c for p, c in colors.items() if p in pathways}
        if present:
            entries |= {family: None} | present
    return entries


def resolved_sign(sobol, prefix=""):
    """+1 or -1 per (flux, factor) of the ``prefix`` inputs where the rank regression and the
    elementary effects agree on one direction, 0 where they do not."""
    srrc, conf, mu, mu_star = (np.nan_to_num(sobol[f"{prefix}{k}"])
                               for k in ("SRRC", "SRRC_conf", "mu", "mu_star"))
    sign = np.sign(srrc)
    resolved = ((np.abs(srrc) > conf) & (np.sign(mu) == sign)
                & (np.abs(mu) >= MONOTONE * mu_star) & (mu_star > 0))
    return np.where(resolved, sign, 0).astype(int)


def total_indices(d, sobol, groups, min_index=MIN_INDEX, key="ST"):
    """The randomized tokens of every tracer whose total index reaches ``min_index`` for some
    flux column (all of them when None), against the columns, read at each column's first
    reaction. Returns (tokens, tracer of each, whether the experiments measure each, token labels,
    ST (columns x tokens), ST of each measured-flux input (columns x measured fluxes), the
    measured fluxes' names, and the resolved sign of the tokens and of the measured fluxes).
    Another ``key`` reads that result instead, e.g. ``"ST_conf"`` or ``"SRRC"``, for the same
    tokens; the unsigned ones clipped at 0."""
    st = np.zeros((len(sobol["flux_names"]), len(sobol["token_index"])))
    st[:, sobol["varied"]] = np.nan_to_num(sobol["ST"]).clip(min=0)
    low = None if key in SIGNED else 0
    values, signs = np.zeros_like(st), np.zeros(st.shape, int)
    values[:, sobol["varied"]] = np.nan_to_num(sobol[key]).clip(min=low)
    signs[:, sobol["varied"]] = resolved_sign(sobol)
    rows = [g["members"][0] for g in groups]
    patterns = np.array([pattern(t) for t in d["token_index"][:, 0]])
    tokens = np.concatenate([np.intersect1d(np.flatnonzero(patterns == p), sobol["varied"])
                             for p in TRACER_STYLES])
    if min_index is not None:
        tokens = tokens[st[np.ix_(rows, tokens)].max(0) >= min_index]
    labels = [f"{display(met)} {iso}" for _, met, iso in d["token_index"][tokens]]
    return (tokens, d["token_index"][tokens, 0], measured_tokens(d)[tokens], labels,
            values[np.ix_(rows, tokens)],
            np.nan_to_num(sobol[f"flux_{key}"])[rows].clip(min=low),
            # The measured fluxes that were varied: none when they were held fixed.
            [str(n) for n in sobol["flux_names"][sobol["fixed_idx"]]][:sobol["flux_ST"].shape[1]],
            signs[np.ix_(rows, tokens)], resolved_sign(sobol, "flux_")[rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where cross_attention.py wrote its .npz files")
    parser.add_argument("--model", default=KEY, help=f"Model key (default {KEY})")
    parser.add_argument("--sobol", default="_naive",
                        help="Suffix of the Sobol result to read (default _naive, sobol_naive.py)")
    parser.add_argument("--hub-min-strength", type=float, default=HUB_MIN_STRENGTH,
                        help=f"Weighted degree that makes a token a hub (default {HUB_MIN_STRENGTH})")
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The experimental decomposition fixes which tokens are measured and the merged columns.
    key = args.model
    with np.load(results_path(args.results_dir, key, "experimental")) as f:
        d = dict(f)
    with np.load(args.results_dir / f"{key}_sobol{args.sobol}.npz") as f:
        sobol = dict(f)
    pathways = reaction_pathways(key)
    _, groups = flux_columns(d, pathways)
    _, tracers, measured, labels, st, _, _, sign, _ = total_indices(d, sobol, groups)
    signed = {k: total_indices(d, sobol, groups, key=k)[4] for k in ("SRRC", "PRCC", "mu", "mu_star")}
    informed = st.max(1) >= MIN_INDEX
    groups, st, sign = [g for g, keep in zip(groups, informed) if keep], st[informed], sign[informed]
    signed = {k: v[informed] for k, v in signed.items()}
    flux_pathway = {g["label"]: g["pathway"] for g in groups}
    unknown = {pathway_of(r, pathways) for g in groups for r in g["names"]} - set(PATHWAY_COLORS)
    if unknown:
        raise ValueError(f"pathways without a family color: {sorted(unknown)}")

    is_measured = {f"{label} | {t}": m for label, t, m in zip(labels, tracers, measured)}
    # A node per (tracer, token): the same isotopologue on two tracers is two inputs.
    nodes = [f"{label} | {tracer}" for label, tracer in zip(labels, tracers)]
    graph = bipartite_graph(st, nodes, list(flux_pathway), THRESHOLD)
    fluxes = [n for n in graph if n[0] == "target"]
    tokens = [n for n in graph if n[0] == "source"]
    top = hubs(graph, n=None, min_strength=args.hub_min_strength)
    highlighted = {f"{label} | {t}" for t in set(tracers) for label in HIGHLIGHTS.get(pattern(t), [])}
    cell = {(n, f["label"]): (i, j) for j, n in enumerate(nodes) for i, f in enumerate(groups)}
    edge_sign = {(token, flux): sign[cell[token[1], flux[1]]] for token, flux in
                 ((u, v) if u[0] == "source" else (v, u) for u, v in graph.edges)}

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_bipartite_graph(
        graph, ax=ax, node_size=FLUX_SATELLITE_SIZE,
        node_colors={n: PATHWAY_COLORS[flux_pathway[n[1]]] for n in fluxes},
        node_styles={n: tracer_style(n[1].split(" | ")[1]) for n in tokens},
        hub_labels={n: n[1].split(" | ")[0] for n in tokens}, source_gap=TOKEN_GAP,
        bold={n for n in tokens if n[1] in highlighted},
        edge_colors={e: SIGN_COLORS[s] for e, s in edge_sign.items()},
        legend=family_legend({flux_pathway[n[1]] for n in fluxes}, dict.fromkeys(tracers)))
    save_figure(fig, args.out_dir / f"sensitivity_graph_{key}.svg")

    edges = []
    for u, v, w in graph.edges(data="weight"):
        token, flux = (u, v) if u[0] == "source" else (v, u)
        label, tracer = token[1].split(" | ")
        at = cell[token[1], flux[1]]
        edges.append({"tracer": tracer_label(tracer), "token": label,
                      "measured": is_measured[token[1]], "flux": flux[1],
                      "pathway": flux_pathway[flux[1]], "ST": st[at], "weight": w,
                      "sign": edge_sign[token, flux]} | {k: v[at] for k, v in signed.items()})
    strength, degree = dict(graph.degree(weight="weight")), dict(graph.degree())
    hub_rows = [{"rank": i + 1, "tracer": tracer_label(n[1].split(" | ")[1]),
                 "token": n[1].split(" | ")[0], "measured": is_measured[n[1]],
                 "strength": strength[n], "degree": degree[n]}
                for i, n in enumerate(top)]
    for rows, stem in ((edges, "edges"), (hub_rows, "hubs")):
        csv = args.out_dir / f"sensitivity_graph_{key}_{stem}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
