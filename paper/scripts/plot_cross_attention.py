"""
Clustered heatmaps of which tokens inform which fluxes, one per tracer and model.

Reads the per-token flux contributions from paper/scripts/cross_attention.py (the
cross-attention pushed through the value path and Z, so the contributions sum to the
prediction), averaged over the samples and the ensemble. A figure draws the fluxes as
rows against one tracer's measured labeling tokens as columns (an unmeasured token enters
the model as the missing-value sentinel and informs by identity only), each flux divided
by its largest token in that figure, so 1 marks the flux's strongest input from this
tracer. The CSV also keeps each token's share of the flux over every token -- all tracers'
labeling tokens and the measured-flux tokens -- which the per-figure scaling hides.

Four or more fluxes whose profiles over all tokens are identical (proportional rows of Z,
so the model cannot move them apart) collapse into one column named by their most common
pathway and member count; the members are listed in the groups CSV. Each figure is
biclustered by spectral co-clustering (``--n-biclusters``): fluxes and tokens are sorted by
bicluster so the blocks sit on the diagonal, white lines mark the block edges, and the CSV
records every flux's and token's bicluster. The heatmaps are 8 in tall with square cells,
so the width follows the token count. Instead of tick labels the row and column bands are
named, rows on the left and columns underneath: by hand (``GROUP_NAMES``) directly on the
bands, or else from their members (``group_names``) with the long token-group names
lettered into a key under the map. The
token and flux names are in the CSVs.

Measurement-pool reactions (compartment fractions, not rates), exchanges and dilution
fluxes are dropped. The figures read the experimental-data decomposition by default,
averaged over every growth condition. ``--data validation`` draws the simulated
validation set instead.

Per model, in ``paper/figures/``:
  cross_attention_<key>_<tracer>[_experimental].svg   one clustered heatmap per tracer
  cross_attention_<key>[_experimental].csv            block, token, measured, column,
                                                      reaction, token/flux bicluster,
                                                      share, shown, mean |F|, mean F
  cross_attention_<key>[_experimental]_groups.csv     column label, pathway, members
  cross_attention_<key>[_experimental]_biclusters.csv per tracer and bicluster: its
                                                      reactions and tokens, their pathway
                                                      and metabolite counts, and the block
                                                      mean against the fluxes' other tokens

Usage:
    python paper/scripts/plot_cross_attention.py
    python paper/scripts/plot_cross_attention.py --models s_cerevisiae
    python paper/scripts/plot_cross_attention.py --data validation
"""

import argparse
import collections
import pathlib
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import SpectralCoclustering

from _figure_common import (
    _REPO_ROOT, add_common_args, apply_style, save_figure, selected_models, tracer_label)
from cross_attention import RESULTS_DIR, results_path
from sigmaflux.core.measurement_pools import POOL_PREFIX

DROPPED_PREFIXES = (POOL_PREFIX, "EX_", "DILUT_")
MEASURED_BLOCK = "measured fluxes"
# Largest max-scaled difference at which two flux profiles count as identical.
IDENTICAL_TOL = 1e-6
# Identical fluxes merge into one column only in groups at least this large.
MIN_MERGED = 4
FIG_HEIGHT = 9.45    # inches, before the tight crop trims a named T-cell map to ~8
CBAR_HEIGHT = 2.0    # inches, top-aligned right of the heatmap
CMAP = "viridis"

NAME_FONT = 8
ROW_NAME_CHARS = 20  # row names without an '&' wrap at this many characters

JOIN = " · "          # between the parts of a generated group name

# Every token metabolite (compartment and stereo suffix dropped, lower case) -> display name
# and class. Amino acids are classed by the precursor they are made from; urea-cycle
# intermediates by their cycle.
METABOLITES = {
    "g6p": ("G6P", "Glycolytic intermediates"), "f6p": ("F6P", "Glycolytic intermediates"),
    "fbp": ("FBP", "Glycolytic intermediates"), "fdp": ("FBP", "Glycolytic intermediates"),
    "dhap": ("DHAP", "Glycolytic intermediates"), "g3p": ("GAP", "Glycolytic intermediates"),
    "glyc3p": ("Gly3P", "Glycolytic intermediates"), "3pg": ("3PG", "Glycolytic intermediates"),
    "pep": ("PEP", "Glycolytic intermediates"),
    "6pg": ("6PG", "Pentose phosphates"), "r5p": ("R5P", "Pentose phosphates"),
    "x5p": ("X5P", "Pentose phosphates"), "s7p": ("S7P", "Pentose phosphates"),
    "e4p": ("E4P", "Pentose phosphates"),
    "pyr": ("Pyr", "Pyruvate & lactate"), "lac": ("Lac", "Pyruvate & lactate"),
    "accoa": ("AcCoA", "Pyruvate & lactate"),
    "cit": ("Cit", "TCA intermediates"), "acon_c": ("Aco", "TCA intermediates"),
    "akg": ("αKG", "TCA intermediates"), "succ": ("Succ", "TCA intermediates"),
    "fum": ("Fum", "TCA intermediates"), "mal": ("Mal", "TCA intermediates"),
    "glu": ("Glu", "Glutamate-family amino acids"), "gln": ("Gln", "Glutamate-family amino acids"),
    "pro": ("Pro", "Glutamate-family amino acids"), "arg": ("Arg", "Glutamate-family amino acids"),
    "orn": ("Orn", "Urea-cycle intermediates"), "citr": ("Citr", "Urea-cycle intermediates"),
    "asp": ("Asp", "Aspartate-family amino acids"), "thr": ("Thr", "Aspartate-family amino acids"),
    "met": ("Met", "Aspartate-family amino acids"), "lys": ("Lys", "Aspartate-family amino acids"),
    "ile": ("Ile", "Aspartate-family amino acids"),
    "ala": ("Ala", "Pyruvate-family amino acids"), "val": ("Val", "Pyruvate-family amino acids"),
    "leu": ("Leu", "Pyruvate-family amino acids"),
    "3c3hmp": ("IPM", "Pyruvate-family amino acids"),     # isopropylmalate, Leu precursor
    "ser": ("Ser", "Serine-family amino acids"), "gly": ("Gly", "Serine-family amino acids"),
    "phe": ("Phe", "Aromatic amino acids"), "tyr": ("Tyr", "Aromatic amino acids"),
    "trp": ("Trp", "Aromatic amino acids"),
    "his": ("Histidine", "Histidine"),
    "orot": ("Orotate", "Pyrimidine precursor"),
}
# Mammalian cells cannot make these (Tyr only from essential Phe), so glucose carbon does
# not reach them there; in those models they form their own class.
ESSENTIAL_AMINO_ACIDS = {"his", "ile", "leu", "lys", "met", "phe", "thr", "trp", "val", "tyr"}
MAMMALIAN_MODELS = {"m_musculus_t_cell"}
# A class makes it into a token-group name when it holds at least this share of the
# group's tokens (the largest class always does), at most MAX_CLASSES of them.
CLASS_MIN_SHARE = 0.2
MAX_CLASSES = 3

# Pathway labels too generic to name a flux group by, unless nothing else is in it.
CATCH_ALL = {"Transport", "Other", "Biomass", "Unassigned", "Pseudoreaction"}

# Hand-given names for a figure's biclusters, {bicluster: (flux group, token group)}:
# conceptual rather than exhaustive, and drawn straight onto the bands. They interpret the
# members in the biclusters CSV and are tied to that clustering (--n-biclusters 6, seeded,
# measured tokens only), so rename them if either changes. Figures without an entry are
# named from their members by group_names.
GROUP_NAMES = {
    ("m_musculus_t_cell", "glc__D_e_[1, 1, 0, 0, 0, 0]_1.0", "experimental"): {
        0: ("Glycolysis", "Hexose phosphates"),
        1: ("TCA cycle & nitrogen metabolism", "TCA intermediates"),
        2: ("PPP & biosynthetic export", "Pyruvate & lactate"),
        3: ("Nucleotide & protein synthesis", "Nucleotide precursors"),
        4: ("Redox shuttles", "Anaplerotic TCA label"),
        5: ("Respiration & anaplerosis", "Glutamate"),
    },
}

# Per-reaction subsystems come from the GEM workbooks' Reactions sheet.
GEM_WORKBOOKS = {
    "s_cerevisiae": "S_cerevisiae/scGEM.xlsx",
    "i_orientalis": "I_orientalis/imIsor_GEM.xlsx",
    "m_musculus_t_cell": "M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx",
}
# The E. coli core workbook labels every reaction "One System"; these follow the
# reaction list in models/atomistic/E_coli_core/E_coli_core_reaction_mapping.csv.
E_COLI_CORE_PATHWAYS = {
    **{f"R{i}": "Glycolysis / Gluconeogenesis" for i in range(1, 8)},
    "R2x": "Glycolysis / Gluconeogenesis",
    **{f"R{i}": "Pentose phosphate pathway" for i in range(8, 15)},
    "R15": "Entner-Doudoroff pathway", "R16": "Entner-Doudoroff pathway",
    "R17": "Pyruvate metabolism",
    **{f"R{i}": "Citric acid cycle" for i in range(18, 26)},
    "GlxShnt1": "Glyoxylate shunt", "GlxShnt2": "Glyoxylate shunt",
    **{f"R{i}": "Anaplerotic reactions" for i in range(26, 30)},
    "R30": "Acetate metabolism",
    **{f"R{i}": "Amino acid metabolism" for i in range(35, 58)},
    "R58": "One-carbon metabolism", "R59": "One-carbon metabolism",
    "R60": "Oxidative phosphorylation", "R61": "Oxidative phosphorylation",
    "R62": "Transhydrogenase", "R63": "ATP maintenance",
    **{r: "Transport" for r in ("gluc_up", "ac_out", "co2_out", "o2_in", "R72", "R73")},
}
BIOMASS_REACTIONS = {"BIOMASS", "PROTEIN_c", "RNA_c", "DNA_c", "LIPID_c", "CARB_c"}


def strip_compartment(met_id):
    return re.sub(r"_[a-z]$", "", met_id)


def metabolite_name(met_id):
    """``'phe__L_c'`` -> ``'phe'``: compartment and stereo suffix dropped for display."""
    return re.sub(r"__[LD]$", "", strip_compartment(met_id))


def tracer_slug(tracer):
    return re.sub(r"[^A-Za-z0-9]+", "_", tracer_label(tracer).replace("¹³C", "13C")).strip("_")


def reaction_pathways(key):
    if key == "e_coli_core":
        return dict(E_COLI_CORE_PATHWAYS)
    path = _REPO_ROOT / "models" / "atomistic" / GEM_WORKBOOKS[key]
    df = pd.read_excel(path, sheet_name="Reactions", usecols=["Rxn ID", "subsystem"]).dropna()
    return dict(zip(df["Rxn ID"].astype(str).str.strip(), df["subsystem"].astype(str).str.strip()))


def pathway_of(rxn, pathways):
    if rxn in BIOMASS_REACTIONS:
        return "Biomass"
    return pathways.get(rxn) or "Other"


def flux_columns(d, pathways):
    """Each token's share of a flux's |F| (tokens x columns), identical fluxes merged, and the groups.

    Identity is judged on the max-scaled profile, where the tolerance is independent of the
    token count.
    """
    shown = np.array([not n.startswith(DROPPED_PREFIXES) for n in d["flux_names"]])
    rows = np.flatnonzero(shown & ~d["z_zero_rows"] & (d["F_abs"].max(1) > 0))
    F = d["F_abs"][rows]
    profiles = F / F.max(1, keepdims=True)
    shares = F / F.sum(1, keepdims=True)
    if len(rows) > 1:
        ids = fcluster(linkage(profiles, method="single", metric="chebyshev"),
                       t=IDENTICAL_TOL, criterion="distance")
    else:
        ids = np.ones(len(rows), dtype=int)

    sizes = collections.Counter(ids)
    ids = [g if sizes[g] >= MIN_MERGED else -(i + 1) for i, g in enumerate(ids)]
    groups, seen = [], collections.Counter()
    for g in dict.fromkeys(ids):
        pos = np.flatnonzero(np.asarray(ids) == g)
        names = [str(n) for n in d["flux_names"][rows[pos]]]
        pathway = collections.Counter(pathway_of(n, pathways) for n in names).most_common(1)[0][0]
        label = names[0] if len(pos) == 1 else f"{pathway} ({len(pos)})"
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label} #{seen[label]}"
        groups.append(dict(label=label, pathway=pathway, members=rows[pos], names=names,
                           profile=shares[pos[0]]))
    return np.stack([g["profile"] for g in groups], axis=1), groups


def measured_tokens(d):
    """Validation samples cover every token; experimental ones only what was measured."""
    return d["measured"].any(0) if "measured" in d else np.ones(len(d["token_index"]), dtype=bool)


def tracer_matrix(d, matrix, tracer):
    """One tracer's measured tokens and the columns x tokens map, each column divided by its
    strongest of them. Only measured tokens are kept: an unmeasured one informs by identity only."""
    tokens = np.flatnonzero(d["token_index"][:, 0] == tracer)
    kept = tokens[measured_tokens(d)[tokens]]
    return kept, (matrix[kept] / matrix[kept].max(0)).T


def bicluster(matrix, n_clusters):
    """Spectral co-clustering: row and column bicluster labels, and the order that puts
    bicluster k's rows and columns together, blocks on the diagonal and strongest first."""
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0).fit(matrix)
    rows, cols = model.row_labels_, model.column_labels_
    r_order = np.lexsort((-matrix.mean(1), rows))
    c_order = np.lexsort((-matrix.mean(0), cols))
    return rows, cols, r_order, c_order


def top(items, n, skip=()):
    """The *n* most common *items*, ``', '``-joined; *skip* only if nothing else is left."""
    ranked = [k for k, _ in collections.Counter(items).most_common()]
    kept = [k for k in ranked if k not in skip] or ranked
    return JOIN.join(kept[:n])


def metabolite_class(met_id, mammalian):
    """(display name, class) of a token metabolite, from METABOLITES."""
    key = metabolite_name(met_id).lower()
    name, cls = METABOLITES.get(key, (metabolite_name(met_id), "Other"))
    if mammalian and key in ESSENTIAL_AMINO_ACIDS:
        cls = "Essential amino acids"
    return name, cls


def mass_range(masses):
    """``[1, 2, 3, 5]`` -> ``'M+1–3, 5'``: consecutive isotopologues as one run."""
    masses, runs = sorted(masses), []
    for m in masses:
        if runs and m == runs[-1][1] + 1:
            runs[-1][1] = m
        else:
            runs.append([m, m])
    return "M+" + ", ".join(f"{a}–{b}" if b > a else f"{a}" for a, b in runs)


def token_group_name(token_ids, mammalian):
    """Name a token group by its main metabolite classes, each metabolite with the
    isotopologues the group holds of it."""
    by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, met, iso in token_ids:
        name, cls = metabolite_class(met, mammalian)
        by_class[cls][name].append(int(iso.split("+")[1]))
    ranked = sorted(by_class.items(), key=lambda kv: -sum(map(len, kv[1].values())))
    kept = [kv for i, kv in enumerate(ranked[:MAX_CLASSES])
            if i == 0 or sum(map(len, kv[1].values())) >= CLASS_MIN_SHARE * len(token_ids)]
    return "\n".join(f"{cls}: " + ", ".join(f"{m} {mass_range(ms)}" for m, ms in mets.items())
                     for cls, mets in kept)


def group_names(groups, flux_bc, token_bc, token_ids, pathways, mammalian):
    """``{bicluster: (flux-group label, token-group label)}``.

    A flux group is named by its two most common pathways, catch-alls aside; a token group
    by its main metabolite classes (``token_group_name``).
    """
    names = {}
    for k in np.union1d(flux_bc, token_bc):
        reactions = [n for g, b in zip(groups, flux_bc) if b == k for n in g["names"]]
        rows = top((pathway_of(r, pathways) for r in reactions), 2, CATCH_ALL) if reactions else ""
        t = np.flatnonzero(token_bc == k)
        cols = token_group_name(token_ids[t], mammalian) if len(t) else ""
        names[k] = (rows, cols)
    return names


def counts(items):
    """``'a (3); b (1)'`` for *items*, most common first."""
    return "; ".join(f"{k} ({n})" for k, n in collections.Counter(items).most_common())


def bicluster_summary(tracer, shown, flux_bc, token_bc, groups, token_ids, pathways):
    """One row per bicluster: its fluxes and tokens, what they are, and how distinct it is.

    ``block_mean`` is the mean shown value inside the bicluster's diagonal block and
    ``flux_mean_elsewhere`` its fluxes' mean over every other token, so their gap says how
    much more these fluxes draw on these tokens than on the rest.
    """
    out = []
    for k in np.unique(flux_bc):
        f, t = flux_bc == k, token_bc == k
        reactions = [n for g, keep in zip(groups, f) if keep for n in g["names"]]
        mets = [strip_compartment(token_ids[i][1]) for i in np.flatnonzero(t)]
        out.append(dict(
            tracer=tracer, bicluster=int(k),
            n_columns=int(f.sum()), n_reactions=len(reactions), n_tokens=int(t.sum()),
            block_mean=shown[np.ix_(f, t)].mean() if t.any() else np.nan,
            flux_mean_elsewhere=shown[np.ix_(f, ~t)].mean() if (~t).any() else np.nan,
            pathways=counts(pathway_of(r, pathways) for r in reactions),
            metabolites=counts(mets),
            reactions=";".join(reactions),
            tokens=";".join(f"{strip_compartment(token_ids[i][1])} {token_ids[i][2]}"
                            for i in np.flatnonzero(t)),
        ))
    return out


def stacked(label, width):
    """*label* with each of its lines wrapped to *width* characters."""
    return "\n".join(textwrap.fill(line, width, break_long_words=False)
                     for line in label.split("\n"))


def row_label(name):
    """A row name broken after its '&' when it has one, else wrapped."""
    if " & " in name:
        return name.replace(" & ", " &\n", 1)
    return stacked(name, ROW_NAME_CHARS)


def band_centers(labels):
    """``{bicluster: center}`` along an axis already sorted by bicluster."""
    return {k: np.flatnonzero(labels == k).mean() + 0.5 for k in np.unique(labels)}


def plot_tracer(lab, rows, cols, r_order, c_order, title, names=None, keyed=False):
    """Biclustered heatmap of fluxes (rows) against one tracer's labeling tokens.

    The fluxes share the fixed height and square cells set the width. *names*, a
    ``{bicluster: (flux group, token group)}`` dict, labels the row bands on the left and
    the column bands underneath; without it the map carries no labels.
    """
    corner_w = 0.5
    cell = FIG_HEIGHT / len(r_order)
    widths = [len(c_order) * cell, corner_w]
    fig = plt.figure(figsize=(sum(widths), FIG_HEIGHT))
    gs = fig.add_gridspec(1, 2, width_ratios=widths, wspace=0.01)

    ax = fig.add_subplot(gs[0, 0])
    # pcolormesh, not imshow: each cell stays a vector patch in the SVG instead of one bitmap.
    im = ax.pcolormesh(lab[r_order][:, c_order], cmap=CMAP, vmin=0, vmax=1,
                       edgecolors="face", linewidth=0.3)   # edges hide anti-alias seams
    for edges, line in ((np.flatnonzero(np.diff(rows[r_order])) + 1, ax.axhline),
                        (np.flatnonzero(np.diff(cols[c_order])) + 1, ax.axvline)):
        for e in edges:
            line(e, color="white", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_axis_off()

    ax.set_title(title, fontsize=11)
    if names:
        for k, y in band_centers(rows[r_order]).items():
            ax.text(-1, y, row_label(names[k][0]), ha="right", va="center",
                    fontsize=NAME_FONT)
        sorted_cols, key = cols[c_order], []
        for letter, (k, x) in zip("ABCDEFGHIJ", band_centers(sorted_cols).items()):
            if keyed:
                # Generated names are too long for a band: a letter under it, the name in a
                # key below that.
                label = letter
                key.append(f"{letter}   {'; '.join(names[k][1].split(chr(10)))}")
            else:
                width_pt = (sorted_cols == k).sum() * cell * 72
                label = stacked(names[k][1], max(8, int(width_pt / (0.55 * NAME_FONT))))
            ax.text(x, -0.008, label, ha="center", va="top", fontsize=NAME_FONT,
                    transform=ax.get_xaxis_transform())
        if key:
            ax.text(0, -0.04, "\n".join(key), ha="left", va="top", fontsize=NAME_FONT,
                    linespacing=1.5, transform=ax.transAxes)

    side = fig.add_subplot(gs[0, 1])
    side.set_axis_off()
    frac = CBAR_HEIGHT / FIG_HEIGHT
    cbar = fig.colorbar(im, cax=side.inset_axes([0.1, 1 - frac, 0.2, frac]))
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=7)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--data", choices=["validation", "experimental"], default="experimental")
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where cross_attention.py wrote its .npz files")
    parser.add_argument("--n-biclusters", type=int, default=6,
                        help="Spectral co-clustering biclusters per figure (default: 6)")
    args = parser.parse_args()
    suffix = "" if args.data == "validation" else f"_{args.data}"
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for m in selected_models(args):
        with np.load(results_path(args.results_dir, m["key"], args.data)) as f:
            d = dict(f)
        pathways = reaction_pathways(m["key"])
        matrix, groups = flux_columns(d, pathways)
        col_labels = [g["label"] for g in groups]

        n_lab = len(d["token_index"])
        tok_names = [f"{strip_compartment(met)} {iso}" for _, met, iso in d["token_index"]]
        meas_names = [str(n) for n in d["flux_names"][d["fixed_idx"]]]
        meas_rows = np.arange(n_lab, matrix.shape[0])

        title_suffix = f" ({args.data})" if suffix else ""
        measured = measured_tokens(d)

        records, summaries = [], []
        for tracer in dict.fromkeys(d["token_index"][:, 0]):
            tokens = np.flatnonzero(d["token_index"][:, 0] == tracer)
            kept, shown = tracer_matrix(d, matrix, tracer)
            flux_bc, token_bc, r_order, c_order = bicluster(shown, args.n_biclusters)
            hand = GROUP_NAMES.get((m["key"], tracer, args.data))
            fig = plot_tracer(shown, flux_bc, token_bc, r_order, c_order,
                              f"{m['label']} — {tracer_label(tracer)}{title_suffix}",
                              hand or group_names(groups, flux_bc, token_bc,
                                                  d["token_index"][kept], pathways,
                                                  m["key"] in MAMMALIAN_MODELS),
                              keyed=hand is None)
            save_figure(fig, args.out_dir /
                        f"cross_attention_{m['key']}_{tracer_slug(tracer)}{suffix}.svg")
            all_bc = np.full(len(tokens), -1)
            all_bc[measured[tokens]] = token_bc
            records.append((tracer_label(tracer), tokens, [tok_names[t] for t in tokens],
                            all_bc, flux_bc, measured[tokens]))
            summaries += bicluster_summary(tracer_label(tracer), shown, flux_bc, token_bc,
                                           groups, d["token_index"][kept], pathways)
        records.append((MEASURED_BLOCK, meas_rows, meas_names, np.full(len(meas_rows), -1),
                        np.full(len(groups), -1), np.ones(len(meas_rows), dtype=bool)))

        rows = []
        for block, tokens, names, token_bc, flux_bc, is_meas in records:
            for g, col, fbc in zip(groups, matrix.T, flux_bc):
                for rxn_idx, rxn in zip(g["members"], g["names"]):
                    rows.append(pd.DataFrame({
                        "block": block, "token": names, "measured": is_meas,
                        "column": g["label"], "reaction": rxn,
                        "token_bicluster": token_bc, "flux_bicluster": fbc,
                        "share": col[tokens],
                        "shown": np.where(is_meas, col[tokens] / col[tokens[is_meas]].max(),
                                          np.nan),
                        "mean_abs_F": d["F_abs"][rxn_idx, tokens],
                        "mean_F": d["F_signed"][rxn_idx, tokens],
                    }))
        csv = args.out_dir / f"cross_attention_{m['key']}{suffix}.csv"
        pd.concat(rows).to_csv(csv, index=False)
        print(f"Saved: {csv}")

        groups_csv = args.out_dir / f"cross_attention_{m['key']}{suffix}_groups.csv"
        pd.DataFrame({"column": col_labels,
                      "pathway": [g["pathway"] for g in groups],
                      "n_reactions": [len(g["names"]) for g in groups],
                      "reactions": [";".join(g["names"]) for g in groups]}
                     ).to_csv(groups_csv, index=False)
        print(f"Saved: {groups_csv}")

        bc_csv = args.out_dir / f"cross_attention_{m['key']}{suffix}_biclusters.csv"
        pd.DataFrame(summaries).to_csv(bc_csv, index=False)
        print(f"Saved: {bc_csv}")


if __name__ == "__main__":
    main()
