"""
Does the model attend to the isotopomers that actually carry the information?

The existing attention analyses aggregate the ~540 tokens down to metabolites, taking a
max or mean over each metabolite's (isotopomer, tracer) tokens. That is the right
granularity for a network-wide map, but it destroys the one thing that makes a ¹³C
experiment interpretable: *which* mass isotopomer is informative. This script keeps the
tokens at M+n resolution, and it keeps the **measured-flux tokens** too, which the
metabolite-aggregating scripts drop entirely.

The textbook case is the split of pyruvate between carboxylation and decarboxylation:

    PC_c    pyr + hco3       -> oaa        C3 + C1 -> C4, all three labelled carbons kept
    PDH_m   pyr + coa + nad  -> accoa + co2   C3 -> C2, one carbon leaves as CO2

Under U-¹³C glucose, pyruvate is a mixture of M+3 and M+0. Carboxylation therefore writes
a **malate M+3** (the intact 3-carbon unit plus an unlabelled bicarbonate); the
dehydrogenase route writes an acetyl unit that condenses onto OAA and, after a turn of the
cycle, shows up as **malate M+2**. So malate M+3 vs M+2 is the direct readout of the PC:PDH
split, and a model that has learned the biochemistry should attend there when predicting
those two fluxes.

Usage:
    python paper/analysis/attention/tca_attention.py \\
        paper/.results/i_orientalis_20260614/attention/flux_token_attn.npz

Writes tca_attention.svg and tca_attention.csv next to the npz.
"""

import argparse
import pathlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.use("Agg")
mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

# Black -> amber -> gold, built with seaborn's blend_palette so the ramp is anchored on the
# project's amber (#c17429) rather than borrowing inferno/hot, which detour through purple
# or red. Same family as the `dark:<hex>` palettes used by the predicted-flux panels.
BLACK_GOLD = sns.blend_palette(["#000000", "#c17429", "#f7d354"], as_cmap=True)

COLORS = {"cmap": BLACK_GOLD, "flux_token": "#ffffff"}

# Tracers are stored in the token label as their raw labelling vector; map them to the
# names used in the configs, and give each a colour for the strip above the heatmap.
TRACERS = {
    "[1, 1, 1, 1, 1, 1]": ("U-$^{13}$C glucose (50%)", "#1B6F6A"),
    "[1, 1, 0, 0, 0, 0]": ("1,2-$^{13}$C glucose (50%)", "#7fb3d5"),
}
FLUX_TRACER = ("measured flux", "#c17429")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Reaction groups, in pathway order. Reactions absent from an organism's model are skipped
# rather than erroring, so the same lists serve both organisms.
GROUPS = {
    "tca": [
        "PYK_c", "PYRtps_m",                    # pyruvate supply
        "PC_c", "PDH_m",                        # the carboxylation / decarboxylation split
        "CS_m", "ACONTa_m", "ACONTb_m",         # citrate -> isocitrate
        "ICDHx_m", "AKGDH_m", "SUCOAS_m",       # oxidative span
        "SUCDq7_m", "FUM_m", "MDH_m",           # succinate -> malate -> OAA
        "ME1_m", "ME2_m", "PPCK_c",             # malate / OAA decarboxylating exits
        "ICL_1_c", "MALS_c",                    # glyoxylate shunt
    ],
    # Upper glycolysis keeps the carbon skeleton intact (C6 -> 2 x C3); the split at aldolase
    # is where a hexose M+6 becomes two triose M+3.
    "glycolysis": [
        "HEX1_c", "PGI_c", "PFK_c", "FBP_c", "FBA_c", "TPI_c",
        "GAPD_c", "PGK_c", "PGM_c", "ENO_c", "PYK_c",
    ],
    # Canonical net-flux steps of upper glycolysis + PPP in one figure. Excluded on purpose:
    # FBP_c (gluconeogenic reverse of PFK), FBA3_c / SBP_c (sedoheptulose-bisphosphate
    # bypass), and TKThalf_c / TKT1h_c / TKT2h_c (this model splits transketolase into half
    # reactions through a tktc2_c carrier, which is a modelling device, not a net step).
    "glycolysis_ppp": [
        "HEX1_c", "PGI_c", "PFK_c", "FBA_c", "TPI_c",            # hexose -> triose
        "GAPD_c", "PGK_c", "PGM_c", "ENO_c", "PYK_c",            # lower glycolysis
        "G6PDH2i_c", "PGL_c", "GND_c",                           # oxidative PPP (CO2 out)
        "RPE_c", "RPI_c", "TALA_c",                              # pentose interconv. + transaldolase
    ],
    # The oxidative branch decarboxylates (GND releases CO2, so a hexose M+6 yields a pentose
    # M+5); the non-oxidative branch (TKT/TALA) reshuffles carbon between C3/C4/C5/C6/C7 pools
    # and is what puts non-M+0/M+6 species back into F6P.
    "ppp": [
        "G6PDH2i_c", "PGL_c", "GND_c",          # oxidative, CO2-releasing
        "RPE_c", "RPI_c",                       # pentose interconversion
        "TKThalf_c", "TKT1h_c", "TKT2h_c",      # transketolase
        "TALA_c", "FBA3_c", "SBP_c",            # transaldolase / sedoheptulose bypass
        "PGI_c", "PFK_c",                       # the glycolytic reactions they feed back into
    ],
}

# Pathway each reaction belongs to, so combined figures can be blocked and labelled by
# pathway rather than relying on list order being read correctly.
PATHWAY_OF = {
    **{r: "glycolysis" for r in ["HEX1_c", "PGI_c", "PFK_c", "FBP_c", "FBA_c", "TPI_c",
                                 "GAPD_c", "PGK_c", "PGM_c", "ENO_c", "PYK_c"]},
    **{r: "oxidative PPP" for r in ["G6PDH2i_c", "PGL_c", "GND_c"]},
    **{r: "non-ox PPP" for r in ["RPE_c", "RPI_c", "TALA_c", "TKThalf_c", "TKT1h_c",
                                 "TKT2h_c", "FBA3_c", "SBP_c"]},
}

# Metabolites whose isotopomers report on each group.
FOCUS = {
    "tca": ["mal__L", "oaa", "cit", "akg", "succ", "fum", "pyr", "asp__L", "glu__L"],
    "glycolysis": ["g6p", "f6p", "fdp", "dhap", "g3p", "3pg", "pep", "pyr"],
    "ppp": ["g6p", "6pgc", "r5p", "s7p", "e4p", "f6p", "fdp", "g3p", "pep", "pyr"],
    "glycolysis_ppp": ["g6p", "f6p", "fdp", "dhap", "g3p", "3pg", "pep", "pyr",
                       "6pgc", "r5p", "s7p", "e4p"],
}

# The isotopomer each group's split is classically read from, printed as a reminder.
READOUT = {
    "tca": ("malate M+2 vs M+3 under U-13C: PC keeps all three pyruvate carbons (-> M+3), "
            "PDH loses one as CO2 and its acetyl unit enters the cycle (-> M+2)"),
    "glycolysis": ("hexose M+6 vs triose M+3 under U-13C: aldolase halves an intact M+6 "
                   "into two M+3, so a triose pool that is not M+3/M+0 has been reshuffled"),
    "glycolysis_ppp": ("hexose/triose M+3 and M+6 track the intact glycolytic split; F6P "
                       "M+1/M+2/M+4 and the pentoses report PPP recycling, and under "
                       "[1,2-13C] glucose M+1 vs M+2 in the C3 pools separates oxidative "
                       "PPP (C1 lost as CO2) from glycolysis"),
    "ppp": ("M+1 vs M+2 in the C3 pools (pep, pyr, 3pg) under [1,2-13C] glucose: glycolysis "
            "keeps both labelled carbons (-> M+2), while the oxidative PPP releases C1 as "
            "CO2 and yields M+1. F6P carrying M+1/M+2/M+4 is non-oxidative PPP recycling"),
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", type=pathlib.Path, help="flux_token_attn.npz from show_attention.py")
    p.add_argument("--top-tokens", type=int, default=22,
                   help="Tokens to show, chosen by peak attention across the reaction set")
    p.add_argument("--group", default="tca", choices=list(GROUPS),
                   help="Reaction group to analyse")
    p.add_argument("--focus-only", action="store_true",
                   help="Restrict labeling tokens to the group's FOCUS metabolites")
    p.add_argument("--metabolite", default=None,
                   help="Metabolite for the per-isotopomer evidence table")
    p.add_argument("--features", type=pathlib.Path, default=None,
                   help="features.csv (default: <npz>/../../features.csv)")
    p.add_argument("--mid-xlsx", type=pathlib.Path, default=None,
                   help="Measured MID xlsx for the across-condition variation column")
    p.add_argument("--tracer-sheet", default="U-13C glucose (50%)")
    p.add_argument("--mixing", default="quadrature",
                   choices=["quadrature", "convex", "signed"],
                   help="How to project latent-query attention onto reactions; see "
                        "load_matrix. convex sums to 1 per reaction, signed does not.")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    return p.parse_args()


def load_matrix(npz_path: pathlib.Path, mixing: str = "convex") -> pd.DataFrame:
    """Reaction x token attention, projected from the latent queries onto reactions.

    Attention itself is defined per *latent query*: `attn_mean` has one softmax row per
    free null-space coordinate y_k (plus reversibility queries, dropped upstream), so each
    row sums to exactly 1. Reactions are not queries — they are recovered from the latent
    coordinates through v = v_particular + Z·y — so a reaction-level quantity has to be
    projected, and how you project changes what it means.

    signed   `Z @ attn_mean`, as saved by show_attention.py. This is the first-order
             sensitivity dv_i/dy_k * alpha_kj summed over k. Z is signed, so rows do NOT
             sum to 1 (they sum to sum_k Z_ik: here −2.06…+1.54, negative for 91 of 387
             reactions) and opposite-signed latent dims cancel *before* the abs() is taken
             — up to 96% of the mass for the worst decile of reactions. Useful as a
             sensitivity, but it is not attention and must not be read as a share.

    convex   row-normalise |Z| into mixing weights: w_ik = |Z_ik| / sum_k |Z_ik|, then
             P = W @ attn_mean. A convex combination of softmaxes, so it sums to 1 and is a
             genuine distribution over tokens. No cancellation.

    quadrature (default)  combine in quadrature instead: w_ik = Z_ik^2 / sum_k Z_ik^2.
             Treating the latent coordinates as independent, Var(v_i) = sum_k Z_ik^2, so
             these weights are exactly the *share of the flux's variance* contributed by
             each latent dimension. Sign-free by construction, sums to 1, and marginally
             more reaction-specific than |Z| (mean row-row r 0.35 vs 0.41).

    signed vs quadrature rankings agree to Spearman ~0.77; |Z| vs Z^2 to ~0.97, so the
    choice between the two sign-free variants is not load-bearing. Note the median flux has
    an effective dependence on only ~1 latent dimension, which is why all three coincide
    for most reactions and diverge only for the minority that span several.
    """
    d = np.load(npz_path, allow_pickle=True)
    index = [str(x) for x in d["flux_names"]]
    columns = [str(x) for x in d["token_labels"]]

    if mixing == "signed":
        mat = np.abs(d["flux_token"])
    else:
        w = np.abs(d["Z"]) if mixing == "convex" else d["Z"] ** 2
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        mat = w @ d["attn_mean"]
    return pd.DataFrame(mat, index=index, columns=columns)


def pretty_names(mid_xlsx) -> dict:
    """model metabolite stem -> the short name used in the experimental sheets (fdp -> FBP)."""
    org = "I_orientalis" if "I_orientalis" in str(mid_xlsx) else "S_cerevisiae"
    csv = _REPO_ROOT / "models" / "atomistic" / org / f"{org}_mapping.csv"
    if not csv.exists():
        return {}
    m = pd.read_csv(csv)
    m["base"] = m.model_id.astype(str).str.replace(r"_[a-z]$", "", regex=True)
    return m.drop_duplicates("base").set_index("base").experimental_id.astype(str).to_dict()


def axis_labels(columns, rows, names: dict):
    """Readable tick labels, plus the tracer of each column for the colour strip."""
    xs, tracers = [], []
    for c in columns:
        met, mid, trac = parse_token(c)
        if trac == "flux":
            xs.append(met)
            tracers.append(None)
        elif c.startswith("other labeling"):
            xs.append(c)
            tracers.append(None)
        else:
            xs.append(f"{names.get(met, met)} M+{mid}")
            tracers.append(trac)
    # Compartment suffixes only help when the same enzyme appears in two compartments.
    stems = [re.sub(r"_[a-z]$", "", r) for r in rows]
    ys = stems if len(set(stems)) == len(stems) else list(rows)
    return xs, ys, tracers


def token_kind(label: str) -> str:
    return "flux" if label.endswith("[flux]") else "labeling"


def parse_token(label: str) -> tuple:
    """'mal__L M+2 [U]' -> ('mal__L', 2, 'U'); flux tokens -> (name, None, 'flux')."""
    if label.endswith("[flux]"):
        return label.replace(" [flux]", ""), None, "flux"
    m = re.match(r"^(\S+)\s+M\+(\d+)\s+\[(.*)\]$", label)
    if not m:
        return label, None, ""
    return m.group(1), int(m.group(2)), m.group(3)


def select(attn: pd.DataFrame, reactions, focus_mets, n_top: int):
    rows = [r for r in reactions if r in attn.index]
    missing = [r for r in reactions if r not in attn.index]
    sub = attn.loc[rows]

    if focus_mets:
        keep = [c for c in sub.columns
                if token_kind(c) == "flux" or parse_token(c)[0] in focus_mets]
        sub = sub[keep]

    # Row-normalise before ranking: each reaction's attention sums to 1 over all tokens, so
    # without this the ranking is dominated by whichever reactions attend most sharply.
    frac = sub.div(sub.sum(axis=1), axis=0)
    # Always retain the flux tokens — they are the point of including them — plus the
    # labeling tokens with the highest peak share across this reaction set.
    flux_cols = [c for c in frac.columns if token_kind(c) == "flux"]
    lab_cols = [c for c in frac.columns if token_kind(c) == "labeling"]
    top_lab = frac[lab_cols].max(axis=0).sort_values(ascending=False).head(n_top).index.tolist()

    order = sorted(top_lab, key=lambda c: (parse_token(c)[0], parse_token(c)[1] or 0))
    # Positional selection: token labels repeat (a metabolite's _c and _m tokens strip to the
    # same name), so frac[order] would pull every column matching each duplicated label.
    want = list(dict.fromkeys(order + flux_cols))
    pos, seen = [], set()
    for name in want:
        for j, c in enumerate(frac.columns):
            if c == name and j not in seen:
                pos.append(j); seen.add(j); break
    shown = frac.iloc[:, pos]

    # Shares are of each reaction's total over ALL tokens, so the displayed columns sum to
    # well under 100%. The remainder is kept as a column in the CSV so the table is complete
    # and auditable, but it is NOT plotted: at up to ~77% it would dominate the colour scale
    # and flatten every cell that the figure exists to show.
    n_hidden = len(lab_cols) - len(order)
    complete = shown.copy()
    complete[f"other labeling ({n_hidden})"] = 1.0 - shown.sum(axis=1)
    return shown, complete, rows, missing, flux_cols


def plot(frac: pd.DataFrame, flux_cols, out_path: pathlib.Path, label: str,
         names: dict = None):
    names = names or {}
    xs, ys, tracers = axis_labels(frac.columns, frac.index, names)
    pathways = [PATHWAY_OF.get(r, "") for r in frac.index]

    # Thin strip above the heatmap encodes which tracer each column came from, so the tick
    # labels can stay short instead of carrying a raw labelling vector.
    #
    # The colour bar gets its own gridspec column rather than being placed by seaborn. Left
    # to itself, seaborn carves the colour bar out of the heatmap axes only, which makes that
    # axes narrower than the tracer strip above it and leaves the strip overhanging to the
    # right. Reserving a column keeps both axes exactly the same width.
    fig = plt.figure(figsize=(max(6.5, 0.30 * frac.shape[1] + 3.0),
                              0.34 * frac.shape[0] + 2.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.045, 1], width_ratios=[1, 0.022],
                          hspace=0.02, wspace=0.06)
    cax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=cax)
    cbar_ax = fig.add_subplot(gs[1, 1])

    sns.heatmap(frac * 100, ax=ax, cmap=COLORS["cmap"], linewidths=0.4, linecolor="white",
                cbar_ax=cbar_ax, cbar_kws={"label": label}, annot=False)
    for j, t in enumerate(tracers):
        colour = TRACERS.get(t, (None, FLUX_TRACER[1]))[1] if t is not None else FLUX_TRACER[1]
        cax.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=colour, edgecolor="white", lw=0.4))
    cax.set_xlim(0, frac.shape[1]); cax.set_ylim(0, 1); cax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for _, c in list(TRACERS.values()) + [FLUX_TRACER]]
    labels = [n for n, _ in list(TRACERS.values()) + [FLUX_TRACER]]
    cax.legend(handles, labels, ncol=len(labels), frameon=False, fontsize=8,
               loc="lower left", bbox_to_anchor=(0.0, 1.15), handlelength=1.2,
               columnspacing=1.2, handletextpad=0.5)

    if flux_cols:
        ax.axvline(frac.shape[1] - len(flux_cols), color=COLORS["flux_token"], lw=2.0)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels(xs, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_yticklabels(ys, rotation=0, fontsize=9)

    if any(pathways):
        start = 0
        for i in range(1, len(pathways) + 1):
            if i == len(pathways) or pathways[i] != pathways[start]:
                if start > 0:
                    ax.axhline(start, color="white", lw=2.2)
                if pathways[start]:
                    # Left of the y tick labels, offset in points so the colour bar can sit
                    # tight against the heatmap on the right.
                    ax.annotate(pathways[start], xy=(0, (start + i) / 2),
                                xycoords=("axes fraction", "data"),
                                xytext=(-62, 0), textcoords="offset points",
                                rotation=90, va="center", ha="center", fontsize=9)
                start = i

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def isotopomer_evidence(attn: pd.DataFrame, metabolite: str, features_csv, mid_xlsx,
                        tracer_sheet: str, reactions):
    """For one metabolite, line up all three kinds of evidence per isotopomer.

    attention  — what the trained model actually leans on (share of the reaction's total)
    Fisher     — what the D-optimal selection judged informative *on the simulated prior*,
                 and whether the isotopomer therefore made it into the model's inputs
    measured   — how much the isotopomer actually moves across growth conditions in the
                 real data, in units of its own replicate SD

    The three can disagree, and where they do is the interesting part: an isotopomer that
    the biochemistry says is diagnostic is useless if it does not vary, and invisible to the
    model if selection dropped it.
    """
    print(f"\n=== Evidence per {metabolite} isotopomer ===")
    rows = {}

    # Attention, summed over tracers and compartments, averaged over the reaction set.
    # Indexed positionally: token labels repeat (a metabolite's _c and _m tokens both strip
    # to the same name), so label-based selection would return a frame instead of a column.
    frac_all = attn.div(attn.sum(axis=1), axis=0)
    keep_rows = [i for i, r in enumerate(frac_all.index) if r in set(reactions)]
    for j, c in enumerate(attn.columns):
        met, mid, _ = parse_token(c)
        if met == metabolite and mid is not None:
            rows.setdefault(mid, {}).setdefault("attn", 0.0)
            rows[mid]["attn"] += float(frac_all.iloc[keep_rows, j].mean()) * 100

    if features_csv and pathlib.Path(features_csv).exists():
        f = pd.read_csv(features_csv)
        f["base"] = f.metabolite_id.astype(str).str.replace(r"_[a-z]$", "", regex=True)
        for mid_lbl, g in f[f.base == metabolite].groupby("isotopomer"):
            n = int(str(mid_lbl).replace("M+", ""))
            rows.setdefault(n, {})
            rows[n]["nats"] = float(g.standalone_nats.max())
            rows[n]["selected"] = bool(g.included.any())

    if mid_xlsx and pathlib.Path(mid_xlsx).exists():
        sheet = pd.read_excel(mid_xlsx, sheet_name=tracer_sheet)
        # Experimental compound names do not follow the model ids ("FBP" for fdp_c), so map
        # them explicitly rather than by string prefix.
        cand = []
        map_csv = _REPO_ROOT / "models" / "atomistic" / ("I_orientalis" if "I_orientalis" in str(mid_xlsx)
                                           else "S_cerevisiae")
        map_csv = map_csv / f"{map_csv.name}_mapping.csv"
        if map_csv.exists():
            mp = pd.read_csv(map_csv)
            mp["base"] = mp.model_id.astype(str).str.replace(r"_[a-z]$", "", regex=True)
            cand = mp.loc[mp.base == metabolite, "experimental_id"].astype(str).unique().tolist()
        cand = [c for c in cand if c in set(sheet["Compound"].astype(str))]
        if cand:
            sub = sheet[sheet["Compound"].astype(str) == cand[0]]
            pat = re.compile(r"^([A-Z])(\d+)_Rep(\d+)$")
            groups = {}
            for c in sub.columns:
                m = pat.match(str(c))
                if m:
                    groups.setdefault(int(m.group(2)), []).append(c)
            for _, r in sub.iterrows():
                n = int(r["C_Label"])
                means = [np.mean([r[c] for c in cols]) for cols in groups.values()]
                sd = np.mean([np.std([r[c] for c in cols], ddof=1) for cols in groups.values()])
                rows.setdefault(n, {})
                rows[n]["range"] = max(means) - min(means)
                rows[n]["snr"] = (max(means) - min(means)) / sd if sd > 0 else np.nan

    hdr = (f"  {'M+n':>4} {'attention%':>11} {'Fisher nats':>12} {'in model':>9} "
           f"{'meas. range':>12} {'range/repSD':>12}")
    print(hdr, "\n ", "-" * (len(hdr) - 2))
    for n in sorted(rows):
        r = rows[n]
        sel = r.get("selected")
        print(f"  M+{n:<2} {r.get('attn', float('nan')):11.2f} {r.get('nats', float('nan')):12.2f} "
              f"{('yes' if sel else 'NO') if sel is not None else '?':>9} "
              f"{r.get('range', float('nan')):12.3f} {r.get('snr', float('nan')):12.1f}")
    print("  (attention is the mean over the reaction set; blank = no such token in the model)")


def main():
    args = parse_args()
    attn = load_matrix(args.npz, args.mixing)
    reactions = GROUPS[args.group]
    focus_mets = FOCUS[args.group] if args.focus_only else None
    frac, complete, rows, missing, flux_cols = select(attn, reactions, focus_mets,
                                                     args.top_tokens)
    out_dir = args.out_dir or args.npz.parent

    print(f"{attn.shape[0]} reactions x {attn.shape[1]} tokens in {args.npz.name}")
    n_flux_all = sum(token_kind(c) == "flux" for c in attn.columns)
    print(f"  token kinds: {attn.shape[1] - n_flux_all} labeling + {n_flux_all} measured-flux")
    print(f"  group '{args.group}': showing {len(rows)} of {len(reactions)} reactions"
          + (f"; not in this model: {missing}" if missing else ""))

    features = args.features or (args.npz.parent.parent / "features.csv")
    mid_xlsx = args.mid_xlsx
    if mid_xlsx is None:
        org = "I_orientalis" if "i_orientalis" in str(args.npz) else "S_cerevisiae"
        mid_xlsx = _REPO_ROOT / ("data/I_orientalis" if org == "I_orientalis"
                                 else "data/S_cervesiae") / f"{org}_C.xlsx"

    complete.to_csv(out_dir / f"attention_{args.group}.csv")
    label = ("% of reaction's |sensitivity|" if args.mixing == "signed"
             else "% of reaction's attention")
    plot(frac, flux_cols, out_dir / f"attention_{args.group}.svg", label,
         pretty_names(mid_xlsx))

    print(f"\n=== Row completeness (shares are of each reaction's total over all "
          f"{attn.shape[1]} tokens) ===")
    cap = 100 * frac.sum(axis=1)
    print(f"  plotted columns capture {cap.mean():.1f}% on average "
          f"(range {cap.min():.0f}-{cap.max():.0f}%); the rest is on the {attn.shape[1] - frac.shape[1]} "
          f"tokens not shown, kept as 'other labeling' in the CSV so its rows sum to 100%.")

    print("\n=== Top-attended token per reaction ===")
    for r in frac.index:
        top = frac.loc[r].sort_values(ascending=False).head(3)
        print(f"  {r:10s} " + "  ".join(f"{c} {100 * v:.1f}%" for c, v in top.items()))

    if flux_cols:
        fl = frac[flux_cols].sum(axis=1) * 100
        print(f"\n=== Share of attention on measured-flux tokens ({len(flux_cols)}) ===")
        for r, v in fl.sort_values(ascending=False).items():
            print(f"  {r:10s} {v:5.1f}%")

    print(f"\n=== Classical readout for '{args.group}' ===\n  {READOUT[args.group]}")
    default_met = {"tca": "mal__L", "glycolysis": "fdp", "ppp": "f6p",
                   "glycolysis_ppp": "f6p"}[args.group]
    for met in ([args.metabolite] if args.metabolite else [default_met]):
        isotopomer_evidence(attn, met, features, mid_xlsx, args.tracer_sheet, rows)
    print(f"\nSaved: {out_dir / f'attention_{args.group}.csv'}")


if __name__ == "__main__":
    main()
