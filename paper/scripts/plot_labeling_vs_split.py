"""
Measured labeling of a token against the flux split it reports, across organisms.

A labeling fraction has no units: it reports how carbon divides at a branch point, not how fast
it flows. So each x axis is a split, the SigmaFlux-predicted flux into one branch over the flux
reaching the branch point, unitless and comparable across organisms. The PPP split is G6PDH over
glucose phosphorylation (hexokinase, or the PTS in E. coli): G6P has no other outlet, so it equals
G6PDH / (G6PDH + PGI). Oxygen uptake per glucose measures how much carbon is respired through the
TCA cycle, whose turns scramble acetyl-CoA's M+2 into M+1 and dilute first-turn M+2 citrate. RNA
synthesis per glucose draws ribose through the non-oxidative PPP, which recombines labeled and
unlabeled carbon into F6P M+3 on partially labeled U-13C-glucose.

The panels are the PPP readouts the sensitivity map names (Pyr M+2, and the upstream F6P, 3PG and
PEP M+2), and the other relations found across organisms.

One panel per (tracer, token, split) triple; each point is one experimental condition, and only
triples with two or more organisms are drawn. y is the normalized fraction, the measured
isotopologue fraction divided by the tracer enrichment (1,2-13C-glucose is 50% for the yeasts, 100%
for E. coli and the T cell), bars its replicate standard deviation. A panel with at least
``MIN_FIT`` conditions draws a least-squares line through them, pooled over organisms (in log x on
a log axis), with its 95% bootstrap confidence band.

A pair is drawn only when every organism contributing to it predicts every reaction of the split
with validation R² > ``MIN_R2`` (the ``r2_ai`` column of mfa_benchmark_r2_per_reaction_<key>.csv,
from plot_mfa_benchmark.py); otherwise it is skipped and the failing reactions are printed.

In ``paper/figures/``:
  labeling_vs_split_<tracer>_<token>_<split>.svg   one scatter per drawn triple
  labeling_vs_split.csv                            model, condition, gc, tracer, token, met,
                                                   iso, split, ratio, measured, measured_std,
                                                   enrichment
"""
import argparse
import glob
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _figure_common import (
    MODELS,
    add_common_args,
    apply_style,
    run_dir_for,
    save_figure,
    selected_models,
)

from sigmaflux.utils.read_reference_flux import read_predicted_fluxes

TRACERS = {"1,2-13C-glucose": "[1, 1, 0, 0, 0, 0]", "U-13C-glucose": "[1, 1, 1, 1, 1, 1]"}
MIN_R2 = 0.8
MIN_FIT = 3     # conditions a panel needs for a fitted line
COLORS = {
    "e_coli_core": "#4c72b0",
    "s_cerevisiae": "#dd8452",
    "i_orientalis": "#c44e52",
    "m_musculus_t_cell": "#55a868",
}
MARKERS = {"B": "o", "C": "s", "N": "^", "P": "D", "A": "P"}
# Split -> model -> (branch, reactions summed into the flux reaching the node); a model without
# the split is left out.
SPLITS = {
    "oxPPP": {
        "e_coli_core": ("R8", ("gluc_up",)),
        "s_cerevisiae": ("G6PDH2i_c", ("HEX1_c",)),
        "i_orientalis": ("G6PDH2i_c", ("HEX1_c",)),
        "m_musculus_t_cell": ("G6PDH2i_c", ("HEX1_c",)),
    },
    # Oxygen taken up per glucose: how much carbon is respired through the TCA cycle.
    "respiration": {k: ("O2t_c_e", ("HEX1_c",))
                    for k in ("s_cerevisiae", "i_orientalis", "m_musculus_t_cell")},
    "rna_synthesis": {k: ("RNA_c", ("HEX1_c",))
                      for k in ("s_cerevisiae", "i_orientalis", "m_musculus_t_cell")},
}
SPLIT_NAMES = {"oxPPP": "PPP / glucose uptake",
               "respiration": "O₂ uptake / glucose uptake",
               "rna_synthesis": "RNA synthesis / glucose uptake"}
LOG_X = {"respiration"}   # per glucose, a long tail
PAIRS = [   # (tracer, metabolite, isotopologue, split)
    ("1,2-13C-glucose", "pyr", "M+2", "oxPPP"),                  # pyruvate and the PPP
    ("1,2-13C-glucose", "f6p", "M+2", "oxPPP"),                  # the sensitivity map's PPP
    ("1,2-13C-glucose", "3pg", "M+2", "oxPPP"),                  # readouts, upstream of pyruvate
    ("1,2-13C-glucose", "pep", "M+2", "oxPPP"),
    ("1,2-13C-glucose", "asp", "M+1", "respiration"),            # TCA turns scramble acetyl-CoA
    ("1,2-13C-glucose", "cit", "M+2", "respiration"),            # M+2 into M+1 and dilute M+2
    ("U-13C-glucose", "f6p", "M+3", "rna_synthesis"),            # non-oxidative PPP scrambling
]
TOKEN_NAMES = {"pyr": "Pyr", "cit": "Cit", "f6p": "F6P", "3pg": "3PG", "pep": "PEP", "asp": "Asp"}


def metabolite(met_id):
    """``'ser__L'``, ``'g6p_c'``, ``'Pyr'`` -> ``'ser'``, ``'g6p'``, ``'pyr'``."""
    return re.sub(r"(__[LD])?(_[a-z])?$", "", str(met_id)).lower()


def enrichment(tracer):
    return float(tracer.rsplit("_", 1)[1])


def predicted_splits(model, conditions):
    """(condition, gc) -> {split: branch / flux reaching the node}, signed fluxes."""
    out = {}
    for cond, gcs in conditions.items():
        hits = sorted(glob.glob(str(run_dir_for(model) / "predictions" /
                                    f"predictions_*_{cond}_*.csv")))
        if not hits:
            continue
        pred = read_predicted_fluxes(hits[-1])
        for gc in gcs:
            if gc in pred.columns:
                out[cond, gc] = {
                    name: pred.at[a, gc] / sum(pred.at[b, gc] for b in node)
                    for name, ids in SPLITS.items() if model["key"] in ids
                    for a, node in [ids[model["key"]]]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_csv(args.out_dir / "experimental_mid_tokens.csv").assign(
        met=lambda t: t["Metabolite"].map(metabolite))

    rows, r2 = [], {}
    for model in selected_models(args):
        key = model["key"]
        mine = tokens[tokens["key"] == key]
        splits = predicted_splits(model, mine.groupby("condition")["gc"].unique().to_dict())
        r2[key] = pd.read_csv(args.out_dir / f"mfa_benchmark_r2_per_reaction_{key}.csv"
                              ).set_index("reaction")["r2_ai"]
        for tracer, met, iso, split in PAIRS:
            hit = mine[mine["Tracer"].str.contains(TRACERS[tracer], regex=False)
                       & (mine["met"] == met) & (mine["Isotopomer"] == iso)]
            for t in hit.itertuples():
                ratio = splits.get((t.condition, t.gc), {}).get(split)
                if ratio is not None:
                    rows.append({"model": key, "condition": t.condition, "gc": t.gc,
                                 "tracer": tracer, "token": f"{TOKEN_NAMES[met]} {iso}",
                                 "met": met, "iso": iso,
                                 "split": split,
                                 "ratio": ratio, "measured": t.measured,
                                 "measured_std": t.measured_std,
                                 "enrichment": enrichment(t.Tracer)})
    df = pd.DataFrame(rows)
    csv = args.out_dir / "labeling_vs_split.csv"
    df.to_csv(csv, index=False)
    print(f"Saved: {csv}")

    labels = {m["key"]: m["label"] for m in MODELS}
    for (tracer, token, split), sub in df.groupby(["tracer", "token", "split"], sort=False):
        if sub["model"].nunique() < 2:
            continue
        failing = [f"{labels[k]} {r} ({r2[k].get(r, float('nan')):.2f})"
                   for k in sub["model"].unique()
                   for r in {SPLITS[split][k][0], *SPLITS[split][k][1]}
                   if not r2[k].get(r, float("nan")) > MIN_R2]
        if failing:
            print(f"Skipped {tracer} {token} vs {split}: validation R² <= {MIN_R2} for "
                  f"{', '.join(failing)}")
            continue

        sub = sub.assign(organism=sub["model"].map(labels),
                         y=sub["measured"] / sub["enrichment"],
                         y_std=sub["measured_std"] / sub["enrichment"])
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        for k, g in sub.groupby("model"):
            ax.errorbar(g["ratio"], g["y"], yerr=g["y_std"], fmt="none", ecolor=COLORS[k],
                        elinewidth=0.8, capsize=0)
        if len(sub) >= MIN_FIT:
            sns.regplot(data=sub, x="ratio", y="y", scatter=False, logx=split in LOG_X, ci=95,
                        seed=0, color="0.35", line_kws={"linewidth": 1}, ax=ax)
        sns.scatterplot(data=sub, x="ratio", y="y", hue="organism", style="condition",
                        palette={labels[k]: c for k, c in COLORS.items() if k in labels},
                        markers=MARKERS, s=50, ax=ax)
        if split in LOG_X:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(SPLIT_NAMES[split])
        ax.set_ylabel(f"Normalized fraction\n({token})")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
        sns.despine(ax=ax)
        name = re.sub(r"[^A-Za-z0-9]+", "_", f"{tracer} {token.replace('α', 'a')} {split}")
        save_figure(fig, args.out_dir / f"labeling_vs_split_{name}.svg")


if __name__ == "__main__":
    main()
