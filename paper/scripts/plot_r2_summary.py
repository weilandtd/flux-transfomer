"""
R² of SigmaFlux against experimental data, every organism and condition on one axis.

  r2_flux.svg          predicted vs published MFA fit, one bar per (organism, condition)
  r2_labeling.svg      predicted vs measured isotopologue fraction, one bar per
                        (organism, condition), height = median over that condition's
                        tracers, with the per-tracer values as points
  r2_labeling_per_tracer.csv   the per-tracer R² behind the second figure

Two views of the same predictions, and they answer different questions. The flux panel
asks whether we agree with a *second estimate* of the fluxes -- a published MFA fit, with
its own identifiability problems. The labeling panel asks whether we reproduce the
measurement itself. Labeling depends on flux ratios, so a state can miss the other
estimate and still explain the data.

Labeling R² is computed **per tracer** rather than pooled, because a condition's tracers
are separate experiments on the same cells: pooling them hides a tracer that the model
reproduces poorly behind one it does well, and the spread across tracers is itself worth
seeing. The bar is the median so one bad tracer cannot carry the summary, and every tracer
is drawn as a point so the reader sees how many there are and how far apart they sit.

Both R² are linear and signed, the same definition as ``per_flux_r2`` in the simulation
benchmark and ``_r2_linear`` in plot_experimental_predictions.py, so R² means one thing
across the paper.

Reads what the two experimental scripts already wrote into --out-dir:
``experimental_flux_summary.csv`` and ``experimental_mid_tokens.csv``. Run those first.

Usage:
    python paper/scripts/plot_r2_summary.py
    python paper/scripts/plot_r2_summary.py --models s_cerevisiae,i_orientalis
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    CONDITION_LABEL, MODELS_BY_KEY, TRACER_MARKERS, add_common_args, apply_style,
    condition_color, save_figure, selected_models, tracer_label)
from _figure_common import COLORS as _SHARED_COLORS

apply_style()

COLORS = {**_SHARED_COLORS, "point": "#333333"}   # per-tracer markers on the bars

#: Legend entries in order. Hue is the resolved condition *name*, and its color comes
#: from _figure_common.condition_color, which the experimental scatter panels use too --
#: so a condition is the same color here as in the panel it summarizes.
CONDITION_ORDER = ["batch", "C-limited", "N-limited", "P-limited", "naive", "activated"]

#: Letter -> name for the organisms CONDITION_LABEL says nothing about, whose letters are
#: nutrient limitations.
LIMITATION_NAME = {"B": "batch", "C": "C-limited", "N": "N-limited", "P": "P-limited"}


#: x tick text, matching the NRMSE violin in plot_mfa_benchmark.py so the two read as
#: one figure set. Only the T cell needs wrapping at this width.
SHORT_LABEL = {
    "e_coli_core": "E. coli",
    "m_musculus_t_cell": "M. musculus\n(T cell)",
}

#: Same aspect as the NRMSE violin: four organisms with several bars each need the width.
FIGSIZE = (3.5 * 1.8, 3.5)


def condition_name(key, condition):
    """How this organism names the condition, e.g. ('m_musculus_t_cell', 'N') -> naive."""
    named = CONDITION_LABEL.get(key, {})
    if condition in named:
        return named[condition]
    return LIMITATION_NAME.get(condition, condition)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    return p.parse_args()


def r2_linear(pred, measured):
    """Signed R² in linear space; NaN when the measurements carry no variance."""
    if len(measured) < 2:
        return np.nan
    ss_tot = ((measured - measured.mean()) ** 2).sum()
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ((pred - measured) ** 2).sum() / ss_tot


def load_flux_r2(out_dir, keys):
    """One R² per (model, condition), straight out of the flux panel's summary."""
    path = out_dir / "experimental_flux_summary.csv"
    if not path.is_file():
        raise SystemExit(f"no {path}; run plot_experimental_predictions.py first")
    df = pd.read_csv(path)
    df = df[df["key"].isin(keys)][["model", "key", "condition", "r2"]].copy()
    df["name"] = [condition_name(k, c) for k, c in zip(df["key"], df["condition"])]
    df["color"] = [condition_color(k, c) for k, c in zip(df["key"], df["condition"])]
    return df


def load_labeling_r2(out_dir, keys):
    """R² per (model, condition, tracer), recomputed from the plotted token table."""
    path = out_dir / "experimental_mid_tokens.csv"
    if not path.is_file():
        raise SystemExit(f"no {path}; run plot_experimental_mid.py first")
    tok = pd.read_csv(path)
    tok = tok[tok["key"].isin(keys)]
    rows = []
    for (model, key, cond, tracer), g in tok.groupby(
            ["model", "key", "condition", "Tracer"], sort=False):
        rows.append(dict(
            model=model, key=key, condition=cond, name=condition_name(key, cond),
            color=condition_color(key, cond),
            tracer=tracer_label(tracer), n_tokens=len(g),
            r2=r2_linear(g["sim"].to_numpy(), g["measured"].to_numpy())))
    return pd.DataFrame(rows).dropna(subset=["r2"])


def _order(df, models):
    """Model order as MODELS lists them, and the condition names actually present."""
    keys = [m["key"] for m in models if m["key"] in set(df["key"])]
    labels = [SHORT_LABEL.get(k, MODELS_BY_KEY[k]["label"]) for k in keys]
    conds = [c for c in CONDITION_ORDER if c in set(df["name"])]
    return keys, labels, conds


def _finish(ax, df, conds, ylabel):
    """Shared axis furniture: zero line, limits that admit a negative R², legend.

    Returns the condition legend so a caller adding a second one can re-attach it --
    ``ax.legend()`` replaces the axes' legend rather than appending to it.
    """
    # A bar chart must not clip a negative R² -- an arm that predicts worse than the
    # mean is exactly what this figure exists to show, so the axis follows the data.
    lo = min(0.0, float(df["r2"].min()) * 1.1)
    ax.set_ylim(lo, 1.0)
    if lo < 0:
        ax.axhline(0.0, color=COLORS["reference"], lw=0.8, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    hue_color = dict(zip(df["name"], df["color"]))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=hue_color[c], edgecolor="none")
               for c in conds]
    # Outside on the right, one condition per row: bars run close to the top of the panel
    # across most organisms, so no in-axes corner stays reliably clear, and stacking the
    # entries keeps them in the same order as the bars within each group.
    legend = ax.legend(handles, conds,
                       frameon=False, fontsize=9, loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), ncol=1,
                       handlelength=1.0, handleheight=1.0, labelspacing=0.6)
    sns.despine(ax=ax)
    return legend


#: Fraction of the tick spacing one organism's whole group of bars may occupy.
GROUP_WIDTH = 0.8


def bar_positions(df, keys, conds):
    """x of every (organism, condition) bar, each organism's group centred on its tick.

    seaborn's dodge reserves a slot for every hue level in every group, so an organism
    measured under one condition gets one bar and several blank slots, pushed to whichever
    side its condition sits on in the hue order. Here each organism is given only as many
    slots as it has conditions, centred on its tick. Bar width is set by the organism with
    the most conditions, so widths stay comparable across the axis.
    """
    per_key = {k: [c for c in conds if not df[(df["key"] == k) & (df["name"] == c)].empty]
               for k in keys}
    width = GROUP_WIDTH / max(len(v) for v in per_key.values())
    pos = {}
    for i, k in enumerate(keys):
        present = per_key[k]
        for j, c in enumerate(present):
            pos[(k, c)] = i + (j - (len(present) - 1) / 2) * width
    return pos, width


def plot_flux(df, models, out_dir):
    keys, labels, conds = _order(df, models)
    pos, width = bar_positions(df, keys, conds)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for _, r in df.iterrows():
        ax.bar(pos[(r["key"], r["name"])], r["r2"], width=width,
               color=r["color"], linewidth=0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    _finish(ax, df, conds, r"Flux $R^2$ vs published MFA")
    save_figure(fig, out_dir / "r2_flux.svg")


def plot_labeling(df, models, out_dir):
    keys, labels, conds = _order(df, models)
    pos, width = bar_positions(df, keys, conds)
    # One marker per tracer, assigned over every tracer in the figure rather than per
    # organism, so a shape means the same thing in every column. Enrichment is part of the
    # identity: 50% and 100% U-13C behave differently, so they are not one tracer.
    tracers = sorted(df["tracer"].unique())
    marker = {t: TRACER_MARKERS[i % len(TRACER_MARKERS)] for i, t in enumerate(tracers)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (key, cond), g in df.groupby(["key", "name"], sort=False):
        x = pos[(key, cond)]
        # Median, not mean: with two or three tracers per condition one poorly reproduced
        # tracer would drag a mean well below where most of the data sits.
        ax.bar(x, g["r2"].median(), width=width, color=g["color"].iloc[0], linewidth=0)
        # Spread by rank rather than jittered: with two or three points the shapes have to
        # be told apart, and a random offset can overlap them.
        offs = np.linspace(-width / 4, width / 4, len(g)) if len(g) > 1 else [0.0]
        for off, (_, r) in zip(offs, g.sort_values("tracer").iterrows()):
            ax.scatter(x + off, r["r2"], s=16, marker=marker[r["tracer"]],
                       facecolor="none", edgecolor=COLORS["point"], linewidth=0.9,
                       zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    cond_legend = _finish(ax, df, conds, r"Labeling $R^2$ per tracer")
    # Two legends: color is condition (right), shape is tracer (below). The first has to
    # be re-attached as an artist, because ax.legend() replaces ax.legend_ rather than
    # adding to it.
    ax.add_artist(cond_legend)
    handles = [plt.Line2D([], [], marker=marker[t], linestyle="", markerfacecolor="none",
                          markeredgecolor=COLORS["point"], markeredgewidth=0.9, label=t)
               for t in tracers]
    tracer_legend = ax.legend(handles=handles, frameon=False, fontsize=7,
                              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
                              handlelength=1.0, columnspacing=1.0)
    save_figure(fig, out_dir / "r2_labeling.svg",
                extra_artists=(cond_legend, tracer_legend))


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = selected_models(args)
    keys = {m["key"] for m in models}

    flux = load_flux_r2(out_dir, keys)
    lab = load_labeling_r2(out_dir, keys)

    print("── Flux R² vs published MFA ──")
    for _, r in flux.iterrows():
        print(f"  {r['model']:20s} {r['name']:11s} {r['r2']:8.4f}")
    plot_flux(flux, models, out_dir)

    print("\n── Labeling R² per tracer ──")
    for (model, cond), g in lab.groupby(["model", "name"], sort=False):
        print(f"  {model:20s} {cond:11s} median {g['r2'].median():7.4f}  "
              f"({len(g)} tracers: {', '.join(f'{v:.3f}' for v in g['r2'])})")
    plot_labeling(lab, models, out_dir)

    path = out_dir / "r2_labeling_per_tracer.csv"
    lab.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
