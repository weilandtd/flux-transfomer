"""
SigmaFlux's predicted labeling against the measured ¹³C data it was inferred from.

Per model:
  experimental_mid_<key>.svg      predicted vs measured isotopologue fraction, log-log,
                                   one point per (condition, growth condition, tracer,
                                   metabolite, isotopomer), with error bars on both axes;
                                   no R² text on the panel -- it lives in the summary CSV
Across all models:
  experimental_mid_summary.csv    per-(model, condition, growth condition) R², RMSE, MAE,
                                   worst error, median |z| against replicate spread, the
                                   fraction of tokens inside one replicate SD, and the
                                   fraction whose measurement falls in the predicted
                                   interval
  experimental_mid_tokens.csv     every plotted token, all models pooled

This closes the loop the flux panels leave open. plot_experimental_predictions.py asks
whether our fluxes agree with a *published MFA fit* -- a second estimate, with its own
identifiability problems, on the same data. This asks whether they reproduce the
measurement itself. Labeling depends on flux *ratios*, so a state can be far from another
estimate in flux space and still explain the data, or close to it and not -- which is
exactly why both panels are needed.

Reads the token tables paper/scripts/experimental_mid.py writes into each run's
``experimental_mid/`` directory; it does no simulation of its own, so a change of floor,
scale or annotation costs a second rather than another pass over the whole bootstrap. Run
the compute step first (paper/submit/submit_experimental_mid.sh).

The x error bar is the measured replicate spread -- the floor on how well any method can
do. The y bar is the bootstrap interval of the prediction, propagated through the EMU pass
draw by draw (see experimental_mid.py).

Axes are log-log: measured fractions run over several decades, and a linear panel puts
most tokens in the bottom corner. ``--min-fraction`` sets the floor below which a token is
dropped rather than drawn, defaulting to 1e-3: below that a fraction is far under the
replicate spread the measurement itself carries, so the point says more about the noise
floor than about the prediction. Anything non-positive is dropped regardless -- a log axis
cannot place it.

Each panel reports R² per condition, on the fractions in linear space -- the same
definition the flux panels and the simulation benchmark use, so R² means one thing across
the paper. RMSE and the interval-coverage fraction stay in the summary CSV.

A model with no token table yet is skipped with a warning rather than failing the run, so
the figure set fills itself in as the compute step lands.

Usage:
    python paper/scripts/plot_experimental_mid.py
    python paper/scripts/plot_experimental_mid.py --models m_musculus_t_cell
    python paper/scripts/plot_experimental_mid.py --min-fraction 0
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    CONDITION_COLORS, CONDITION_LABEL, COLORS, PREDICTION_SPECS, TRACER_MARKERS,
    add_common_args, apply_style, axis_floor, error_offsets, run_dir_for, save_figure,
    selected_models, tracer_label)
from experimental_mid import tokens_path

#: Hard floor for the log axes. Only a backstop: the limit is set from the data
#: (:func:`_figure_common.axis_floor`), and this stops a single near-zero fraction from
#: stretching the panel over a dozen empty decades.
MID_FLOOR = 1e-5

apply_style()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--min-fraction", type=float, default=1e-3,
                   help="Drop tokens whose measured or predicted fraction is at or below "
                        "this; below the replicate spread a point describes the noise "
                        "floor, not the prediction (default: 1e-3)")
    return p.parse_args()


def drop_unplottable(df, min_fraction):
    """Tokens a log axis can show, plus how many were dropped.

    A measured fraction of exactly zero -- the instrument saw nothing at that mass -- has
    no position on a log axis. It is dropped rather than floored, because flooring would
    draw it at whatever the axis bottom happens to be and invite reading a value off it.
    """
    keep = (df["measured"] > min_fraction) & (df["sim"] > min_fraction)
    return df[keep].copy(), int((~keep).sum())


def _r2(pred, measured):
    """Linear R² of predicted against measured fractions.

    Linear, not on log10 of the fractions the axes show: this is the same definition as
    ``_r2_linear`` in plot_experimental_predictions.py and ``per_flux_r2`` in the
    benchmark, so R² means one thing across the paper. It is therefore carried by the
    large fractions, which the log axes deliberately are not.
    """
    if len(measured) < 2:
        return float("nan")
    ss_res = ((pred - measured) ** 2).sum()
    ss_tot = ((measured - measured.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def summarize(model, df):
    """Per-(condition, growth condition) agreement statistics."""
    rows = []
    for (cond, gc), grp in df.groupby(["condition", "gc"], sort=False):
        err = grp["error"].to_numpy()
        sd = grp["measured_std"].to_numpy()
        inside = ((grp["Value_lo"] <= grp["measured"]) &
                  (grp["measured"] <= grp["Value_hi"]))
        rows.append(dict(
            model=model["label"], key=model["key"], condition=cond, gc=int(gc),
            label=CONDITION_LABEL.get(model["key"], {}).get(cond, cond),
            n_tokens=len(grp),
            r2=_r2(grp["sim"].to_numpy(), grp["measured"].to_numpy()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mae=float(np.mean(np.abs(err))),
            max_abs_error=float(np.max(np.abs(err))),
            median_abs_z=float(np.median(np.abs(grp["z"].to_numpy()))),
            frac_within_1sd=float(np.mean(np.abs(err) <= np.nan_to_num(sd, nan=np.inf))),
            frac_measured_in_interval=float(inside.mean()),
            median_interval_width=float((grp["Value_hi"] - grp["Value_lo"]).median()),
        ))
    return rows


def plot_experimental_mid(model, df, summary_rows, out_dir):
    """Predicted vs measured MID fraction, log-log, one figure per model.

    Color encodes condition (as before); marker shape encodes tracer, so a mixed-tracer
    panel (e.g. S. cerevisiae's U-13C + 1,2-13C glucose) distinguishes the two without a
    second color scale. Tracer names are rewritten into literature notation
    (:func:`_figure_common.tracer_label`) rather than the internal
    ``<met>_[<pattern>]_<enrichment>`` id.
    """
    names = CONDITION_LABEL.get(model["key"], {})
    tidy = df.assign(label=[names.get(c, c) for c in df["condition"]],
                     tracer_disp=df["Tracer"].map(tracer_label))
    palette = {names.get(c, c): CONDITION_COLORS.get(c, "#777777")
               for c in tidy["condition"].unique()}
    tracers = sorted(tidy["tracer_disp"].unique())
    marker_map = {t: TRACER_MARKERS[i % len(TRACER_MARKERS)] for i, t in enumerate(tracers)}

    lim_lo = axis_floor(tidy["measured"], tidy["sim"], floor=MID_FLOOR)
    # A simulated fraction above 1 is not a MID -- the flux state that produced it is
    # infeasible. The axis stretches to show it rather than clipping it out of sight.
    lim_hi = float(max(1.0, tidy["measured"].max(), tidy["sim"].max()))
    n_over = int((tidy["sim"] > 1.0).sum())
    if n_over:
        print(f"  {n_over} simulated fraction(s) above 1 — those flux states are not "
              f"feasible; axis extended to {lim_hi:.3g}")

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=COLORS["reference"], lw=1.0, ls="--", zorder=1)

    # Error bars first, so markers sit on top. x is the measured replicate spread -- the
    # floor on how well any method can do -- and y the bootstrap interval of the
    # prediction, so both sides show the uncertainty they actually carry.
    for cond, grp in tidy.groupby("condition", sort=False):
        x, y = grp["measured"].to_numpy(), grp["sim"].to_numpy()
        sd = grp["measured_std"].fillna(0.0).to_numpy()
        xerr = error_offsets(x, x - sd, x + sd, lim_lo, lim_hi)
        yerr = error_offsets(y, grp["Value_lo"].to_numpy(), grp["Value_hi"].to_numpy(),
                             lim_lo, lim_hi)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=CONDITION_COLORS.get(cond, "#777777"), elinewidth=0.6,
                    alpha=0.35, capsize=0, zorder=1.5)

    sns.scatterplot(data=tidy, x="measured", y="sim", hue="label", palette=palette,
                    style="tracer_disp", markers=marker_map,
                    s=14, linewidth=0, alpha=0.8, ax=ax, zorder=2, legend=False)

    # Pooled per condition, not per growth condition: a panel legend with one line per
    # (condition, growth condition) would not fit, and the CSV keeps the finer split.
    # R² stays in the CSV, not on the panel.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured isotopologue fraction")
    ax.set_ylabel("SigmaFlux simulated isotopologue fraction")

    # Two legends -- color (condition) and marker shape (tracer) -- since seaborn's
    # combined hue+style legend gets cluttered once both vary; built manually from
    # scratch (not read off the (legend=False) scatter) so their handles show a flat
    # color/shape swatch each, with no accidental cross-encoding.
    cond_handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
                    for label, color in palette.items()]
    cond_legend = ax.legend(handles=cond_handles, frameon=False, fontsize=7,
                            loc="upper left", markerscale=1.6, title=None)
    ax.add_artist(cond_legend)
    if len(tracers) > 1:
        tracer_handles = [plt.Line2D([], [], marker=marker_map[t], linestyle="",
                                     color="#555555", label=t) for t in tracers]
        ax.legend(handles=tracer_handles, frameon=False, fontsize=6,
                 loc="lower right", markerscale=1.3, title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"experimental_mid_{model['key']}.svg")
    return summary_rows


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in selected_models(args) if m["key"] in PREDICTION_SPECS]

    print(f"── Predicted vs measured isotopologues (min fraction {args.min_fraction:g}) ──")
    summary, plotted = [], []
    for model in models:
        path = tokens_path(run_dir_for(model, args.run_suffix))
        if not path.is_file():
            print(f"  {model['label']}: skipped — no {path}; "
                  f"run paper/submit/submit_experimental_mid.sh first")
            continue
        df, n_dropped = drop_unplottable(pd.read_csv(path), args.min_fraction)
        if df.empty:
            print(f"  {model['label']}: skipped — every token is at or below "
                  f"{args.min_fraction:g}")
            continue
        print(f"\n{model['label']}: {len(df)} token(s) plotted, {n_dropped} below the floor")
        summary.extend(plot_experimental_mid(model, df, summarize(model, df), out_dir))
        plotted.append(df.assign(model=model["label"], key=model["key"]))

    if summary:
        path = out_dir / "experimental_mid_summary.csv"
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"Saved: {path}")
        path = out_dir / "experimental_mid_tokens.csv"
        pd.concat(plotted, ignore_index=True).to_csv(path, index=False)
        print(f"Saved: {path}")
    else:
        print("No model had a token table — nothing written.")


if __name__ == "__main__":
    main()
