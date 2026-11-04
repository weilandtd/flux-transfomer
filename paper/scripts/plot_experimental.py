"""
The flux transformer on real experimental ¹³C data: fluxes against the published MFA fits,
labeling against the measurements, one panel per (organism, condition).

Per (organism, condition):
  experimental_flux_<key>_<cond>.svg   predicted vs published MFA flux, log-log, one point
                                        per (growth condition, reaction)
  experimental_mid_<key>_<cond>.svg    predicted vs measured isotopologue fraction, one
                                        point per (growth condition, tracer, metabolite,
                                        isotopomer)
Across all models:
  r2_flux.svg                          flux R² vs traditional MFA, one bar per condition
  r2_labeling.svg                      labeling R² vs measurement, one bar per condition,
                                        length = median over that condition's tracers
  experimental_flux_summary.csv        per-(model, condition) R², interval overlap, point
                                        count, reaction count
  experimental_mid_summary.csv         per-(model, condition, growth condition) R², RMSE,
                                        MAE, worst error, median |z| against replicate
                                        spread, fraction inside one replicate SD and inside
                                        the predicted interval
  experimental_mid_tokens.csv          every plotted token, all models pooled
  r2_labeling_per_tracer.csv           the per-tracer R² behind r2_labeling.svg

Two views of the same predictions, answering different questions. The flux panels ask
whether we agree with a *second estimate* of the fluxes -- a published MFA fit, with its own
identifiability problems. The labeling panels ask whether we reproduce the measurement
itself. Labeling depends on flux ratios, so a state can miss the other estimate and still
explain the data, or match it and not.

By default (``--reactions mfa``) the flux panels are restricted to reactions **MFA**
resolves on held-out simulations (R² >= R2_THRESHOLD in the simulation benchmark -- see
plot_mfa_benchmark.py): where MFA cannot determine a flux the reference value is itself
unreliable. ``--reactions all`` skips this filter (and the benchmark HDF5s it needs).

Every R² is ``_figure_common.r2_linear``: signed and linear, the same definition as
``per_flux_r2`` in the simulation benchmark, so R² means one thing across the paper. In
linear space it is carried by the fastest reactions and the largest fractions, whatever
scale the axes are drawn on. The interval-overlap count stays in the summary CSV as the
statistic that uses both sides' published uncertainty.

Labeling R² is computed per tracer, because a condition's tracers are separate experiments
on the same cells; the bar is their median, so one poorly reproduced tracer cannot carry the
summary. Tracers stay visible in the scatter panels as marker shapes.

Flux axes are log-log -- fluxes span ~8 orders of magnitude -- with error bars on both
axes; an interval spanning zero is truncated to one decade. Labeling axes are linear from
zero; the x error bar is the measured replicate spread, the y bar the bootstrap interval of
the prediction propagated through the EMU pass (see experimental_mid.py).

Reads each run's predictions/ and ``experimental_mid/`` token table
(paper/submit/submit_inference.sh, paper/submit/submit_experimental_mid.sh). A model missing
either is skipped with a warning, so the figure set fills itself in as runs land.

Usage:
    python paper/scripts/plot_experimental.py
    python paper/scripts/plot_experimental.py --models e_coli_core --reactions all
    python paper/scripts/plot_experimental.py --min-fraction 1e-3
    python paper/scripts/plot_experimental.py --run-suffix 20260807
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sigmaflux.utils import magnitude_bounds

from _figure_common import (
    COLORS, METHOD_COLORS, PREDICTION_SPECS, R2_THRESHOLD, TRACER_MARKERS,
    add_common_args, apply_style, axis_floor, benchmark_metrics,
    build_reference_comparison, condition_name, error_offsets, panel_name, r2_linear,
    resolve_organism, run_dir_for, save_figure, selected_models, tracer_label)
from experimental_mid import tokens_path

apply_style()

TEAL = METHOD_COLORS["ai"]

#: (organism, condition) whose R² bar is drawn in the dark teal; the rest are lighter.
HIGHLIGHT = ("i_orientalis", "B")

#: One (bar, text) color pair per organism, in bar order, all lighter than the highlight.
SPECIES_TEALS = [("#4F9893", "white"), ("#7DB4AF", "#1a1a1a"),
                 ("#A9CFCB", "#1a1a1a"), ("#D2E8E5", "#1a1a1a")]

FLUX_UNITS = r"[mmol gDW$^{-1}$ h$^{-1}$]"

#: Fixed margins (figure fraction) so every scatter panel has the same axes box.
SCATTER_MARGINS = dict(left=0.27, right=0.93, bottom=0.24, top=0.9)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--reactions", choices=["mfa", "all"], default="mfa",
                   help="'mfa' (default) restricts to reactions MFA resolves in the "
                        "simulation benchmark (R² >= R2_THRESHOLD); 'all' skips the "
                        "benchmark and uses every reaction the reference crosswalk maps.")
    p.add_argument("--min-fraction", type=float, default=0.0,
                   help="Drop tokens whose measured or predicted fraction is below this; "
                        "below the replicate spread a point describes the noise floor, "
                        "not the prediction (default: 0, draw every token)")
    return p.parse_args()


# ── Flux data ─────────────────────────────────────────────────────────────────

def literature_df(model, tbl, spec, run_suffix=None):
    """Tidy pred-vs-reference frame for one model, or None when it cannot be built."""
    run = run_dir_for(model, run_suffix)
    if not list(run.glob("predictions/predictions_*.csv")):
        print(f"  {model['label']}: skipped — no predictions/ in {run.name}; "
              f"run paper/submit/submit_inference.sh {run}")
        return None
    if tbl is not None:
        # The panel is defined as the fluxes MFA resolves, so an empty set is not a
        # plotting failure but a statement about the run: MFA cleared R² >= R2_THRESHOLD
        # on nothing.
        mfa_rxns = set(tbl.index[tbl["r2_mfa"] >= R2_THRESHOLD])
        if not mfa_rxns:
            print(f"  {model['label']}: skipped — no reaction has MFA R² >= {R2_THRESHOLD} "
                  f"in {run.name}, so the MFA-identifiable set is empty")
            return None
    else:
        # --reactions all, or no mfa_benchmark run for this model -- fall back to
        # whatever reactions its own reference-flux crosswalk maps.
        print(f"  {model['label']}: using its reference-flux crosswalk's reaction set "
              f"as-is (no MFA-identifiability filter)")
        mfa_rxns = None
    try:
        org = resolve_organism(run)
        df = build_reference_comparison(run, org, mfa_rxns)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"  {model['label']}: skipped — {type(exc).__name__}: {exc}")
        return None

    df = df[df["condition"].isin(spec["conditions"])]
    if spec["select"] == "max_D":
        # The highest dilution rate per condition, read off the reference's own BIOMASS
        # row (build_plot_df already resolved it into D), never a hardcoded index.
        df = df[df["D"] == df.groupby("condition")["D"].transform("max")]
    elif spec["select"] == "wt":
        # Wild type is growth condition 1 in data/E_coli/strain_index.csv; the other 20
        # growth conditions are single-gene knockouts, not a like-for-like comparison.
        df = df[df["gc"] == 1]
    if df.empty:
        print(f"  {model['label']}: skipped — no points for "
              f"conditions {spec['conditions']}")
        return None
    return df.assign(model=model["label"], key=model["key"])


def _interval_overlap(df):
    """Reactions whose predicted interval overlaps the reference MFA's, as ``(n, total)``.

    Two intervals overlap iff each starts before the other ends. Computed on **signed**
    fluxes, like R²: a flux predicted in the wrong direction is not agreement, and the
    published exchange bounds are already oriented into the model's convention upstream
    (``pair_fluxes``). Where MFA cannot determine a flux -- on E. coli core its own 95%
    interval for R2x spans 0.05 to 39.85 -- a squared deviation from its point estimate
    measures nothing, while this uses the uncertainty both sides report.
    """
    lo_p, hi_p = df["pred_lo"].to_numpy(), df["pred_hi"].to_numpy()
    lo_r, hi_r = df["ref_lo"].to_numpy(), df["ref_hi"].to_numpy()
    overlap = (lo_p <= hi_r) & (lo_r <= hi_p)
    return int(np.sum(overlap)), int(np.sum(np.isfinite(lo_p) & np.isfinite(lo_r)))


def flux_summary(model, cond, grp):
    """One experimental_flux_summary.csv row."""
    n_ov, n_cmp = _interval_overlap(grp)
    return dict(model=model["label"], key=model["key"], condition=cond,
                label=condition_name(model["key"], cond),
                r2=r2_linear(grp["pred"].to_numpy(), grp["ref"].to_numpy()),
                n_overlap=n_ov, n_compared=n_cmp,
                frac_overlap=(n_ov / n_cmp if n_cmp else float("nan")),
                n_points=len(grp), n_reactions=grp["reaction"].nunique(),
                n_sign_opposed=int((~grp["sign_agrees"]).sum()))


# ── Labeling data ─────────────────────────────────────────────────────────────

def drop_unplottable(df, min_fraction):
    """Tokens at or above the floor, plus how many were dropped.

    Inclusive, so a measured zero -- the instrument saw nothing at that mass -- is drawn
    at the origin rather than discarded.
    """
    lo = max(min_fraction, 0.0)
    keep = (df["measured"] >= lo) & (df["sim"] >= lo)
    return df[keep].copy(), int((~keep).sum())


def mid_summary(model, df):
    """Per-(condition, growth condition) labeling agreement statistics."""
    rows = []
    for (cond, gc), grp in df.groupby(["condition", "gc"], sort=False):
        err = grp["error"].to_numpy()
        sd = grp["measured_std"].to_numpy()
        inside = ((grp["Value_lo"] <= grp["measured"]) &
                  (grp["measured"] <= grp["Value_hi"]))
        rows.append(dict(
            model=model["label"], key=model["key"], condition=cond, gc=int(gc),
            label=condition_name(model["key"], cond),
            n_tokens=len(grp),
            r2=r2_linear(grp["sim"].to_numpy(), grp["measured"].to_numpy()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mae=float(np.mean(np.abs(err))),
            max_abs_error=float(np.max(np.abs(err))),
            median_abs_z=float(np.median(np.abs(grp["z"].to_numpy()))),
            frac_within_1sd=float(np.mean(np.abs(err) <= np.nan_to_num(sd, nan=np.inf))),
            frac_measured_in_interval=float(inside.mean()),
            median_interval_width=float((grp["Value_hi"] - grp["Value_lo"]).median()),
        ))
    return rows


def labeling_r2_per_tracer(model, df):
    """R² per (condition, tracer)."""
    rows = []
    for (cond, tracer), g in df.groupby(["condition", "Tracer"], sort=False):
        rows.append(dict(
            model=model["label"], key=model["key"], condition=cond,
            name=condition_name(model["key"], cond), tracer=tracer_label(tracer),
            n_tokens=len(g), r2=r2_linear(g["sim"].to_numpy(), g["measured"].to_numpy())))
    return pd.DataFrame(rows).dropna(subset=["r2"])


# ── Scatter panels ────────────────────────────────────────────────────────────

def _panel_title(ax, key, cond):
    ax.set_title(panel_name(key, cond), loc="left", fontsize=9)


def plot_flux_panel(model, cond, grp, out_dir):
    """Flux transformer vs the published MFA fit for one condition, log-log."""
    x, y = grp["ref"].abs().to_numpy(), grp["pred"].abs().to_numpy()
    ref_lo, ref_hi, _ = magnitude_bounds(grp["ref_lo"], grp["ref_hi"])
    pred_lo, pred_hi, _ = magnitude_bounds(grp["pred_lo"], grp["pred_hi"])
    lim_lo = axis_floor(pd.Series(x), pd.Series(y))
    lim_hi = float(np.nanmax([x.max(), y.max()])) * 2.0

    fig, ax = plt.subplots(figsize=(3.5, 3.5), gridspec_kw=SCATTER_MARGINS)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=COLORS["reference"], lw=1.0, ls="--", zorder=1)
    ax.errorbar(x, y, xerr=error_offsets(x, ref_lo, ref_hi, lim_lo, lim_hi),
                yerr=error_offsets(y, pred_lo, pred_hi, lim_lo, lim_hi),
                fmt="none", ecolor=TEAL, elinewidth=0.6, alpha=0.35, capsize=0, zorder=1.5)
    ax.scatter(x, y, s=18, color=TEAL, linewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    # Mathtext, not a literal Unicode superscript: Arial has no glyph for U+207B and
    # silently falls back to a "Last Resort" placeholder box in the saved SVG.
    ax.set_xlabel("Traditional MFA\n" + FLUX_UNITS)
    ax.set_ylabel("Flux transformer\n" + FLUX_UNITS)
    _panel_title(ax, model["key"], cond)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"experimental_flux_{model['key']}_{cond}.svg", tight=False)


def plot_mid_panel(model, cond, grp, marker, out_dir):
    """Predicted vs measured isotopologue fraction for one condition; shape is tracer."""
    x, y = grp["measured"].to_numpy(), grp["sim"].to_numpy()
    lim_lo = 0.0
    # The top follows the data rather than sitting at 1, or most of the panel is empty.
    # A simulated fraction above 1 is not a MID -- the flux state that produced it is
    # infeasible -- and is shown rather than clipped out of sight.
    lim_hi = float(max(x.max(), y.max())) * 1.05
    n_over = int((y > 1.0).sum())
    if n_over:
        print(f"  {n_over} simulated fraction(s) above 1 — those flux states are not "
              f"feasible; axis extended to {lim_hi:.3g}")

    fig, ax = plt.subplots(figsize=(3.5, 3.5), gridspec_kw=SCATTER_MARGINS)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=COLORS["reference"], lw=1.0, ls="--", zorder=1)
    sd = grp["measured_std"].fillna(0.0).to_numpy()
    ax.errorbar(x, y, xerr=error_offsets(x, x - sd, x + sd, lim_lo, lim_hi),
                yerr=error_offsets(y, grp["Value_lo"].to_numpy(), grp["Value_hi"].to_numpy(),
                                   lim_lo, lim_hi),
                fmt="none", ecolor=TEAL, elinewidth=0.6, alpha=0.35, capsize=0, zorder=1.5)
    tracers = sorted(grp["Tracer"].map(tracer_label).unique())
    for tracer, g in grp.groupby(grp["Tracer"].map(tracer_label)):
        ax.scatter(g["measured"], g["sim"], s=14, marker=marker[tracer], color=TEAL,
                   linewidth=0, alpha=0.8, zorder=2, label=tracer)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured isotopologue fraction")
    ax.set_ylabel("Flux transformer\nisotopologue fraction")
    _panel_title(ax, model["key"], cond)
    if len(tracers) > 1:
        ax.legend(frameon=False, fontsize=6, loc="lower right", markerscale=1.3)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"experimental_mid_{model['key']}_{cond}.svg", tight=False)


# ── R² bars ───────────────────────────────────────────────────────────────────

def bar_order(df):
    """(key, condition) pairs: organisms by their best R², conditions by descending R²
    within each."""
    best = df.groupby("key")["r2"].max()
    rows = sorted(df.itertuples(), key=lambda r: (-best[r.key], r.key, -r.r2))
    return [(r.key, r.condition) for r in rows]


def plot_r2(df, order, xlabel, path):
    """One horizontal bar per (organism, condition) in ``order``, its name written on
    the bar; pairs missing from ``order`` follow in their own bar_order."""
    order = order + [o for o in bar_order(df) if o not in order]
    rows = list(df.set_index(["key", "condition"]).reindex(order).dropna(subset=["r2"])
                .reset_index().itertuples())
    species = list(dict.fromkeys(r.key for r in rows))
    shade = {k: SPECIES_TEALS[i % len(SPECIES_TEALS)] for i, k in enumerate(species)}
    height = 0.42 * len(rows) + 0.85
    fig, ax = plt.subplots(figsize=(3.0, height), gridspec_kw=dict(
        left=0.07, right=0.92, bottom=0.85 / height, top=1 - 0.1 / height))
    for y, r in enumerate(rows):
        fill, text = (TEAL, "white") if (r.key, r.condition) == HIGHLIGHT else shade[r.key]
        ax.barh(y, r.r2, height=0.8, color=fill, linewidth=0)
        ax.text(0.02, y, panel_name(r.key, r.condition), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=text)
    ax.invert_yaxis()
    ax.set_yticks([])
    # Never clip a negative R²: an arm that predicts worse than the mean is what this
    # figure exists to show.
    lo = min(0.0, float(df["r2"].min()) * 1.1)
    ax.set_xlim(lo, 1.0)
    if lo < 0:
        ax.axvline(0.0, color=COLORS["reference"], lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax, left=True)
    save_figure(fig, path, tight=False)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in selected_models(args) if m["key"] in PREDICTION_SPECS]

    print("── Per-reaction R² (MFA-identifiability filter) ──")
    tables = {}
    if args.reactions == "mfa":
        for m in models:
            tbl = benchmark_metrics(m, args.run_suffix)
            if tbl is not None:
                tables[m["key"]] = tbl

    print("\n── Fluxes vs published MFA ──")
    flux_rows = []
    for model in models:
        df = literature_df(model, tables.get(model["key"]), PREDICTION_SPECS[model["key"]],
                           args.run_suffix)
        if df is None:
            continue
        for cond, grp in df.groupby("condition", sort=False):
            plot_flux_panel(model, cond, grp, out_dir)
            flux_rows.append(flux_summary(model, cond, grp))

    order = []
    print(f"\n── Predicted vs measured isotopologues (min fraction "
          f"{args.min_fraction:g}) ──")
    mid_rows, tokens, per_tracer = [], [], []
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
        print(f"  {model['label']}: {len(df)} token(s) plotted, {n_dropped} below the floor")
        # One shape per tracer over the whole organism, so a shape means the same tracer
        # in every one of its panels.
        tracers = sorted(df["Tracer"].map(tracer_label).unique())
        marker = {t: TRACER_MARKERS[i % len(TRACER_MARKERS)] for i, t in enumerate(tracers)}
        for cond, grp in df.groupby("condition", sort=False):
            plot_mid_panel(model, cond, grp, marker, out_dir)
        mid_rows.extend(mid_summary(model, df))
        tokens.append(df.assign(model=model["label"], key=model["key"]))
        per_tracer.append(labeling_r2_per_tracer(model, df))

    if flux_rows:
        flux = pd.DataFrame(flux_rows)
        print("\n── Flux R² vs published MFA ──")
        for _, r in flux.iterrows():
            print(f"  {r['model']:20s} {r['label']:11s} {r['r2']:8.4f}")
        path = out_dir / "experimental_flux_summary.csv"
        flux.to_csv(path, index=False)
        print(f"Saved: {path}")
        order = bar_order(flux)
        plot_r2(flux, order, "Flux transformer vs.\n" r"traditional MFA $R^2$",
                out_dir / "r2_flux.svg")
    else:
        print("No model had usable prediction data — no flux outputs written.")

    if mid_rows:
        lab = pd.concat(per_tracer, ignore_index=True)
        print("\n── Labeling R² per tracer ──")
        for (model, cond), g in lab.groupby(["model", "name"], sort=False):
            print(f"  {model:20s} {cond:11s} median {g['r2'].median():7.4f}  "
                  f"({len(g)} tracers: {', '.join(f'{v:.3f}' for v in g['r2'])})")
        for name, frame in (("experimental_mid_summary.csv", pd.DataFrame(mid_rows)),
                            ("experimental_mid_tokens.csv",
                             pd.concat(tokens, ignore_index=True)),
                            ("r2_labeling_per_tracer.csv", lab)):
            frame.to_csv(out_dir / name, index=False)
            print(f"Saved: {out_dir / name}")
        median = lab.groupby(["key", "condition"], sort=False)["r2"].median().reset_index()
        # Same bar order as the flux plot, so the two read side by side.
        plot_r2(median, order, "Flux transformer vs.\n" r"measured labeling $R^2$",
                out_dir / "r2_labeling.svg")
    else:
        print("No model had a token table — no labeling outputs written.")


if __name__ == "__main__":
    main()
