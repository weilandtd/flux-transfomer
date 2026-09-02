"""
SigmaFlux predictions vs published MFA fits, on real experimental ¹³C data.

Per model:
  experimental_flux_<key>.svg      SigmaFlux vs reference, log-log, one point per
                                    (condition, growth condition, reaction)
Across all models:
  experimental_flux_summary.csv    per-(model, condition) R², interval overlap,
                                    point count, reaction count

By default (``--reactions mfa``) each model is restricted to reactions **MFA** resolves
on held-out simulations (R² >= R2_THRESHOLD in the simulation benchmark — see
plot_mfa_benchmark.py). The x axis here is a published MFA fit, so where MFA cannot
determine a flux the reference value is itself unreliable and agreement with it would
mean nothing. ``--reactions all`` skips this filter (and the benchmark HDF5s it needs)
entirely, for a model that has predictions but no mfa_benchmark/ run yet.

Each panel reports R² per condition (``_r2_linear``), on signed fluxes in linear space.
That is the same definition as ``per_flux_r2`` and ``R2_THRESHOLD`` in the simulation
benchmark, so R² means one thing across the paper.

Read it knowing what it weights: squared error in linear space is carried by whichever
reactions run fastest, and on these panels the five largest terms are 40-90% of it. The
interval-overlap count -- how many reactions have a predicted 95% interval compatible
with the reference MFA's own, which is the question both sides' published uncertainty
can answer -- stays in the summary CSV alongside it.

Axes are log-log for display -- fluxes span ~8 orders of magnitude across these
organisms -- while both R² and the overlap test are computed on signed, linear fluxes.
Error bars are drawn identically on both axes: an interval that spans zero has no
positive floor on a log magnitude axis, so its bar is truncated to one decade rather
than run to the axis edge, with no caret marking the open end.

A model whose predictions or (for --reactions mfa) benchmark data is missing is skipped
with a warning rather than failing the run, so the figure set fills itself in as runs
land.

Usage:
    python paper/scripts/plot_experimental_predictions.py
    python paper/scripts/plot_experimental_predictions.py --models e_coli_core
    python paper/scripts/plot_experimental_predictions.py --models e_coli_core --reactions all
    python paper/scripts/plot_experimental_predictions.py --run-suffix 20260807
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sigmaflux.utils import magnitude_bounds

from _figure_common import (
    COLORS, CONDITION_COLORS, CONDITION_LABEL, PREDICTION_SPECS, R2_THRESHOLD,
    add_common_args, apply_style, axis_floor, benchmark_metrics,
    build_reference_comparison, error_offsets, resolve_organism, run_dir_for,
    save_figure, selected_models)

apply_style()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--reactions", choices=["mfa", "all"], default="mfa",
                   help="'mfa' (default) restricts to reactions MFA resolves in the "
                        "simulation benchmark (R² >= R2_THRESHOLD); 'all' skips the "
                        "benchmark and uses every reaction the reference crosswalk maps.")
    return p.parse_args()


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
        # build_plot_df(..., None) still restricts to reactions shared with predictions,
        # so this is exactly the mapped reaction set, with no MFA-identifiability filter.
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
    fluxes, like the R² it replaces: a flux predicted in the wrong direction has no
    business counting as agreement, and the published exchange bounds are already oriented
    into the model's convention upstream (``pair_fluxes``), so a surviving sign
    disagreement is real.

    Reported in the summary CSV rather than on the panel, which shows R². It answers a
    question R² cannot: where MFA itself cannot determine a flux -- on E. coli core its
    own 95% interval for R2x spans 0.05 to 39.85 -- a squared deviation from its point
    estimate measures nothing, while asking whether the two intervals are compatible uses
    the uncertainty both sides actually report.
    """
    lo_p, hi_p = df["pred_lo"].to_numpy(), df["pred_hi"].to_numpy()
    lo_r, hi_r = df["ref_lo"].to_numpy(), df["ref_hi"].to_numpy()
    overlap = (lo_p <= hi_r) & (lo_r <= hi_p)
    return int(np.sum(overlap)), int(np.sum(np.isfinite(lo_p) & np.isfinite(lo_r)))


def _r2_linear(pred, ref):
    """Standard linear-space R², on **signed** fluxes.

    Signed, not magnitudes: a flux predicted in the wrong direction is wrong, and scoring
    |v| would credit it as a perfect hit. Exchange reactions are oriented into the model's
    convention upstream, so a surviving sign disagreement is real.
    """
    if len(ref) < 2:
        return float("nan")
    ss_res = ((pred - ref) ** 2).sum()
    ss_tot = ((ref - ref.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def plot_experimental_flux(model, df, out_dir):
    """SigmaFlux vs the published MFA reference, log-log, one figure per model."""
    names = CONDITION_LABEL.get(model["key"], {})
    ref_lo, ref_hi, _ = magnitude_bounds(df["ref_lo"], df["ref_hi"])
    pred_lo, pred_hi, _ = magnitude_bounds(df["pred_lo"], df["pred_hi"])

    tidy = pd.DataFrame({
        "reference":  df["ref"].abs().to_numpy(),
        "sigmaflux":  df["pred"].abs().to_numpy(),
        "key":        df["condition"].to_numpy(),
        "condition":  [names.get(c, c) for c in df["condition"]],
        "ref_lo": ref_lo, "ref_hi": ref_hi, "pred_lo": pred_lo, "pred_hi": pred_hi,
    })

    lim_lo = axis_floor(tidy["reference"], tidy["sigmaflux"])
    lim_hi = float(np.nanmax([tidy["reference"].max(), tidy["sigmaflux"].max()])) * 2.0

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=COLORS["reference"], lw=1.0, ls="--", zorder=1)

    # Error bars first, so the markers sit on top of them. Drawn per condition because each
    # carries its own color; seaborn has no per-point xerr/yerr equivalent.
    for key, grp in tidy.groupby("key", sort=False):
        x, y = grp["reference"].to_numpy(), grp["sigmaflux"].to_numpy()
        color = CONDITION_COLORS.get(key, "#777777")
        xerr = error_offsets(x, grp["ref_lo"].to_numpy(), grp["ref_hi"].to_numpy(),
                              lim_lo, lim_hi)
        yerr = error_offsets(y, grp["pred_lo"].to_numpy(), grp["pred_hi"].to_numpy(),
                              lim_lo, lim_hi)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color,
                    elinewidth=0.6, alpha=0.35, capsize=0, zorder=1.5)

    palette = {names.get(k, k): CONDITION_COLORS.get(k, "#777777")
               for k in tidy["key"].unique()}
    # Sign disagreement is reported as a count in the summary CSV rather than as a second
    # marker style: encoding it here made the panel hard to read for a case that is rare
    # (zero points on the MFA-identifiable set) once exchanges are correctly oriented.
    sns.scatterplot(data=tidy, x="reference", y="sigmaflux", hue="condition",
                    palette=palette, s=18, linewidth=0, ax=ax, zorder=2)

    # One figure per condition, not a single value pooled across them -- conditions can
    # differ a lot in agreement, and pooling would hide that.
    summary_rows, lines = [], []
    for cond, grp in df.groupby("condition", sort=False):
        label = names.get(cond, cond)
        n_ov, n_cmp = _interval_overlap(grp)
        r2 = _r2_linear(grp["pred"].to_numpy(), grp["ref"].to_numpy())
        lines.append(f"{label}: $R^2$ = {r2:.2f}")
        # The overlap count stays in the CSV -- it is no longer what the panel reports,
        # but it is the only statistic here that uses both sides' published uncertainty.
        summary_rows.append(dict(model=model["label"], key=model["key"], condition=cond,
                                 label=label, r2=r2,
                                 n_overlap=n_ov, n_compared=n_cmp,
                                 frac_overlap=(n_ov / n_cmp if n_cmp else float("nan")),
                                 n_points=len(grp),
                                 n_reactions=grp["reaction"].nunique(),
                                 n_sign_opposed=int((~grp["sign_agrees"]).sum())))

    ax.text(0.97, 0.04, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=COLORS["box"],
                      boxstyle="round,pad=0.25"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    # Mathtext, not a literal Unicode superscript: Arial has no glyph for U+207B and
    # silently falls back to a "Last Resort" placeholder box in the saved SVG.
    ax.set_xlabel("Reference MFA\n" + r"[mmol gDW$^{-1}$ h$^{-1}$]")
    ax.set_ylabel("SigmaFlux\n" + r"[mmol gDW$^{-1}$ h$^{-1}$]")
    ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=1.4)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"experimental_flux_{model['key']}.svg")
    return summary_rows


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

    print("\n── Predictions vs published reference ──")
    summary = []
    for model in models:
        spec = PREDICTION_SPECS[model["key"]]
        df = literature_df(model, tables.get(model["key"]), spec, args.run_suffix)
        if df is None:
            print(f"experimental_flux_{model['key']}: skipped — no usable data")
            continue
        summary.extend(plot_experimental_flux(model, df, out_dir))

    if summary:
        path = out_dir / "experimental_flux_summary.csv"
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"Saved: {path}")
    else:
        print("No model had usable prediction data — nothing written.")


if __name__ == "__main__":
    main()
