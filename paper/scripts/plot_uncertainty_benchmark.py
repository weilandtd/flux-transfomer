"""
Missing-metabolite and measurement-noise sensitivity -- line plots.

Reads what ``sigmaflux-benchmark-uncertainty`` writes --
``<run_dir>/uncertainty_benchmark/<condition>/{ai,mfa}/`` -- and draws three
single-panel line figures, one per quantity swept (metabolites observed, MID noise,
flux noise). y is median per-reaction NRMSE: for each reaction, RMSE across the
scored samples normalised by that reaction's own range (max - min) across those
same samples, then the median of that across reactions -- not per-sample RMSE
(which would let a few volatile samples, rather than a few badly-recovered
reactions, drive the number), and not raw RMSE (which would let whichever reactions
happen to carry the largest absolute flux dominate the median -- fluxes here span
several orders of magnitude, e.g. mmol/gDW/h exchange fluxes vs. biomass-precursor
fluxes). Normalised, the number reads as "typical error, as a fraction of a
reaction's own dynamic range" and is comparable across reactions and models.

Colour is METHOD (SigmaFlux/MFA, ``plot_test_performance.COLORS`` -- the paper's own
convention); marker + linestyle is MODEL, via seaborn's ``style``. The condition grid
itself is never re-derived here -- imported straight from
:mod:`sigmaflux.pipeline.run_uncertainty_benchmark`, so a change to the grid there
cannot silently desync from what this script expects to find on disk.

Two different sources of spread, matching how the underlying data was collected:
  missing-metabolite points -- each of the (up to) 20 replicate metabolite draws
      IS an independent real run, so the error band is their spread directly.
  noise points -- one real draw per level, so the error band comes from bootstrap-
      resampling that draw's scored samples (:func:`_bootstrap_nrmse`).
Both are fed to seaborn as raw per-replicate/per-draw rows and rendered with the
same ``errorbar=('pi', 95)``, so one plotting code path serves both figures.

MFA is scored over EVERY fit (``load_benchmark(..., require_converged=False)``),
same as the bar-chart version this replaces -- see that rationale in
``sigmaflux.pipeline.run_uncertainty_benchmark``'s module docstring (the
solved-sample pool already excludes the fits MFA could not solve at baseline; this
is a *further*, per-condition acceptance test that would leave too few fits to say
anything for some organism/condition combinations).

Outputs (--out-dir, default paper/figures):
  uncertainty_missing_metabolites_nrmse.svg
  uncertainty_midnoise_nrmse.svg
  uncertainty_fluxnoise_nrmse.svg
  uncertainty_benchmark_summary.csv    every plotted point, long format
  uncertainty_paired_tests.csv         per-condition Wilcoxon signed-rank, AI vs MFA
                                        per-reaction NRMSE (paired by reaction: both
                                        arms predict the same reaction set)

Usage:
    python paper/scripts/plot_uncertainty_benchmark.py
    python paper/scripts/plot_uncertainty_benchmark.py --models e_coli_core
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from _figure_common import (  # noqa: E402
    add_common_args, apply_style, run_dir_for, save_figure, selected_models)
from plot_test_performance import COLORS, LABEL, load_benchmark  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402
from sigmaflux.pipeline.run_uncertainty_benchmark import (  # noqa: E402
    DEFAULT_FLUX_STD, DEFAULT_LAB_STD, _build_conditions)

apply_style()

N_BOOT = 200   # bootstrap resamples behind each noise-axis error band

MISSING_ORDER = ["25%", "50%", "75%", "100%"]           # x_cat order; reversed to plot
LAB_ORDER = ["0.1%", "1%", "2%", "5%"]
FLUX_ORDER = ["0.1%", "5%", "10%", "25%"]


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--uncertainty-subdir", default="uncertainty_benchmark",
                        help="Subdirectory of each run holding the condition "
                             "folders (default: uncertainty_benchmark)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap resamples per noise-axis point (default: {N_BOOT})")
    parser.add_argument("--n-met-reps", type=int, default=None,
                        help="Metabolite-draw replicate count the grid was run "
                             "with (default: read from the driver's own default)")
    return parser


def _long_frame(run_dir, subdir, condition_name, method, flux_df):
    """The (sample, reaction, gt, pred) frame for one (condition, method) cell, or
    ``None`` if it was never run."""
    d = run_dir / subdir / condition_name / method
    if not list(d.glob("pred_*_flux.h5")):
        return None
    df, _ = load_benchmark(d, flux_df, require_converged=False)
    return df


def _per_reaction_nrmse(gt, pred):
    """NRMSE across samples, one value per row (reaction): RMSE normalised by that
    reaction's own range (max - min) across the scored samples, so reactions of very
    different flux magnitude are directly comparable -- shared by the point
    estimate, the bootstrap, and the paired test so all three define it identically.
    A reaction with zero range (constant across every scored sample) gives NaN,
    same convention as ``plot_test_performance.per_flux_r2``'s zero-variance guard.
    gt/pred are (reactions, samples) arrays.
    """
    with np.errstate(invalid="ignore"):
        rmse = np.sqrt(np.nanmean((pred - gt) ** 2, axis=1))
        rng = np.nanmax(gt, axis=1) - np.nanmin(gt, axis=1)
    return np.where(rng > 0, rmse / rng, np.nan)


def _point_estimate(df):
    """Median per-reaction NRMSE from one real, unresampled draw."""
    wide = df.pivot(index="reaction", columns="sample", values=["gt", "pred"])
    nrmse = _per_reaction_nrmse(wide["gt"].to_numpy(), wide["pred"].to_numpy())
    return float(np.nanmedian(nrmse))


def _bootstrap_nrmse(df, n_boot, seed):
    """``n_boot`` median-per-reaction-NRMSE draws, resampling *samples* (not rows)
    with replacement -- the correct unit, since a resampled column must carry every
    reaction's (gt, pred) pair for that sample together."""
    gt = df.pivot(index="reaction", columns="sample", values="gt")
    pred = df.pivot(index="reaction", columns="sample", values="pred")
    gt, pred = gt.align(pred, join="inner", axis=None)
    valid = gt.notna().to_numpy() & pred.notna().to_numpy()
    gt_arr = np.where(valid, gt.to_numpy(), np.nan)
    pred_arr = np.where(valid, pred.to_numpy(), np.nan)
    n_samples = gt_arr.shape[1]

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_samples, size=n_samples)
        nrmse = _per_reaction_nrmse(gt_arr[:, draw], pred_arr[:, draw])
        out.append(float(np.nanmedian(nrmse)))
    return out


def _model_n_reactions(run_dir):
    _, flux_df, _ = load_data(str(run_dir))
    flux_df.columns = range(flux_df.shape[1])
    return flux_df, flux_df.shape[0]


def build_missing_data(model, run_dir, subdir, flux_df, conditions):
    """One row per (method, replicate) real draw -- the error band's raw material."""
    rows = []
    for c in (c for c in conditions if "missing" in c.studies):
        for method in ("ai", "mfa"):
            df = _long_frame(run_dir, subdir, c.name, method, flux_df)
            if df is None:
                continue
            rows.append({"model": model["key"], "Model": model["label"], "method": method,
                        "Method": LABEL[method], "x_cat": f"{round(c.keep_met_frac * 100)}%",
                        "rep": c.keep_met_seed, "nrmse": _point_estimate(df)})
    return rows


def build_noise(model, run_dir, subdir, flux_df, conditions, axis, n_boot, seed0):
    """One row per (method, level, bootstrap draw) -- axis is 'lab' or 'flux'."""
    if axis == "lab":
        cells = sorted((c for c in conditions if "noise" in c.studies
                        and c.flux_std == DEFAULT_FLUX_STD), key=lambda c: c.labeling_std)
        x_of = lambda c: f"{c.labeling_std * 100:g}%"   # noqa: E731
    else:
        cells = sorted((c for c in conditions if "noise" in c.studies
                        and c.labeling_std == DEFAULT_LAB_STD), key=lambda c: c.flux_std)
        x_of = lambda c: f"{c.flux_std * 100:g}%"        # noqa: E731

    rows, seed = [], seed0
    for c in cells:
        for method in ("ai", "mfa"):
            df = _long_frame(run_dir, subdir, c.name, method, flux_df)
            if df is None:
                continue
            seed += 1
            for nrmse in _bootstrap_nrmse(df, n_boot, seed):
                rows.append({"model": model["key"], "Model": model["label"], "method": method,
                            "Method": LABEL[method], "x_cat": x_of(c), "nrmse": nrmse})
    return rows


def build_paired_tests(model, run_dir, subdir, flux_df, conditions):
    """Per-condition Wilcoxon signed-rank test, AI vs MFA per-reaction NRMSE.

    Paired by reaction identity: both arms predict the same reaction set, so
    reaction r's AI NRMSE and MFA NRMSE are a genuine matched pair -- not an
    independent-groups comparison, and not dependent on the two arms having scored
    the identical sample columns (each arm's NRMSE is already collapsed across its
    own samples before the two are compared).
    """
    rows = []
    for c in conditions:
        ai = _long_frame(run_dir, subdir, c.name, "ai", flux_df)
        mfa = _long_frame(run_dir, subdir, c.name, "mfa", flux_df)
        if ai is None or mfa is None:
            continue
        ai_w = ai.pivot(index="reaction", columns="sample", values=["gt", "pred"])
        mfa_w = mfa.pivot(index="reaction", columns="sample", values=["gt", "pred"])
        ai_nrmse = pd.Series(_per_reaction_nrmse(ai_w["gt"].to_numpy(), ai_w["pred"].to_numpy()),
                             index=ai_w.index)
        mfa_nrmse = pd.Series(_per_reaction_nrmse(mfa_w["gt"].to_numpy(), mfa_w["pred"].to_numpy()),
                              index=mfa_w.index)
        shared = ai_nrmse.index.intersection(mfa_nrmse.index)
        if len(shared) < 6:      # scipy.stats.wilcoxon's own practical floor
            continue
        diff = (ai_nrmse[shared] - mfa_nrmse[shared]).to_numpy()
        diff = diff[np.isfinite(diff)]
        if len(diff) < 6 or np.allclose(diff, 0):
            continue
        stat, p = stats.wilcoxon(diff)
        rows.append({"model": model["key"], "Model": model["label"], "condition": c.name,
                    "n_paired": len(diff), "median_nrmse_diff_ai_minus_mfa": float(np.median(diff)),
                    "wilcoxon_stat": float(stat), "wilcoxon_p": float(p)})
    return rows


def _plot(df, order, reverse, xlabel, out_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.lineplot(data=df, x="x_cat", y="nrmse", hue="Method", style="Model",
                hue_order=[LABEL["ai"], LABEL["mfa"]],
                palette={LABEL["ai"]: COLORS["ai"], LABEL["mfa"]: COLORS["mfa"]},
                markers=True, dashes=True, err_style="band", errorbar=("pi", 95),
                sort=False, ax=ax)
    lo, hi = -0.5, len(order) - 0.5
    ax.set_xlim(hi, lo) if reverse else ax.set_xlim(lo, hi)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median per-reaction NRMSE")
    ax.legend(frameon=False, fontsize=6, loc="best")
    sns.despine(ax=ax)
    save_figure(fig, out_path)


def plot_all(df, out_dir):
    # (study, category order, x reversed?, x-axis label, output-name tag)
    specs = (("missing", MISSING_ORDER, True, "Metabolites observed", "missing_metabolites"),
            ("lab_noise", LAB_ORDER, False, "MID noise (% absolute)", "midnoise"),
            ("flux_noise", FLUX_ORDER, False, "Flux noise (% relative)", "fluxnoise"))
    for study, order, reverse, xlabel, tag in specs:
        sub = df[df["study"] == study].copy()
        sub["x_cat"] = pd.Categorical(sub["x_cat"], categories=order, ordered=True).codes
        _plot(sub, order, reverse, xlabel, out_dir / f"uncertainty_{tag}_nrmse.svg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_args(p).parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("uncertainty_*"):
        f.unlink()

    models = selected_models(args)
    conditions = _build_conditions(args.n_met_reps) if args.n_met_reps else _build_conditions()
    print("── Uncertainty benchmark (missing metabolites, noise) ──")

    all_rows, paired_rows = [], []
    for model in models:
        run_dir = run_dir_for(model, args.run_suffix)
        subdir = run_dir / args.uncertainty_subdir
        if not subdir.exists():
            print(f"  {model['label']}: skipped — no {subdir}")
            continue
        flux_df, _ = _model_n_reactions(run_dir)

        missing_rows = build_missing_data(model, run_dir, args.uncertainty_subdir, flux_df, conditions)
        lab_rows = build_noise(model, run_dir, args.uncertainty_subdir, flux_df,
                               conditions, "lab", args.n_boot, seed0=0)
        flux_rows = build_noise(model, run_dir, args.uncertainty_subdir, flux_df,
                                conditions, "flux", args.n_boot, seed0=10_000)
        for rows, study in ((missing_rows, "missing"), (lab_rows, "lab_noise"),
                            (flux_rows, "flux_noise")):
            for r in rows:
                r["study"] = study
            all_rows += rows
        paired_rows += build_paired_tests(model, run_dir, args.uncertainty_subdir,
                                          flux_df, conditions)
        n_cells = len({(r["study"], r["x_cat"], r["method"])
                      for r in missing_rows + lab_rows + flux_rows})
        print(f"  {model['label']:26s} {n_cells} (study, level, method) cells found")

    if not all_rows:
        print("No model had uncertainty-benchmark data — nothing written.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "uncertainty_benchmark_summary.csv", index=False)
    print(f"Saved: {out_dir / 'uncertainty_benchmark_summary.csv'}")

    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(out_dir / "uncertainty_paired_tests.csv", index=False)
        print(f"Saved: {out_dir / 'uncertainty_paired_tests.csv'}")

    plot_all(df, out_dir)


if __name__ == "__main__":
    main()
