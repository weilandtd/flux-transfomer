"""
Missing-metabolite and measurement-noise sensitivity -- line plots.

Reads what ``sigmaflux-benchmark-uncertainty`` writes --
``<run_dir>/uncertainty_benchmark/<condition>/{ai,mfa}/`` -- and draws three
single-panel line figures, one per quantity swept (metabolites observed, MID noise,
flux noise). y is the number of reactions the arm **resolves**: R² across the scored
samples at or above ``R2_THRESHOLD``, the same bar the simulation benchmark's Venn
diagrams use, so "resolved" means one thing across the paper. The sweep then reads as
"how many reactions survive as the measurement degrades" rather than as a change in
typical error, and needs no normalisation to be comparable across reactions.

y is that count as a **fraction** of the reactions the cell scores, so models of very
different size share one axis -- E. coli core has 79 reactions where I. orientalis has
510, and the raw counts sit in separate bands with no common scale. The summary CSV
carries ``n_reactions`` beside every point for the absolute number.

The line is the **median** over the replicates (missing-metabolite axis) or the
bootstrap draws (noise axes), matching the percentile error band around it, so a single
pathological draw moves neither.

Colour is METHOD (SigmaFlux/MFA, ``plot_test_performance.COLORS`` -- the paper's own
convention); marker + linestyle is MODEL, via seaborn's ``style``. The condition grid
itself is never re-derived here -- imported straight from
:mod:`sigmaflux.pipeline.run_uncertainty_benchmark`, so a change to the grid there
cannot silently desync from what this script expects to find on disk.

Two different sources of spread, matching how the underlying data was collected:
  missing-metabolite points -- each of the (up to) 20 replicate metabolite draws
      IS an independent real run, so the error band is their spread directly.
  noise points -- one real draw per level, so the error band comes from bootstrap-
      resampling that draw's scored samples (:func:`_bootstrap_resolved`).
Both are fed to seaborn as raw per-replicate/per-draw rows and rendered with the
same ``errorbar=('pi', 95)``, so one plotting code path serves both figures.

MFA is scored over every fit, as everywhere ``load_benchmark`` is used.

Outputs (--out-dir, default paper/figures):
  uncertainty_missing_metabolites_resolved.svg
  uncertainty_midnoise_resolved.svg
  uncertainty_fluxnoise_resolved.svg
  uncertainty_benchmark_summary.csv    every plotted point, long format
  uncertainty_paired_tests.csv         per-condition Wilcoxon signed-rank, AI vs MFA
                                        per-reaction R² (paired by reaction: both
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
from plot_test_performance import (  # noqa: E402
    COLORS, LABEL, R2_THRESHOLD, load_benchmark)
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
    df, _ = load_benchmark(d, flux_df)
    return df


def _per_reaction_r2(gt, pred):
    """R² across samples, one value per row (reaction).

    Shared by the point estimate, the bootstrap and the paired test, so all three
    define it identically, and identical to ``plot_test_performance.per_flux_r2``
    including its zero-variance guard: a reaction constant across every scored sample
    has no variance to explain and gives NaN rather than a score.

    gt/pred are (reactions, samples) arrays.
    """
    with np.errstate(invalid="ignore"):
        ss_res = np.nansum((pred - gt) ** 2, axis=1)
        ss_tot = np.nansum((gt - np.nanmean(gt, axis=1, keepdims=True)) ** 2, axis=1)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def _n_resolved(gt, pred):
    """How many of this cell's reactions it resolves, i.e. reach R² >= R2_THRESHOLD.

    A reaction with no ground-truth variance scores NaN and so is never counted as
    resolved -- the same zero-variance guard ``per_flux_r2`` applies.
    """
    return int(np.sum(_per_reaction_r2(gt, pred) >= R2_THRESHOLD))


def _frac_resolved(gt, pred):
    """:func:`_n_resolved` over every reaction the cell scores.

    A fraction rather than the count, so models of very different size read against
    each other on one axis -- E. coli core has 79 reactions where I. orientalis has
    510, and counts put those curves in separate bands of the panel with no common
    scale.

    A reaction with no ground-truth variance can never be resolved but stays in the
    denominator: it is a reaction the method did not resolve, and both arms carry the
    identical denominator, so it shifts a curve without distorting its shape or the
    comparison between arms.
    """
    r2 = _per_reaction_r2(gt, pred)
    return float(np.sum(r2 >= R2_THRESHOLD) / len(r2)) if len(r2) else np.nan


def _point_estimate(df):
    """Fraction of reactions resolved in one real, unresampled draw."""
    wide = df.pivot(index="reaction", columns="sample", values=["gt", "pred"])
    return _frac_resolved(wide["gt"].to_numpy(), wide["pred"].to_numpy())


def _bootstrap_resolved(df, n_boot, seed):
    """``n_boot`` resolved-reaction fractions, resampling *samples* (not rows)
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
        out.append(_frac_resolved(gt_arr[:, draw], pred_arr[:, draw]))
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
                        "rep": c.keep_met_seed,
                        "frac_resolved": _point_estimate(df),
                        "n_reactions": int(df["reaction"].nunique())})
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
            n_reactions = int(df["reaction"].nunique())
            for frac in _bootstrap_resolved(df, n_boot, seed):
                rows.append({"model": model["key"], "Model": model["label"], "method": method,
                            "Method": LABEL[method], "x_cat": x_of(c),
                            "frac_resolved": frac, "n_reactions": n_reactions})
    return rows


def build_paired_tests(model, run_dir, subdir, flux_df, conditions):
    """Per-condition Wilcoxon signed-rank test, AI vs MFA per-reaction R².

    Paired by reaction identity: both arms predict the same reaction set, so
    reaction r's AI R² and MFA R² are a genuine matched pair -- not an
    independent-groups comparison, and not dependent on the two arms having scored
    the identical sample columns (each arm's R² is already collapsed across its
    own samples before the two are compared).

    The panels report a count of resolved reactions, which is one number per cell and
    so has nothing to pair; the per-reaction R² behind that count does, and is what is
    tested here.
    """
    rows = []
    for c in conditions:
        ai = _long_frame(run_dir, subdir, c.name, "ai", flux_df)
        mfa = _long_frame(run_dir, subdir, c.name, "mfa", flux_df)
        if ai is None or mfa is None:
            continue
        ai_w = ai.pivot(index="reaction", columns="sample", values=["gt", "pred"])
        mfa_w = mfa.pivot(index="reaction", columns="sample", values=["gt", "pred"])
        ai_r2 = pd.Series(_per_reaction_r2(ai_w["gt"].to_numpy(), ai_w["pred"].to_numpy()),
                          index=ai_w.index)
        mfa_r2 = pd.Series(_per_reaction_r2(mfa_w["gt"].to_numpy(), mfa_w["pred"].to_numpy()),
                           index=mfa_w.index)
        shared = ai_r2.index.intersection(mfa_r2.index)
        if len(shared) < 6:      # scipy.stats.wilcoxon's own practical floor
            continue
        diff = (ai_r2[shared] - mfa_r2[shared]).to_numpy()
        diff = diff[np.isfinite(diff)]
        if len(diff) < 6 or np.allclose(diff, 0):
            continue
        stat, p = stats.wilcoxon(diff)
        rows.append({"model": model["key"], "Model": model["label"], "condition": c.name,
                    "n_paired": len(diff), "median_r2_diff_ai_minus_mfa": float(np.median(diff)),
                    "n_resolved_ai": int((ai_r2[shared] >= R2_THRESHOLD).sum()),
                    "n_resolved_mfa": int((mfa_r2[shared] >= R2_THRESHOLD).sum()),
                    "wilcoxon_stat": float(stat), "wilcoxon_p": float(p)})
    return rows


def _plot(df, order, reverse, xlabel, out_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.lineplot(data=df, x="x_cat", y="frac_resolved", hue="Method", style="Model",
                estimator="median",
                hue_order=[LABEL["ai"], LABEL["mfa"]],
                palette={LABEL["ai"]: COLORS["ai"], LABEL["mfa"]: COLORS["mfa"]},
                markers=True, dashes=True, err_style="band", errorbar=("pi", 95),
                sort=False, ax=ax)
    lo, hi = -0.5, len(order) - 0.5
    ax.set_xlim(hi, lo) if reverse else ax.set_xlim(lo, hi)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Median fraction of reactions with R² $\\geq$ {R2_THRESHOLD}")
    ax.set_ylim(0, 1)
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
        _plot(sub, order, reverse, xlabel, out_dir / f"uncertainty_{tag}_resolved.svg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_args(p).parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only this script's own outputs. A blanket uncertainty_* sweep also deletes
    # plot_uncertainty_validation.py's and plot_uncertainty_vs_mfa.py's figures, which
    # share the prefix but not this script's inputs.
    for pattern in ("uncertainty_*_resolved.svg", "uncertainty_*_nrmse.svg",
                    "uncertainty_benchmark_summary.csv", "uncertainty_paired_tests.csv"):
        for f in out_dir.glob(pattern):
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
