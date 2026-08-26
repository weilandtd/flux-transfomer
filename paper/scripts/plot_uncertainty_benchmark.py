"""
Missing-metabolite and measurement-noise sensitivity — bar plots.

Reads what ``sigmaflux-benchmark-uncertainty`` writes --
``<run_dir>/uncertainty_benchmark/<condition>/{ai,mfa}/`` -- and for each (model,
study) pair draws one bar chart: x = condition level, one bar per method (SigmaFlux
/ MFA) in the paper's established method colours (``plot_test_performance.COLORS``),
y = number of reactions with per-reaction R² > ``R2_THRESHOLD`` (0.9).

Error bars are a 95% bootstrap percentile interval, resampling the scored validation
samples with replacement (``N_BOOT`` draws) and recomputing the reaction count on
each draw -- the standard way to get an error bar on a threshold-count statistic
without new raw data. Mirrors the barplot convention already used in
``paper/analysis/observability/ablation_grid.py`` (``estimator='mean'``,
``errorbar=('pi', ...)`` over per-replicate rows), just with ``hue='Method'``.

MFA is scored over EVERY fit here (``load_benchmark(..., require_converged=False)``),
unlike Figure 2's ``benchmark_r2()`` which scores only χ²-accepted fits: at 100
samples/condition the accepted-only filter would leave too few fits to say anything
for some organisms (S. cerevisiae accepts under 10% of its fits at the default noise
level -- see the ``mfa_benchmark_characterization`` project notes). The MFA counts
here therefore read lower than Figure 2's published ones; the comparison ACROSS
conditions is still like-for-like.

Outputs (--out-dir, default paper/figures), one bar chart per (model, study):
  uncertainty_missing_metabolites_<model>.svg
  uncertainty_lab_noise_<model>.svg
  uncertainty_flux_noise_<model>.svg
  uncertainty_benchmark_summary.csv   bootstrap mean + 95% PI behind every bar

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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from _figure_common import (  # noqa: E402
    add_common_args, apply_style, run_dir_for, save_figure, selected_models)
from plot_test_performance import COLORS, LABEL, R2_THRESHOLD, load_benchmark  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402

apply_style()

N_BOOT = 200   # bootstrap resamples (over scored samples) behind each error bar

#: (condition, x-axis category label), in plot order -- study membership and
#: condition names mirror sigmaflux.pipeline.run_uncertainty_benchmark.CONDITIONS
#: exactly; see that module's docstring for why 1e-6 stands in for a literal 0% noise.
_GRID = {
    "missing_metabolites": [
        ("met025_lab001_flux005", "25%"),
        ("met050_lab001_flux005", "50%"),
        ("met075_lab001_flux005", "75%"),
        ("met100_lab001_flux005", "100%"),
    ],
    "lab_noise": [
        ("met100_lab000_flux005", "0%"),
        ("met100_lab001_flux005", "1%"),
        ("met100_lab005_flux005", "5%"),
    ],
    "flux_noise": [
        ("met100_lab001_flux000", "0%"),
        ("met100_lab001_flux005", "5%"),
        ("met100_lab001_flux025", "25%"),
    ],
}
X_LABEL = {
    "missing_metabolites": "Metabolites observed",
    "lab_noise": "MID noise (% absolute)",
    "flux_noise": "Flux noise (% relative)",
}


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--uncertainty-subdir", default="uncertainty_benchmark",
                        help="Subdirectory of each run holding the condition "
                             "folders (default: uncertainty_benchmark)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap resamples per bar (default: {N_BOOT})")
    return parser


def _bootstrap_counts(df, n_boot, seed):
    """Bootstrap distribution of #reactions with R² > R2_THRESHOLD.

    Resamples the scored *samples* with replacement (not individual rows), so a
    resampled column carries every reaction's (gt, pred) pair for that sample
    together -- the correct unit for a flux-recovery bootstrap. Vectorised: the
    python loop is only over ``n_boot``, each iteration is plain numpy array ops
    over a (reactions, samples) matrix.

    Args:
        df: long frame with columns [sample, reaction, gt, pred], as returned by
            ``plot_test_performance.load_benchmark``.
        n_boot: number of bootstrap resamples.
        seed: RNG seed, for reproducibility across runs.
    """
    gt = df.pivot(index="reaction", columns="sample", values="gt")
    pred = df.pivot(index="reaction", columns="sample", values="pred")
    gt, pred = gt.align(pred, join="inner", axis=None)
    valid = gt.notna().to_numpy() & pred.notna().to_numpy()
    gt_arr = np.where(valid, gt.to_numpy(), np.nan)
    pred_arr = np.where(valid, pred.to_numpy(), np.nan)
    n_samples = gt_arr.shape[1]

    rng = np.random.default_rng(seed)
    counts = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        draw = rng.integers(0, n_samples, size=n_samples)
        g, p = gt_arr[:, draw], pred_arr[:, draw]
        n_valid = np.sum(~np.isnan(g), axis=1)
        mean = np.nanmean(g, axis=1)
        with np.errstate(invalid="ignore"):
            ss_res = np.nansum((p - g) ** 2, axis=1)
            ss_tot = np.nansum((g - mean[:, None]) ** 2, axis=1)
            r2 = np.where((ss_tot > 0) & (n_valid >= 2), 1.0 - ss_res / ss_tot, np.nan)
        counts[b] = int(np.nansum(r2 > R2_THRESHOLD))
    return counts


def score_condition(run_dir, subdir, condition, method, flux_df, n_boot, seed):
    """Bootstrap replicate rows for one (condition, method) cell, or ``None`` if unrun."""
    d = run_dir / subdir / condition / method
    if not list(d.glob("pred_*_flux.h5")):
        return None
    df, _ = load_benchmark(d, flux_df, require_converged=False)
    counts = _bootstrap_counts(df, n_boot, seed)
    return counts, int(df["sample"].nunique())


def build_replicates(models, run_suffix, subdir, n_boot):
    """Long DataFrame, one row per (model, study, x_cat, method, bootstrap draw)."""
    rows, seed = [], 0
    for model in models:
        run_dir = run_dir_for(model, run_suffix)
        if not (run_dir / subdir).exists():
            print(f"  {model['label']}: skipped — no {run_dir / subdir}")
            continue
        _, flux_df, _ = load_data(str(run_dir))
        flux_df.columns = range(flux_df.shape[1])

        cache = {}
        for study, conditions in _GRID.items():
            for condition, x_cat in conditions:
                for method in ("ai", "mfa"):
                    key = (condition, method)
                    if key not in cache:
                        seed += 1
                        cache[key] = score_condition(
                            run_dir, subdir, condition, method, flux_df, n_boot, seed)
                    res = cache[key]
                    if res is None:
                        continue
                    counts, n_scored = res
                    rows += [{"model": model["key"], "label": model["label"],
                             "study": study, "x_cat": x_cat, "condition": condition,
                             "method": method, "Method": LABEL[method],
                             "n_scored": n_scored, "n_reactions": c} for c in counts]
        n_found = sum(1 for v in cache.values() if v is not None)
        print(f"  {model['label']:26s} {n_found}/{len(cache)} (condition, method) cells found")
    return pd.DataFrame(rows)


def plot_bars(replicates, model, study, out_dir):
    sub = replicates[(replicates["model"] == model["key"]) & (replicates["study"] == study)]
    if sub.empty:
        return
    order = [x for _, x in _GRID[study]]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.barplot(data=sub, x="x_cat", y="n_reactions", order=order,
               hue="Method", hue_order=[LABEL["ai"], LABEL["mfa"]],
               palette={LABEL["ai"]: COLORS["ai"], LABEL["mfa"]: COLORS["mfa"]},
               estimator="mean", errorbar=("pi", 95), err_kws={"linewidth": 1.2},
               capsize=0.15, ax=ax)
    ax.set_xlabel(X_LABEL[study])
    ax.set_ylabel(f"Reactions with $R^2$ > {R2_THRESHOLD}")
    ax.legend(frameon=False, fontsize=8, title=None)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"uncertainty_{study}_{model['key']}.svg")


def _clear_old_outputs(out_dir):
    """Remove every uncertainty_* artifact so a re-run never leaves stale files
    (an old model x study combination, or the earlier line-plot figures)."""
    for f in out_dir.glob("uncertainty_*"):
        f.unlink()
        print(f"Removed: {f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_args(p).parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_old_outputs(out_dir)

    models = selected_models(args)
    print("── Uncertainty benchmark (missing metabolites, noise) ──")
    replicates = build_replicates(models, args.run_suffix, args.uncertainty_subdir, args.n_boot)

    if replicates.empty:
        print("No model had uncertainty-benchmark data — nothing written.")
        return

    summary = (replicates.groupby(["model", "label", "study", "x_cat", "method", "n_scored"])
              ["n_reactions"]
              .agg(mean="mean", pi_lo=lambda s: s.quantile(0.025), pi_hi=lambda s: s.quantile(0.975))
              .reset_index())
    summary.to_csv(out_dir / "uncertainty_benchmark_summary.csv", index=False)
    print(f"Saved: {out_dir / 'uncertainty_benchmark_summary.csv'}")

    for model in models:
        for study in _GRID:
            plot_bars(replicates, model, study, out_dir)


if __name__ == "__main__":
    main()
