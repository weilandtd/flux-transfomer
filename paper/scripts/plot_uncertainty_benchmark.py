"""
Missing-metabolite and measurement-noise sensitivity — aggregation + figures.

Reads what ``sigmaflux-benchmark-uncertainty`` writes --
``<run_dir>/uncertainty_benchmark/<condition>/{ai,mfa}/`` -- and reduces it to one
tidy row per (model, condition, method): median per-reaction R², the fraction of
reactions above ``R2_THRESHOLD``, how many samples were scored, and -- for the mfa
method -- the χ² goodness-of-fit acceptance rate read from its ``*_convergence.csv``.

MFA is scored over EVERY fit here (``load_benchmark(..., require_converged=False)``),
unlike Figure 2's ``benchmark_r2()`` which scores only χ²-accepted fits: at 100
samples/condition the accepted-only filter would leave too few fits to say anything
for some organisms (S. cerevisiae accepts under 10% of its fits at the default noise
level -- see ``mfa_benchmark_characterization`` project notes). So the MFA numbers
here read lower than Figure 2's published one; the comparison ACROSS conditions is
still like-for-like, and ``uncertainty_mfa_acceptance.svg`` is what makes that
filter's own sensitivity visible instead of silently hiding it.

Outputs (--out-dir, default paper/figures):
  uncertainty_missing_metabolites.svg   median R² vs % metabolites observed
  uncertainty_noise.svg                 median R² vs noise level (one-at-a-time cross)
  uncertainty_mfa_acceptance.svg        χ²-acceptance rate vs condition, MFA only
  uncertainty_benchmark_summary.csv     the tidy table behind all three

Usage:
    python paper/scripts/plot_uncertainty_benchmark.py
    python paper/scripts/plot_uncertainty_benchmark.py --models e_coli_core,s_cerevisiae
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
from plot_test_performance import (  # noqa: E402
    R2_THRESHOLD, converged_samples, load_benchmark, per_flux_r2)
from sigmaflux.ai.data import load_data  # noqa: E402

apply_style()

#: Per-model line colour, shared across all three figures. Method is distinguished
#: by linestyle (solid = SigmaFlux, dashed = MFA), not colour, so a reader tracks
#: one model across both cues rather than needing an 8-entry colour legend.
MODEL_COLORS = {
    "e_coli_core":       "#c1440e",
    "s_cerevisiae":      "#1B6F6A",
    "i_orientalis":      "#2C7BB6",
    "m_musculus_t_cell": "#6a3d9a",
}
COLORS = {
    "reference": "#AAAAAA",
}
METHOD_STYLE = {"ai": "-", "mfa": "--"}
METHOD_LABEL = {"ai": "SigmaFlux", "mfa": "MFA"}

#: (condition, study, x value, x label) -- study membership and x mirror
#: sigmaflux.pipeline.run_uncertainty_benchmark.CONDITIONS exactly; the baseline
#: condition appears once per study it belongs to; see that module's docstring for
#: why 1e-6 stands in for a literal 0% noise.
_GRID = [
    ("met100_lab001_flux005", "missing",    100, "100%"),
    ("met075_lab001_flux005", "missing",    75,  "75%"),
    ("met050_lab001_flux005", "missing",    50,  "50%"),
    ("met025_lab001_flux005", "missing",    25,  "25%"),
    ("met100_lab000_flux005", "lab_noise",  0,   "0%"),
    ("met100_lab001_flux005", "lab_noise",  1,   "1%"),
    ("met100_lab005_flux005", "lab_noise",  5,   "5%"),
    ("met100_lab001_flux000", "flux_noise", 0,   "0%"),
    ("met100_lab001_flux005", "flux_noise", 5,   "5%"),
    ("met100_lab001_flux025", "flux_noise", 25,  "25%"),
]
_X_PREFIX = {"missing": "", "lab_noise": "MID ", "flux_noise": "Flux "}
NOISE_ORDER = ["MID 0%", "MID 1%", "MID 5%", "Flux 0%", "Flux 5%", "Flux 25%"]
ALL_ORDER = ["25%", "50%", "75%", "100%"] + NOISE_ORDER


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--uncertainty-subdir", default="uncertainty_benchmark",
                        help="Subdirectory of each run holding the condition "
                             "folders (default: uncertainty_benchmark)")
    return parser


def score_condition(run_dir, subdir, condition, method, flux_df):
    """One tidy row for a (condition, method) cell, or ``None`` if it wasn't run."""
    d = run_dir / subdir / condition / method
    if not list(d.glob("pred_*_flux.h5")):
        return None
    df, _ = load_benchmark(d, flux_df, require_converged=False)
    r2 = per_flux_r2(df)
    n_scored = int(df["sample"].nunique())
    row = {
        "median_r2": float(np.median(r2)) if len(r2) else float("nan"),
        "frac_above_threshold": float((r2 >= R2_THRESHOLD).mean()) if len(r2) else float("nan"),
        "n_scored": n_scored,
    }
    if method == "mfa":
        accepted = converged_samples(d)
        row["accept_rate"] = (len(accepted) / n_scored
                              if accepted is not None and n_scored else float("nan"))
    return row


def build_summary(models, run_suffix, subdir):
    """Tidy DataFrame: one row per (model, condition, method) that was actually run."""
    rows = []
    for model in models:
        run_dir = run_dir_for(model, run_suffix)
        if not (run_dir / subdir).exists():
            print(f"  {model['label']}: skipped — no {run_dir / subdir}")
            continue
        _, flux_df, _ = load_data(str(run_dir))
        flux_df.columns = range(flux_df.shape[1])

        cache = {}
        for name, study, x, xlabel in _GRID:
            for method in ("ai", "mfa"):
                key = (name, method)
                if key not in cache:
                    cache[key] = score_condition(run_dir, subdir, name, method, flux_df)
                res = cache[key]
                if res is None:
                    continue
                rows.append({
                    "model": model["key"], "label": model["label"], "condition": name,
                    "study": study, "x": x, "x_cat": _X_PREFIX[study] + xlabel,
                    "method": method, **res,
                })
        n = sum(1 for k in cache if cache[k] is not None)
        print(f"  {model['label']:26s} {n}/{len(cache)} (condition, method) cells found")
    return pd.DataFrame(rows)


def _plot_lines(ax, df, x_col, order=None):
    """One line per (model, method) -- shared by all three line figures."""
    for (model, method), grp in df.groupby(["model", "method"], sort=False):
        label = f"{grp['label'].iloc[0]} ({METHOD_LABEL[method]})"
        grp = grp.sort_values(x_col) if order is None else (
            grp.set_index(x_col).reindex(order))
        ax.plot(grp[x_col] if order is None else grp.index, grp["median_r2"],
                color=MODEL_COLORS[model], linestyle=METHOD_STYLE[method],
                marker="o", ms=4, lw=1.5, label=label)


def plot_missing_metabolites(df, out_dir):
    sub = df[df["study"] == "missing"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _plot_lines(ax, sub, "x")
    ax.set_xlim(20, 105)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.set_xlabel("Metabolites observed")
    ax.set_ylabel(r"Median flux $R^2$")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "uncertainty_missing_metabolites.svg")


def plot_noise(df, out_dir):
    sub = df[df["study"].isin(("lab_noise", "flux_noise"))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _plot_lines(ax, sub, "x_cat", order=NOISE_ORDER)
    ax.axvline(2.5, color=COLORS["reference"], lw=0.8, ls=":", zorder=0)
    ax.set_xticks(range(len(NOISE_ORDER)))
    ax.set_xticklabels(NOISE_ORDER, rotation=40, ha="right")
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel(r"Median flux $R^2$")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "uncertainty_noise.svg")


def plot_acceptance(df, out_dir):
    """χ²-acceptance rate across the whole grid, MFA only -- the second readout
    scoring all fits (rather than accepted-only) buys us."""
    sub = df[(df["method"] == "mfa") & df["accept_rate"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for model, grp in sub.groupby("model", sort=False):
        grp = grp.set_index("x_cat").reindex(ALL_ORDER).reset_index()
        ax.plot(grp["x_cat"], grp["accept_rate"], color=MODEL_COLORS[model],
                marker="o", ms=4, lw=1.5, label=grp["label"].iloc[0])
    ax.axvline(3.5, color=COLORS["reference"], lw=0.8, ls=":", zorder=0)
    ax.axvline(6.5, color=COLORS["reference"], lw=0.8, ls=":", zorder=0)
    ax.set_xticks(range(len(ALL_ORDER)))
    ax.set_xticklabels(ALL_ORDER, rotation=40, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$\chi^2$ acceptance rate")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "uncertainty_mfa_acceptance.svg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_args(p).parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = selected_models(args)
    print("── Uncertainty benchmark (missing metabolites, noise) ──")
    df = build_summary(models, args.run_suffix, args.uncertainty_subdir)

    if df.empty:
        print("No model had uncertainty-benchmark data — nothing written.")
        return

    df.to_csv(out_dir / "uncertainty_benchmark_summary.csv", index=False)
    print(f"Saved: {out_dir / 'uncertainty_benchmark_summary.csv'}")

    plot_missing_metabolites(df, out_dir)
    plot_noise(df, out_dir)
    plot_acceptance(df, out_dir)


if __name__ == "__main__":
    main()
