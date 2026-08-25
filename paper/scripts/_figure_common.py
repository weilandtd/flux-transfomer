"""
Shared plumbing for the Figure 2 scripts: model registry, run-dir resolution,
per-reaction R² loading, and plot styling.

Both paper/scripts/plot_mfa_benchmark.py and paper/scripts/plot_experimental_predictions.py
import from here so they share exactly one source of truth for "what a model is" and
"how its benchmark R² is computed" -- the benchmark HDF5s, not output/identifiability.csv,
so neither script silently depends on plot_test_performance.py having been run first
against the same run directory.
"""

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Both analysis dirs, since this script lives outside either of them.
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "benchmark"))
from analysis_predicted_fluxes import (  # noqa: E402
    CONDITION_COLORS, FLUX_MIN_ABS, _resolve_organism, build_plot_df)
from plot_test_performance import (  # noqa: E402
    LABEL, R2_THRESHOLD, load_benchmark, per_flux_r2)
from plot_test_performance import COLORS as METHOD_COLORS  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402

# ── What to draw ──────────────────────────────────────────────────────────────
# `run` is relative to the repo root.
MODELS = [
    dict(key="e_coli_core", label="E. coli core",
         run="paper/.results/e_coli_core_20260822"),
    dict(key="s_cerevisiae", label="S. cerevisiae",
         run="paper/.results/s_cerevisiae_20260822"),
    dict(key="i_orientalis", label="I. orientalis",
         run="paper/.results/i_orientalis_20260822"),
    dict(key="m_musculus_t_cell", label="M. musculus T cell",
         run="paper/.results/m_musculus_t_cell_20260822"),
]
MODELS_BY_KEY = {m["key"]: m for m in MODELS}

R2_FLOOR = -0.2          # R² at or below this piles up on the axis floor
COLORS = {
    "reference": "#AAAAAA",   # threshold and identity lines
    "box":       "#cccccc",   # annotation box edge
}


def apply_style():
    mpl.use("Agg")
    mpl.rcParams.update({"font.family": "Arial", "font.size": 12})


def add_common_args(parser, models_help="Comma-separated model keys (default: all)"):
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=_REPO_ROOT / "paper" / "figures",
                        help="Where the output files go (default: paper/figures)")
    parser.add_argument("--models", default=None, help=models_help)
    parser.add_argument("--run-suffix", default=None,
                        help="Replace the date suffix of every run directory in MODELS, "
                             "so a different set of runs can be plotted without editing "
                             "the file")
    return parser


def selected_models(args):
    """MODELS filtered to --models, in the order given (default: all, in MODELS order)."""
    if not args.models:
        return MODELS
    keys = [k.strip() for k in args.models.split(",")]
    unknown = [k for k in keys if k not in MODELS_BY_KEY]
    if unknown:
        raise SystemExit(f"Unknown model key(s) {unknown}. "
                         f"Known: {list(MODELS_BY_KEY)}")
    return [MODELS_BY_KEY[k] for k in keys]


def run_dir_for(model, run_suffix=None):
    """Absolute run directory for a model, with the date suffix optionally swapped."""
    run = model["run"]
    if run_suffix:
        run = f"{run.rsplit('_', 1)[0]}_{run_suffix}"
    return (_REPO_ROOT / run).resolve()


def benchmark_r2(model, run_suffix=None):
    """Per-reaction R² for both methods, or None when the model has no benchmark yet.

    Computed from the benchmark HDF5s rather than read from ``output/identifiability.csv``
    so the figure has one source of truth and does not silently depend on
    plot_test_performance.py having been run first -- and on the same run directory.
    """
    run = run_dir_for(model, run_suffix)
    missing = [d for d in ("mfa_benchmark", "ai_benchmark")
               if not list(run.glob(f"{d}/pred_*_flux.h5"))]
    if missing:
        print(f"  {model['label']}: skipped — no predictions in {', '.join(missing)}")
        return None

    _, flux_df, _ = load_data(str(run))
    flux_df.columns = range(flux_df.shape[1])
    out = {}
    for method, sub in (("mfa", "mfa_benchmark"), ("ai", "ai_benchmark")):
        df, n_failed = load_benchmark(run / sub, flux_df)
        n = df["sample"].nunique()
        print(f"  {model['label']:26s} {LABEL[method]:10s} "
              f"{n} samples ({n_failed} unsolved), {df['reaction'].nunique()} reactions")
        out[f"r2_{method}"] = per_flux_r2(df)
    return pd.DataFrame(out).dropna()


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
