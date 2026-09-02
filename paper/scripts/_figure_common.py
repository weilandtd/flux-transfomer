"""
Shared plumbing for the Figure 2 scripts: model registry, run-dir resolution,
per-reaction R² loading, and plot styling.

Both paper/scripts/plot_mfa_benchmark.py and paper/scripts/plot_experimental_predictions.py
import from here so they share exactly one source of truth for "what a model is" and
"how its benchmark R² is computed" -- the benchmark HDF5s, not output/identifiability.csv,
so neither script silently depends on plot_test_performance.py having been run first
against the same run directory.
"""

import glob
import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sigmaflux.pipeline._paths import load_config, resolve_path
from sigmaflux.utils import (
    pair_fluxes, read_predicted_fluxes, read_reference_fluxes, reference_orientation)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from plot_test_performance import (  # noqa: E402
    LABEL, R2_THRESHOLD, drop_pool_reactions, load_benchmark, per_flux_max_abs,
    per_flux_nrmse, per_flux_r2, per_flux_rmse)
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

# ── This project's data conventions ───────────────────────────────────────────
# These describe how *these* runs and workbooks are laid out, not anything intrinsic to
# the file formats, so they live here rather than in sigmaflux.utils: another dataset may
# use different condition letters, a different sentinel, or none at all.
CONDITIONS = ["C", "N", "P", "A", "B"]

# Published bounds at or beyond this magnitude are the fit's "unbounded" sentinel (these
# workbooks write +/-1e8): the flux is not identifiable, and the bound is not a measurement.
UNBOUNDED_FLUX = 1e6

# Reactions with |flux| at or below this are excluded from the log-log figures.
FLUX_MIN_ABS = 1e-6

CONFIG_DIR = _REPO_ROOT / "paper" / "configs"

# Per-condition base colors. The letter means different things per organism (N is
# nitrogen-limited for the yeasts, naive for the T cell), so callers relabel; the color is
# keyed on the letter regardless.
CONDITION_COLORS = {
    "C": "#f3c47f",   # golden
    "N": "#7fb3d5",   # light blue
    "P": "#2c9c8f",   # teal
    "A": "#c17429",   # burnt orange -- activated T cell
    "B": "#9b7fb3",   # muted purple -- batch
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


def benchmark_metrics(model, run_suffix=None):
    """Per-reaction R², RMSE and NRMSE for both methods, or None when the model has no
    benchmark yet.

    Computed from the benchmark HDF5s rather than read from ``output/identifiability.csv``
    so the figure has one source of truth and does not silently depend on
    plot_test_performance.py having been run first -- and on the same run directory.

    Every metric comes from the same ``load_benchmark()`` call per method, so they share
    the identical chi2-accepted sample set -- no second, differently-filtered pass. The
    NRMSE denominator is the exception: it is each reaction's max |ground-truth flux|
    over the *whole* population, shared by both methods, because the two arms accept
    different sample sets and a per-method denominator would not be one scale. The final
    ``dropna()`` means a reaction missing R² (e.g. zero GT variance) is also dropped from
    the RMSE columns, keeping one reaction population across every column.
    """
    run = run_dir_for(model, run_suffix)
    missing = [d for d in ("mfa_benchmark", "ai_benchmark")
               if not list(run.glob(f"{d}/pred_*_flux.h5"))]
    if missing:
        print(f"  {model['label']}: skipped — no predictions in {', '.join(missing)}")
        return None

    _, flux_df, _ = load_data(str(run))
    flux_df.columns = range(flux_df.shape[1])
    # A pool's drain is pinned and its legs sum to it, so they are compartment fractions
    # rather than rates -- and the MFA arm has no such reactions to be compared against.
    # plot_test_performance.py already drops them; this is the same population.
    flux_df = drop_pool_reactions(flux_df, run)
    denom = per_flux_max_abs(flux_df)
    out = {}
    for method, sub in (("mfa", "mfa_benchmark"), ("ai", "ai_benchmark")):
        df, n_failed = load_benchmark(run / sub, flux_df)
        n = df["sample"].nunique()
        print(f"  {model['label']:26s} {LABEL[method]:10s} "
              f"{n} samples ({n_failed} unsolved), {df['reaction'].nunique()} reactions")
        out[f"r2_{method}"] = per_flux_r2(df)
        out[f"rmse_{method}"] = per_flux_rmse(df)
        out[f"nrmse_{method}"] = per_flux_nrmse(df, denom)
    return pd.DataFrame(out).dropna()


def _config_for_run(run_dir):
    """The organism config a run directory came from, by ``<config-stem>_<date>``.

    Longest stem first, so ``i_orientalis_ko_20260614`` picks i_orientalis_ko rather than
    i_orientalis.
    """
    stems = sorted(CONFIG_DIR.glob("*.yaml"), key=lambda p: -len(p.stem))
    for cfg_path in stems:
        if run_dir.name.startswith(cfg_path.stem):
            return cfg_path
    raise ValueError(f"Cannot detect organism config from '{run_dir.name}'. "
                     f"Known: {[p.stem for p in stems]}")


def resolve_organism(run_dir, config=None):
    """Experimental-data paths and media bounds for a run, read from its organism YAML."""
    config = pathlib.Path(config) if config else _config_for_run(run_dir)
    cfg, root = load_config(config)
    exp = cfg.get("experiment")
    if not exp:
        raise ValueError(f"{config} has no 'experiment:' block "
                         "(data_dir / reference_flux / gem_xlsx)")
    return dict(
        prefix    = cfg["name"],
        data      = exp["data_dir"],
        reference = exp["reference_flux"],
        gem_xlsx  = exp.get("gem_xlsx"),
        media     = cfg.get("sampling", {}).get("media", {}),
        root      = root,
    )


def build_reference_comparison(run_dir, org, panel_rxns):
    """Signed pred-vs-reference frame for every condition a run has predictions for.

    Locates this project's files and hands them to :mod:`sigmaflux.utils.read_reference_flux`,
    which does the reading and pairing but knows nothing about where they live.
    """
    # A bare filename sits next to the experimental data; anything with a directory in
    # it is resolved against the project root instead, so a reference shared between two
    # platforms (the E. coli LC-MS and GC-MS arms quote the same published MFA fit) does
    # not have to be duplicated into both data directories.
    reference = org["reference"]
    rel = reference if os.path.dirname(reference) else os.path.join(org["data"], reference)
    ref_path = pathlib.Path(resolve_path(rel, org["root"]))
    if not ref_path.exists():
        raise FileNotFoundError(f"reference_flux not found: {ref_path}")
    orientation = reference_orientation(org.get("media"))
    parts = []

    for cond in CONDITIONS:
        hits = sorted(glob.glob(str(run_dir / "predictions" /
                                    f"predictions_{org['prefix']}_{cond}_*.csv")))
        if not hits:
            print(f"  condition {cond}: no predictions CSV — skipping")
            continue
        ref_value = read_reference_fluxes(ref_path, cond, unbounded=UNBOUNDED_FLUX)
        if ref_value is None:
            print(f"  condition {cond}: no reference column — skipping")
            continue

        pred = {k: read_predicted_fluxes(hits[-1], c)
                for k, c in (("value", "Value"), ("lo", "Value_lo"), ("hi", "Value_hi"))}
        ref = {
            "value": ref_value,
            "lo": read_reference_fluxes(ref_path, cond, "_mfaLB", unbounded=UNBOUNDED_FLUX),
            "hi": read_reference_fluxes(ref_path, cond, "_mfaUB", unbounded=UNBOUNDED_FLUX),
        }
        # Growth rate D = the biomass reaction flux per growth condition (h⁻¹).
        rates = (ref_value.loc["BIOMASS"].abs().round(2).to_dict()
                 if "BIOMASS" in ref_value.index else {})

        parts.append(pair_fluxes(pred, ref, condition=cond, orientation=orientation,
                                 reactions=panel_rxns, growth_rates=rates,
                                 min_abs=FLUX_MIN_ABS))

    return pd.concat(parts, ignore_index=True)


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
