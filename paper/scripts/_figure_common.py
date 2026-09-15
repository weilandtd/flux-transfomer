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
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sigmaflux.pipeline._paths import load_config, resolve_path
from sigmaflux.utils import (
    pair_fluxes, read_predicted_fluxes, read_reference_fluxes, reference_orientation)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from plot_test_performance import (  # noqa: E402
    LABEL, R2_THRESHOLD, drop_pool_reactions, load_benchmark, per_flux_nrmse,
    per_flux_r2, per_flux_rms, per_flux_rmse)
from plot_test_performance import COLORS as METHOD_COLORS  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402

# ── What to draw ──────────────────────────────────────────────────────────────
# `run` is relative to the repo root.
MODELS = [
    dict(key="e_coli_core", label="E. coli core",
         run="paper/.results/e_coli_core_malatefix_20260914"),
    dict(key="s_cerevisiae", label="S. cerevisiae",
         run="paper/.results/s_cerevisiae_malatefix_20260914"),
    dict(key="i_orientalis", label="I. orientalis",
         run="paper/.results/i_orientalis_malatefix_20260914"),
    dict(key="m_musculus_t_cell", label="M. musculus T cell",
         run="paper/.results/m_musculus_t_cell_malatefix_20260914"),
]
MODELS_BY_KEY = {m["key"]: m for m in MODELS}

R2_FLOOR = -0.2          # R² at or below this piles up on the axis floor
COLORS = {
    "reference": "#AAAAAA",   # threshold and identity lines
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


# Which conditions and growth condition go into each model's experimental panel.
# "max_D" keeps only the highest dilution rate per condition; "wt" keeps only the
# wild-type strain (gc=1, condition B); "all" keeps every one. Shared by the flux and
# the MID panels so both describe the same set of measured states.
PREDICTION_SPECS = {
    # The Long & Antoniewicz reference is a 21-strain knockout panel (condition 'B',
    # gc 1..21 per data/E_coli/strain_index.csv); reindexed onto our own reaction ids
    # by models/atomistic/build_e_coli_reference_flux.py. Wild type only, not the
    # knockouts, is the like-for-like comparison the other models make.
    "e_coli_core":       dict(conditions=["B"],           select="wt"),
    "s_cerevisiae":      dict(conditions=["C", "N", "P"], select="max_D"),
    "i_orientalis":      dict(conditions=["C", "N", "P"], select="max_D"),
    # Naive (N) dropped: its measured glucose/lactate/gln are rejected by both MFA
    # fits (see the tcell_naive_extflux_unreliable note), so the reference itself is
    # unreliable there -- activated is the only condition worth plotting against.
    "m_musculus_t_cell": dict(conditions=["A"],           select="all"),
}

# The condition letter means different things per organism: N is nitrogen-limited for
# the yeasts and naive for the T cell, so the legend cannot be derived from the letter.
CONDITION_LABEL = {
    "m_musculus_t_cell": {"N": "naive", "A": "activated"},
}

# Literature-style names for the tracer metabolite ids seen in tracer strings below.
_TRACER_METABOLITE_NAMES = {
    "glc__D_e": "glucose", "Gluc_e": "glucose", "gln__L_e": "glutamine",
}

#: Marker shapes cycled by distinct tracer within a panel, so tracer is visually
#: distinguishable from condition (which is colored) without a second color scale.
TRACER_MARKERS = ["o", "^", "s", "D", "P", "X"]


def tracer_label(tracer_str):
    """Literature-style label for an internal tracer id.

    ``'glc__D_e_[1, 1, 0, 0, 0, 0]_0.5'`` -> ``'1,2-¹³C-glucose (50%)'``;
    ``'glc__D_e_[1, 1, 1, 1, 1, 1]_0.5'`` -> ``'U-¹³C-glucose (50%)'``.
    Falls back to the raw string for anything that doesn't match the
    ``<metabolite>_[<pattern>]_<enrichment>`` shape this codebase generates tracer ids in.
    """
    m = re.match(r"^(.+?)_\[([\d,\s]+)\]_([\d.]+)$", str(tracer_str))
    if not m:
        return str(tracer_str)
    met_id, pattern_str, enrichment = m.groups()
    pattern = [int(x) for x in pattern_str.split(",")]
    positions = [i + 1 for i, v in enumerate(pattern) if v == 1]
    name = _TRACER_METABOLITE_NAMES.get(met_id, met_id)
    pct = f"{float(enrichment) * 100:.0f}%"
    tag = "U" if len(positions) == len(pattern) else ",".join(str(p) for p in positions)
    return f"{tag}-¹³C-{name} ({pct})"


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
    NRMSE denominator is the exception: it is each reaction's root-mean-square flux over
    the *whole* population, shared by both methods, because the two arms accept different
    sample sets and a per-method denominator would not be one scale. The final
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
    denom = per_flux_rms(flux_df)
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
        # Most Shen-et-al workbooks report uptake as an unsigned magnitude, needing
        # reference_orientation()'s model-bounds-derived flip. A workbook that already
        # reports signed values sets this so build_reference_comparison() skips it.
        reference_signed = exp.get("reference_signed", False),
        root      = root,
        cfg       = cfg,
    )


def reference_path(org):
    """Absolute path to a run's published reference-flux workbook.

    A bare filename sits next to the experimental data; anything with a directory in it
    is resolved against the project root instead, so a reference one organism borrows
    from another's data directory (the genome-scale E. coli arm quotes the GC-MS
    workbook's published fit) does not have to be duplicated into both.
    """
    reference = org["reference"]
    rel = reference if os.path.dirname(reference) else os.path.join(org["data"], reference)
    path = pathlib.Path(resolve_path(rel, org["root"]))
    if not path.exists():
        raise FileNotFoundError(f"reference_flux not found: {path}")
    return path


def growth_rates(ref_value):
    """Growth condition -> dilution rate (h⁻¹), the reference's own BIOMASS row."""
    if ref_value is None or "BIOMASS" not in ref_value.index:
        return {}
    return ref_value.loc["BIOMASS"].abs().round(2).to_dict()


def build_reference_comparison(run_dir, org, panel_rxns):
    """Signed pred-vs-reference frame for every condition a run has predictions for.

    Locates this project's files and hands them to :mod:`sigmaflux.utils.read_reference_flux`,
    which does the reading and pairing but knows nothing about where they live.
    """
    ref_path = reference_path(org)
    orientation = {} if org.get("reference_signed") else reference_orientation(org.get("media"))
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
        parts.append(pair_fluxes(pred, ref, condition=cond, orientation=orientation,
                                 reactions=panel_rxns,
                                 growth_rates=growth_rates(ref_value),
                                 min_abs=FLUX_MIN_ABS))

    return pd.concat(parts, ignore_index=True)


def error_offsets(value, lo, hi, lim_lo, lim_hi):
    """Per-point (lower, upper) error-bar lengths on a log magnitude axis.

    A NaN lower bound means the interval has no positive floor -- it spans zero, or the
    published fit never bounded it -- so the bar is drawn all the way to the axis edge,
    because that is where it actually reaches. Cutting it to a fixed stub instead makes it
    indistinguishable from a genuinely tight interval, which is the worse failure. The
    axis floor comes from the data (:func:`axis_floor`), so a truthful bar spans the
    decades actually in play rather than every decade the shared floor allows.

    A bound of zero on a log axis is the same situation and clips to the same place; a
    missing bound contributes no bar at all.
    """
    value = np.asarray(value, dtype=float)
    lo = np.where(np.isnan(lo) & np.isfinite(hi), lim_lo, lo)
    lower = np.where(np.isfinite(lo), value - np.clip(lo, lim_lo, value), 0.0)
    upper = np.where(np.isfinite(hi), np.clip(hi, value, lim_hi) - value, 0.0)
    # A point outside the axis limits clips to the wrong side and yields a negative
    # length, which matplotlib rejects outright. Floor both: the point is off-axis, so
    # the honest bar is no bar.
    return np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])


def axis_floor(*series, decades=0.5, floor=FLUX_MIN_ABS):
    """Lower axis limit: half a decade below the smallest positive plotted magnitude.

    Set from the data rather than from the shared ``floor``, which sits decades below
    anything some panels plot and would leave most of them empty. Non-positive values are
    ignored -- they have no place on a log axis and the caller drops them -- and the
    result never goes below ``floor``.
    """
    values = np.concatenate([np.abs(np.asarray(s, dtype=float)).ravel() for s in series])
    positive = values[np.isfinite(values) & (values > 0)]
    if not positive.size:
        return floor
    return max(floor, float(positive.min()) / 10.0 ** decades)


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
