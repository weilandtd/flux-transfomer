"""
Stage 2 experiment: does a median-seeking aggregate beat the ensemble mean for SC?

Compares four ways to collapse the SC ensemble into one flux state -- mean, median,
medoid, medoid_convex (see sigmaflux.ai.uncertainty._medoid_weights) -- on real SC data,
reporting both feasibility and R^2 against the published/x3cflux reference flux, using
the *exact* same methodology as plot_experimental.py (MFA-identifiability
reaction filter, orientation, max-D selection, signed-flux R^2) so the numbers are
directly comparable to experimental_flux_summary.csv's mean-aggregate R^2.

No retrain, no SLURM: a handful of EMU forward passes on the existing checkpoint.

Usage:
    python paper/analysis/diagnostics/compare_aggregation_sc.py
"""
import sys
import pathlib

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))

from sigmaflux.ai.inference import load_predictor
from sigmaflux.ai.uncertainty import sample_inputs
from sigmaflux.pipeline._benchmark import build_mfa_context
from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path, load_config
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, load_condition_inputs, predict_draws)
from sigmaflux.utils import read_reference_fluxes
from sigmaflux.utils.read_reference_flux import pair_fluxes, reference_orientation

from validate_inferred_fluxes import (  # noqa: E402
    aggregate_state, feasibility, report_feasibility, split_predictions,
    as_state_frames,
)
from _figure_common import (  # noqa: E402
    MODELS_BY_KEY, R2_THRESHOLD, FLUX_MIN_ABS, benchmark_metrics, growth_rates,
    PREDICTION_SPECS, r2_linear,
)

RUN_DIR = _REPO_ROOT / "paper/.results/s_cerevisiae_20260917"
CONFIG = _REPO_ROOT / "paper/configs/s_cerevisiae.yaml"
DATA_DIR = _REPO_ROOT / "data/S_cervesiae"
HOWS = ["mean", "median", "medoid", "medoid_convex"]


def main():
    cfg, root = load_config(str(CONFIG), None)
    mapping_csv = resolve_path(mapping_csv_for(cfg), root)
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None

    print(f"Loading ensemble from {RUN_DIR/'ensemble'} ...")
    sf = load_predictor(str(CONFIG), str(RUN_DIR / "ensemble"), device="cpu")

    ext_file = _find_ext_flux_file(str(DATA_DIR))
    conditions = _discover_conditions(str(DATA_DIR))
    ref_path = _REPO_ROOT / "data/S_cervesiae/Shen_et_al_flux_data_SC.xlsx"

    # Same MFA-identifiability reaction filter plot_experimental.py uses.
    model = MODELS_BY_KEY["s_cerevisiae"]
    tbl = benchmark_metrics(model, run_suffix="sentinelfix_20260912")
    mfa_rxns = set(tbl.index[tbl["r2_mfa"] >= R2_THRESHOLD]) if tbl is not None else None
    print(f"MFA-identifiable reactions (R2 >= {R2_THRESHOLD}): "
          f"{len(mfa_rxns) if mfa_rxns else 'none -- no filter'}")

    orientation = reference_orientation(model.get("media"))

    from cobra.io.json import load_json_model
    cobra_model = load_json_model(str(RUN_DIR / "model_prepared_for_mfa.json"))

    rows = []
    per_how_frames = {how: [] for how in HOWS}
    mfa_model = reaction_ids = labeling_to = labeling_index = None

    for cond_letter in PREDICTION_SPECS["s_cerevisiae"]["conditions"]:
        if cond_letter not in conditions:
            continue
        mid_file = conditions[cond_letter]
        ref_value = read_reference_fluxes(str(ref_path), cond_letter)
        if ref_value is None:
            print(f"  {cond_letter}: no reference column -- skipping")
            continue
        gc = int(ref_value.columns.max())  # max_D, matches PREDICTION_SPECS select="max_D"

        df_L, df_Vm, df_err = load_condition_inputs(
            mid_file, ext_file, cond_letter, gc, mapping_csv, flux_names, model=cobra_model)

        rng = np.random.default_rng(42)
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=rng, n_bootstrap=1)
        n_draws = df_L_draws["Sample"].max()
        raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))
        flux_wide, rev_wide, rev_predicted = split_predictions(
            raw, sf, [r.id for r in cobra_model.reactions])

        if mfa_model is None:
            v0 = flux_wide.iloc[0]
            mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
                cobra_model, cfg["labeling"]["tracers"], v0,
                "compare_agg", with_backward=False, restrict_targets_to=None)

        print(f"\n=== {cond_letter}{gc} (max D) ===")
        for how in HOWS:
            v, r_ = aggregate_state(flux_wide, rev_wide, n_draws, how)
            feas = feasibility(mfa_model, *as_state_frames(v, r_, n_draws), rev_predicted)
            passed = report_feasibility(feas, f"[{how}]")

            pred = {"value": v.to_frame(name=gc), "lo": None, "hi": None}
            joined = pair_fluxes(pred, {"value": ref_value, "lo": None, "hi": None},
                                 condition=cond_letter, orientation=orientation,
                                 reactions=mfa_rxns, growth_rates=growth_rates(ref_value),
                                 min_abs=FLUX_MIN_ABS)
            per_how_frames[how].append(joined)
            r2_val = r2_linear(joined["pred"], joined["ref"]) if len(joined) else float("nan")

            print(f"  {how:14s} R2_vs_reference = {r2_val:+.4f}  (n={len(joined)})  "
                  f"feasible={passed}   max|Sv|={feas['max_Sv'].max():.3g}   "
                  f"worst_bound={feas['worst_bound'].max():.3g}")
            rows.append(dict(Condition=cond_letter, GrowthCondition=gc, how=how,
                             r2_vs_reference=r2_val, n_reactions=len(joined),
                             feasible=passed, max_Sv=feas["max_Sv"].max(),
                             worst_bound=feas["worst_bound"].max()))

    print("\n=== Pooled across C/N/P (max D), matching experimental_flux_summary.csv's scope ===")
    for how in HOWS:
        pooled = pd.concat(per_how_frames[how], ignore_index=True) if per_how_frames[how] else pd.DataFrame()
        r2_pooled = r2_linear(pooled["pred"], pooled["ref"]) if len(pooled) else float("nan")
        print(f"  {how:14s} pooled R2 = {r2_pooled:+.4f}  (n={len(pooled)})")
        rows.append(dict(Condition="pooled", GrowthCondition=-1, how=how,
                         r2_vs_reference=r2_pooled, n_reactions=len(pooled),
                         feasible=None, max_Sv=None, worst_bound=None))

    out = pd.DataFrame(rows)
    out_path = RUN_DIR / "aggregation_comparison_s_cerevisiae.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
