"""
Stage 2 experiment: does a median-seeking aggregate beat the ensemble mean for SC?

Compares four ways to collapse the SC ensemble into one flux state --
mean, median, medoid, medoid_convex (see sigmaflux.ai.uncertainty._medoid_weights) --
on real SC data, reporting both feasibility and R^2 against the published/x3cflux
reference flux at the max-dilution-rate growth condition per nutrient condition,
matching PREDICTION_SPECS["s_cerevisiae"].

No retrain, no SLURM: a handful of EMU forward passes on the existing checkpoint.

Usage:
    python paper/analysis/diagnostics/compare_aggregation_sc.py
"""
import glob
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

from validate_inferred_fluxes import (  # noqa: E402
    aggregate_state, feasibility, report_feasibility, split_predictions,
    as_state_frames,
)

RUN_DIR = _REPO_ROOT / "paper/.results/s_cerevisiae_20260831"
CONFIG = _REPO_ROOT / "paper/configs/s_cerevisiae.yaml"
DATA_DIR = _REPO_ROOT / "data/S_cervesiae"
HOWS = ["mean", "median", "medoid", "medoid_convex"]


def r2(pred: pd.Series, ref: pd.Series) -> float:
    both = pd.concat([pred, ref], axis=1, keys=["pred", "ref"]).dropna()
    if len(both) < 2:
        return float("nan")
    ss_res = ((both["pred"] - both["ref"]) ** 2).sum()
    ss_tot = ((both["ref"] - both["ref"].mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def main():
    cfg, root = load_config(str(CONFIG), None)
    mapping_csv = resolve_path(mapping_csv_for(cfg), root)
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None

    print(f"Loading ensemble from {RUN_DIR/'ensemble'} ...")
    sf = load_predictor(str(CONFIG), str(RUN_DIR / "ensemble"), device="cpu")

    ext_file = _find_ext_flux_file(str(DATA_DIR))
    conditions = _discover_conditions(str(DATA_DIR))

    ref_path = _REPO_ROOT / "data/S_cervesiae/Shen_et_al_flux_data_SC.xlsx"

    rows = []
    mfa_model = reaction_ids = labeling_to = labeling_index = None
    cobra_model = None
    from cobra.io.json import load_json_model
    cobra_model = load_json_model(str(RUN_DIR / "model_prepared_for_mfa.json"))

    for cond_letter in ["C", "N", "P"]:
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

        ref_series = ref_value[gc].dropna()

        print(f"\n=== {cond_letter}{gc} (max D) ===")
        for how in HOWS:
            v, r_ = aggregate_state(flux_wide, rev_wide, n_draws, how)
            feas = feasibility(mfa_model, *as_state_frames(v, r_, n_draws), rev_predicted)
            passed = report_feasibility(feas, f"[{how}]")
            r2_val = r2(v, ref_series)
            print(f"  {how:14s} R2_vs_reference = {r2_val:+.4f}   feasible={passed}   "
                  f"max|Sv|={feas['max_Sv'].max():.3g}   worst_bound={feas['worst_bound'].max():.3g}")
            rows.append(dict(Condition=cond_letter, GrowthCondition=gc, how=how,
                             r2_vs_reference=r2_val, feasible=passed,
                             max_Sv=feas["max_Sv"].max(), worst_bound=feas["worst_bound"].max()))

    out = pd.DataFrame(rows)
    out_path = RUN_DIR / "aggregation_comparison_s_cerevisiae.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
