"""
End-to-end test of the traditional MFA pipeline using Shen et al. I. orientalis data.

Loads MID and external flux data from xlsx files via the utils package, bridges
through read_df_data, builds MFA pipeline objects, fits fluxes via gradient
descent, and saves predictions as CSV.

Run: pytest -s tests/test_core_mfa.py 
"""

import os
import sys

import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.nn import MSELoss
from cobra.io.json import load_json_model
from cobra.flux_analysis import pfba

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.core.model import MFAModel
from sigmaflux.core.experiment import Tracer, Experiment
from sigmaflux.core.data import MassIsotopomerData, FluxData
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.utils import read_xlsx_mid_data, read_xlsx_extflux
from sigmaflux.utils.read_df_data import (
    mid_df_to_mass_isotopomer_table,
    extflux_df_to_flux_table,
)


# ============================================================
# USER CONFIGURATION
# ============================================================

MODEL_JSON        = "models/I_orientalis/imIsor_annotated.json"
MID_FILE_TEMPLATE = "data/I_orientalis/I_orientalis_{condition}.xlsx"
EXT_FILE          = "data/I_orientalis/I_orientalis_ExtFlux.xlsx"
MAPPING_CSV       = "models/I_orientalis/I_orientalis_mapping.csv"

# Conditions and biological replicates to run.
# Each key is a condition label; values are lists of growth condition (GC) numbers.
CONDITIONS = {
    "C": [1],
    "N": [1],
}

# Extracellular entry point and tracer definitions
TRACER_MET_ID      = "glc__D_e"
TRACER_UC_FRACTION = 0.5               
TRACER_UC_PATTERN  = [1, 1, 1, 1, 1, 1]
TRACER_12_FRACTION = 0.5               
TRACER_12_PATTERN  = [1, 1, 0, 0, 0, 0]

LR                = 1e-4
EPOCHS            = 1000
LAMBDA0           = 1.0
FLUX_SCALER       = 200
MAX_SAMPLING_FLUX = 50.0   # mmol/gDW/h hard cap on all reaction bounds during sampling
RANDOM_SEED       = 256      # seed for COBRA sampler; set None to disable

OUT_DIR = "SigmaFlux_output/mfa_traditional"

# ============================================================
# END USER CONFIGURATION
# ============================================================


def run_mfa_pipeline(condition, gc):
    def _p(rel):
        return os.path.join(_REPO_ROOT, rel)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 1. Load model ----
    cobra_model = load_json_model(_p(MODEL_JSON))

    # ---- 2. Load raw staging DataFrames ----
    df_L = read_xlsx_mid_data(
        xlsx_path        = _p(MID_FILE_TEMPLATE.format(condition=condition)),
        mapping_csv      = _p(MAPPING_CSV),
        growth_condition = gc,
    )
    df_Vm, df_err = read_xlsx_extflux(
        xlsx_path        = _p(EXT_FILE),
        condition        = condition,
        growth_condition = gc,
        return_errors    = True,
    )

    # ---- 3. Convert to pipeline-ready tables ----
    mid_table  = mid_df_to_mass_isotopomer_table(df_L)
    flux_table = extflux_df_to_flux_table(df_Vm, df_err)

    # Keep only reactions that exist in the COBRA model
    model_rxn_ids = {r.id for r in cobra_model.reactions}
    flux_table    = flux_table[flux_table['reaction'].isin(model_rxn_ids)].reset_index(drop=True)

    # ---- 3b. Constrain cobra bounds for measured reactions before sampling ----
    original_bounds = {}
    for _, row in flux_table.iterrows():
        rxn = cobra_model.reactions.get_by_id(row['reaction'])
        original_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
        mean, std = float(row['mean']), float(row['std'])
        cobra_mean = -mean if rxn.upper_bound <= 0 else mean
        lb = max(cobra_mean - 2 * std, rxn.lower_bound)
        ub = min(cobra_mean + 2 * std, rxn.upper_bound)
        if lb < ub:
            rxn.lower_bound, rxn.upper_bound = lb, ub

    # ---- 3b-i. Global sampling cap for all remaining reactions ----
    for rxn in cobra_model.reactions:
        if rxn.id not in original_bounds:
            original_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
        rxn.lower_bound = max(rxn.lower_bound, -MAX_SAMPLING_FLUX)
        rxn.upper_bound = min(rxn.upper_bound,  MAX_SAMPLING_FLUX)

    # ---- 3c. Create MFA model (sampling uses constrained bounds) ----
    mfa_model = MFAModel(cobra_model, "IO", lambda0=LAMBDA0, seed=RANDOM_SEED)

    # Restore original cobra bounds so the MFA optimizer is not artificially limited
    for rxn_id, (lb, ub) in original_bounds.items():
        rxn = cobra_model.reactions.get_by_id(rxn_id)
        rxn.lower_bound, rxn.upper_bound = lb, ub

    # ---- 3d. Convert flux_table means to COBRA sign convention ----
    for idx, row in flux_table.iterrows():
        rxn = cobra_model.reactions.get_by_id(row['reaction'])
        if rxn.upper_bound <= 0:
            flux_table.at[idx, 'mean'] = -row['mean']

    # ---- 3e. Save sampled initialization to csv ----
    rxn_ids   = [r.id for r in mfa_model.reactions]
    init_vals = mfa_model.initial_flux_values.cpu().numpy()
    init_df   = pd.DataFrame({'Reaction': rxn_ids, 'Initial Flux': init_vals})
    os.makedirs(_p(OUT_DIR), exist_ok=True)
    init_path = os.path.join(_p(OUT_DIR), f"init_fluxes_{condition}{gc}_{timestamp}.csv")
    init_df.to_csv(init_path, index=False)
    print(f"Sampled initialization saved -> {init_path}")

    glc_ix = mfa_model._flux_hash.get("EX_glc__D_e")
    bio_ix = mfa_model._flux_hash.get("BIOMASS")
    print(f"Init   EX_glc__D_e: {init_vals[glc_ix]:.4f}  |  BIOMASS: {init_vals[bio_ix]:.4f}")

    # ---- 4. Build MFA pipeline objects ----
    mid_data  = MassIsotopomerData(model=mfa_model, table=mid_table)
    flux_data = FluxData(mfa_model, flux_table)
    tracer_uc = Tracer(mfa_model.metabolites.get_by_id(TRACER_MET_ID), TRACER_UC_FRACTION, TRACER_UC_PATTERN)
    tracer_12 = Tracer(mfa_model.metabolites.get_by_id(TRACER_MET_ID), TRACER_12_FRACTION, TRACER_12_PATTERN)
    experiment = Experiment(mid_data, [tracer_uc, tracer_12], fluxes=flux_data)
    mfa_model.prepare([experiment])

    # ---- 5. Fit ----
    optimizer = torch.optim.Adam(mfa_model.parameters(), lr=LR)
    loss_log  = fit_mfa_model(
        mfa_model, MSELoss(), MSELoss(), optimizer, epochs=EPOCHS,
        flux_scaler=FLUX_SCALER,
    )

    # ---- 6. Extract and save flux predictions ----
    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_values  = mfa_model.fluxes.detach().cpu().numpy()
    predictions  = pd.DataFrame({'Flux Name': reaction_ids, 'Value': flux_values})

    os.makedirs(_p(OUT_DIR), exist_ok=True)
    out_path = os.path.join(_p(OUT_DIR), f"predictions_{condition}{gc}_{timestamp}.csv")
    predictions.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(predictions)} rows)")
    print(f"Starting loss: {loss_log[0]:.5f}  |  Final loss: {loss_log[-1]:.5f}")

    # ---- diagnostic: mass balance + key fluxes ----
    residual  = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().abs().max().item()
    flux_vals = mfa_model.fluxes.detach().cpu().numpy()
    glc_measured = flux_table[flux_table['reaction'] == 'EX_glc__D_e']['mean'].values
    bio_measured = flux_table[flux_table['reaction'] == 'BIOMASS']['mean'].values
    print(f"Max mass balance residual: {residual:.4f}")
    print(f"Predicted EX_glc__D_e: {flux_vals[glc_ix]:.4f}  (measured COBRA: {glc_measured[0]:.4f})" if len(glc_measured) else f"Predicted EX_glc__D_e: {flux_vals[glc_ix]:.4f}")
    print(f"Predicted BIOMASS:     {flux_vals[bio_ix]:.4f}  (measured: {bio_measured[0]:.4f})" if len(bio_measured) else f"Predicted BIOMASS: {flux_vals[bio_ix]:.4f}")

    return predictions, init_df, loss_log


def run_all_conditions():
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_pred = {}
    results_init = {}

    def _p(rel):
        return os.path.join(_REPO_ROOT, rel)

    for condition, gc_list in CONDITIONS.items():
        pred_wide = None
        init_wide = None
        for gc in gc_list:
            predictions, init_df, _ = run_mfa_pipeline(condition, gc)
            col = f"{condition}{gc}"
            if pred_wide is None:
                pred_wide = predictions.rename(columns={"Value": col}).set_index("Flux Name")
                init_wide = init_df.rename(columns={"Initial Flux": col}).set_index("Reaction")
            else:
                pred_wide[col] = predictions.set_index("Flux Name")["Value"]
                init_wide[col] = init_df.set_index("Reaction")["Initial Flux"]
        results_pred[condition] = pred_wide.reset_index()
        results_init[condition] = init_wide.reset_index()

    os.makedirs(_p(OUT_DIR), exist_ok=True)
    pred_path = os.path.join(_p(OUT_DIR), f"predictions_{timestamp}.xlsx")
    init_path = os.path.join(_p(OUT_DIR), f"init_fluxes_{timestamp}.xlsx")

    with pd.ExcelWriter(pred_path, engine="openpyxl") as writer:
        for condition, df in results_pred.items():
            df.to_excel(writer, sheet_name=condition, index=False)
    with pd.ExcelWriter(init_path, engine="openpyxl") as writer:
        for condition, df in results_init.items():
            df.to_excel(writer, sheet_name=condition, index=False)

    print(f"Predictions  -> {pred_path}")
    print(f"Init fluxes  -> {init_path}")
    return results_pred, results_init


def test_core_mfa():
    results_pred, _ = run_all_conditions()
    for _, df in results_pred.items():
        assert len(df) > 0
        assert "Flux Name" in df.columns
