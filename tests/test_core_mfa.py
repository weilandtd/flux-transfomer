"""
End-to-end test of the traditional MFA pipeline using real I. orientalis data.

Loads MID and external flux data from xlsx files via the utils package, bridges
through read_df_data, builds MFA pipeline objects, fits fluxes via gradient
descent, and saves predictions as CSV.

Run: pytest SigmaFlux/tests/test_core_mfa.py
"""

import os
import sys

import pandas as pd
import torch
from torch.nn import MSELoss
from cobra.io.json import load_json_model

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

MODEL_JSON  = "models/I_orientalis/imIsor_annotated.json"
MID_FILE    = "data/I_orientalis/I_orientalis_C.xlsx"
EXT_FILE    = "data/I_orientalis/I_orientalis_ExtFlux.xlsx"
MAPPING_CSV = "models/I_orientalis/I_orientalis_mapping.csv"
CONDITION   = "C"
GC          = 1                     # growth condition number

# Tracer: [13C6]-glucose; glc__D_c is the intracellular entry point in the imIsor model
TRACER_MET_ID  = "glc__D_c"
TRACER_ENRICH  = 0.1                # fraction of labeled tracer in infusate
TRACER_PATTERN = [1, 1, 1, 1, 1, 1]

LR     = 1e-3
EPOCHS = 200

OUT_DIR = "SigmaFlux_output/mfa_traditional"

# ============================================================
# END USER CONFIGURATION
# ============================================================


def run_mfa_pipeline():
    def _p(rel):
        return os.path.join(_REPO_ROOT, rel)

    # ---- 1. Load model ----
    cobra_model    = load_json_model(_p(MODEL_JSON))
    initial_fluxes = pd.Series(1.0, index=[r.id for r in cobra_model.reactions])
    mfa_model      = MFAModel(cobra_model, "IO", initial_flux_distribution=initial_fluxes)

    # ---- 2. Load raw staging DataFrames ----
    df_L = read_xlsx_mid_data(
        xlsx_path        = _p(MID_FILE),
        mapping_csv      = _p(MAPPING_CSV),
        growth_condition = GC,
    )
    df_Vm, df_err = read_xlsx_extflux(
        xlsx_path        = _p(EXT_FILE),
        condition        = CONDITION,
        growth_condition = GC,
        return_errors    = True,
    )

    # ---- 3. Bridge to pipeline-ready tables ----
    mid_table  = mid_df_to_mass_isotopomer_table(df_L)
    flux_table = extflux_df_to_flux_table(df_Vm, df_err)

    # Keep only reactions that exist in the model 
    model_rxn_ids = {r.id for r in mfa_model.reactions}
    flux_table    = flux_table[flux_table['reaction'].isin(model_rxn_ids)].reset_index(drop=True)

    # ---- 4. Build MFA pipeline objects ----
    mid_data  = MassIsotopomerData(model=mfa_model, table=mid_table)
    flux_data = FluxData(mfa_model, flux_table)
    tracer    = Tracer(
        mfa_model.metabolites.get_by_id(TRACER_MET_ID),
        TRACER_ENRICH,
        TRACER_PATTERN,
    )
    experiment = Experiment(mid_data, [tracer], fluxes=flux_data)
    mfa_model.prepare([experiment])

    # ---- 5. Fit ----
    optimizer = torch.optim.Adam(mfa_model.parameters(), lr=LR)
    loss_log  = fit_mfa_model(
        mfa_model, MSELoss(), MSELoss(), optimizer, epochs=EPOCHS,
    )

    # ---- 6. Extract and save flux predictions ----
    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_values  = mfa_model.fluxes.detach().cpu().numpy()
    predictions  = pd.DataFrame({'Flux Name': reaction_ids, 'Value': flux_values})

    os.makedirs(_p(OUT_DIR), exist_ok=True)
    out_path = os.path.join(_p(OUT_DIR), f"predictions_{CONDITION}{GC}.csv")
    predictions.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(predictions)} rows)")
    print(f"Final loss: {loss_log[-1]:.5f}")

    return predictions, loss_log


def test_core_mfa():
    predictions, loss_log = run_mfa_pipeline()
    assert len(predictions) > 0
    assert set(predictions.columns) >= {'Flux Name', 'Value'}
    assert loss_log[-1] < loss_log[0], "Loss did not decrease — check tracer/data config"
