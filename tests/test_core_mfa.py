"""
Lightweight smoke test for the traditional MFA pipeline.

Runs 10 epochs on the toy one-carbon model using the synthetic test fixtures in
data/test/. The full real-data benchmark lives in paper/pipeline/run_mfa_data.py
and is submitted to the cluster via paper/submit/submit_mfa_data_io.sh.
"""

import os
import sys

import pandas as pd
import torch
from cobra.io.json import load_json_model
from torch.nn import MSELoss

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_TEST_DIR, os.pardir))

TOY_MODEL = os.path.join(_PROJECT_DIR, "models", "toy", "one_carbon_annotated.json")
TOY_LABELING = os.path.join(_PROJECT_DIR, "data", "test", "toy_test_labeling.csv")
TOY_FLUXES = os.path.join(_PROJECT_DIR, "data", "test", "toy_test_fluxes.csv")


def _load_toy_data(mfa_model):
    df = pd.read_csv(TOY_LABELING)
    mid_table = (
        df.groupby(["id", "mid"])["enrichment"]
        .agg(mean="mean", std="std")
        .reset_index()
        .fillna(0.01)
    )

    df_flux = pd.read_csv(TOY_FLUXES)
    flux_table = (
        df_flux.groupby("id")["flux"]
        .agg(mean="mean", std="std")
        .reset_index()
        .rename(columns={"id": "reaction"})
        .fillna(0.01)
    )
    reaction_ids = {r.id for r in mfa_model.reactions}
    flux_table = flux_table[flux_table["reaction"].isin(reaction_ids)].reset_index(drop=True)
    return mid_table, flux_table


def test_core_mfa():
    cobra_model = load_json_model(TOY_MODEL)
    initial_fluxes = pd.Series(1.0, index=[r.id for r in cobra_model.reactions])
    mfa_model = MFAModel(cobra_model, "TOY", initial_flux_distribution=initial_fluxes)

    mid_table, flux_table = _load_toy_data(mfa_model)

    mid_data = MassIsotopomerData(model=mfa_model, table=mid_table)
    flux_data = FluxData(mfa_model, flux_table)
    tracer = Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1])
    experiment = Experiment(mid_data, [tracer], fluxes=flux_data)
    mfa_model.prepare([experiment])

    optimizer = torch.optim.Adam(mfa_model.parameters(), lr=1e-3)
    loss_log = fit_mfa_model(mfa_model, MSELoss(), MSELoss(), optimizer, epochs=10)

    assert len(loss_log) == 10
    assert loss_log[-1] < max(loss_log), "Loss never decreased from peak — pipeline broken"

    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_values = mfa_model.fluxes.detach().cpu().numpy()
    assert len(flux_values) == len(reaction_ids)
