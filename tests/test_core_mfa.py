"""
Smoke test for the constrained MFA pipeline (null-space L-BFGS-B).

Runs a short optimization on the toy one-carbon model using synthetic test
fixtures in data/test/.  The full benchmark lives in
paper/pipeline/run_mfa_benchmark.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from cobra.io.json import load_json_model

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
    mfa_model.prepare([experiment], with_backward=True)

    loss_log = fit_mfa_model(mfa_model, max_iter=50, tol=1e-3)

    chi2, converged, n_iter = loss_log
    assert np.isfinite(chi2), f"chi2 is not finite: {chi2}"
    assert n_iter > 0, "Optimizer did not run any iterations"

    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_values = mfa_model.fluxes.detach().cpu().numpy()
    assert len(flux_values) == len(reaction_ids)

    # Mass balance should be satisfied to numerical precision
    sv = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().cpu().numpy()
    assert np.max(np.abs(sv)) < 1e-6, f"S·v not zero: max|S·v|={np.max(np.abs(sv)):.2e}"
