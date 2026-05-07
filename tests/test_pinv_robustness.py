"""Tests for pseudo-inverse robustness in MFAModel.forward().

These tests are written TDD-style: both should FAIL before the production
fixes are applied to model.py, and PASS after.
"""
import os
import pytest
import torch
from unittest.mock import patch

import pandas as pd
from cobra.io.json import load_json_model
from sigmaflux.core.model import MFAModel
from sigmaflux.core.experiment import Tracer, Experiment
from sigmaflux.core.data import MassIsotopomerData, FluxData

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
YEAST_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "I_orientalis", "imIsor_annotated.json")


@pytest.fixture(scope="module")
def yeast_mfa_model():
    """Prepared MFAModel using the I. orientalis yeast model and yeast test dataset."""
    model = load_json_model(YEAST_MODEL_PATH)
    # Maranas-format models have _out metabolites not mass-balanced in COBRA,
    # so use uniform unit fluxes as initial distribution.
    initial_fluxes = pd.Series(1.0, index=[r.id for r in model.reactions])
    mfa_model = MFAModel(model, "IO_robustness", initial_flux_distribution=initial_fluxes)

    prefix = "yeast_test"
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "test", f"{prefix}_labeling.csv"))
    data_df = df.groupby(["id", "mid"]).mean()[["enrichment"]]
    data_df["std"] = df.groupby(["id", "mid"]).std()["enrichment"]
    data_df.columns = ["mean", "std"]
    data = MassIsotopomerData(mfa_model, data_df.reset_index())

    flux_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "test", f"{prefix}_fluxes.csv"))
    flux_agg = flux_df.groupby(["id"]).mean()[["flux"]]
    flux_agg["std"] = flux_df.groupby(["id"]).std()["flux"]
    flux_agg.columns = ["mean", "std"]
    flux_agg = flux_agg.reset_index().rename(columns={"id": "reaction"})
    reaction_ids = {r.id for r in model.reactions}
    flux_data = FluxData(mfa_model, flux_agg[flux_agg["reaction"].isin(reaction_ids)].reset_index(drop=True))

    tracer = Tracer(mfa_model.metabolites.glc__D_e, 0.1, [1, 1, 1, 1, 1, 1])
    experiment = Experiment(data, [tracer], fluxes=flux_data)
    mfa_model.prepare([experiment])
    return mfa_model


def test_forward_raises_on_degenerate_matrix(yeast_mfa_model):
    """forward() must raise RuntimeError when pinv fails (degenerate flux sample)."""
    with patch("torch.linalg.pinv", side_effect=torch.linalg.LinAlgError("injected")):
        with pytest.raises(RuntimeError, match="Degenerate A matrix"):
            yeast_mfa_model()


def test_forward_raises_clearly_on_nan_flux(yeast_mfa_model):
    """forward() must raise RuntimeError with a clear message when fluxes contain NaN."""
    original = yeast_mfa_model.fluxes.data.clone()
    yeast_mfa_model.fluxes.data[:] = float("nan")
    try:
        with pytest.raises(RuntimeError, match="Non-finite flux values"):
            yeast_mfa_model()
    finally:
        yeast_mfa_model.fluxes.data[:] = original
