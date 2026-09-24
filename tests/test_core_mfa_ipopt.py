"""
Smoke test for the full-flux-space Ipopt MFA fitter.

Uses test_core_mfa.py's toy one-carbon model setup. Checks S·v=0 feasibility and
the FVA/reversibility box bounds on the fitted flux state.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from cobra.io.json import load_json_model

pytest.importorskip("cyipopt", reason="optional native-ipopt backend not installed")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.mfa_ipopt import fit_mfa_model_ipopt
from sigmaflux.core.model import MFAModel

from test_core_mfa import TOY_FLUXES, TOY_LABELING, TOY_MODEL, _load_toy_data


def _build_toy_model():
    cobra_model = load_json_model(TOY_MODEL)
    initial_fluxes = pd.Series(1.0, index=[r.id for r in cobra_model.reactions])
    mfa_model = MFAModel(cobra_model, "TOY", initial_flux_distribution=initial_fluxes)

    mid_table, flux_table = _load_toy_data(mfa_model)
    mid_data = MassIsotopomerData(model=mfa_model, table=mid_table)
    flux_data = FluxData(mfa_model, flux_table)
    tracer = Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1])
    experiment = Experiment(mid_data, [tracer], fluxes=flux_data)
    mfa_model.prepare([experiment], with_backward=True)
    return mfa_model


def test_core_mfa_ipopt():
    mfa_model = _build_toy_model()

    loss_log = fit_mfa_model_ipopt(mfa_model, max_iter=200, tol=1e-6)

    chi2, n_iter = loss_log["loss"], loss_log["n_iter"]
    assert np.isfinite(chi2), f"chi2 is not finite: {chi2}"
    assert n_iter > 0, "Optimizer did not run any iterations"

    v = mfa_model.fluxes.detach().cpu().numpy()
    r = mfa_model.reversibility.detach().cpu().numpy()

    sv = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().cpu().numpy()
    assert np.max(np.abs(sv)) < 1e-4, f"S·v not zero: max|S·v|={np.max(np.abs(sv)):.2e}"

    lb, ub = mfa_model.get_flux_bounds()
    lb, ub = lb.cpu().numpy(), ub.cpu().numpy()
    assert np.all(v >= lb - 1e-6), "flux below its lower bound"
    assert np.all(v <= ub + 1e-6), "flux above its upper bound"

    irrev = mfa_model._reversibility_mask.cpu().numpy() == 0
    assert np.allclose(r[irrev], 0.0, atol=1e-6)
    assert np.all(r[~irrev] >= mfa_model.min_rev_flux - 1e-6)


def test_native_and_ipopt_score_the_same_measurement_layout():
    """fit_mfa_model and fit_mfa_model_ipopt report the same n_labeling/n_fluxes
    from the same toy model and produce finite loss values."""
    model_lbfgsb = _build_toy_model()
    model_ipopt = _build_toy_model()

    res_lbfgsb = fit_mfa_model(model_lbfgsb, max_iter=200, tol=1e-6)
    res_ipopt = fit_mfa_model_ipopt(model_ipopt, max_iter=200, tol=1e-6)

    assert res_lbfgsb["n_labeling"] == res_ipopt["n_labeling"]
    assert res_lbfgsb["n_fluxes"] == res_ipopt["n_fluxes"]
    assert np.isfinite(res_lbfgsb["loss"]) and np.isfinite(res_ipopt["loss"])
