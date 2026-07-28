"""
Integration tests for the AI inference pipeline using the toy one-carbon model.

The ``toy_run`` fixture (see conftest.py) builds a complete but minimal
FluxDecoder checkpoint (random weights, correct stoichiometric structure)
without running training. Tests verify output shape, mass-balance and
fixed-flux propagation.
"""

import numpy as np
import pandas as pd

from sigmaflux.ai.data import load_stoichiometric_matrix
from sigmaflux.ai.inference import SigmaFlux

from conftest import TOY_CONFIG_YAML as _TOY_CONFIG_YAML
from conftest import TOY_MODEL_JSON as _TOY_MODEL_JSON


# ── helpers ───────────────────────────────────────────────────────────────────

def make_toy_df_L(n_samples: int = 2) -> pd.DataFrame:
    """Long-format labeling DataFrame compatible with SigmaFlux.predict."""
    rng = np.random.default_rng(7)
    records = []
    for s in range(n_samples):
        # Serine: 3 carbons → M+1, M+2, M+3 (exclude M+0)
        ser_w = rng.dirichlet(np.ones(4))          # M+0..M+3
        for iso, frac in zip(['M+1', 'M+2', 'M+3'], ser_w[1:]):
            records.append({'Tracer Name': 'labeled-Ser', 'Sample': s,
                            'Metabolite ID': 'Ser', 'Isotopomer': iso,
                            'Fraction': float(frac)})
        # Glycine: 2 carbons → M+1, M+2 (exclude M+0)
        gly_w = rng.dirichlet(np.ones(3))          # M+0..M+2
        for iso, frac in zip(['M+1', 'M+2'], gly_w[1:]):
            records.append({'Tracer Name': 'labeled-Ser', 'Sample': s,
                            'Metabolite ID': 'Gly', 'Isotopomer': iso,
                            'Fraction': float(frac)})
    return pd.DataFrame(records)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_predict_output_shape(toy_run):
    """Predictions have the right number of rows and finite values."""
    sf     = SigmaFlux(_TOY_CONFIG_YAML, toy_run['checkpoint'])
    df_L   = make_toy_df_L(n_samples=2)
    result = sf.predict(df_L)

    _, cobra_model = load_stoichiometric_matrix(_TOY_MODEL_JSON)
    n_fluxes = len(cobra_model.reactions)   # 23

    assert isinstance(result, pd.DataFrame)
    assert {'Flux Name', 'Value'}.issubset(result.columns)
    # n_fluxes flux rows + n_fluxes reversibility rows, per sample × 2 samples
    assert len(result) == 2 * n_fluxes * 2
    assert result['Value'].notna().all()
    assert np.isfinite(result['Value'].values).all()


def test_mass_balance(toy_run):
    """Predicted fluxes satisfy S @ v ≈ 0 (guaranteed by NullSpaceDecoder)."""
    S, cobra_model = load_stoichiometric_matrix(_TOY_MODEL_JSON)
    flux_names = [r.id for r in cobra_model.reactions]
    n_fluxes   = len(flux_names)

    sf     = SigmaFlux(_TOY_CONFIG_YAML, toy_run['checkpoint'])
    result = sf.predict(make_toy_df_L(n_samples=1))   # single sample → no Sample col

    # First n_fluxes rows are the flux predictions; the rest are reversibilities
    flux_vals = result.iloc[:n_fluxes]['Value'].values   # (23,)
    sv        = S @ flux_vals
    assert np.abs(sv).max() < 1e-4, f"Max |Sv| = {np.abs(sv).max():.2e}"


def test_fixed_fluxes_affect_prediction(toy_run):
    """
    Providing fixed_fluxes changes predictions vs. providing none.

    RotationFusion zeroes the non-S y-components of y_norm (those not
    constrained by measured reactions), replacing y_raw's signal there.
    Predictions with df_Vm therefore differ from predictions without it.
    """
    sf   = SigmaFlux(_TOY_CONFIG_YAML, toy_run['checkpoint'])
    df_L = make_toy_df_L(n_samples=1)

    df_Vm = pd.DataFrame({
        'Sample':    [0, 0],
        'Flux Name': ['EX_Ser_in', 'EX_Gly_src'],
        'Value':     [1.0, 1.0],
    })

    res_no_vm   = sf.predict(df_L)['Value'].values
    res_with_vm = sf.predict(df_L, df_Vm=df_Vm)['Value'].values

    assert not np.allclose(res_no_vm, res_with_vm, atol=1e-6), (
        "Predictions are identical with and without measured fluxes — "
        "fixed_fluxes may not be reaching rotation_fusion."
    )
