"""sigmaflux-infer --fbapro: every member draw is projected, in the right shape."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sigmaflux.ai.inference import _fbapro_decoder
from sigmaflux.core.fbapro import fbapro
from sigmaflux.pipeline.run_inference_workflow import fbapro_draws

# One metabolite, A -> B -> C: steady state forces v1 = v2.
S = np.array([[1.0, -1.0, 0.0]])
FLUX_NAMES = ["v0", "v1", "v2"]


class StubEnsemble:
    flux_names = FLUX_NAMES

    def project_knockouts(self, blocks, knockout_rxn_ids):
        vfixed = np.full(len(FLUX_NAMES), np.nan)
        vfixed[[FLUX_NAMES.index(r) for r in knockout_rxn_ids]] = 0.0
        return fbapro(blocks.reshape(-1, 3), S=S, vfixed=vfixed).reshape(blocks.shape)


def _raw(n_members=2, n_samples=3, seed=0):
    """``predict_draws``-shaped frame: per member, per sample, fluxes then reversibility."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_members):
        for s in range(n_samples):
            for i, name in enumerate(FLUX_NAMES):
                rows.append({"Sample": s, "Member": m, "Flux Name": name,
                             "Value": rng.normal(), "Type": "flux"})
            rows.append({"Sample": s, "Member": m, "Flux Name": "v1",
                         "Value": rng.normal(), "Type": "reversibility"})
    return pd.DataFrame(rows)


def test_every_member_draw_is_steady_state_and_pinned():
    raw = _raw()
    out = fbapro_draws(StubEnsemble(), raw, 3, ["v2"])
    wide = out[out["Type"] == "flux"]["Value"].to_numpy().reshape(-1, 3)
    np.testing.assert_allclose(S @ wide.T, 0.0, atol=1e-12)
    np.testing.assert_allclose(wide[:, 2], 0.0, atol=1e-12)


def test_reversibilities_are_untouched():
    raw = _raw()
    out = fbapro_draws(StubEnsemble(), raw, 3, [])
    rev = raw["Type"] == "reversibility"
    np.testing.assert_array_equal(out.loc[rev, "Value"], raw.loc[rev, "Value"])


def test_a_single_model_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="ensemble"):
        fbapro_draws(StubEnsemble(), _raw().drop(columns="Member"), 3, [])


def _decoder():
    """In -> X, X -> v1 | v2, with the inflow pinned at 1 through ``flux_offset``."""
    return SimpleNamespace(Z_full=torch.tensor([[0.0], [-1.0], [1.0]]),
                           flux_offset=torch.tensor([1.0, 1.0, 0.0]))


def test_decoder_projection_keeps_the_offset():
    v = np.array([[1.0, 0.3, 0.7]])
    free = np.full(3, np.nan)
    np.testing.assert_allclose(_fbapro_decoder(_decoder(), v, free, None), v)


def test_decoder_knockout_reroutes_around_the_pinned_inflow():
    v = np.array([[1.0, 0.3, 0.7]])
    ko = np.array([np.nan, np.nan, 0.0])
    np.testing.assert_allclose(_fbapro_decoder(_decoder(), v, ko, None), [[1.0, 1.0, 0.0]],
                               atol=1e-12)
