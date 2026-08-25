"""Tests for the sign-bounded Beta resampler used at inference (run_inference_workflow.py).

An exchange flux is either uptake or secretion in a given condition, never both, so
resampling it for the inference ensemble must not put mass on the wrong side of zero --
see _resample_flux's docstring.
"""

import numpy as np
import pandas as pd
import pytest

from sigmaflux.pipeline.run_inference_workflow import _expand_samples, _resample_flux


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestResampleFlux:
    """Direct tests of the per-flux Beta draw."""

    def test_positive_value_stays_positive(self, rng):
        draws = _resample_flux(8.576, 0.012, 5000, rng)
        assert np.all(draws > 0)

    def test_negative_value_stays_negative(self, rng):
        draws = _resample_flux(-11.378, 3.353, 5000, rng)
        assert np.all(draws < 0)

    def test_zero_value_stays_nonnegative(self, rng):
        """The case that motivated the change: a flux measured as exactly zero (e.g.
        S. cerevisiae EX_glyc_e under carbon limitation) must not resample as an uptake."""
        draws = _resample_flux(0.0, 0.011, 5000, rng)
        assert np.all(draws >= 0)

    def test_draws_bounded_by_signed_upper_limit(self, rng):
        """Every draw lies within [0, M] scaled by the measured sign, M = the truncated
        law's 97.5th percentile."""
        from scipy.stats import truncnorm
        value, err = -0.130229, 0.3866
        a = -abs(value) / err
        M = truncnorm(a, np.inf, loc=abs(value), scale=err).ppf(0.975)
        draws = _resample_flux(value, err, 5000, rng)
        assert np.all(draws <= 0)
        assert np.all(draws >= -M)

    def test_well_measured_flux_recovers_value_and_error(self, rng):
        """|value| / err = 715 for this real E. coli glucose-uptake measurement -- far
        into the regime where the truncation at zero is irrelevant and the Beta draw
        should reproduce a plain Normal(value, err)."""
        value, err = -8.576, 0.012
        draws = _resample_flux(value, err, 20000, rng)
        assert draws.mean() == pytest.approx(value, abs=0.01 * abs(value))
        assert draws.std() == pytest.approx(err, rel=0.05)

    @pytest.mark.parametrize("ratio", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    def test_alpha_beta_always_positive(self, ratio, rng):
        """kappa = x(1-x)/v - 1 is bounded below by ~2.17 at ratio == 0 and grows from
        there, so alpha and beta never go non-positive -- there is no infeasible branch
        to fall back on across the range of |value|/err actually seen in the data."""
        value, err = ratio, 1.0
        draws = _resample_flux(value, err, 10, rng)
        assert np.all(np.isfinite(draws))


class TestExpandSamples:
    """Integration through _expand_samples: sign preservation and fixed-flux passthrough."""

    @pytest.fixture
    def frames(self):
        df_L = pd.DataFrame({"Sample": [1, 2], "dummy": ["a", "b"]})
        df_Vm = pd.DataFrame({
            "Sample":    [1, 1, 2, 2],
            "Flux Name": ["EX_glc__D_e", "EX_glyc_e", "EX_glc__D_e", "EX_glyc_e"],
            "Value":     [-10.0, 0.0, -12.0, 0.0],
        })
        df_err = pd.DataFrame({
            "Sample":    [1, 1, 2, 2],
            "Flux Name": ["EX_glc__D_e", "EX_glyc_e", "EX_glc__D_e", "EX_glyc_e"],
            "Value":     [0.5, 0.02, 0.6, 0.0],   # EX_glyc_e err==0 for rep 2 -> held fixed
        })
        return df_L, df_Vm, df_err

    def test_resampled_fluxes_keep_measured_sign(self, frames, rng):
        df_L, df_Vm, df_err = frames
        n = 200
        _, df_Vm_exp = _expand_samples(df_L, df_Vm, df_err, n_samples=n, rng=rng)

        glc = df_Vm_exp[df_Vm_exp["Flux Name"] == "EX_glc__D_e"]["Value"]
        glyc_rep1 = df_Vm_exp[
            (df_Vm_exp["Sample"] < n) & (df_Vm_exp["Flux Name"] == "EX_glyc_e")
        ]["Value"]
        assert (glc < 0).all()
        assert (glyc_rep1 >= 0).all()

    def test_zero_error_flux_is_held_fixed(self, frames, rng):
        df_L, df_Vm, df_err = frames
        n = 50
        _, df_Vm_exp = _expand_samples(df_L, df_Vm, df_err, n_samples=n, rng=rng)

        # Bio-rep 2 synthetic ids are n..2n-1; EX_glyc_e has err==0 there, so it must
        # stay at its measured value of 0.0 rather than being resampled.
        glyc_rep2 = df_Vm_exp[
            (df_Vm_exp["Sample"] >= n) & (df_Vm_exp["Flux Name"] == "EX_glyc_e")
        ]["Value"]
        assert (glyc_rep2 == 0.0).all()

    def test_synthetic_sample_ids_and_count(self, frames, rng):
        df_L, df_Vm, df_err = frames
        n = 10
        df_L_exp, df_Vm_exp = _expand_samples(df_L, df_Vm, df_err, n_samples=n, rng=rng)

        assert df_L_exp["Sample"].nunique() == 2 * n
        assert sorted(df_L_exp["Sample"].unique()) == list(range(2 * n))
        assert set(df_Vm_exp["Sample"].unique()) == set(df_L_exp["Sample"].unique())
