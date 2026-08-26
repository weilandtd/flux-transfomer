"""Tests for the sign-bounded Beta resampler used at inference (sigmaflux.ai.uncertainty).

An exchange flux is either uptake or secretion in a given condition, never both, so
resampling it for the inference ensemble must not put mass on the wrong side of zero --
see resample_flux's docstring.
"""

import numpy as np
import pandas as pd
import pytest

from sigmaflux.ai.uncertainty import resample_flux, sample_inputs

_resample_flux = resample_flux


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


class TestSampleInputs:
    """Integration through sample_inputs: replicate pass-through, sign preservation,
    and fixed-flux passthrough. Three replicates, matching the real bio-rep count."""

    @pytest.fixture
    def frames(self):
        df_L = pd.DataFrame({
            "Sample":        [1, 1, 2, 2, 3, 3],
            "Tracer Name":   ["T"] * 6,
            "Metabolite ID": ["m"] * 6,
            "Isotopomer":    ["M+1", "M+2"] * 3,
            "Fraction":      [0.6, 0.4, 0.7, 0.3, 0.5, 0.5],
        })
        df_Vm = pd.DataFrame({
            "Sample":    [1, 1],
            "Flux Name": ["EX_glc__D_e", "EX_glyc_e"],
            "Value":     [-10.0, 0.0],
        })
        df_err = pd.DataFrame({
            "Sample":    [1, 1],
            "Flux Name": ["EX_glc__D_e", "EX_glyc_e"],
            "Value":     [0.5, 0.02],
        })
        return df_L, df_Vm, df_err

    def test_resampled_fluxes_keep_measured_sign(self, frames, rng):
        df_L, df_Vm, df_err = frames
        _, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng)

        glc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glc__D_e"]["Value"]
        glyc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glyc_e"]["Value"]
        assert (glc < 0).all()
        assert (glyc >= 0).all()

    def test_zero_error_flux_is_held_fixed(self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_err = df_err.assign(Value=[0.5, 0.0])   # EX_glyc_e now has no usable error
        _, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng)

        glyc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glyc_e"]["Value"]
        assert (glyc == 0.0).all()

    def test_replicate_rows_are_passed_through_unperturbed(self, frames, rng):
        """The MID side has no synthetic noise model any more -- each of the 3
        replicate rows must appear in df_L_draws exactly as measured."""
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng)

        for s, rep in enumerate(sorted(df_L["Sample"].unique())):
            got = (df_L_draws[df_L_draws["Sample"] == s]
                   .set_index("Isotopomer")["Fraction"])
            want = (df_L[df_L["Sample"] == rep]
                   .set_index("Isotopomer")["Fraction"])
            assert got["M+1"] == pytest.approx(want["M+1"])
            assert got["M+2"] == pytest.approx(want["M+2"])

    def test_last_sample_is_the_unperturbed_maximum_likelihood_input(self, frames, rng):
        """Sample n_reps carries the measured fluxes and the replicate-mean MIDs as-is.

        That row is what the reported point estimate is evaluated at, so any perturbation
        leaking into it would silently bias every prediction.
        """
        df_L, df_Vm, df_err = frames
        n_reps = df_L["Sample"].nunique()
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng)

        mle_flux = df_Vm_draws[df_Vm_draws["Sample"] == n_reps].set_index("Flux Name")["Value"]
        assert mle_flux["EX_glc__D_e"] == -10.0
        assert mle_flux["EX_glyc_e"] == 0.0

        mle_mid = df_L_draws[df_L_draws["Sample"] == n_reps].set_index("Isotopomer")["Fraction"]
        assert mle_mid["M+1"] == pytest.approx((0.6 + 0.7 + 0.5) / 3)
        assert mle_mid["M+2"] == pytest.approx((0.4 + 0.3 + 0.5) / 3)

    def test_sample_ids_cover_every_replicate_plus_the_mle_row(self, frames, rng):
        df_L, df_Vm, df_err = frames
        n_reps = df_L["Sample"].nunique()
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng)

        assert sorted(df_L_draws["Sample"].unique()) == list(range(n_reps + 1))
        assert set(df_Vm_draws["Sample"].unique()) == set(df_L_draws["Sample"].unique())
