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


#: Small draw count so these single-tracer cases stay readable; the draw count itself is
#: covered by TestBootstrapReplicates.
N_DRAWS = 3


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
        _, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=N_DRAWS)

        glc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glc__D_e"]["Value"]
        glyc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glyc_e"]["Value"]
        assert (glc < 0).all()
        assert (glyc >= 0).all()

    def test_zero_error_flux_is_held_fixed(self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_err = df_err.assign(Value=[0.5, 0.0])   # EX_glyc_e now has no usable error
        _, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=N_DRAWS)

        glyc = df_Vm_draws[df_Vm_draws["Flux Name"] == "EX_glyc_e"]["Value"]
        assert (glyc == 0.0).all()

    def test_drawn_rows_are_measured_replicates_passed_through_unperturbed(self, frames, rng):
        """The MID side has no synthetic noise model: every drawn row must be one of the
        measured replicates verbatim, not an interpolation or a perturbation of them."""
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=50)

        measured = {(r["M+1"], r["M+2"]) for _, r in
                    df_L.pivot_table(index="Sample", columns="Isotopomer",
                                     values="Fraction").iterrows()}
        drawn = df_L_draws[df_L_draws["Sample"] < 50].pivot_table(
            index="Sample", columns="Isotopomer", values="Fraction")
        for _, row in drawn.iterrows():
            assert (row["M+1"], row["M+2"]) in measured

    def test_last_sample_is_the_unperturbed_maximum_likelihood_input(self, frames, rng):
        """Sample n_bootstrap carries the measured fluxes and replicate-mean MIDs as-is.

        That row is what the reported point estimate is evaluated at, so any perturbation
        leaking into it would silently bias every prediction.
        """
        df_L, df_Vm, df_err = frames
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=N_DRAWS)

        mle_flux = df_Vm_draws[df_Vm_draws["Sample"] == N_DRAWS].set_index("Flux Name")["Value"]
        assert mle_flux["EX_glc__D_e"] == -10.0
        assert mle_flux["EX_glyc_e"] == 0.0

        mle_mid = df_L_draws[df_L_draws["Sample"] == N_DRAWS].set_index("Isotopomer")["Fraction"]
        assert mle_mid["M+1"] == pytest.approx((0.6 + 0.7 + 0.5) / 3)
        assert mle_mid["M+2"] == pytest.approx((0.4 + 0.3 + 0.5) / 3)

    def test_sample_ids_cover_every_draw_plus_the_mle_row(self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=N_DRAWS)

        assert sorted(df_L_draws["Sample"].unique()) == list(range(N_DRAWS + 1))
        assert set(df_Vm_draws["Sample"].unique()) == set(df_L_draws["Sample"].unique())


class TestBootstrapReplicates:
    """sample_inputs' n_bootstrap: cross-tracer replicate combinations.

    Two tracers are two cultures measured on two sheets, so replicate 1 of one has nothing
    to do with replicate 1 of the other. Values here encode which replicate each row came
    from (tracer A: r/10, tracer B: r/100), so a draw can be decoded back to the (rep_A,
    rep_B) pair it used. Index-pairing is not an option any more -- it is a mistake, not a
    trade-off -- so there is no "off" path left to compare against.
    """

    @pytest.fixture
    def frames(self):
        rows = []
        for tracer, scale in (("A", 10.0), ("B", 100.0)):
            n_reps = 5 if tracer == "A" else 3      # deliberately unequal
            for rep in range(1, n_reps + 1):
                rows.append(dict(Sample=rep, **{"Tracer Name": tracer},
                                 **{"Metabolite ID": "m"}, Isotopomer="M+1",
                                 Fraction=rep / scale))
        df_L = pd.DataFrame(rows)
        df_Vm = pd.DataFrame({"Sample": [1], "Flux Name": ["EX_glc__D_e"], "Value": [-10.0]})
        df_err = pd.DataFrame({"Sample": [1], "Flux Name": ["EX_glc__D_e"], "Value": [0.5]})
        return df_L, df_Vm, df_err

    def _pairs(self, df_L_draws, n_bootstrap):
        """Decode each synthetic replicate back to the (rep_A, rep_B) it drew."""
        drawn = df_L_draws[df_L_draws["Sample"] < n_bootstrap]
        wide = drawn.pivot_table(index="Sample", columns="Tracer Name", values="Fraction")
        return list(zip(np.round(wide["A"] * 10).astype(int),
                        np.round(wide["B"] * 100).astype(int)))

    def test_a_nonpositive_draw_count_is_rejected(self, frames, rng):
        """Zero draws would leave only the MLE row, so every sigma would come out 0.0 and
        the reported interval would collapse to a point without anything failing."""
        df_L, df_Vm, df_err = frames
        with pytest.raises(ValueError, match="at least 1"):
            sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=0)

    def test_sample_ids_are_the_draws_plus_the_mle_row(self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=20)

        assert sorted(df_L_draws["Sample"].unique()) == list(range(21))
        assert set(df_Vm_draws["Sample"].unique()) == set(df_L_draws["Sample"].unique())

    def test_every_draw_holds_exactly_one_replicate_per_tracer(self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=30)

        drawn = df_L_draws[df_L_draws["Sample"] < 30]
        counts = drawn.groupby(["Sample", "Tracer Name"]).size()
        assert set(counts.unique()) == {1}                       # one token per tracer here
        assert len(counts) == 30 * 2

    def test_pairings_are_not_locked_to_a_shared_replicate_index(self, frames, rng):
        """The point of the change: index-pairing reaches 3 of the 5x3 combinations."""
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=400)

        pairs = set(self._pairs(df_L_draws, 400))
        assert pairs == {(a, b) for a in range(1, 6) for b in range(1, 4)}
        assert len(pairs) == 15

    def test_each_tracer_draws_uniformly_over_its_own_replicates(self, frames, rng):
        """Unequal replicate counts must not leak: A has 5 replicates, B has 3."""
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=2000)

        a_reps, b_reps = zip(*self._pairs(df_L_draws, 2000))
        a_freq = np.bincount(a_reps, minlength=6)[1:] / 2000
        b_freq = np.bincount(b_reps, minlength=4)[1:] / 2000
        assert np.allclose(a_freq, 1 / 5, atol=0.03)
        assert np.allclose(b_freq, 1 / 3, atol=0.03)

    def test_mle_row_is_the_replicate_mean_per_tracer_regardless_of_bootstrapping(
            self, frames, rng):
        df_L, df_Vm, df_err = frames
        df_L_draws, _ = sample_inputs(df_L, df_Vm, df_err, rng, n_bootstrap=50)

        mle = df_L_draws[df_L_draws["Sample"] == 50].set_index("Tracer Name")["Fraction"]
        assert mle["A"] == pytest.approx(np.mean([1, 2, 3, 4, 5]) / 10)
        assert mle["B"] == pytest.approx(np.mean([1, 2, 3]) / 100)
