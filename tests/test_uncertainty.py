"""Tests for sigmaflux.ai.uncertainty: the MID noise model, input draws and summarisation.

The flux-side resampler has its own file (test_flux_resampling.py); what is covered here is
the labeling side and the reduction of raw draws to a reported estimate.
"""

import numpy as np
import pandas as pd
import pytest

from sigmaflux.ai.uncertainty import (
    MidNoiseModel, coverage_by_reaction, fit_mid_noise_model, sample_inputs,
    summarize_draws)


def _replicate_frames(a, b, n_points=400, n_reps=3, n_frames=4, seed=0):
    """Synthetic replicate frames drawn from a known sigma^2 = a + b*p(1-p)."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_frames):
        p = rng.uniform(0.01, 0.99, n_points)
        sigma = np.sqrt(a + b * p * (1 - p))
        rows = []
        for rep in range(n_reps):
            rows.append(pd.DataFrame({
                "Sample":        rep + 1,
                "Tracer Name":   "T",
                "Metabolite ID": [f"m{i}" for i in range(n_points)],
                "Isotopomer":    "M+1",
                "Fraction":      p + sigma * rng.standard_normal(n_points),
            }))
        frames.append(pd.concat(rows, ignore_index=True))
    return frames


class TestMidNoiseModel:

    def test_sigma_follows_the_multinomial_shape(self):
        m = MidNoiseModel(a=1e-4, b=4e-3)
        assert m.sigma(0.0) == pytest.approx(0.01)
        assert m.sigma(1.0) == pytest.approx(0.01)
        # Widest in the middle, where p(1-p) peaks.
        assert m.sigma(0.5) > m.sigma(0.1) > m.sigma(0.0)

    def test_sigma_is_clipped_to_the_unit_interval(self):
        """Fractions can come back marginally outside [0, 1]; the variance must stay real."""
        m = MidNoiseModel(a=1e-4, b=4e-3)
        assert np.isfinite(m.sigma([-0.05, 1.05])).all()

    def test_fit_recovers_known_coefficients(self):
        a, b = 2e-5, 9e-4
        fitted = fit_mid_noise_model(_replicate_frames(a, b, n_points=600, n_frames=6))
        assert fitted.a == pytest.approx(a, abs=2e-5)
        assert fitted.b == pytest.approx(b, rel=0.35)

    def test_coefficients_are_non_negative(self):
        """A variance component cannot be negative, and the floor is what a near-zero
        fraction's sigma rests on -- a negative fit there would produce NaN sigmas."""
        fitted = fit_mid_noise_model(_replicate_frames(0.0, 1e-3))
        assert fitted.a >= 0.0
        assert fitted.b >= 0.0

    def test_needs_replicates(self):
        single = _replicate_frames(1e-5, 1e-3, n_points=10, n_reps=1, n_frames=1)
        with pytest.raises(ValueError, match="replicates"):
            fit_mid_noise_model(single)


class TestSampleInputsLabeling:

    @pytest.fixture
    def df_L(self):
        rng = np.random.default_rng(1)
        rows = []
        for rep in range(1, 4):
            for met in ("m1", "m2"):
                frac = rng.dirichlet(np.ones(4))
                rows.append(pd.DataFrame({
                    "Sample": rep, "Tracer Name": "T", "Metabolite ID": met,
                    "Isotopomer": [f"M+{i}" for i in range(4)], "Fraction": frac,
                }))
        return pd.concat(rows, ignore_index=True)

    @pytest.fixture
    def flux_frames(self):
        vm = pd.DataFrame({"Sample": [1], "Flux Name": ["EX_glc__D_e"], "Value": [-5.0]})
        err = pd.DataFrame({"Sample": [1], "Flux Name": ["EX_glc__D_e"], "Value": [0.2]})
        return vm, err

    def test_draws_stay_on_the_composition(self, df_L, flux_frames):
        """Each metabolite's isotopomers must still sum to 1 after perturbation.

        A diagonal draw breaks the constraint; renormalisation is what puts it back, and
        without it the model would be fed compositions that cannot exist.
        """
        vm, err = flux_frames
        draws, _ = sample_inputs(df_L, vm, err, MidNoiseModel(1e-4, 4e-3),
                                 n_draws=30, rng=np.random.default_rng(0))
        sums = draws.groupby(["Sample", "Tracer Name", "Metabolite ID"])["Fraction"].sum()
        assert np.allclose(sums.to_numpy(), 1.0)

    def test_draws_are_non_negative(self, df_L, flux_frames):
        vm, err = flux_frames
        draws, _ = sample_inputs(df_L, vm, err, MidNoiseModel(1e-3, 1e-2),
                                 n_draws=50, rng=np.random.default_rng(0))
        assert (draws["Fraction"] >= 0).all()

    def test_spread_scales_with_the_standard_error_not_the_replicate_sd(self, df_L,
                                                                        flux_frames):
        """The model is fed the replicate mean, so the input uncertainty is sigma/sqrt(n).

        Tripling the replicate count must shrink the draw spread, not leave it alone --
        this is the correction that separates an interval on the estimate from one on a
        single measurement.
        """
        vm, err = flux_frames
        noise = MidNoiseModel(1e-3, 0.0)      # constant sigma, so the check is clean
        wide = pd.concat([df_L.assign(Sample=r) for r in range(1, 4)], ignore_index=True)
        many = pd.concat([df_L.assign(Sample=r) for r in range(1, 28)], ignore_index=True)

        def spread(frame):
            d, _ = sample_inputs(frame, vm, err, noise, n_draws=400,
                                 rng=np.random.default_rng(0))
            d = d[d["Sample"] < 400]
            return d.groupby(["Metabolite ID", "Isotopomer"])["Fraction"].std().mean()

        # 9x the replicates -> 3x tighter, up to renormalisation and sampling noise.
        assert spread(many) == pytest.approx(spread(wide) / 3.0, rel=0.35)


class TestSummarizeDraws:

    def _raw(self, n_draws=40, n_members=5, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_draws + 1):
            for m in range(n_members):
                # Members disagree by a fixed offset; draws add their own scatter.
                base = 5.0 + 0.5 * m
                value = base if s == n_draws else base + rng.normal(0, 0.2)
                rows.append(dict(Sample=s, Member=m, Type="flux",
                                 **{"Flux Name": "R1", "Value": value}))
        return pd.DataFrame(rows)

    def test_point_estimate_is_the_ensemble_median_at_the_mle_input(self):
        out = summarize_draws(self._raw(), n_draws=40, group_cols=["Type", "Flux Name"])
        assert len(out) == 1
        # Members sit at 5.0, 5.5, 6.0, 6.5, 7.0 on the unperturbed row.
        assert out["Value"].iloc[0] == pytest.approx(6.0)

    def test_interval_brackets_the_point_estimate(self):
        out = summarize_draws(self._raw(), n_draws=40, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["Value_lo"] < row["Value"] < row["Value_hi"]

    def test_components_separate_member_spread_from_input_spread(self):
        """sigma_model is member disagreement at a fixed input; sigma_input is how far the
        consensus moves with the input. Here members are spread far wider than the draws,
        so the model term must dominate."""
        out = summarize_draws(self._raw(), n_draws=40, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["sigma_model"] > row["sigma_input"]
        assert row["sigma_model"] == pytest.approx(np.std([5.0, 5.5, 6.0, 6.5, 7.0],
                                                          ddof=1), rel=0.01)

    def test_flux_and_reversibility_of_one_reaction_stay_separate(self):
        """A reaction appears under both Types, so grouping on the name alone would merge
        two different quantities into one interval."""
        raw = self._raw()
        raw = pd.concat([raw, raw.assign(Type="reversibility", Value=raw["Value"] * 0.1)],
                        ignore_index=True)
        out = summarize_draws(raw, n_draws=40, group_cols=["Type", "Flux Name"])
        assert set(out["Type"]) == {"flux", "reversibility"}
        assert len(out) == 2

    def test_single_model_has_no_member_spread(self):
        raw = self._raw(n_members=1).drop(columns="Member")
        out = summarize_draws(raw, n_draws=40, group_cols=["Type", "Flux Name"])
        assert out["sigma_model"].iloc[0] == 0.0


class TestCoverageByReaction:

    def test_counts_only_the_true_value_inside_the_interval(self):
        pred = pd.DataFrame({
            "reaction": ["A", "A", "B", "B"],
            "sample":   [0, 1, 0, 1],
            "Value_lo": [0.0, 0.0, 0.0, 0.0],
            "Value_hi": [1.0, 1.0, 1.0, 1.0],
        })
        truth = pd.DataFrame({
            "reaction": ["A", "A", "B", "B"],
            "sample":   [0, 1, 0, 1],
            "gt":       [0.5, 0.5, 0.5, 9.0],    # B misses once
        })
        cov = coverage_by_reaction(pred, truth)
        assert cov["A"] == pytest.approx(1.0)
        assert cov["B"] == pytest.approx(0.5)

    def test_a_sign_error_is_not_covered(self):
        """Scoring magnitudes would count -0.5 as inside [0, 1]; signed scoring must not."""
        pred = pd.DataFrame({"reaction": ["A"], "sample": [0],
                             "Value_lo": [0.0], "Value_hi": [1.0]})
        truth = pd.DataFrame({"reaction": ["A"], "sample": [0], "gt": [-0.5]})
        assert coverage_by_reaction(pred, truth)["A"] == pytest.approx(0.0)
