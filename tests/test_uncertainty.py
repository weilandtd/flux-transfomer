"""Tests for sigmaflux.ai.uncertainty: replicate-based input draws and the composed
interval (ensemble + measurement).

The flux-side resampler and sample_inputs' replicate handling have their own file
(test_flux_resampling.py); what is covered here is summarize_draws' composition of the
two variance sources, and coverage_by_reaction.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sigmaflux.ai.uncertainty import coverage_by_reaction, summarize_draws


class TestSummarizeDraws:

    def _raw(self, n_reps=3, n_members=5, seed=0):
        """Synthetic (Sample, Member, Value) draws: sample n_reps is the MLE row,
        samples 0..n_reps-1 are replicate draws. Members disagree by a fixed offset;
        draws add their own scatter."""
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_reps + 1):
            for m in range(n_members):
                base = 5.0 + 0.5 * m
                value = base if s == n_reps else base + rng.normal(0, 0.2)
                rows.append(dict(Sample=s, Member=m, Type="flux",
                                 **{"Flux Name": "R1", "Value": value}))
        return pd.DataFrame(rows)

    def test_point_estimate_is_the_ensemble_mean_at_the_mle_input(self):
        out = summarize_draws(self._raw(n_members=5), n_draws=3, group_cols=["Type", "Flux Name"])
        assert len(out) == 1
        # Members sit at 5.0, 5.5, 6.0, 6.5, 7.0 on the unperturbed row.
        assert out["Value"].iloc[0] == pytest.approx(6.0)

    def test_point_estimate_is_the_mean_not_the_median(self):
        """The two coincide on a symmetric ensemble, so this skews one member.

        The distinction is load-bearing: a member predicts ``v = y @ Z_full.T`` and so
        lies in the null space, and a mean of those is still a flux state -- a
        per-reaction median is not, and measurably leaves it.
        """
        raw = self._raw(n_members=5)
        outlier = (raw["Member"] == 4)
        raw.loc[outlier, "Value"] = raw.loc[outlier, "Value"] + 10.0

        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"]).iloc[0]
        # Members now sit at 5.0, 5.5, 6.0, 6.5, 17.0: mean 8.0, median 6.0.
        assert out["Value"] == pytest.approx(8.0)

    def test_interval_brackets_the_point_estimate(self):
        out = summarize_draws(self._raw(), n_draws=3, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["Value_lo"] < row["Value"] < row["Value_hi"]

    def test_interval_is_the_bias_corrected_bootstrap_percentile(self):
        """The draws are a nonparametric bootstrap, so the interval is their percentile --
        shifted by Efron's z0 because the bootstrap distribution does not centre on the
        point estimate, which is evaluated at the maximum-likelihood input instead."""
        n_draws = 400
        raw = self._raw(n_reps=n_draws, n_members=5, seed=3)
        out = summarize_draws(raw, n_draws=n_draws, group_cols=["Type", "Flux Name"]).iloc[0]

        boot = raw[raw["Sample"] < n_draws]["Value"].to_numpy()
        z0 = norm.ppf((boot < out["Value"]).mean())
        q = norm.cdf(2 * z0 + np.array([norm.ppf(0.025), norm.ppf(0.975)]))
        lo, hi = np.quantile(boot, q)
        assert out["Value_lo"] == pytest.approx(lo)
        assert out["Value_hi"] == pytest.approx(hi)

    def test_interval_is_asymmetric_when_the_draws_are(self):
        """The point of the change: a symmetric band could not report a skewed spread.

        Draws are pushed hard to one side, so the two half-widths must differ; the old
        ``point +/- t * sigma`` form made them equal by construction.
        """
        n_draws = 400
        raw = self._raw(n_reps=n_draws, n_members=5, seed=5)
        drawn = raw["Sample"] < n_draws
        # A long right tail on one draw in five, nothing on the left.
        skew = drawn & (raw["Sample"] % 5 == 0)
        raw.loc[skew, "Value"] = raw.loc[skew, "Value"] + 20.0

        row = summarize_draws(raw, n_draws=n_draws, group_cols=["Type", "Flux Name"]).iloc[0]
        upper, lower = row["Value_hi"] - row["Value"], row["Value"] - row["Value_lo"]
        assert upper > 3 * lower, f"expected a right-skewed interval, got {lower=} {upper=}"

    def test_a_degenerate_group_collapses_to_the_point_estimate(self):
        """Every draw identical means nothing was resampled into a spread; the interval is
        a point, not a zero-width artefact of a quantile on a constant."""
        raw = self._raw(n_reps=5, n_members=3)
        raw["Value"] = 4.0
        row = summarize_draws(raw, n_draws=5, group_cols=["Type", "Flux Name"]).iloc[0]
        assert row["Value_lo"] == pytest.approx(4.0)
        assert row["Value_hi"] == pytest.approx(4.0)

    def test_components_separate_member_spread_from_input_spread(self):
        """sigma_model is member disagreement at the MLE input; sigma_input is how far
        the consensus moves across replicates. Members are spread far wider here than
        the replicate draws, so the model term must dominate."""
        out = summarize_draws(self._raw(n_members=5), n_draws=3, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["sigma_model"] > row["sigma_input"]
        assert row["sigma_model"] == pytest.approx(np.std([5.0, 5.5, 6.0, 6.5, 7.0], ddof=1),
                                                    rel=0.01)

    def test_flux_and_reversibility_of_one_reaction_stay_separate(self):
        """A reaction appears under both Types, so grouping on the name alone would merge
        two different quantities into one interval."""
        raw = self._raw()
        raw = pd.concat([raw, raw.assign(Type="reversibility", Value=raw["Value"] * 0.1)],
                        ignore_index=True)
        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"])
        assert set(out["Type"]) == {"flux", "reversibility"}
        assert len(out) == 2

    def test_single_model_has_no_member_spread(self):
        raw = self._raw(n_members=1).drop(columns="Member")
        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"])
        assert out["sigma_model"].iloc[0] == 0.0

    def test_sigma_total_is_the_pooled_spread_not_assumed_additive(self):
        """sigma_total is measured directly over every pooled (replicate, member) draw,
        not sqrt(sigma_model^2 + sigma_input^2) -- that identity need not hold and is
        not imposed."""
        raw = self._raw()
        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"]).iloc[0]
        pooled = raw[raw["Sample"] < 3].groupby("Flux Name")["Value"].std(ddof=1).iloc[0]
        assert out["sigma_total"] == pytest.approx(pooled)


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
