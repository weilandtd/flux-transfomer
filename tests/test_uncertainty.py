"""Tests for sigmaflux.ai.uncertainty: replicate-based input draws and the composed
interval (ensemble + measurement + learned aleatoric uncertainty).

The flux-side resampler and sample_inputs' replicate handling have their own file
(test_flux_resampling.py); what is covered here is summarize_draws' composition of the
three variance sources, and coverage_by_reaction.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as tdist

from sigmaflux.ai.uncertainty import coverage_by_reaction, summarize_draws


class TestSummarizeDraws:

    def _raw(self, n_reps=3, n_members=5, seed=0, sigma=None):
        """Synthetic (Sample, Member, Value[, Sigma]) draws: sample n_reps is the MLE
        row, samples 0..n_reps-1 are replicate draws. Members disagree by a fixed
        offset; draws add their own scatter."""
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_reps + 1):
            for m in range(n_members):
                base = 5.0 + 0.5 * m
                value = base if s == n_reps else base + rng.normal(0, 0.2)
                row = dict(Sample=s, Member=m, Type="flux", **{"Flux Name": "R1", "Value": value})
                if sigma is not None:
                    row["Sigma"] = sigma
                rows.append(row)
        return pd.DataFrame(rows)

    def test_point_estimate_is_the_ensemble_median_at_the_mle_input(self):
        out = summarize_draws(self._raw(n_members=5), n_draws=3, group_cols=["Type", "Flux Name"])
        assert len(out) == 1
        # Members sit at 5.0, 5.5, 6.0, 6.5, 7.0 on the unperturbed row.
        assert out["Value"].iloc[0] == pytest.approx(6.0)

    def test_interval_brackets_the_point_estimate(self):
        out = summarize_draws(self._raw(), n_draws=3, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["Value_lo"] < row["Value"] < row["Value_hi"]

    def test_interval_is_a_t_interval_on_the_pooled_spread(self):
        """Not the empirical percentile of the draws -- with as few as 3 replicates the
        2.5th/97.5th percentiles sit at the sample extrema. sigma is estimated from a
        small sample, so the quantile is Student-t on the replicate axis' degrees of
        freedom (n_draws - 1), which is now the thinner axis against ~20 members.
        """
        n_reps = 3
        raw = self._raw(n_reps=n_reps, n_members=20)
        out = summarize_draws(raw, n_draws=n_reps, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        sigma_mi = raw[raw["Sample"] < n_reps].groupby("Flux Name")["Value"].std(ddof=1).iloc[0]
        half = tdist.ppf(0.975, n_reps - 1) * sigma_mi
        assert row["Value_hi"] - row["Value"] == pytest.approx(half)
        assert row["Value"] - row["Value_lo"] == pytest.approx(half)

    def test_degrees_of_freedom_follow_the_replicate_count(self):
        """Fewer replicates means the pooled spread is less well determined and the
        interval must widen, even holding the member count fixed."""
        few = summarize_draws(self._raw(n_reps=2), n_draws=2,
                              group_cols=["Type", "Flux Name"]).iloc[0]
        many = summarize_draws(self._raw(n_reps=8), n_draws=8,
                               group_cols=["Type", "Flux Name"]).iloc[0]
        assert (few["Value_hi"] - few["Value_lo"]) > (many["Value_hi"] - many["Value_lo"])

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

    def test_no_sigma_column_falls_back_to_ensemble_and_measurement_only(self):
        """A checkpoint with no variance head produces no Sigma column at all; the
        interval must still be well formed, with a zero aleatoric contribution."""
        raw = self._raw(sigma=None)
        assert "Sigma" not in raw.columns
        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"])
        assert out["sigma_aleatoric"].iloc[0] == 0.0
        assert out["Value_lo"].iloc[0] < out["Value"].iloc[0] < out["Value_hi"].iloc[0]

    def test_learned_aleatoric_variance_widens_the_interval(self):
        """A model with a variance head reports Sigma per (sample, member); a larger
        learned sigma must widen the interval, since it is a genuinely separate source
        from ensemble/replicate disagreement (nonzero even at one exact input)."""
        narrow = summarize_draws(self._raw(sigma=0.05), n_draws=3,
                                 group_cols=["Type", "Flux Name"]).iloc[0]
        wide = summarize_draws(self._raw(sigma=2.0), n_draws=3,
                               group_cols=["Type", "Flux Name"]).iloc[0]
        assert wide["sigma_aleatoric"] > narrow["sigma_aleatoric"] > 0
        assert (wide["Value_hi"] - wide["Value_lo"]) > (narrow["Value_hi"] - narrow["Value_lo"])

    def test_components_combine_in_quadrature_not_additively(self):
        """sigma_total^2 = sigma_mi^2 + sigma_aleatoric^2 -- the aleatoric term is not
        simply added to the pooled spread linearly."""
        sigma = 1.0
        raw = self._raw(sigma=sigma)
        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"]).iloc[0]
        sigma_mi = raw[raw["Sample"] < 3].groupby("Flux Name")["Value"].std(ddof=1).iloc[0]
        expected_total = np.sqrt(sigma_mi ** 2 + sigma ** 2)
        half = tdist.ppf(0.975, 2) * expected_total
        assert out["Value_hi"] - out["Value"] == pytest.approx(half)


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
