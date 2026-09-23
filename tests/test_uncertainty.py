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

from sigmaflux.ai.uncertainty import (
    _aggregate_draws, coverage_by_reaction, summarize_draws)


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

    def test_point_estimate_is_medoid_convex_not_the_plain_mean(self):
        """The two coincide on a symmetric ensemble, so this skews one member.

        The distinction is load-bearing: a member predicts ``v = y @ Z_full.T`` and so
        lies in the null space, and a convex combination of those (the plain mean, or
        medoid_convex's simplex weights) is still a flux state -- a per-reaction median
        is not, and measurably leaves it. medoid_convex is chosen over the plain mean
        because it concentrates weight near the coordinatewise median instead of
        spreading uniformly, so a single outlier member pulls it far less.
        """
        raw = self._raw(n_members=5)
        outlier = (raw["Member"] == 4)
        raw.loc[outlier, "Value"] = raw.loc[outlier, "Value"] + 10.0

        out = summarize_draws(raw, n_draws=3, group_cols=["Type", "Flux Name"]).iloc[0]
        # Members now sit at 5.0, 5.5, 6.0, 6.5, 17.0: mean 8.0, coordinatewise median
        # 6.0. With a single key, medoid_convex's simplex weights minimize distance to
        # that median in one dimension, which a convex combination can only do by
        # concentrating on the member(s) nearest it -- here, sitting exactly on it.
        assert out["Value"] == pytest.approx(6.0)
        assert out["Value"] != pytest.approx(8.0)   # not the plain mean

    def test_point_estimate_uses_one_joint_weight_vector_across_keys(self):
        """medoid_convex solves one set of member weights from the *whole* flux state,
        not an independent choice per reaction -- so it stays far closer to a non-outlier
        member's values on every key at once, unlike a per-reaction median would.
        """
        rows = []
        # Three members, two reactions. Member 2 is a severe outlier on R2 only.
        for s in (0, 1):  # MLE row is sample 1 (n_draws=1)
            for m, (r1, r2) in enumerate([(0.0, 0.0), (4.0, 4.0), (2.0, 100.0)]):
                rows.append(dict(Sample=s, Member=m, Type="flux", **{"Flux Name": "R1", "Value": r1}))
                rows.append(dict(Sample=s, Member=m, Type="flux", **{"Flux Name": "R2", "Value": r2}))
        raw = pd.DataFrame(rows)

        out = summarize_draws(raw, n_draws=1, group_cols=["Type", "Flux Name"]).set_index("Flux Name")
        # The plain mean would put R2 at (0+4+100)/3 = 34.67, dominated by the outlier.
        # medoid_convex's joint weights keep R2 close to the non-outlier members' range.
        assert out.loc["R2", "Value"] < 20.0
        assert out.loc["R1", "Value"] == pytest.approx(2.0, abs=2.5)

    def test_interval_brackets_the_point_estimate(self):
        out = summarize_draws(self._raw(), n_draws=3, group_cols=["Type", "Flux Name"])
        row = out.iloc[0]
        assert row["Value_lo"] < row["Value"] < row["Value_hi"]

    def test_interval_is_the_percentile_of_the_aggregated_draws(self):
        """The interval is the uncertainty of the reported value, so its bootstrap
        distribution is one aggregate per draw -- not every member's own prediction,
        which would answer the wider question of where a single member lands."""
        n_draws = 400
        raw = self._raw(n_reps=n_draws, n_members=5, seed=3)
        out = summarize_draws(raw, n_draws=n_draws, group_cols=["Type", "Flux Name"]).iloc[0]

        agg = _aggregate_draws(raw[raw["Sample"] < n_draws], ["Type", "Flux Name"])
        boot = agg["Value"].to_numpy()
        assert len(boot) == n_draws            # one value per draw, not per (draw, member)
        z0 = norm.ppf(np.clip((boot < out["Value"]).mean(),
                              1 / (2 * n_draws), 1 - 1 / (2 * n_draws)))
        q = norm.cdf(2 * z0 + np.array([norm.ppf(0.025), norm.ppf(0.975)]))
        lo, hi = np.quantile(boot, q)
        assert out["Value_lo"] == pytest.approx(lo)
        assert out["Value_hi"] == pytest.approx(hi)

    def test_interval_is_narrower_than_the_per_member_spread(self):
        """The distinction that motivated the change: members disagree far more than the
        aggregate moves, so an interval built from pooled member draws would be several
        times wider and would not describe the value we report."""
        n_draws = 200
        raw = self._raw(n_reps=n_draws, n_members=5, seed=7)
        row = summarize_draws(raw, n_draws=n_draws, group_cols=["Type", "Flux Name"]).iloc[0]
        pooled = raw[raw["Sample"] < n_draws]["Value"].std(ddof=1)
        assert row["Value_hi"] - row["Value_lo"] < 2 * 1.96 * pooled

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
        """sigma_model is member disagreement at the MLE input, a diagnostic; sigma_input
        is how far the aggregate moves across draws, which is what the interval reports.
        Members are spread far wider here than the replicate draws, so the diagnostic term
        must dominate -- and that gap is exactly why only one of them sets the width."""
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

    def test_sigma_input_is_the_standard_error_of_the_reported_value(self):
        """sigma_input is the spread of the aggregate across draws, so it matches the
        interval it accompanies rather than describing a different quantity."""
        n_draws = 200
        raw = self._raw(n_reps=n_draws, n_members=5, seed=11)
        out = summarize_draws(raw, n_draws=n_draws, group_cols=["Type", "Flux Name"]).iloc[0]
        agg = _aggregate_draws(raw[raw["Sample"] < n_draws], ["Type", "Flux Name"])
        assert out["sigma_input"] == pytest.approx(agg["Value"].std(ddof=1))
        assert "sigma_total" not in out.index


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
