"""Tests for per-tracer Fisher token selection.

Selection used to pool the tracer axis: a candidate token was one (metabolite, isotopomer)
whose Fisher Gram summed the information every tracer carried, so a metabolite was included
or excluded for all tracers at once. These cover the per-tracer replacement -- the token
identity, the mapping filter, the CSV schema, and how the mask is re-applied at training,
including against a tracer-less features.csv written before the change.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from cobra.io.json import load_json_model

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.ai import features as feat_lib
from sigmaflux.ai.utils import load_fisher_token_scores

TOY_MODEL = os.path.join(_REPO_ROOT, "models", "atomistic", "toy", "one_carbon_annotated.json")

TRACERS = ["Ser_in_[1]_1.0", "Ser_in_[1]_0.5"]
TOKENS = [(t, m, i) for t in TRACERS for m in ("Ser_c", "Gly_c") for i in ("M+1", "M+2")]


def _labeling_df(n_samples=40, seed=0):
    """Labeling frame with the production (Tracer, Metabolite, Isotopomer) row index."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random((len(TOKENS), n_samples)),
        index=pd.MultiIndex.from_tuples(TOKENS, names=["Tracer", "Metabolite", "Isotopomer"]),
        columns=range(n_samples),
    )


class TestTokenIdentity:

    def test_tokens_are_keyed_per_tracer(self):
        model = load_json_model(TOY_MODEL)
        rxn = [r.id for r in model.reactions]
        n = 40
        flux_df = pd.DataFrame(np.random.default_rng(1).random((len(rxn), n)), index=rxn)

        _, token_states, _ = feat_lib.build_theta_and_token_states(
            _labeling_df(n), flux_df, TOY_MODEL, n)

        assert set(token_states) == set(TOKENS)
        # One column per token, not one per tracer: the tracer is part of the identity now,
        # so a token's Gram is rank-1 rather than a sum over the tracer axis.
        assert all(v.shape == (n, 1) for v in token_states.values())

    def test_mapping_filter_matches_the_metabolite_component(self, tmp_path):
        csv = tmp_path / "mapping.csv"
        csv.write_text("experimental_id,model_id\nserine,Ser_c\n")
        token_states = {k: np.zeros((3, 1)) for k in TOKENS}

        kept = feat_lib.apply_mapping_filter(token_states, str(csv))

        assert set(kept) == {k for k in TOKENS if k[1] == "Ser_c"}
        assert {k[0] for k in kept} == set(TRACERS)   # both tracers survive the filter


class TestLoadFisherTokenScores:
    """The mask is re-applied to labeling_df at training time; these pin how it keys."""

    def _features_csv(self, tmp_path, rows, columns):
        path = tmp_path / "features.csv"
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return str(path)

    def test_per_tracer_csv_selects_different_tokens_per_tracer(self, tmp_path):
        # Ser_c/M+1 is informative under the first tracer only.
        rows = [(t, m, i, (t == TRACERS[0] and m == "Ser_c"), 1.0)
                for t, m, i in TOKENS]
        csv = self._features_csv(
            tmp_path, rows,
            ["tracer", "metabolite_id", "isotopomer", "included", "standalone_nats"])

        _, mask = load_fisher_token_scores(csv, _labeling_df())

        selected = [tok for tok, keep in zip(TOKENS, mask.numpy()) if keep]
        assert selected == [(TRACERS[0], "Ser_c", "M+1"), (TRACERS[0], "Ser_c", "M+2")]

    def test_tracerless_csv_still_applies_to_every_tracer(self, tmp_path):
        """A features.csv written before per-tracer selection had no tracer column and
        applied identically to each tracer; existing run directories must keep working."""
        csv = self._features_csv(
            tmp_path,
            [("Ser_c", "M+1", True, 2.0), ("Ser_c", "M+2", False, 0.0),
             ("Gly_c", "M+1", False, 0.0), ("Gly_c", "M+2", False, 0.0)],
            ["metabolite_id", "isotopomer", "included", "standalone_nats"])

        _, mask = load_fisher_token_scores(csv, _labeling_df())

        selected = [tok for tok, keep in zip(TOKENS, mask.numpy()) if keep]
        assert selected == [(t, "Ser_c", "M+1") for t in TRACERS]

    def test_scores_align_to_labeling_df_row_order(self, tmp_path):
        rows = [(t, m, i, True, float(ix)) for ix, (t, m, i) in enumerate(TOKENS)]
        csv = self._features_csv(
            tmp_path, rows,
            ["tracer", "metabolite_id", "isotopomer", "included", "standalone_nats"])

        scores, _ = load_fisher_token_scores(csv, _labeling_df())

        assert scores.tolist() == [float(i) for i in range(len(TOKENS))]

    def test_selecting_nothing_raises_instead_of_emptying_the_token_set(self, tmp_path):
        """A stale CSV keyed on tracers that no longer exist used to leave a zero-row
        labeling_df and fail much later, with nothing pointing back here."""
        csv = self._features_csv(
            tmp_path,
            [("some_other_tracer", "Ser_c", "M+1", True, 2.0)],
            ["tracer", "metabolite_id", "isotopomer", "included", "standalone_nats"])

        with pytest.raises(ValueError, match="selects none of the"):
            load_fisher_token_scores(csv, _labeling_df())
