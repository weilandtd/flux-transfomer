"""KO-selection strategies: "all" (screen_nonessential, unchanged), "random", and
"uncorrelated" (pFBA change-vector greedy pivoted Cholesky selection).

Toy-model helpers mirror tests/test_reusable_sampler.py's style but are kept local
(no cross-file private imports).
"""

import os
import sys

import cobra
import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.pipeline.sample_flux import (  # noqa: E402
    _normalize_ko_strategy,
    _select_uncorrelated_ko,
    greedy_pivoted_cholesky_select,
    screen_nonessential,
    select_ko_reactions,
)


def _reaction(model, rid, stoich, lb, ub):
    r = cobra.Reaction(rid)
    r.bounds = (lb, ub)
    model.add_reactions([r])
    r.add_metabolites({model.metabolites.get_by_id(m): c for m, c in stoich.items()})
    return r


def _branched_model(with_alt_route=True):
    """IN -> A -> B -> C -> OUT, with a parallel B->C route (R3) unless disabled."""
    m = cobra.Model("toy")
    m.add_metabolites([cobra.Metabolite(i) for i in ("A", "B", "C")])
    _reaction(m, "IN", {"A": 1}, 0, 10)
    _reaction(m, "R1", {"A": -1, "B": 1}, 0, 10)
    _reaction(m, "R2", {"B": -1, "C": 1}, 0, 10)
    if with_alt_route:
        _reaction(m, "R3", {"B": -1, "C": 1}, 0, 10)
    _reaction(m, "OUT", {"C": -1}, 1, 10)
    return m


def _wide_model(n_parallel=6):
    """IN -> A, plus n_parallel redundant A->OUT routes -- gives screen_nonessential
    a candidate pool with more than one non-essential reaction to sub-sample from.
    """
    m = cobra.Model("wide")
    m.add_metabolites([cobra.Metabolite("A")])
    _reaction(m, "IN", {"A": 1}, 0, 10)
    for i in range(n_parallel):
        _reaction(m, f"OUT{i}", {"A": -1}, 0, 10)
    m.reactions.get_by_id("OUT0").lower_bound = 1  # forces overall flux, keeps model non-trivial
    return m


def _two_subnetwork_model():
    """Two independent branched subnetworks, each with its own redundant route -- lets
    _select_uncorrelated_ko be tested against a case with two genuinely distinct
    "directions" of change, plus one within-subnetwork redundant pair.
    """
    m = _branched_model(with_alt_route=True)
    m.add_metabolites([cobra.Metabolite(i) for i in ("D", "E")])
    _reaction(m, "IN2", {"D": 1}, 0, 10)
    _reaction(m, "R4", {"D": -1, "E": 1}, 0, 10)
    _reaction(m, "R5", {"D": -1, "E": 1}, 0, 10)
    _reaction(m, "OUT2", {"E": -1}, 1, 10)
    return m


# ── _normalize_ko_strategy ───────────────────────────────────────────────────

def test_normalize_ko_strategy_legacy_and_disabled():
    assert _normalize_ko_strategy(True) == "all"
    assert _normalize_ko_strategy(False) is None
    assert _normalize_ko_strategy(None) is None


@pytest.mark.parametrize("value", ["all", "random", "uncorrelated"])
def test_normalize_ko_strategy_passthrough(value):
    assert _normalize_ko_strategy(value) == value


def test_normalize_ko_strategy_rejects_unknown_string():
    with pytest.raises(ValueError, match="Unknown sampling.ko value"):
        _normalize_ko_strategy("pca")


# ── select_ko_reactions: "all" ───────────────────────────────────────────────

def test_select_ko_reactions_all_matches_screen_nonessential():
    model = _wide_model()
    assert select_ko_reactions(model, "all", None, seed=0) == screen_nonessential(model)


# ── select_ko_reactions: "random" ────────────────────────────────────────────

def test_select_ko_reactions_random_requires_n_ko():
    model = _wide_model()
    with pytest.raises(ValueError, match="sampling.n_ko is required"):
        select_ko_reactions(model, "random", None, seed=0)


def test_select_ko_reactions_random_respects_n_ko_and_is_reproducible():
    model = _wide_model()
    nonessential = screen_nonessential(model)
    n_ko = 3
    assert 0 < n_ko < len(nonessential)

    a = select_ko_reactions(model, "random", n_ko, seed=42)
    b = select_ko_reactions(model, "random", n_ko, seed=42)

    assert a == b, "same seed should reproduce the same selection"
    assert len(a) == n_ko
    assert len(set(a)) == n_ko, "no duplicates"
    assert set(a) <= set(nonessential)


def test_select_ko_reactions_random_degrades_to_all_when_n_ko_covers_pool():
    model = _wide_model()
    nonessential = screen_nonessential(model)

    result = select_ko_reactions(model, "random", n_ko=len(nonessential) + 5, seed=0)

    assert result == nonessential, "n_ko >= pool size should return the pool unpermuted"


# ── greedy_pivoted_cholesky_select ───────────────────────────────────────────

def _clustered_correlation_matrix(n_clusters=3, per_cluster=5, n_features=200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_clusters, n_features))
    rows = np.vstack([
        base[c] + 0.01 * rng.normal(size=(per_cluster, n_features))
        for c in range(n_clusters)
    ])
    return np.corrcoef(rows), n_clusters, per_cluster


def test_greedy_pivoted_cholesky_select_picks_one_per_cluster_first():
    C, n_clusters, per_cluster = _clustered_correlation_matrix()

    selected = greedy_pivoted_cholesky_select(C, n_clusters)

    assert len(selected) == n_clusters
    assert len(set(selected)) == n_clusters
    clusters_hit = {i // per_cluster for i in selected}
    assert clusters_hit == set(range(n_clusters)), \
        "k == n_clusters should pick one representative per cluster"


def test_greedy_pivoted_cholesky_select_caps_at_n_without_duplicates():
    C, n_clusters, per_cluster = _clustered_correlation_matrix()
    n = n_clusters * per_cluster

    selected = greedy_pivoted_cholesky_select(C, n + 50)

    assert len(selected) <= n
    assert len(set(selected)) == len(selected)


def test_greedy_pivoted_cholesky_select_identity_matrix():
    C = np.eye(10)
    selected = greedy_pivoted_cholesky_select(C, 4)
    assert len(selected) == 4
    assert len(set(selected)) == 4


# ── select_ko_reactions / _select_uncorrelated_ko: "uncorrelated" ───────────

def test_select_ko_reactions_uncorrelated_requires_n_ko():
    model = _wide_model()
    with pytest.raises(ValueError, match="sampling.n_ko is required"):
        select_ko_reactions(model, "uncorrelated", None, seed=0)


def test_select_ko_reactions_uncorrelated_degrades_to_all_when_n_ko_covers_pool():
    model = _wide_model()
    nonessential = screen_nonessential(model)

    result = select_ko_reactions(model, "uncorrelated", n_ko=len(nonessential) + 5, seed=0)

    assert result == nonessential
    # degrading to "all" must skip the pFBA/correlation machinery entirely


def test_select_uncorrelated_ko_prefers_distinct_subnetworks():
    """R2/R3 are a redundant pair in one subnetwork, R4/R5 a redundant pair in the
    other. With n_ko=2 the least-mutually-correlated pick should take one reaction
    from each subnetwork rather than two reactions from the same redundant pair.
    """
    model = _two_subnetwork_model()
    candidates = screen_nonessential(model)

    chosen = _select_uncorrelated_ko(model, candidates, n_ko=2, seed=0)

    assert len(chosen) == 2
    subnet_1 = {"R2", "R3"}
    subnet_2 = {"R4", "R5"}
    assert chosen[0] in subnet_1 or chosen[0] in subnet_2
    hit_subnets = {"1" if r in subnet_1 else "2" for r in chosen}
    assert hit_subnets == {"1", "2"}, \
        f"expected one reaction from each subnetwork, got {chosen}"


def test_select_uncorrelated_ko_handles_uninformative_candidate():
    """R3 is a redundant route the WT pFBA (minimal total flux) solution doesn't use,
    so knocking it out leaves the pFBA solution completely unchanged (a zero-variance
    change vector). This must not crash the selection.
    """
    model = _branched_model(with_alt_route=True)
    candidates = screen_nonessential(model)
    assert "R3" in candidates

    chosen = _select_uncorrelated_ko(model, candidates, n_ko=2, seed=0)

    assert len(chosen) <= 2
    assert set(chosen) <= set(candidates)
    assert "R3" not in chosen, "the uninformative (zero-variance) candidate must be filtered out"
