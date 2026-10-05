"""``--shuffle-targets``: the null-model control for Gate G7.

Permuting flux/reversibility samples relative to labeling samples is what turns a
trained ensemble into the null model an attribution map is checked against -- any
structure that survives this pairing is an artifact of the pipeline, not the biology.
The permutation must (a) actually decouple the two, (b) keep flux and reversibility
paired with each other so S.v = 0 still holds for every target, and (c) be
reproducible from the seed so an ensemble's members share one broken pairing.
"""

import numpy as np
import pandas as pd

from sigmaflux.pipeline.train_ai_model import shuffle_targets


def _frames(n_samples=20, n_flux=5, n_rev=3, seed=0):
    rng = np.random.default_rng(seed)
    flux = pd.DataFrame(rng.normal(size=(n_flux, n_samples)))
    rev = pd.DataFrame(rng.normal(size=(n_rev, n_samples)))
    return flux, rev


def test_permutation_actually_moves_samples():
    flux, rev = _frames()
    shuffled_flux, _ = shuffle_targets(flux, rev, seed=1)
    assert not np.array_equal(shuffled_flux.values, flux.values)
    # every original column still appears, just relabeled
    assert {tuple(shuffled_flux[c]) for c in shuffled_flux.columns} == \
           {tuple(flux[c]) for c in flux.columns}


def test_flux_and_reversibility_permuted_together():
    """Decoupled from labeling, but not from each other -- each target stays real."""
    flux, rev = _frames()
    shuffled_flux, shuffled_rev = shuffle_targets(flux, rev, seed=7)
    # column i of the shuffled frames must be some single original column j for both
    orig_cols = {j: (tuple(flux[j]), tuple(rev[j])) for j in flux.columns}
    for i in shuffled_flux.columns:
        pair = (tuple(shuffled_flux[i]), tuple(shuffled_rev[i]))
        assert pair in orig_cols.values()


def test_reproducible_from_seed():
    """An ensemble's members must all see the same broken pairing."""
    flux, rev = _frames()
    a_flux, a_rev = shuffle_targets(flux.copy(), rev.copy(), seed=42)
    b_flux, b_rev = shuffle_targets(flux.copy(), rev.copy(), seed=42)
    pd.testing.assert_frame_equal(a_flux, b_flux)
    pd.testing.assert_frame_equal(a_rev, b_rev)


def test_different_seeds_give_different_pairings():
    flux, rev = _frames(n_samples=50)
    a_flux, _ = shuffle_targets(flux.copy(), rev.copy(), seed=1)
    b_flux, _ = shuffle_targets(flux.copy(), rev.copy(), seed=2)
    assert not np.array_equal(a_flux.values, b_flux.values)


def test_columns_renumbered_like_load_data():
    """Downstream code (subsampling, split_indices) indexes positionally by 0..n-1."""
    flux, rev = _frames()
    shuffled_flux, shuffled_rev = shuffle_targets(flux, rev, seed=3)
    assert list(shuffled_flux.columns) == list(range(flux.shape[1]))
    assert list(shuffled_rev.columns) == list(range(rev.shape[1]))
