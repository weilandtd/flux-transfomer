"""Tests for HSIC sensitivity index computation."""

import numpy as np
import pytest

from sigmaflux.analysis.sensitivity import hsic_indices, hsic_unbiased, rbf_kernel

N = 200
SEED = 42


def test_rbf_kernel_shape_and_diagonal():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(30)
    K = rbf_kernel(x)
    assert K.shape == (30, 30)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    min_eigval = np.linalg.eigvalsh(K).min()
    assert min_eigval >= -1e-10, f"kernel not PSD: min_eigval={min_eigval:.3e}"


def test_rbf_kernel_multidim():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((20, 5))
    K = rbf_kernel(x)
    assert K.shape == (20, 20)
    assert np.allclose(np.diag(K), 1.0)


def test_hsic_independent():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, 1))
    Y = rng.standard_normal((N, 5))
    results = hsic_indices(X, Y)
    R = results.loc["X_0", "R"]
    assert abs(R) < 0.1, f"Independent X,Y should have |R| < 0.1, got R={R:.3f}"


def test_hsic_dependent():
    rng = np.random.default_rng(SEED + 1)
    X = rng.standard_normal((N, 1))
    Y = X + 0.1 * rng.standard_normal((N, 1))
    results = hsic_indices(X, Y)
    R = results.loc["X_0", "R"]
    assert abs(R) > 0.5, f"Dependent X,Y should have |R| > 0.5, got R={R:.3f}"


def test_hsic_normalized_bounds():
    rng = np.random.default_rng(SEED + 2)
    X = rng.standard_normal((N, 3))
    Y = X[:, [0]] + 0.1 * rng.standard_normal((N, 1))
    results = hsic_indices(X, Y)
    for name in results.index:
        R = results.loc[name, "R"]
        if not np.isnan(R):
            assert -1.0 <= R <= 1.0, f"{name}: R={R:.4f} out of [-1, 1]"
    pos_mask = results["hsic"] > 0
    assert (results.loc[pos_mask, "eta"] >= 0).all()


def test_hsic_input_names_propagate():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((50, 2))
    Y = rng.standard_normal((50, 4))
    names = ["flux_A", "flux_B"]
    results = hsic_indices(X, Y, input_names=names)
    assert set(results.index) == set(names)
    assert list(results.columns) == ["hsic", "eta", "R"]


def test_hsic_constant_input():
    rng = np.random.default_rng(SEED)
    X_const = np.ones((50, 1))
    Y = rng.standard_normal((50, 3))
    results = hsic_indices(X_const, Y)
    assert abs(results.loc["X_0", "hsic"]) < 1e-10, (
        f"Constant input should have hsic ≈ 0, got {results.loc['X_0','hsic']:.3e}"
    )


def test_hsic_ordering():
    """The input most correlated with Y should rank first by eta."""
    rng = np.random.default_rng(SEED + 3)
    n = 300
    x_strong = rng.standard_normal(n)
    x_weak = rng.standard_normal(n)
    x_indep = rng.standard_normal(n)
    X = np.column_stack([x_strong, x_weak, x_indep])
    Y = (x_strong + 0.5 * x_weak + 0.1 * rng.standard_normal(n)).reshape(-1, 1)
    results = hsic_indices(X, Y, input_names=["strong", "weak", "indep"])
    assert results.index[0] == "strong", (
        f"Expected 'strong' ranked first, got: {list(results.index)}"
    )


def test_max_samples_subsets_rows():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((200, 5))
    Y = rng.standard_normal((200, 3))
    results = hsic_indices(X, Y, max_samples=50, seed=7)
    assert results.shape == (5, 3)
    assert list(results.columns) == ["hsic", "eta", "R"]
    assert not results["hsic"].isna().any()


def test_n_jobs_matches_sequential():
    rng = np.random.default_rng(SEED + 10)
    X = rng.standard_normal((100, 12))
    Y = rng.standard_normal((100, 4))
    seq = hsic_indices(X, Y, n_jobs=1, chunk_size=3)
    par = hsic_indices(X, Y, n_jobs=2, chunk_size=3)
    seq = seq.sort_index()
    par = par.sort_index()
    np.testing.assert_allclose(
        seq[["hsic", "eta"]].values, par[["hsic", "eta"]].values,
        rtol=1e-5, atol=1e-12,
    )
    valid = ~np.isnan(seq["R"].values)
    np.testing.assert_allclose(seq["R"].values[valid], par["R"].values[valid], rtol=1e-5)
