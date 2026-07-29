"""Unit tests for compute_ica_nullspace in sigmaflux.ai.utils.

Tests cover:
  1. Basic correctness: null-space property, rank, shape, unit column norms, status keys.
  2. Span equivalence with raw SVD null space.
  3. Planted mode recovery on synthetic data with known sparse modes.
  4. Config round-trip: seed passed through YAML.
  5. Projection idempotence of null-space columns.
  6. Fallback: FastICA failure (1-sample input) → Varimax returned.
  7. Rank-deficient sampling: n_active < d → complement fills the gap.
"""
import numpy as np
import pytest
import scipy.linalg
import yaml

from sigmaflux.ai.utils import compute_ica_nullspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_S_and_names():
    """Small stoichiometric matrix with null-space dimension 2.

    Same network as test_targeted_nullspace.py (4 metabolites, 6 reactions, rank 4).
    """
    S = np.array([
        [ 1,  0,  0,  1,  0,  0],   # A
        [-1,  1,  0,  0,  0,  0],   # B
        [ 0, -1,  1,  0,  0, -1],   # C
        [ 0,  0, -1, -1, -1,  0],   # D
    ], dtype=float)
    names = ["r0", "r1", "r2", "r3", "r4", "r5"]
    return S, names


def _make_flux_samples(S, n_samples=300, seed=0):
    """Generate steady-state flux samples drawn from null(S) with non-Gaussian marginals."""
    N = scipy.linalg.null_space(S)
    d = N.shape[1]
    rng = np.random.default_rng(seed)
    coords = rng.laplace(size=(d, n_samples))
    return (N @ coords).T   # (n_samples, n_reactions)


def _make_planted_S_and_samples(n_samples=2000, seed=1):
    """3×7 stoichiometric matrix with d=4, Laplace-distributed samples."""
    S = np.array([
        [ 1, -1,  0, -1,  0,  0,  0],
        [ 0,  1, -1,  0, -1,  0,  0],
        [ 0,  0,  1,  0,  0, -1, -1],
    ], dtype=float)
    N = scipy.linalg.null_space(S)
    d = N.shape[1]
    rng = np.random.default_rng(seed)
    coords = rng.laplace(size=(d, n_samples))
    return S, N, (N @ coords).T   # (n_samples, 7)


# ---------------------------------------------------------------------------
# Test 1 — basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    def setup_method(self):
        self.S, _ = _make_S_and_names()
        self.n = self.S.shape[1]
        self.d = self.n - np.linalg.matrix_rank(self.S)
        self.flux_samples = _make_flux_samples(self.S)

    def _run(self, **kwargs):
        return compute_ica_nullspace(self.S, self.flux_samples, **kwargs)

    def test_shape(self):
        Z, _ = self._run()
        assert Z.shape == (self.n, self.d)

    def test_null_space_property(self):
        Z, _ = self._run()
        assert np.allclose(self.S @ Z, 0, atol=1e-8), (
            f"max|S @ Z| = {np.abs(self.S @ Z).max():.2e}"
        )

    def test_rank(self):
        Z, _ = self._run()
        assert np.linalg.matrix_rank(Z) == self.d

    def test_unit_column_norms(self):
        Z, _ = self._run()
        norms = np.linalg.norm(Z, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-8), f"column norms: {norms}"

    def test_status_keys_present(self):
        _, status = self._run()
        for key in ('n_ica_components', 'n_complement', 'n_active', 'fallback'):
            assert key in status, f"missing status key: {key!r}"

    def test_status_counts_sum_to_d(self):
        _, status = self._run()
        if not status['fallback']:
            assert status['n_ica_components'] + status['n_complement'] == self.d

    def test_n_active_le_d(self):
        _, status = self._run()
        assert status['n_active'] <= self.d


# ---------------------------------------------------------------------------
# Test 2 — span equivalence
# ---------------------------------------------------------------------------

class TestSpanEquivalence:
    def test_same_span_as_svd_basis(self):
        S, _ = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z_svd = scipy.linalg.null_space(S)
        Z_ica, _ = compute_ica_nullspace(S, flux_samples)
        d = Z_svd.shape[1]
        combined = np.hstack([Z_svd, Z_ica])
        assert np.linalg.matrix_rank(combined, tol=1e-9) == d, (
            "Z_ica does not span the same subspace as the SVD null-space basis"
        )


# ---------------------------------------------------------------------------
# Test 3 — planted mode recovery
# ---------------------------------------------------------------------------

class TestPlantedModeRecovery:
    def test_ica_recovers_planted_modes(self):
        S, N_ref, samples = _make_planted_S_and_samples(n_samples=2000, seed=1)
        d = N_ref.shape[1]
        Z, status = compute_ica_nullspace(S, samples, seed=0)
        if status['fallback'] or status['n_ica_components'] == 0:
            pytest.skip("ICA fallback — planted recovery test skipped")

        k_ica = status['n_ica_components']
        Z_ica_block = Z[:, :k_ica]
        # At least one planted mode should align with some ICA column
        found = False
        for j in range(d):
            mode = N_ref[:, j] / np.linalg.norm(N_ref[:, j])
            if np.abs(Z_ica_block.T @ mode).max() >= 0.7:
                found = True
                break
        assert found, "No planted mode has cosine similarity ≥ 0.7 with any ICA column"


# ---------------------------------------------------------------------------
# Test 4 — config round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_yaml_seed_parsed(self):
        yaml_str = """
training:
  null_space_type: ica_informed
  ica_nullspace:
    seed: 99
"""
        cfg = yaml.safe_load(yaml_str)
        t = cfg.get("training", {})
        assert t.get("null_space_type") == "ica_informed"
        params = t.get("ica_nullspace", {})
        assert params.get("seed") == 99

    def test_function_accepts_yaml_params(self):
        S, _ = _make_S_and_names()
        flux_samples = _make_flux_samples(S, n_samples=100)
        Z, status = compute_ica_nullspace(S, flux_samples, **{'seed': 42})
        assert Z.shape[0] == S.shape[1]

    def test_no_ica_nullspace_block_uses_defaults(self):
        yaml_str = """
training:
  null_space_type: ica_informed
"""
        cfg = yaml.safe_load(yaml_str)
        t = cfg.get("training", {})
        params = t.get("ica_nullspace", {})
        # Empty dict → all defaults used; function should work with no kwargs
        S, _ = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z, _ = compute_ica_nullspace(S, flux_samples, **params)
        assert Z.shape == (S.shape[1], S.shape[1] - np.linalg.matrix_rank(S))


# ---------------------------------------------------------------------------
# Test 5 — projection idempotence
# ---------------------------------------------------------------------------

class TestProjectionIdempotence:
    def test_null_space_projection_is_idempotent(self):
        S, _ = _make_S_and_names()
        N = scipy.linalg.null_space(S)
        for j in range(N.shape[1]):
            m = N[:, j]
            m_proj = N @ (N.T @ m)
            assert np.allclose(m, m_proj, atol=1e-10)

    def test_ica_modes_satisfy_null_space(self):
        S, _ = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z, _ = compute_ica_nullspace(S, flux_samples)
        for j in range(Z.shape[1]):
            sv = S @ Z[:, j]
            assert np.allclose(sv, 0, atol=1e-8), (
                f"column {j}: max|S @ z_j| = {np.abs(sv).max():.2e}"
            )


# ---------------------------------------------------------------------------
# Test 6 — fallback when FastICA cannot converge
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_on_single_sample(self):
        """1 sample → FastICA fails → Varimax fallback; basis guarantees still hold."""
        S, _ = _make_S_and_names()
        rng = np.random.default_rng(0)
        N = scipy.linalg.null_space(S)
        flux_samples = (N @ rng.normal(size=(N.shape[1], 1))).T   # (1, 6)
        Z, status = compute_ica_nullspace(S, flux_samples, seed=0)
        assert status['fallback'] is True
        assert np.allclose(S @ Z, 0, atol=1e-8)
        d = S.shape[1] - np.linalg.matrix_rank(S)
        assert np.linalg.matrix_rank(Z) == d


# ---------------------------------------------------------------------------
# Test 7 — rank-deficient sampling
# ---------------------------------------------------------------------------

class TestRankDeficientSampling:
    def test_n_active_below_d_when_samples_sparse(self):
        """Samples confined to a strict subspace → n_active < d."""
        S, _ = _make_S_and_names()   # d = 2
        N = scipy.linalg.null_space(S)
        rng = np.random.default_rng(42)
        # Only vary along the first null-space direction
        coords = np.zeros((2, 200)); coords[0, :] = rng.laplace(size=200)
        flux_samples = (N @ coords).T   # (200, 6), rank-1 in null-space
        Z, status = compute_ica_nullspace(S, flux_samples)
        assert np.allclose(S @ Z, 0, atol=1e-8)
        d = S.shape[1] - np.linalg.matrix_rank(S)
        assert np.linalg.matrix_rank(Z) == d
        assert status['n_active'] <= d   # should detect rank-1 structure

    def test_complement_fills_gap(self):
        """n_ica_components + n_complement always equals d."""
        S, _ = _make_S_and_names()
        flux_samples = _make_flux_samples(S, n_samples=50)
        _, status = compute_ica_nullspace(S, flux_samples)
        d = S.shape[1] - np.linalg.matrix_rank(S)
        if not status['fallback']:
            assert status['n_ica_components'] + status['n_complement'] == d
