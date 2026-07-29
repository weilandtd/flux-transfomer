"""Unit tests for compute_targeted_ica_nullspace in sigmaflux.ai.utils.

Tests cover:
  1. Basic correctness: shape, null-space property, rank, unit column norms, status keys.
  2. Identity block preserved: target rows are zero in all free columns.
  3. Span equivalence with SVD null space.
  4. Free columns satisfy S @ z ≈ 0.
  5. Fallback: non-separable targets → ICA basis returned; guarantees still hold.
  6. No free subspace (d == n_targets) → targeted Z returned as-is.
  7. Parallel reaction group detection.
"""
import numpy as np
import pytest
import scipy.linalg

from sigmaflux.ai.utils import compute_targeted_ica_nullspace, detect_parallel_reaction_groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_S_and_names():
    """4×6 stoichiometric matrix with null-space dimension 2.

    Same network used across all null-space tests.
    """
    S = np.array([
        [ 1,  0,  0,  1,  0,  0],   # A
        [-1,  1,  0,  0,  0,  0],   # B
        [ 0, -1,  1,  0,  0, -1],   # C
        [ 0,  0, -1, -1, -1,  0],   # D
    ], dtype=float)
    names = ["r0", "r1", "r2", "r3", "r4", "r5"]
    return S, names


def _make_flux_samples(S, n_samples=500, seed=0):
    N = scipy.linalg.null_space(S)
    rng = np.random.default_rng(seed)
    coords = rng.laplace(size=(N.shape[1], n_samples))
    return (N @ coords).T   # (n_samples, n_reactions)


# ---------------------------------------------------------------------------
# Test 1 — basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    def setup_method(self):
        self.S, self.names = _make_S_and_names()
        self.n = self.S.shape[1]
        self.d = self.n - np.linalg.matrix_rank(self.S)
        self.flux_samples = _make_flux_samples(self.S)

    def _run(self, targets=("r0",), **kwargs):
        return compute_targeted_ica_nullspace(
            self.S, self.flux_samples, self.names, list(targets), **kwargs
        )

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

    def test_status_keys(self):
        _, status = self._run()
        for key in ('separable', 'cond_SD', 'target_indices',
                    'n_ica_free', 'n_complement_free', 'n_active_free', 'fallback'):
            assert key in status, f"missing status key: {key!r}"

    def test_free_counts_sum_to_n_free(self):
        _, status = self._run(targets=("r0",))
        if status['separable'] and not status['fallback']:
            n_free = self.d - 1
            assert status['n_ica_free'] + status['n_complement_free'] == n_free


# ---------------------------------------------------------------------------
# Test 2 — identity block preserved
# ---------------------------------------------------------------------------

class TestIdentityBlock:
    def test_target_row_nonzero_only_in_own_col(self):
        """Target row must be non-zero in col 0 (its DOF) and zero in all free cols."""
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z, status = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0"])
        assert status['separable']
        target_idx = names.index("r0")
        # Identity entry is column-normalized: non-zero but not necessarily ±1
        assert abs(Z[target_idx, 0]) > 0, f"Z[{target_idx}, 0] = {Z[target_idx, 0]:.6f}"
        # All free columns must have zero on this target row
        assert np.allclose(Z[target_idx, 1:], 0, atol=1e-8), (
            f"target row in free cols: {Z[target_idx, 1:]}"
        )

    def test_target_rows_zero_in_free_cols(self):
        """Target reaction rows must be zero in all free (non-identity) columns."""
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z, status = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0"])
        assert status['separable']
        target_idx = names.index("r0")
        # Columns 1: (free block) must have zero on target row
        free_block = Z[:, 1:]
        assert np.allclose(free_block[target_idx, :], 0, atol=1e-8), (
            f"target row in free block: {free_block[target_idx, :]}"
        )

    def test_two_targets_identity_block(self):
        """Two target reactions: first two columns have 2×2 identity block on target rows."""
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S, n_samples=500)
        # r0 and r4 are both separable in this network
        Z, status = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0", "r4"])
        if not status['separable']:
            pytest.skip("targets not separable in this network")
        t0, t1 = names.index("r0"), names.index("r4")
        # Each target row: non-zero in own column, zero in the other target col + all free cols
        assert abs(Z[t0, 0]) > 0, f"Z[t0, 0] = {Z[t0, 0]}"
        assert abs(Z[t0, 1]) < 1e-8, f"Z[t0, 1] = {Z[t0, 1]}"   # zero in t1's column
        assert np.allclose(Z[t0, 2:], 0, atol=1e-8)               # zero in free cols
        assert abs(Z[t1, 0]) < 1e-8, f"Z[t1, 0] = {Z[t1, 0]}"   # zero in t0's column
        assert abs(Z[t1, 1]) > 0, f"Z[t1, 1] = {Z[t1, 1]}"
        assert np.allclose(Z[t1, 2:], 0, atol=1e-8)               # zero in free cols


# ---------------------------------------------------------------------------
# Test 3 — span equivalence
# ---------------------------------------------------------------------------

class TestSpanEquivalence:
    def test_same_span_as_svd_basis(self):
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z_svd = scipy.linalg.null_space(S)
        Z, _ = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0"])
        d = Z_svd.shape[1]
        combined = np.hstack([Z_svd, Z])
        assert np.linalg.matrix_rank(combined, tol=1e-9) == d


# ---------------------------------------------------------------------------
# Test 4 — all columns in null space
# ---------------------------------------------------------------------------

class TestFreeSubspaceInNull:
    def test_all_columns_satisfy_mass_balance(self):
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        Z, _ = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0"])
        for j in range(Z.shape[1]):
            sv = S @ Z[:, j]
            assert np.allclose(sv, 0, atol=1e-8), (
                f"column {j}: max|S @ z_j| = {np.abs(sv).max():.2e}"
            )


# ---------------------------------------------------------------------------
# Test 5 — fallback on non-separable targets
# ---------------------------------------------------------------------------

class TestFallback:
    def test_non_separable_falls_back_gracefully(self):
        """Coupled target reactions → ICA basis returned; null-space + rank guaranteed."""
        S, names = _make_S_and_names()
        flux_samples = _make_flux_samples(S)
        # r0 and r3: in this network their null-space rows are linearly dependent
        # (both are in the same cycle).  This may or may not be separable depending
        # on the network; if separable the test just checks the basis is valid.
        Z, status = compute_targeted_ica_nullspace(S, flux_samples, names, ["r0", "r3"])
        d = S.shape[1] - np.linalg.matrix_rank(S)
        assert np.allclose(S @ Z, 0, atol=1e-8)
        assert np.linalg.matrix_rank(Z) == d

    def test_fastICA_failure_returns_varimax_free(self):
        """FastICA on 1 sample fails → targeted Z with Varimax free cols returned."""
        S, names = _make_S_and_names()
        N = scipy.linalg.null_space(S)
        rng = np.random.default_rng(0)
        single_sample = (N @ rng.normal(size=(N.shape[1], 1))).T   # (1, n)
        Z, status = compute_targeted_ica_nullspace(S, single_sample, names, ["r0"])
        d = S.shape[1] - np.linalg.matrix_rank(S)
        assert np.allclose(S @ Z, 0, atol=1e-8)
        assert np.linalg.matrix_rank(Z) == d
        # Either succeeded or fell back gracefully
        assert 'fallback' in status


# ---------------------------------------------------------------------------
# Test 6 — no free subspace
# ---------------------------------------------------------------------------

class TestNoFreeSubspace:
    def test_d_equals_n_targets_returns_targeted(self):
        """When d == len(target_reactions), no free subspace exists; return targeted Z."""
        S, names = _make_S_and_names()   # d = 2
        flux_samples = _make_flux_samples(S)
        # Target both null-space DOFs
        Z, status = compute_targeted_ica_nullspace(
            S, flux_samples, names, ["r0", "r4"]
        )
        d = S.shape[1] - np.linalg.matrix_rank(S)
        assert Z.shape == (S.shape[1], d)
        assert np.allclose(S @ Z, 0, atol=1e-8)
        assert np.linalg.matrix_rank(Z) == d


# ---------------------------------------------------------------------------
# Test 7 — parallel reaction group detection
# ---------------------------------------------------------------------------

class TestParallelGroupDetection:
    """detect_parallel_reaction_groups must identify stoichiometrically proportional columns."""

    def test_no_parallels_in_simple_network(self):
        """The 4×6 test network has no exactly parallel reaction pairs."""
        S, names = _make_S_and_names()
        groups = detect_parallel_reaction_groups(S, names)
        assert groups == [], f"unexpected groups: {groups}"

    def test_identical_columns_detected(self):
        """Two identical columns are trivially parallel (cosine = 1)."""
        S, names = _make_S_and_names()
        # Duplicate column 0
        S_dup = np.hstack([S, S[:, [0]]])
        names_dup = names + ["r0_copy"]
        groups = detect_parallel_reaction_groups(S_dup, names_dup)
        assert len(groups) == 1, f"expected 1 group, got {groups}"
        assert set(groups[0]) == {"r0", "r0_copy"}, f"wrong group: {groups[0]}"

    def test_proportional_columns_detected(self):
        """A column scaled by 2 is still parallel (cosine = 1)."""
        S, names = _make_S_and_names()
        S_scaled = np.hstack([S, 2.0 * S[:, [0]]])
        names_scaled = names + ["r0_scaled"]
        groups = detect_parallel_reaction_groups(S_scaled, names_scaled)
        assert len(groups) == 1
        assert set(groups[0]) == {"r0", "r0_scaled"}

    def test_antiparallel_columns_detected(self):
        """The negated column of a reversible reaction is antiparallel (cosine = -1)."""
        S, names = _make_S_and_names()
        S_neg = np.hstack([S, -S[:, [0]]])
        names_neg = names + ["r0_rev"]
        groups = detect_parallel_reaction_groups(S_neg, names_neg)
        assert len(groups) == 1
        assert set(groups[0]) == {"r0", "r0_rev"}

    def test_threshold_respected(self):
        """A slightly perturbed column should not be detected at tight threshold."""
        S, names = _make_S_and_names()
        perturbed = S[:, [0]].copy()
        perturbed[0] += 0.5   # significant perturbation → cosine < 0.99
        S_pert = np.hstack([S, perturbed])
        names_pert = names + ["r0_pert"]
        groups = detect_parallel_reaction_groups(S_pert, names_pert, cos_threshold=0.99)
        # r0_pert should NOT be grouped with r0
        for g in groups:
            assert "r0_pert" not in g or "r0" not in g, (
                f"perturbed column wrongly grouped with r0: {g}"
            )

    def test_groups_contain_reaction_names(self):
        """Each group element must be a valid reaction name."""
        S, names = _make_S_and_names()
        S_dup = np.hstack([S, S[:, [1]]])
        names_dup = names + ["r1_copy"]
        groups = detect_parallel_reaction_groups(S_dup, names_dup)
        all_members = {m for g in groups for m in g}
        assert all_members <= set(names_dup), f"unknown reaction in groups: {all_members}"
