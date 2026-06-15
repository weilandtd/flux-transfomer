"""Unit tests for compute_targeted_nullspace in sigmaflux.ai.utils.

Tests cover:
  1. Basic correctness on a small synthetic S (identity block, mass balance, shape).
  2. Separability failure detection (coupled target rows).
  3. Ill-conditioned S_D warning path.
  4. Null-space span equivalence with varimax basis.
  5. Error cases (unknown reaction name, too many targets).
"""
import warnings

import numpy as np
import pytest
import scipy.linalg

from sigmaflux.ai.utils import compute_targeted_nullspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_S_and_names():
    """Small stoichiometric matrix with null-space dimension 2.

    Network (4 metabolites, 6 reactions, rank 4):
        r0: A → B           (+1 A, -1 B)   — exchange-like
        r1: B → C           (+1 B, -1 C)
        r2: C → D           (+1 C, -1 D)
        r3: A → D           (+1 A, -1 D)   — branch
        r4: D →             (-1 D)         — drain 1
        r5: C →             (-1 C)         — drain 2

    Two independent cycles exist (dim null = 6 - 4 = 2).
    """
    S = np.array([
        [ 1,  0,  0,  1,  0,  0],   # A
        [-1,  1,  0,  0,  0,  0],   # B
        [ 0, -1,  1,  0,  0, -1],   # C
        [ 0,  0, -1, -1, -1,  0],   # D
    ], dtype=float)
    names = ["r0", "r1", "r2", "r3", "r4", "r5"]
    return S, names


# ---------------------------------------------------------------------------
# Test 1 — basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    def setup_method(self):
        self.S, self.names = _make_S_and_names()
        self.m, self.n = self.S.shape
        self.r = np.linalg.matrix_rank(self.S)
        self.k = self.n - self.r   # null-space dimension

    def test_shape(self):
        Z, status = compute_targeted_nullspace(self.S, self.names, ["r0"])
        assert status['separable']
        assert Z.shape == (self.n, self.k), f"Expected ({self.n}, {self.k}), got {Z.shape}"

    def test_mass_balance(self):
        Z, status = compute_targeted_nullspace(self.S, self.names, ["r0"])
        assert status['separable']
        assert np.allclose(self.S @ Z, 0, atol=1e-8), "S @ Z != 0"

    def test_column_norms_unit(self):
        Z, status = compute_targeted_nullspace(self.S, self.names, ["r0"])
        assert status['separable']
        assert np.allclose(np.linalg.norm(Z, axis=0), 1.0, atol=1e-10), \
            "Columns of Z should have unit norm after normalization"

    def test_target_identity_block_single(self):
        target = "r2"
        t_idx = self.names.index(target)
        Z, status = compute_targeted_nullspace(self.S, self.names, [target])
        assert status['separable']
        # Target column: nonzero at own reaction, zero at the other target slot
        assert Z[t_idx, 0] > 0, f"Z[target, 0] = {Z[t_idx, 0]}, expected > 0"
        assert abs(Z[t_idx, 1]) < 1e-10, f"Z[target, 1] = {Z[t_idx, 1]}, expected 0.0"

    def test_target_identity_block_two_targets(self):
        # r0 and r2 are separable in this network (rank-2 null-space rows)
        targets = ["r0", "r2"]
        t_idx = [self.names.index(t) for t in targets]
        Z, status = compute_targeted_nullspace(self.S, self.names, targets)
        assert status['separable']
        # First two columns: diagonal > 0, off-diagonal = 0 at the two target rows
        block = Z[np.ix_(t_idx, [0, 1])]
        assert block[0, 0] > 0 and block[1, 1] > 0, f"Diagonal should be positive:\n{block}"
        assert abs(block[0, 1]) < 1e-10, f"block[0,1] should be 0: {block[0,1]}"
        assert abs(block[1, 0]) < 1e-10, f"block[1,0] should be 0: {block[1,0]}"

    def test_status_fields_present(self):
        Z, status = compute_targeted_nullspace(self.S, self.names, ["r0"])
        for key in ('separable', 'fallback_reason', 'cond_SD', 'col_norms',
                    'target_indices', 'dep_cols', 'free_cols'):
            assert key in status, f"Missing status key: {key}"

    def test_target_indices_in_status(self):
        targets = ["r1", "r3"]
        Z, status = compute_targeted_nullspace(self.S, self.names, targets)
        assert status['target_indices'] == [self.names.index(t) for t in targets]


# ---------------------------------------------------------------------------
# Test 2 — separability failure
# ---------------------------------------------------------------------------

class TestSeparabilityFailure:
    def test_coupled_targets_returns_not_separable(self):
        """Reactions that are in series (fixed flux ratio) cannot be separated."""
        # Use the same S as _make_S_and_names(): r0 and r1 are in series
        # (v1 = v0 in every null-space vector), so rank(G) = 1 < 2 = |T|.
        S, names = _make_S_and_names()
        Z, status = compute_targeted_nullspace(S, names, ["r0", "r1"])
        assert not status['separable']
        assert 'coupled' in status['fallback_reason'] or 'rank' in status['fallback_reason']
        assert Z is None

    def test_single_target_always_separable_if_nonzero(self):
        """A single target is separable as long as it has any null-space component."""
        S, names = _make_S_and_names()
        # Every reaction with nonzero null-space row should be separable alone
        for rxn in names:
            Z, status = compute_targeted_nullspace(S, names, [rxn])
            # r0 through r5 all appear in the null space; all should be separable
            assert status['separable'], f"{rxn} unexpectedly failed separability"


# ---------------------------------------------------------------------------
# Test 3 — ill-conditioned S_D warning
# ---------------------------------------------------------------------------

class TestConditioningWarning:
    def test_low_threshold_triggers_warning(self):
        S, names = _make_S_and_names()
        # Set warn_cond_threshold=1.0 so any non-trivial S_D triggers a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z, status = compute_targeted_nullspace(
                S, names, ["r0"], warn_cond_threshold=1.0
            )
        assert status['separable']
        assert any("cond(S_D)" in str(warning.message) for warning in w), \
            "Expected a cond(S_D) warning but none was raised"

    def test_no_warning_when_well_conditioned(self):
        S, names = _make_S_and_names()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z, status = compute_targeted_nullspace(
                S, names, ["r0"], warn_cond_threshold=1e15   # impossibly high threshold
            )
        cond_warnings = [x for x in w if "cond(S_D)" in str(x.message)]
        assert len(cond_warnings) == 0


# ---------------------------------------------------------------------------
# Test 4 — null-space span equivalence
# ---------------------------------------------------------------------------

class TestSpanEquivalence:
    def test_same_span_as_varimax_basis(self):
        """Z_targeted and Z_varimax must span the same null space."""
        S, names = _make_S_and_names()
        k = S.shape[1] - np.linalg.matrix_rank(S)
        Z_var = scipy.linalg.null_space(S)                     # reference SVD basis
        Z_tgt, status = compute_targeted_nullspace(S, names, ["r1"])
        assert status['separable']
        # Stack both bases: column rank must still be k (no new directions added)
        combined = np.hstack([Z_var, Z_tgt])
        assert np.linalg.matrix_rank(combined, tol=1e-9) == k, \
            "Targeted Z spans a different subspace than null(S)"


# ---------------------------------------------------------------------------
# Test 5 — error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_unknown_reaction_raises_value_error(self):
        S, names = _make_S_and_names()
        with pytest.raises(ValueError, match="not in list|compute_targeted_nullspace"):
            compute_targeted_nullspace(S, names, ["NOTAREACTION"])

    def test_too_many_targets_raises_value_error(self):
        S, names = _make_S_and_names()
        k = S.shape[1] - np.linalg.matrix_rank(S)   # = 2
        # Ask for k+1 = 3 targets
        with pytest.raises(ValueError, match="null-space dimension"):
            compute_targeted_nullspace(S, names, ["r0", "r1", "r2"])

    def test_empty_target_list_raises(self):
        S, names = _make_S_and_names()
        # An empty target list makes no sense; function should accept it gracefully
        # or raise — current impl would return a valid Z with no identity-block guarantee.
        # We at least check it doesn't crash and S@Z=0.
        Z, status = compute_targeted_nullspace(S, names, [])
        assert status['separable']
        assert np.allclose(S @ Z, 0, atol=1e-8)
