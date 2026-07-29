"""Tests for boundary-decomposition null-space basis.

Toy network (11 reactions, 9 metabolites) using COBRA sign convention:
  v > 0 for exchange reaction = metabolite leaves cell (secretion).
  For uptake: EX_glc runs at v < 0 (reversible), or use split leg.

Reactions:
  0 EX_glc (rev):  glc_e → ∅   S[glc_e,0]=-1 (v>0 = secretion; reversible so v<0 = uptake)
  1 EX_eth (irr):  eth_e → ∅   S[eth_e,1]=-1
  2 EX_co2 (irr):  co2_e → ∅   S[co2_e,2]=-1
  3 EX_gln (rev):  gln_e → ∅   S[gln_e,3]=-1
  4 EX_biom (irr): biom_e → ∅  S[biom_e,4]=-1
  5 GK (irr):      glc_e → glc             S[glc_e,5]=-1, S[glc,5]=+1
  6 FERM (irr):    glc → 2 eth_e + 2 co2_e + 2 atp
  7 BIOM (irr):    glc + atp → biom_e
  8 GLN_UTK (irr): gln_e → glc             S[gln_e,8]=-1, S[glc,8]=+1
  9 FUTILE (irr):  atp → adp
 10 FUTILE_R (irr): adp → atp

Null-space analysis (rank(S)=8, nullity=3):
  Free params: a=FERM, b=GLN_UTK, t=FUTILE=FUTILE_R
  - BIOM = 2*FERM (atp balance)
  - GK = 3*FERM - GLN_UTK
  - Mode a: fermentation from glucose (EX_glc<0, EX_eth>0, EX_co2>0, EX_biom>0)
  - Mode b: glutamine substitution (EX_gln<0, EX_glc>0; invisible if gln=glc net)
    Actually EX_glc=+b, EX_gln=-b → net exchange visible.
  - Mode t: futile cycle (zero exchange)
Expected: n_throughput=2, n_cycle=1, total=3.
"""

import numpy as np
import pytest

from sigmaflux.ai.boundary_basis import (
    compute_boundary_basis,
    normalize_externals,
    compute_cycle_block,
    select_throughput_coordinates,
)


# ---------------------------------------------------------------------------
# Toy network
# ---------------------------------------------------------------------------

def _toy_network():
    """
    Rows (metabolites): glc_e, eth_e, co2_e, biom_e, gln_e, atp, adp, glc, pyr
    Cols (reactions):   EX_glc(0), EX_eth(1), EX_co2(2), EX_gln(3), EX_biom(4),
                        GK(5), FERM(6), BIOM(7), GLN_UTK(8), FUTILE(9), FUTILE_R(10)
    """
    S = np.array([
        # 0    1    2    3    4    5    6    7    8    9   10
        [-1,   0,   0,   0,   0,  -1,   0,   0,   0,   0,   0],  # glc_e
        [ 0,  -1,   0,   0,   0,   0,   2,   0,   0,   0,   0],  # eth_e
        [ 0,   0,  -1,   0,   0,   0,   2,   0,   0,   0,   0],  # co2_e
        [ 0,   0,   0,   0,  -1,   0,   0,   1,   0,   0,   0],  # biom_e
        [ 0,   0,   0,  -1,   0,   0,   0,   0,  -1,   0,   0],  # gln_e
        [ 0,   0,   0,   0,   0,   0,   2,  -1,   0,  -1,   1],  # atp
        [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  -1],  # adp
        [ 0,   0,   0,   0,   0,   1,  -1,  -1,   1,   0,   0],  # glc
        [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # pyr (unused)
    ], dtype=float)

    rxn_ids = ["EX_glc", "EX_eth", "EX_co2", "EX_gln", "EX_biom",
               "GK", "FERM", "BIOM", "GLN_UTK", "FUTILE", "FUTILE_R"]
    # EX_glc(0) and EX_gln(3) are reversible; all others irreversible.
    rev = np.array([True, False, False, True, False,
                    False, False, False, False, False, False], dtype=bool)
    # First 5 reactions are exchange reactions.
    ext = np.array([True, True, True, True, True,
                    False, False, False, False, False, False], dtype=bool)
    return S, rev, ext, rxn_ids


@pytest.fixture
def toy():
    return _toy_network()


@pytest.fixture
def toy_basis(toy):
    S, rev, ext, rxn_ids = toy
    Z, status = compute_boundary_basis((S, rev, ext), solver="glpk")
    return Z, status, S, rev, ext, rxn_ids


# ---------------------------------------------------------------------------
# Hard-assertion tests
# ---------------------------------------------------------------------------

class TestHardAssertions:
    def test_mass_balance(self, toy_basis):
        Z, status, S, *_ = toy_basis
        residual = np.linalg.norm(S @ Z)
        assert residual < 1e-6 * max(Z.shape[1], 1), f"residual={residual:.2e}"

    def test_full_rank(self, toy_basis):
        Z, status, S, *_ = toy_basis
        expected = S.shape[1] - np.linalg.matrix_rank(S)
        actual = np.linalg.matrix_rank(Z)
        assert actual == expected, f"rank(Z)={actual}, expected nullity={expected}"

    def test_shape(self, toy_basis):
        Z, status, S, *_ = toy_basis
        nullity = S.shape[1] - np.linalg.matrix_rank(S)
        assert Z.shape == (S.shape[1], nullity)

    def test_block_sum(self, toy_basis):
        Z, status, S, *_ = toy_basis
        nullity = S.shape[1] - np.linalg.matrix_rank(S)
        assert status["n_throughput"] + status["n_cycle"] == nullity

    def test_reproducibility(self, toy):
        S, rev, ext, _ = toy
        Z1, _ = compute_boundary_basis((S, rev, ext), solver="glpk")
        Z2, _ = compute_boundary_basis((S, rev, ext), solver="glpk")
        np.testing.assert_allclose(Z1, Z2, atol=1e-9)

    def test_throughput_cycle_independence(self, toy_basis):
        Z, status, *_ = toy_basis
        p = status["n_throughput"]
        if p == 0 or status["n_cycle"] == 0:
            pytest.skip("need both blocks")
        cross = Z[:, :p].T @ Z[:, p:]
        assert np.linalg.norm(cross) < 1e-6

    def test_validation_flags(self, toy_basis):
        _, status, *_ = toy_basis
        assert status["validation"]["mass_balance_ok"]

    def test_fallback_always_false(self, toy_basis):
        _, status, *_ = toy_basis
        assert status["fallback"] is False


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------

class TestBehavioral:
    def test_futile_cycle_in_cycle_block(self, toy_basis):
        Z, status, S, rev, ext, rxn_ids = toy_basis
        p = status["n_throughput"]
        futile_idx   = rxn_ids.index("FUTILE")
        futile_r_idx = rxn_ids.index("FUTILE_R")
        # FUTILE / FUTILE_R must be absent from throughput columns.
        T = Z[:, :p]
        assert np.all(np.abs(T[futile_idx,  :]) < 1e-6), "FUTILE in throughput"
        assert np.all(np.abs(T[futile_r_idx, :]) < 1e-6), "FUTILE_R in throughput"
        # And present in at least one cycle column.
        C = Z[:, p:]
        assert np.any(np.abs(C[futile_idx,  :]) > 1e-6), "FUTILE absent from cycle"
        assert np.any(np.abs(C[futile_r_idx, :]) > 1e-6), "FUTILE_R absent from cycle"

    def test_expected_block_sizes(self, toy_basis):
        # From null-space analysis: 2 throughput + 1 cycle = 3 total.
        _, status, *_ = toy_basis
        assert status["n_throughput"] == 2
        assert status["n_cycle"] == 1

    def test_throughput_has_production_coordinate(self, toy_basis):
        _, status, *_ = toy_basis
        prod = [m for m in status["throughput_meta"] if m["kind"] == "production"]
        assert prod, "Expected at least one production coordinate"

    def test_throughput_has_substitution_coordinate(self, toy_basis):
        _, status, *_ = toy_basis
        sub = [m for m in status["throughput_meta"] if m["kind"] == "substitution"]
        assert sub, "Expected at least one substitution coordinate"

    def test_ferm_in_fermentation_coordinate(self, toy_basis):
        Z, status, S, rev, ext, rxn_ids = toy_basis
        ferm_idx = rxn_ids.index("FERM")
        # At least one throughput column must use the FERM reaction.
        T = Z[:, :status["n_throughput"]]
        assert np.any(np.abs(T[ferm_idx, :]) > 1e-6), \
            "FERM absent from all throughput columns"

    def test_forward_infeasible_flagged(self, toy):
        """Forcing a secretion-only external (no uptake possible) to be an
        uptake coordinate should flag forward_infeasible rather than crash."""
        S, rev, ext, rxn_ids = toy
        # Make EX_glc strictly irreversible secretion-only (can only secrete glc_e).
        # With GK consuming glc_e, there is no way to uptake glc — but all internal
        # fluxes need glc uptake. This forces backward flux through GK (irrev).
        rev_mod = rev.copy()
        rev_mod[0] = False  # EX_glc now irrev: only secretion direction (v>0)
        Z, status = compute_boundary_basis((S, rev_mod, ext), solver="glpk")
        infeasible = [m for m in status["throughput_meta"]
                      if not m["forward_feasible"]]
        assert infeasible, "Expected at least one forward-infeasible coordinate"


# ---------------------------------------------------------------------------
# normalize_externals unit tests
# ---------------------------------------------------------------------------

class TestNormalizeExternals:
    def test_split_count(self, toy):
        S, rev, ext, rxn_ids = toy
        S_aug, *_, split_map, orig_to_aug = normalize_externals(S, rev, ext, rxn_ids)
        # EX_glc (0) and EX_gln (3) are reversible externals → 2 splits.
        assert len(split_map) == 2

    def test_aug_shape(self, toy):
        S, rev, ext, rxn_ids = toy
        S_aug, rev_aug, ext_aug, ids_aug, split_map, orig_to_aug = \
            normalize_externals(S, rev, ext, rxn_ids)
        n_rev_ext = 2
        assert S_aug.shape == (S.shape[0], S.shape[1] + n_rev_ext)

    def test_split_legs_are_irreversible(self, toy):
        S, rev, ext, rxn_ids = toy
        _, rev_aug, _, _, split_map, _ = normalize_externals(S, rev, ext, rxn_ids)
        for j_orig, (j_fwd, j_rev) in split_map.items():
            assert not rev_aug[j_fwd]
            assert not rev_aug[j_rev]

    def test_fwd_rev_sum_to_zero(self, toy):
        """fwd and rev legs should have opposite S columns (sum to zero)."""
        S, rev, ext, rxn_ids = toy
        S_aug, *_, split_map, _ = normalize_externals(S, rev, ext, rxn_ids)
        for j_orig, (j_fwd, j_rev) in split_map.items():
            np.testing.assert_allclose(
                S_aug[:, j_fwd] + S_aug[:, j_rev], 0.0, atol=1e-12
            )


# ---------------------------------------------------------------------------
# compute_cycle_block unit tests
# ---------------------------------------------------------------------------

class TestCycleBlock:
    def _augmented(self, toy):
        S, rev, ext, rxn_ids = toy
        return normalize_externals(S, rev, ext, rxn_ids)

    def test_cycle_in_null_space(self, toy):
        S_aug, _, ext_aug, *_ = self._augmented(toy)
        C, q = compute_cycle_block(S_aug, ext_aug)
        if q > 0:
            assert np.linalg.norm(S_aug @ C) < 1e-8 * q

    def test_cycle_invisible_at_boundary(self, toy):
        S_aug, _, ext_aug, *_ = self._augmented(toy)
        C, q = compute_cycle_block(S_aug, ext_aug)
        ext_rows = np.where(ext_aug)[0]
        if q > 0 and len(ext_rows) > 0:
            assert np.linalg.norm(C[ext_rows, :]) < 1e-8 * q

    def test_cycle_orthonormal(self, toy):
        S_aug, _, ext_aug, *_ = self._augmented(toy)
        C, q = compute_cycle_block(S_aug, ext_aug)
        if q > 0:
            np.testing.assert_allclose(C.T @ C, np.eye(q), atol=1e-10)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_accepts_raw_bool_ext_mask(self, toy):
        S, rev, ext, _ = toy
        Z, status = compute_boundary_basis((S, rev, ext), solver="glpk")
        assert Z.shape[0] == S.shape[1]

    def test_glpk_default_solver(self, toy):
        S, rev, ext, _ = toy
        _, status = compute_boundary_basis((S, rev, ext))
        assert status["solver_used"] == "glpk"

    def test_sparse_S_input(self, toy):
        import scipy.sparse
        S, rev, ext, _ = toy
        Z, _ = compute_boundary_basis((scipy.sparse.csr_matrix(S), rev, ext), solver="glpk")
        assert Z.shape[0] == S.shape[1]

    def test_no_exchange_reactions(self):
        # All DOF lands in the cycle block when there are no externals.
        S = np.array([[1, -1, 0], [0, 1, -1]], dtype=float)
        rev = np.array([False, False, True])
        ext = np.array([False, False, False])
        Z, status = compute_boundary_basis((S, rev, ext), solver="glpk")
        assert status["n_throughput"] == 0
        nullity = S.shape[1] - np.linalg.matrix_rank(S)
        assert status["n_cycle"] == nullity

    def test_single_throughput(self):
        # Minimal open network with explicit extracellular pool:
        #   EX_A (rev exchange): "A_e → ∅" (COBRA: v<0 = uptake of A_e)
        #   GK   (irrev):        A_e → A_in
        #   INT  (irrev):        A_in → B_e
        #   EX_B (irrev):        "B_e → ∅" (secretion of product)
        #
        # S (3 metabolites × 4 reactions):
        #         EX_A  GK   INT  EX_B
        # A_e   [ -1   -1    0    0  ]
        # A_in  [  0   +1   -1    0  ]
        # B_e   [  0    0   +1   -1  ]
        #
        # rank=3, nullity=1; null vector = (-1,1,1,1) (EX_A uptake drives chain).
        S = np.array([
            [-1, -1,  0,  0],   # A_e
            [ 0,  1, -1,  0],   # A_in
            [ 0,  0,  1, -1],   # B_e
        ], dtype=float)
        rev = np.array([True, False, False, False])  # EX_A reversible, rest irrev
        ext = np.array([True, False, False, True])   # EX_A and EX_B are exchanges
        Z, status = compute_boundary_basis((S, rev, ext), solver="glpk")
        assert Z.shape == (4, 1)
        assert status["n_throughput"] == 1
        assert status["n_cycle"] == 0
        assert np.linalg.norm(S @ Z) < 1e-8
