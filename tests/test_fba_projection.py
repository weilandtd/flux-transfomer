"""
Tests for FBAProProjection and FBAproFixedProjection.

Both projections must satisfy Sv = 0.  FBAproFixedProjection must
additionally preserve v[H] = v0[H] for the fixed reaction indices.

Requirement for FBAproFixedProjection: rank(S[:, free]) = rank(S).
If fixed reactions reduce that rank the system is under-determined and
the projection cannot satisfy the stoichiometric constraint exactly.
"""

import numpy as np
import pytest
import torch

from sigmaflux.ai.model import FBAProProjection, FBAproFixedProjection

# ── shared test network ────────────────────────────────────────────────────────
#   3 metabolites, 5 reactions, rank(S) = 3
S_TOY = np.array(
    [[ 1, -1,  0,  0,  1],
     [ 0,  1, -1,  1,  0],
     [ 0,  0,  1, -1, -1]],
    dtype=float,
)

# Fixed-index sets where rank(S[:, free]) = rank(S) = 3
VALID_FIXED = [
    pytest.param([4],    id="H={4}"),
    pytest.param([2, 4], id="H={2,4}"),
    pytest.param([3, 4], id="H={3,4}"),
]

BATCH   = 16
ATOL_SV = 1e-4   # max |Sv| to accept as "in ker(S)"
ATOL_FP = 1e-5   # max deviation for fixed reactions


@pytest.fixture(scope="module")
def random_v():
    torch.manual_seed(0)
    return torch.randn(BATCH, S_TOY.shape[1])


# ── FBAProProjection ──────────────────────────────────────────────────────────

def test_fbapro_stoichiometry(random_v):
    proj = FBAProProjection(S_TOY)
    out  = proj(random_v)
    St   = torch.tensor(S_TOY, dtype=torch.float32)
    sv   = (St @ out.T).abs().max().item()
    assert sv < ATOL_SV, f"|Sv| = {sv:.3e} exceeds tolerance"


# ── FBAproFixedProjection ─────────────────────────────────────────────────────

@pytest.mark.parametrize("fixed_idx", VALID_FIXED)
def test_fbaprofixed_stoichiometry(random_v, fixed_idx):
    proj = FBAproFixedProjection(S_TOY, fixed_idx)
    out  = proj(random_v)
    St   = torch.tensor(S_TOY, dtype=torch.float32)
    sv   = (St @ out.T).abs().max().item()
    assert sv < ATOL_SV, f"H={fixed_idx}  |Sv| = {sv:.3e} exceeds tolerance"


@pytest.mark.parametrize("fixed_idx", VALID_FIXED)
def test_fbaprofixed_preserves_fixed_reactions(random_v, fixed_idx):
    proj = FBAproFixedProjection(S_TOY, fixed_idx)
    out  = proj(random_v)
    fp   = (out[:, fixed_idx] - random_v[:, fixed_idx]).abs().max().item()
    assert fp < ATOL_FP, f"H={fixed_idx}  max|out_H - v0_H| = {fp:.3e}"


def test_fbaprofixed_empty_matches_fbapro(random_v):
    """H=∅ should give the same result as FBAProProjection (within float32 error)."""
    proj_base  = FBAProProjection(S_TOY)
    proj_fixed = FBAproFixedProjection(S_TOY, [])
    diff = (proj_base(random_v) - proj_fixed(random_v)).abs().max().item()
    assert diff < 1e-5, f"H=∅ mismatch vs FBAProProjection: {diff:.3e}"
