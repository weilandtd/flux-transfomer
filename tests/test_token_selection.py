"""
Characterize the metabolite token selection procedure from _train_epoch.

Verifies that top-K selection picks the highest-scoring tokens deterministically
and correctly handles the fallback path when the window covers all tokens.
"""

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

_IO_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "paper", ".results",
    "i_orientalis_20260611", "imisor_transformer.pt",
)
_IO_TOKEN_WINDOW = 512


def _select_met_tokens(n_all_met, token_information, n_met_window):
    """Mirror of the top-K selection block in _train_epoch."""
    if token_information is not None and n_met_window < n_all_met:
        scores = token_information
        return torch.topk(scores, n_met_window, largest=True, sorted=False).indices
    return torch.arange(n_all_met)


# ── synthetic tests ────────────────────────────────────────────────────────────

def test_topk_selects_highest_scoring_tokens():
    """Top-K must select exactly the n_met_window highest-scoring tokens."""
    n_tokens     = 60
    token_window = 15

    token_information = torch.rand(n_tokens)
    met_sel = _select_met_tokens(n_tokens, token_information, token_window)

    assert met_sel.shape[0] == token_window

    expected_top = torch.topk(token_information, token_window, largest=True).indices
    assert set(met_sel.tolist()) == set(expected_top.tolist())


def test_topk_is_deterministic():
    """Calling top-K twice with the same scores must return the same token set."""
    n_tokens          = 60
    token_window      = 15
    token_information = torch.rand(n_tokens)

    sel_a = _select_met_tokens(n_tokens, token_information, token_window)
    sel_b = _select_met_tokens(n_tokens, token_information, token_window)

    assert set(sel_a.tolist()) == set(sel_b.tolist())


def test_topk_fallback_returns_all_tokens():
    """When n_met_window >= n_all_met, all token indices are returned."""
    n_tokens          = 20
    token_information = torch.rand(n_tokens)

    # window larger than pool
    met_sel = _select_met_tokens(n_tokens, token_information, n_tokens)
    assert set(met_sel.tolist()) == set(range(n_tokens))

    # no scores provided
    met_sel_none = _select_met_tokens(n_tokens, None, n_tokens // 2)
    assert set(met_sel_none.tolist()) == set(range(n_tokens))


def test_topk_high_weight_metabolites_always_selected():
    """Tokens in the top-K by score must always appear in the selection."""
    n_metabolites = 20
    n_isotopomers = 3
    n_tokens      = n_metabolites * n_isotopomers   # 60
    token_window  = 15

    token_information = torch.ones(n_tokens)
    token_information[: 5 * n_isotopomers] = 5.0   # first 5 metabolites × 5×

    met_sel = _select_met_tokens(n_tokens, token_information, token_window)

    # The 15 highest-scoring tokens are all from the first 5 metabolites
    assert all(idx < 5 * n_isotopomers for idx in met_sel.tolist())


# ── I. orientalis real-data test ───────────────────────────────────────────────

import pytest

@pytest.mark.skipif(not os.path.exists(_IO_CKPT), reason="I. orientalis checkpoint not found")
def test_topk_io_selects_correct_tokens():
    """
    With the real I. orientalis weights, top-K must select exactly the
    _IO_TOKEN_WINDOW highest-scoring tokens.
    """
    ckpt = torch.load(_IO_CKPT, map_location="cpu", weights_only=False)

    token_information = torch.tensor(ckpt["token_information"], dtype=torch.float32)
    n_tokens          = len(token_information)
    token_window      = _IO_TOKEN_WINDOW

    met_sel = _select_met_tokens(n_tokens, token_information, token_window)

    assert met_sel.shape[0] == token_window

    expected_top = torch.topk(token_information, token_window, largest=True).indices
    assert set(met_sel.tolist()) == set(expected_top.tolist())
