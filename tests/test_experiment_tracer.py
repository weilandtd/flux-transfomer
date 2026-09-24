"""Boundary MIDs for positional tracers.

Regression tests for a bug that made the label count of a boundary EMU depend
only on its *size*, never on which carbons it covered: `calculate_tracer_input`
indexed `tracer.labeling` with `EMU.ix` (0-based sequence positions) where it
meant `EMU.mid` (1-based carbon numbers), and then subtracted a further 1, so the
first entry read `labeling[-1]`.

Uniformly labeled tracers are blind to it -- every carbon carries the same label,
so any index set gives the same count -- which is why the existing suite never
caught it. Only positional tracers such as 1,2-13C glucose expose it, and those
are exactly the tracers chosen to resolve a pathway split.
"""

import os
import sys

import cobra
import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.core.experiment import Tracer, calculate_tracer_input  # noqa: E402
from sigmaflux.expansion.emu import EMU  # noqa: E402


def _emu(mid, n_atoms=6):
    """A boundary EMU on 1-based carbons ``mid`` of an n-carbon metabolite."""
    met = cobra.Metabolite("glc__D_e")
    met.annotation["atom_pattern"] = "abcdef"[:n_atoms]
    return EMU(met, mid)


def _labeled_weight(mid_tensor):
    """The single mass shift carrying all the weight, i.e. M+k."""
    nonzero = torch.nonzero(mid_tensor).flatten().tolist()
    assert len(nonzero) == 1, f"expected a pure mass isotopomer, got {mid_tensor}"
    return nonzero[0]


@pytest.mark.parametrize("mid, expected", [
    ([1, 2], 2),        # both carbons labeled
    ([2, 3], 1),        # only C2 labeled
    ([5, 6], 0),        # neither labeled
    ([1, 2, 3], 2),
    ([4, 5, 6], 0),
    ([3, 4, 5, 6], 0),
    ([1, 2, 3, 4, 5, 6], 2),
])
def test_positional_tracer_counts_its_own_carbons(mid, expected):
    """1,2-13C glucose: the count must follow the EMU's carbons, not its size."""
    tracer = Tracer("glc__D_e", 1.0, [1, 1, 0, 0, 0, 0])
    got = calculate_tracer_input([tracer], _emu(mid))
    assert _labeled_weight(got) == expected
    assert got.shape[0] == len(mid) + 1


def test_same_size_different_carbons_differ():
    """The bug's signature: equal-size EMUs all got the same answer."""
    tracer = Tracer("glc__D_e", 1.0, [1, 1, 0, 0, 0, 0])
    labeled = _labeled_weight(calculate_tracer_input([tracer], _emu([1, 2])))
    unlabeled = _labeled_weight(calculate_tracer_input([tracer], _emu([5, 6])))
    assert labeled != unlabeled, "positional information is being discarded"


@pytest.mark.parametrize("mid", [[1], [1, 2], [4, 5, 6], [1, 2, 3, 4, 5, 6]])
def test_uniform_tracer_labels_every_carbon(mid):
    """U-13C is size-only by construction, and must stay unchanged."""
    tracer = Tracer("glc__D_e", 1.0, [1] * 6)
    got = calculate_tracer_input([tracer], _emu(mid))
    assert _labeled_weight(got) == len(mid)


def test_partial_enrichment_splits_between_m0_and_mk():
    """Enrichment below 1 leaves the balance at M+0."""
    tracer = Tracer("glc__D_e", 0.75, [1, 1, 0, 0, 0, 0])
    got = calculate_tracer_input([tracer], _emu([1, 2]))
    assert got[0].item() == pytest.approx(0.25)
    assert got[2].item() == pytest.approx(0.75)
    assert got.sum().item() == pytest.approx(1.0)
