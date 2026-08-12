"""EMU edge construction: repeated atom-map copies, and symmetry scrambling.

Both are invisible in the usual case -- one copy of an unsymmetric metabolite gives
one edge of weight 1 either way -- so they are pinned here on purpose-built
two-route models where the correct MID is known by hand.

The multiplicity rule is the one this file was written to close. A reaction moving
several molecules of the same metabolite spells each one out in the atom map with
its own pattern (``atp (P1) + amp (P2) -> adp (P1) + adp (P2)``), and every named
copy becomes its own EMU edge. ``get_origin_emus`` used to keep only the first,
dropping the AMP-derived route entirely. Inferring the missing copies from the
stoichiometric coefficient instead is *not* the fix: it guesses which atoms they
carry, and on the T-cell model that guess moved the adenine-nucleotide pool 0.38
MID units away from x3cflux. An under-specified map is warned about, not repaired.
"""

import cobra
import numpy as np
import pandas as pd
import pytest
import torch

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


def _metabolite(met_id, atom_pattern, symmetries=None):
    met = cobra.Metabolite(met_id)
    met.annotation = {"atom_pattern": atom_pattern}
    if symmetries:
        met.annotation["symmetries"] = symmetries
    return met


def _reaction(model, rxn_id, stoichiometry, atom_mapping):
    rxn = cobra.Reaction(rxn_id)
    rxn.lower_bound, rxn.upper_bound = 0, 1000
    model.add_reactions([rxn])
    rxn.add_metabolites(stoichiometry)
    rxn.annotation = {"atom_mapping": atom_mapping}
    return rxn


def _simulate(model, fluxes, targets, tracer_id, labeling, tmp_path):
    """MIDs at *fluxes*, as ``{metabolite_id: array}`` indexed M+0..M+n.

    ``prepare`` can drop ``dAs.pkl``/``dBs.pkl`` into the working directory, so the
    caller's ``tmp_path`` is made current first (see ``tests/conftest.py``).
    """
    mfa_model = MFAModel(model, "test", initial_flux_distribution=pd.Series(fluxes))
    target_emus = {
        EMU(mfa_model.metabolites.get_by_id(met_id), list(range(1, size + 1)))
        for met_id, size in targets.items()
    }
    tracer = Tracer(mfa_model.metabolites.get_by_id(tracer_id), 1.0, labeling)
    experiment = Experiment(mid_data=MassIsotopomerData(), tracers=[tracer],
                            fluxes=None, enrichment=None)
    cwd = tmp_path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        mfa_model.prepare([experiment], targets=target_emus, with_backward=False)
        with torch.no_grad():
            simulated = mfa_model().detach().cpu().numpy()
    finally:
        import os
        os.chdir(cwd)
    return {met_id: simulated[:size + 1, mfa_model._metabolite_hash[met_id]]
            for met_id, size in targets.items()}


# ── Stoichiometric coefficients ───────────────────────────────────────────────

def test_each_mapped_copy_becomes_its_own_edge(tmp_path):
    """Multiplicity lives in the atom map, one named copy per molecule.

    ``ADK1_c`` is written ``atp (P1) + amp (P2) -> adp (P1) + adp (P2)``: the two
    product molecules are named separately, each carrying the pattern of the educt
    it came from. Both must become EMU edges of weight 1. Keeping only the first --
    which is what ``next()`` in ``get_origin_emus`` used to do -- silently discards
    the entire AMP-derived route, and the pool comes out unlabeled.

    Here ``S`` is split into two halves that differ in labeling, so dropping either
    edge is visible: with both, Y is an even mix of the labeled and unlabeled half.
    """
    model = cobra.Model("copies")
    source_in, source, product = (_metabolite("S_in", "abcd"),
                                  _metabolite("S", "abcd"), _metabolite("Y", "ab"))
    model.add_metabolites([source_in, source, product])
    _reaction(model, "EX_S_in", {source_in: 1}, " --> S_in (abcd)")
    _reaction(model, "in_S", {source_in: -1, source: 1}, "S_in (abcd) -> S (abcd)")
    _reaction(model, "split", {source: -1, product: 2},
              "S (abcd) -> Y (ab) + Y (cd)")
    _reaction(model, "EX_Y", {product: -1}, "Y (ab) --> ")

    # C1 and C2 labeled: the two halves of S differ, so a dropped edge shows up.
    mids = _simulate(model, {"EX_S_in": 10., "in_S": 10., "split": 10., "EX_Y": 20.},
                     {"Y": 2}, "S_in", [1, 1, 0, 0], tmp_path)
    assert mids["Y"] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)


def test_underspecified_map_warns_and_is_not_reweighted(tmp_path):
    """A coefficient the atom map does not spell out is reported, not guessed.

    ``CBPS2_m`` moves two ATP and two ADP while naming one of each. The copies it
    does not name could carry any atoms, so inferring them from the coefficient is
    a guess -- and an invisible one. Weighting the edge by the coefficient instead
    of warning moved the whole adenine-nucleotide pool by 0.38 MID units away from
    x3cflux, which is what this test exists to prevent.

    Only the mapped copy carries label, so Y here is a 50/50 mix of the two routes
    even though ``R2`` moves two molecules -- and the warning says so.
    """
    model = cobra.Model("underspecified")
    labeled_in, unlabeled_in, labeled, unlabeled, product = (
        _metabolite(i, "ab") for i in ("L_in", "U_in", "L", "U", "Y"))
    model.add_metabolites([labeled_in, unlabeled_in, labeled, unlabeled, product])
    _reaction(model, "EX_L_in", {labeled_in: 1}, " --> L_in (ab)")
    _reaction(model, "EX_U_in", {unlabeled_in: 1}, " --> U_in (ab)")
    _reaction(model, "in_L", {labeled_in: -1, labeled: 1}, "L_in (ab) -> L (ab)")
    _reaction(model, "in_U", {unlabeled_in: -1, unlabeled: 1}, "U_in (ab) -> U (ab)")
    # Coefficient 2, but only one mapped copy -- the shape CBPS2_m has.
    _reaction(model, "R2", {labeled: -2, product: 2}, "L (ab) -> Y (ab)")
    _reaction(model, "R1", {unlabeled: -1, product: 1}, "U (ab) -> Y (ab)")
    _reaction(model, "EX_Y", {product: -1}, "Y (ab) --> ")

    flux = {"EX_L_in": 20., "in_L": 20., "EX_U_in": 10., "in_U": 10.,
            "R2": 10., "R1": 10., "EX_Y": 30.}
    with pytest.warns(RuntimeWarning, match="atom map names Y 1 time"):
        mids = _simulate(model, flux, {"L": 2, "U": 2, "Y": 2}, "L_in", [1, 1], tmp_path)

    assert mids["Y"] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)


# ── Symmetry ──────────────────────────────────────────────────────────────────

def _symmetric_model(symmetric):
    """``A -> F -> {P+Q, R+S}``, with ``F`` optionally carrying a ``dcba`` symmetry.

    Two consumers, chosen so one fragment is scrambled by the symmetry and the other
    is not: ``P`` takes ``F``'s carbons 1-2, which the flip sends to 4-3, while ``R``
    takes carbons 1 and 4, which the flip maps onto itself.
    """
    model = cobra.Model("symmetry")
    source_in, source = _metabolite("A_in", "abcd"), _metabolite("A", "abcd")
    middle = _metabolite("F", "abcd", ["abcd.dcba"] if symmetric else None)
    fragments = [_metabolite(i, "ab") for i in ("P", "Q", "R", "S")]
    model.add_metabolites([source_in, source, middle, *fragments])

    _reaction(model, "EX_A_in", {source_in: 1}, " --> A_in (abcd)")
    _reaction(model, "in_A", {source_in: -1, source: 1}, "A_in (abcd) -> A (abcd)")
    _reaction(model, "AtoF", {source: -1, middle: 1}, "A (abcd) -> F (abcd)")
    _reaction(model, "F_split12", {middle: -1, fragments[0]: 1, fragments[1]: 1},
              "F (abcd) -> P (ab) + Q (cd)")
    _reaction(model, "F_split14", {middle: -1, fragments[2]: 1, fragments[3]: 1},
              "F (abcd) -> R (ad) + S (bc)")
    for fragment in fragments:
        _reaction(model, f"EX_{fragment.id}", {fragment: -1}, f"{fragment.id} (ab) --> ")
    return model


SYMMETRY_FLUX = {"EX_A_in": 10., "in_A": 10., "AtoF": 10., "F_split12": 5.,
                 "F_split14": 5., "EX_P": 5., "EX_Q": 5., "EX_R": 5., "EX_S": 5.}
SYMMETRY_TARGETS = {"F": 4, "P": 2, "R": 2}


def test_symmetry_scrambles_only_the_orientation_sensitive_fragment(tmp_path):
    """A symmetric pool splits label 50/50 across orientations, and only there.

    With C1 and C2 labeled, ``P`` (carbons 1-2) must go from a pure M+2 to an even
    M+0/M+2 mix once ``F`` is symmetric. ``R`` (carbons 1 and 4) must not move: the
    flip exchanges its two carbons, and one labeled plus one unlabeled is M+1 either
    way. ``F`` itself never moves -- a symmetry cannot change a whole molecule's mass.
    """
    plain = _simulate(_symmetric_model(False), SYMMETRY_FLUX, SYMMETRY_TARGETS,
                      "A_in", [1, 1, 0, 0], tmp_path)
    symmetric = _simulate(_symmetric_model(True), SYMMETRY_FLUX, SYMMETRY_TARGETS,
                          "A_in", [1, 1, 0, 0], tmp_path)

    assert plain["P"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert symmetric["P"] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)

    assert plain["R"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert symmetric["R"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    for mids in (plain, symmetric):
        assert mids["F"] == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0], abs=1e-6)


@pytest.mark.parametrize("symmetric", [False, True])
def test_mids_sum_to_one(symmetric, tmp_path):
    """Every simulated MID normalises, symmetric pools included.

    The two orientations of a symmetry-*invariant* fragment collapse to one origin
    set that still carries the 1/(1+n) weight, so its production term looks halved.
    That is harmless -- the balance is a flux-weighted mean, so scaling an edge whole
    cancels -- and this asserts it stays harmless.
    """
    mids = _simulate(_symmetric_model(symmetric), SYMMETRY_FLUX, SYMMETRY_TARGETS,
                     "A_in", [1, 1, 0, 0], tmp_path)
    for met_id, values in mids.items():
        assert np.sum(values) == pytest.approx(1.0, abs=1e-6), met_id
