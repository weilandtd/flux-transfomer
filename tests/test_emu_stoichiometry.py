"""EMU stoichiometry weights: symmetry scrambling and stoichiometric coefficients.

Both properties are invisible in the usual case -- a reaction moving one copy of an
unsymmetric metabolite weights its edge by 1 either way -- so they are pinned here on
purpose-built two-route models where the correct MID is known by hand.

The coefficient case is a real bug this file was written to close: ``get_subnets``
used to reduce every coefficient to its sign, so ``CBPS2_m`` (two ATP, two ADP, each
named once in the atom map) delivered half the label it should. It was found by
porting the T-cell model to x3cflux, which cannot express a coefficient at all and
forced every one of them to be written out explicitly.
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

def test_coefficient_magnitude_sets_the_mixing_ratio(tmp_path):
    """A reaction moving two copies delivers twice the label of one moving one.

    ``Y`` is fed by two routes carrying *equal flux*: a labeled one with coefficient
    2 whose atom map names the metabolite once (the ``CBPS2_m`` shape) and an
    unlabeled one with coefficient 1. Y therefore draws 20 labeled and 10 unlabeled
    carbons, so it must come out 2/3 labeled. Weighting both edges by 1 gives 1/2,
    which is what this test exists to catch.
    """
    model = cobra.Model("coefficient")
    labeled_in, unlabeled_in, labeled, unlabeled, product = (
        _metabolite(i, "ab") for i in ("L_in", "U_in", "L", "U", "Y"))
    model.add_metabolites([labeled_in, unlabeled_in, labeled, unlabeled, product])

    _reaction(model, "EX_L_in", {labeled_in: 1}, " --> L_in (ab)")
    _reaction(model, "EX_U_in", {unlabeled_in: 1}, " --> U_in (ab)")
    _reaction(model, "in_L", {labeled_in: -1, labeled: 1}, "L_in (ab) -> L (ab)")
    _reaction(model, "in_U", {unlabeled_in: -1, unlabeled: 1}, "U_in (ab) -> U (ab)")
    # Coefficient 2, but the atom map names each metabolite only once.
    _reaction(model, "R2", {labeled: -2, product: 2}, "L (ab) -> Y (ab)")
    _reaction(model, "R1", {unlabeled: -1, product: 1}, "U (ab) -> Y (ab)")
    _reaction(model, "EX_Y", {product: -1}, "Y (ab) --> ")

    mids = _simulate(
        model,
        {"EX_L_in": 20., "in_L": 20., "EX_U_in": 10., "in_U": 10.,
         "R2": 10., "R1": 10., "EX_Y": 30.},
        {"L": 2, "U": 2, "Y": 2}, "L_in", [1, 1], tmp_path,
    )

    assert mids["Y"] == pytest.approx([1 / 3, 0.0, 2 / 3], abs=1e-6)


def test_repeated_name_and_coefficient_agree(tmp_path):
    """Naming the metabolite twice for coefficient 2 is the same as naming it once.

    ``X -> Y (ab) + Y (cd)`` splits into two edges of weight 1; ``X -> Y`` at
    coefficient 2 is one edge of weight 2. Both deliver 2 copies per unit flux, so
    the two spellings of the same chemistry must give the same MID -- which is the
    invariant that keeps the fix from disturbing the reactions already written out.
    """
    def build(spelled_out):
        model = cobra.Model("repeat")
        source_in, source, product = (_metabolite("S_in", "abcd"),
                                      _metabolite("S", "abcd"), _metabolite("Y", "ab"))
        model.add_metabolites([source_in, source, product])
        _reaction(model, "EX_S_in", {source_in: 1}, " --> S_in (abcd)")
        _reaction(model, "in_S", {source_in: -1, source: 1}, "S_in (abcd) -> S (abcd)")
        _reaction(model, "split", {source: -1, product: 2},
                  "S (abcd) -> Y (ab) + Y (cd)" if spelled_out else "S (abcd) -> Y (ab)")
        _reaction(model, "EX_Y", {product: -1}, "Y (ab) --> ")
        return model

    flux = {"EX_S_in": 10., "in_S": 10., "split": 10., "EX_Y": 20.}
    # C1 and C2 labeled: the two halves of S differ, so a wrong edge weight shows up.
    spelled = _simulate(build(True), flux, {"Y": 2}, "S_in", [1, 1, 0, 0], tmp_path)
    implied = _simulate(build(False), flux, {"Y": 2}, "S_in", [1, 1, 0, 0], tmp_path)

    assert spelled["Y"] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)
    # The implied spelling maps only the first copy, so both copies inherit it.
    assert implied["Y"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


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
