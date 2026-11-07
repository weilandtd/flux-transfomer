"""Natural-isotope correction of GC/MS fragment MIDs.

A raw fragment MID carries the natural isotopes of every atom in the ion -- the derivative's
silicon, hydrogen, oxygen and extra carbons, plus natural 13C on backbone carbons the
tracer did not label. The model simulates only tracer label, with unlabeled carbon as pure
12C, so the measurement is corrected the classical way: ``raw = C @ x``, where column ``j``
of ``C`` is the mass distribution of an ion whose backbone holds ``j`` tracer labels.
"""

import re

import numpy as np
from scipy.optimize import nnls

#: Natural abundance by mass shift, IUPAC 2013 representative values.
ISOTOPES = {
    "C": [0.9893, 0.0107],
    "H": [0.999885, 0.000115],
    "N": [0.99636, 0.00364],
    "O": [0.99757, 0.00038, 0.00205],
    "Si": [0.92223, 0.04685, 0.03092],
    "S": [0.9499, 0.0075, 0.0425, 0.0, 0.0001],
}


def parse_formula(formula):
    """``{element: count}`` of *formula*, e.g. ``'C10H26NOSi2+'``; charge ignored."""
    counts = {}
    for element, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula.rstrip("+-")):
        if element not in ISOTOPES:
            raise ValueError(f"{formula!r}: no isotope data for {element!r}")
        counts[element] = counts.get(element, 0) + (int(n) if n else 1)
    return counts


def _distribution(counts):
    """Mass-shift distribution of the atoms in *counts*, by repeated convolution."""
    dist = np.array([1.0])
    for element, n in counts.items():
        for _ in range(n):
            dist = np.convolve(dist, ISOTOPES[element])
    return dist


def correction_matrix(formula, n_backbone, n_obs):
    """``(n_obs, n_backbone + 1)`` matrix mapping tracer label to the raw MID.

    Args:
        formula: elemental formula of the whole ion, backbone carbons included.
        n_backbone: carbons of the ion that come from the metabolite backbone.
        n_obs: number of raw isotopomers measured, M+0..M+(n_obs-1).
    """
    counts = parse_formula(formula)
    if counts.get("C", 0) < n_backbone:
        raise ValueError(f"{formula!r} holds fewer than {n_backbone} carbons")
    counts["C"] -= n_backbone
    rest = _distribution(counts)
    matrix = np.zeros((n_obs, n_backbone + 1))
    for j in range(n_backbone + 1):
        column = np.convolve(rest, _distribution({"C": n_backbone - j}))
        column = column[:max(n_obs - j, 0)]
        matrix[j:j + len(column), j] = column
    return matrix


def correct_mid(raw, formula, n_backbone):
    """Tracer-label MID over M+0..M+n_backbone, from a raw fragment MID.

    Solved by non-negative least squares, then renormalized to sum to one.

    Args:
        raw: raw MID, M+0 upward.
        formula: elemental formula of the whole ion.
        n_backbone: backbone carbons the ion retains.
    """
    raw = np.asarray(raw, dtype=float)
    x, _ = nnls(correction_matrix(formula, n_backbone, len(raw)), raw)
    return x / x.sum()
