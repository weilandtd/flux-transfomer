"""Direct unit tests for the EMU convolution/convolution_derivative machinery.

Covers convolution() and convolution_derivative() over OriginSets of every
parity and length (1, 2, 3, 4, 5, 6 member EMUs), including the odd-length
(>=3) reduction path that convolution_derivative() routes back through
convolution(). These construct OriginSets, core_met_ix, emu_states and
derivative_states directly, bypassing the model/experiment pipeline, so they
run fast and pin correctness independently of any particular organism's
network shape.
"""

import functools

import cobra
import numpy as np
import pytest
import torch

from sigmaflux.core.mfa import convolution, convolution_derivative
from sigmaflux.expansion.emu import EMU, OriginSet

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _emu(index):
    """A distinct single-carbon EMU, for building an n-way OriginSet."""
    met = cobra.Metabolite(f"m{index}")
    met.annotation["atom_pattern"] = "a"
    return EMU(met, [1])


def _bases(n):
    """n distinct, non-trivial 2-entry MIDs (M+0/M+1 split), each summing to 1."""
    return [[0.9 - 0.1 * i, 0.1 + 0.1 * i] for i in range(n)]


def _states(n, bases):
    emus = [_emu(i) for i in range(n)]
    core_met_ix = {1: {emu: i for i, emu in enumerate(emus)}}
    emu_states = {1: torch.tensor(bases, dtype=DTYPE)}
    return emus, core_met_ix, emu_states


def _chained_convolve(vectors):
    """Reference joint MID: repeated np.convolve, associative like the real thing."""
    return functools.reduce(np.convolve, vectors)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_convolution_matches_polynomial_product(n):
    """convolution() is polynomial multiplication of the member MIDs, for any n >= 2."""
    bases = _bases(n)
    emus, core_met_ix, emu_states = _states(n, bases)
    origin_set = OriginSet(emus)

    got = convolution(core_met_ix, emu_states, origin_set, DEVICE, DTYPE)

    expected = _chained_convolve(bases)
    assert got.shape == (n + 1,)
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-9)


def test_convolution_singleton_returns_own_mid_unchanged():
    """The len==1 base case: a lone EMU's MID is already the joint distribution."""
    bases = _bases(1)
    emus, core_met_ix, emu_states = _states(1, bases)
    origin_set = OriginSet(emus)

    got = convolution(core_met_ix, emu_states, origin_set, DEVICE, DTYPE)

    np.testing.assert_allclose(got.numpy(), bases[0])


def _derivative_states(n, n_params):
    """n distinct EMUs, each with a fixed [n_params, 2] derivative block, keyed like emu_states."""
    values = [[[0.05 * (i + 1) * (p + 1), -0.03 * (i + 1) * (p + 1)] for p in range(n_params)]
              for i in range(n)]
    return torch.tensor(values, dtype=DTYPE), values


def _manual_product_rule(bases, derivs_by_param):
    """d(convolution(bases))/d(param) via the Leibniz rule, one array per parameter column."""
    n = len(bases)
    results = []
    for derivs in derivs_by_param:
        total = np.zeros(n + 1)
        for i in range(n):
            others = [bases[j] for j in range(n) if j != i]
            product_of_others = _chained_convolve(others) if others else np.array([1.0])
            total = total + np.convolve(derivs[i], product_of_others)
        results.append(total)
    return np.stack(results)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_convolution_derivative_matches_product_rule(n):
    """convolution_derivative() follows the Leibniz rule for every parity/length tested."""
    n_params = 2
    bases = _bases(n)
    emus, core_met_ix, emu_states = _states(n, bases)
    origin_set = OriginSet(emus)

    deriv_tensor, deriv_values = _derivative_states(n, n_params)
    derivative_states = {1: deriv_tensor}

    got = convolution_derivative(core_met_ix, derivative_states, emu_states, origin_set, DEVICE, DTYPE)

    derivs_by_param = [[deriv_values[i][p] for i in range(n)] for p in range(n_params)]
    expected = _manual_product_rule(bases, derivs_by_param)

    assert got.shape == (n_params, n + 1)
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-9)
