"""The gauge latent must be feasible for any head output, invertible, and differentiable.

Feasibility is the whole point of the layer: it exists so inference never has to repair a
prediction, and a repair operator was measured costing per-reaction R2 0.592 -> -0.005.
So the tests push head outputs far past anything training would produce.
"""

import numpy as np
import pytest
import torch

from sigmaflux.ai.box_nullspace import (
    BoxNullspace,
    chebyshev_center,
    facet_report,
    orthonormalize,
    prepare_box_nullspace,
)

RADIAL = ["sigmoid", "mobius"]


@pytest.fixture(scope="module")
def problem():
    """A small, non-trivial polytope: 3 equalities over 9 coordinates, asymmetric bounds.

    Asymmetric bounds matter -- a box centred on the origin hides sign errors in the
    u/l split, since the two branches of the gauge then agree.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 9))
    lb = np.array([-2.0, -1.0, -5.0, 0.0, -3.0, -1.0, -4.0, -0.5, -2.0])
    ub = np.array([3.0, 4.0, 1.0, 8.0, 2.0, 6.0, 1.0, 2.5, 7.0])
    b = np.zeros(3)
    N = np.linalg.svd(A)[2][3:].T                      # orthonormal null-space basis
    return A, b, lb, ub, N


@pytest.fixture(scope="module")
def layers(problem):
    A, _, lb, ub, N = problem
    return {r: prepare_box_nullspace(N, A, lb, ub, radial=r, verbose=False)
            for r in RADIAL}


@pytest.mark.parametrize("radial", RADIAL)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
def test_any_head_output_lands_strictly_inside(problem, layers, radial, scale):
    """No z, however large, can leave the polytope -- that is the layer's contract.

    Strict interiority holds whenever x_p has slack the dtype can resolve. A coordinate
    pinned closer than that -- I. orientalis has one, POOL_pro__L_m at 0.9999957 of
    [0, 1] -- rounds onto its bound in float32, and the final clamp keeps lb <= x <= ub.
    """
    A, b, lb, ub, _ = problem
    layer = layers[radial]
    z = torch.as_tensor(
        np.random.default_rng(1).normal(size=(64, layer.latent_dim)) * scale
    )
    x = layer(z).numpy()

    assert (x > lb).all(), f"below lb by {(lb - x).max():.3g}"
    assert (x < ub).all(), f"above ub by {(x - ub).max():.3g}"
    assert np.abs(A @ x.T - b[:, None]).max() < 1e-9


@pytest.mark.parametrize("radial", RADIAL)
def test_round_trip_recovers_head_output(problem, layers, radial):
    """inverse(forward(z)) == z, so targets can be encoded into head space."""
    layer = layers[radial]
    z = torch.as_tensor(np.random.default_rng(2).normal(size=(32, layer.latent_dim)))
    back = layer.inverse(layer(z))
    if radial == "mobius":
        assert torch.allclose(back, z, atol=1e-9)
    else:
        # d is only defined up to a positive scale (the map normalises it away), so the
        # direction and the radial coordinate are what must round-trip, not d itself.
        d0, d1 = z[:, :-1], back[:, :-1]
        cos = torch.nn.functional.cosine_similarity(d0, d1, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-9)
        assert torch.allclose(back[:, -1], z[:, -1], atol=1e-9)


@pytest.mark.parametrize("radial", RADIAL)
def test_round_trip_recovers_interior_points(problem, layers, radial):
    """forward(inverse(x)) == x for feasible x, including points close to a facet."""
    _, _, _, _, _ = problem
    layer = layers[radial]
    z = torch.as_tensor(np.random.default_rng(3).normal(size=(32, layer.latent_dim)) * 30)
    x = layer(z)                                        # feasible by construction
    assert torch.allclose(layer(layer.inverse(x)), x, atol=1e-8)


@pytest.mark.parametrize("radial", RADIAL)
def test_gradients_are_correct(problem, layers, radial):
    """gradcheck in float64. The gauge is a max, so ties are non-differentiable; random
    directions make an exact tie a measure-zero event."""
    layer = layers[radial]
    z = torch.as_tensor(
        np.random.default_rng(4).normal(size=(4, layer.latent_dim))
    ).requires_grad_(True)
    assert torch.autograd.gradcheck(layer.forward, (z,), eps=1e-6, atol=1e-7)


@pytest.mark.parametrize("radial", RADIAL)
def test_gradient_survives_near_the_boundary(problem, layers, radial):
    """Targets sit near facets -- median gauge 0.989 on I. orientalis -- so the radial
    gradient there decides trainability. Mobius decays as (1-g)^2, sigmoid as g(1-g)."""
    layer = layers[radial]
    z = torch.as_tensor(
        np.random.default_rng(5).normal(size=(16, layer.latent_dim)) * 1e4
    ).requires_grad_(True)
    layer(z).square().sum().backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().max() > 0


def test_softened_gauge_stays_feasible(problem):
    """LogSumExp >= max, so softening pulls the state further in, never out."""
    A, _, lb, ub, N = problem
    soft = prepare_box_nullspace(N, A, lb, ub, radial="sigmoid", tau=1e-2,
                                 verbose=False)
    z = torch.as_tensor(np.random.default_rng(6).normal(size=(64, soft.latent_dim)) * 1e3)
    x = soft(z).numpy()
    assert (x > lb).all() and (x < ub).all()


def test_chebyshev_centre_is_strictly_interior(problem):
    A, b, lb, ub, N = problem
    x_p, r = chebyshev_center(N, lb, ub)
    assert r > 0
    assert (x_p > lb).all() and (x_p < ub).all()
    assert np.abs(A @ x_p - b).max() < 1e-9
    # r is in units of each coordinate's own interval, so it bounds every scaled slack.
    assert np.minimum((x_p - lb) / (ub - lb), (ub - x_p) / (ub - lb)).min() >= r - 1e-9
    # and the centre must be reachable by the basis the decoder actually has
    assert np.abs(N @ np.linalg.lstsq(N, x_p, rcond=None)[0] - x_p).max() < 1e-9


def test_boundary_origin_is_refused(problem):
    """A gauge cast from the boundary has zero-length rays in some direction."""
    _, _, lb, ub, N = problem
    x_p, _ = chebyshev_center(N, lb, ub)
    on_facet = x_p.copy()
    on_facet[0] = ub[0]
    with pytest.raises(ValueError, match="not strictly inside"):
        BoxNullspace(N, on_facet, lb, ub)


def test_rank_deficient_basis_is_refused(problem):
    _, _, lb, ub, N = problem
    x_p, _ = chebyshev_center(N, lb, ub)
    duplicated = np.hstack([N, N[:, :1]])
    with pytest.raises(ValueError, match="full column rank"):
        BoxNullspace(duplicated, x_p, lb, ub)


def test_basis_that_does_not_span_the_nullspace_is_refused(problem):
    A, _, lb, ub, N = problem
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="not a null-space basis"):
        prepare_box_nullspace(rng.normal(size=N.shape), A, lb, ub, verbose=False)


def test_orthonormalize_preserves_the_targeted_block(problem):
    """The identity block is why measured fluxes each own a coordinate; Gram-Schmidt with
    the targets first must not hand a target's ownership to another column."""
    _, _, _, _, N = problem
    skewed = N.copy()
    skewed[:, 1] += 0.9 * skewed[:, 0]                  # make columns 0 and 1 near-parallel
    Q = orthonormalize(skewed, n_targets=2)
    assert np.abs(Q.T @ Q - np.eye(Q.shape[1])).max() < 1e-10
    # column 0 is untouched up to sign, so target 0 keeps its own direction
    assert abs(abs(np.dot(Q[:, 0], skewed[:, 0] / np.linalg.norm(skewed[:, 0]))) - 1) < 1e-10


def _states_at_gauge(layer, target_g, seed=8, n=256):
    """States whose gauge is exactly *target_g*, via the sigmoid law's rho = g identity."""
    rng = np.random.default_rng(seed)
    d = torch.as_tensor(rng.normal(size=(n, layer.m)))
    s = torch.full((n, 1), float(np.log(target_g) - np.log1p(-target_g)))
    return layer(torch.cat([d, s], dim=-1))


def test_facet_report_flags_boundary_concentration(problem, layers):
    """The report is what tells us whether a model is trainable in this latent at all."""
    _, _, lb, ub, N = problem
    layer = layers["sigmoid"]
    rep = facet_report(N, layer.x_p.numpy(), lb, ub,
                       X=_states_at_gauge(layer, 0.99).numpy())
    assert rep["g_max"] < 1
    assert rep["n_binding_facets"] >= 1
    assert rep["g_median"] == pytest.approx(0.99, abs=1e-6)
    assert rep["frac_g_above_099"] >= 0.0


def test_sigmoid_keeps_more_gradient_where_the_data_sits(problem, layers):
    """The two laws cross at g = 0.5: below it mobius has more gradient, above it sigmoid
    does. Real targets sit at median g 0.952 (E. coli core) and 0.989 (I. orientalis), so
    the crossover is what justifies carrying both arms rather than assuming."""
    _, _, lb, ub, N = problem
    layer = layers["sigmoid"]
    x_p = layer.x_p.numpy()

    below = facet_report(N, x_p, lb, ub, X=_states_at_gauge(layer, 0.2).numpy())
    assert below["mobius_grad_median"] > below["sigmoid_grad_median"]

    for g in (0.952, 0.989, 0.9999):
        rep = facet_report(N, x_p, lb, ub, X=_states_at_gauge(layer, g).numpy())
        assert rep["sigmoid_grad_median"] > rep["mobius_grad_median"], g
    # at the I. orientalis median the advantage is ~90x; assert the order of magnitude
    io = facet_report(N, x_p, lb, ub, X=_states_at_gauge(layer, 0.989).numpy())
    assert io["sigmoid_grad_median"] / io["mobius_grad_median"] > 50


@pytest.mark.parametrize("radial", RADIAL)
def test_float32_stays_feasible(problem, layers, radial):
    """Training runs in float32; feasibility must not depend on float64 buffers."""
    _, _, lb, ub, _ = problem
    layer = layers[radial]
    z = torch.as_tensor(
        np.random.default_rng(9).normal(size=(64, layer.latent_dim)) * 1e3,
        dtype=torch.float32,
    )
    x = layer(z)
    assert x.dtype is torch.float32
    x = x.numpy()
    assert (x > lb).all() and (x < ub).all()
