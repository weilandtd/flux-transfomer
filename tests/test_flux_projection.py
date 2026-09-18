"""The projection layer must return a point of the polytope, and leave one alone.

``FluxProjection`` replaces the post-hoc clip-and-reproject repair that never reached box
feasibility. Its contract is narrow and worth pinning exactly: it is bit-exactly the identity
on ``P``, it lands *strictly* inside rather than on a face (so the float32 cast cannot round
the endpoint back out), and where the greedily selected face set matches a real active-set QP
it is the exact Euclidean projection. So the tests push latents far past anything training
would produce, and use a second network built out of linear chains -- whose ``Z`` rows are
exactly parallel -- because that is what makes the Gram-Schmidt remainder collapse, the one
path where the recursion can stall instead of converging.

Exactness against a QP is deliberately not asserted. Greedy most-violated selection carries
no multiplier-sign check and never drops a constraint, so it is the exact projection when the
selected set is the true active set and not otherwise -- a bound on the gap would be a flaky
test asserting something the algorithm does not promise. What the forward pass needs is a
feasible state and a small move, and those are pinned directly.
"""

import numpy as np
import pytest
import torch

from sigmaflux.ai.flux_projection import FluxProjection, active_set_sizes

DEPTH = 24


# ── Problems ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dense():
    """A generic polytope: orthonormal basis, deliberately asymmetric bounds.

    A box centred on the origin hides sign errors in the upper/lower face split.
    """
    rng = np.random.default_rng(0)
    S = rng.normal(size=(3, 9))
    Z = np.linalg.svd(S)[2][3:].T                      # (9, 6) orthonormal null space
    lb = np.array([-2.0, -1.0, -5.0, 0.0, -3.0, -1.0, -4.0, -0.5, -2.0])
    ub = np.array([3.0, 4.0, 1.0, 8.0, 2.0, 6.0, 1.0, 2.5, 7.0])
    return Z, lb, ub


@pytest.fixture(scope="module")
def chained():
    """A polytope with linear chains, so several rows of Z are exactly parallel.

    Reactions 0-3 carry one flux and 4-6 another, which is how a metabolic chain appears in
    a null-space basis; their rows are identical up to sign, and their opposite faces are
    exactly antiparallel. This is the degenerate geometry that froze the pure no-op form on
    S. cerevisiae, concentrated on exactly such chains.
    """
    rng = np.random.default_rng(3)
    k = 4
    base = rng.normal(size=(3, k))
    Z = np.vstack([
        np.repeat(base[0:1], 4, axis=0),               # chain A, 4 reactions
        np.repeat(base[1:2], 3, axis=0),               # chain B, 3 reactions
        base[2:3], -base[2:3],                         # an antiparallel pair
        rng.normal(size=(3, k)),                       # and some independent rows
    ])
    lb = np.array([-1.0, -0.8, -1.2, -0.9, -2.0, -1.5, -1.8, -3.0, -3.0, -2.0, -1.0, -4.0])
    ub = np.array([2.0, 1.5, 0.4, 3.0, 1.0, 2.5, 0.6, 4.0, 2.0, 1.0, 5.0, 2.0])
    return Z, lb, ub


@pytest.fixture(scope="module", params=["dense", "chained"])
def problem(request):
    return request.getfixturevalue(request.param)


def _layer(problem, **kw):
    Z, lb, ub = problem
    kw.setdefault("depth", DEPTH)
    return FluxProjection(Z, lb, ub, **kw)


def _outside(problem, n=64, scale=1.0, seed=1):
    """Latents whose reconstruction is well outside the box."""
    Z, lb, ub = problem
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(scale=scale, size=(n, Z.shape[1])), dtype=torch.float64)


def _inside(problem, n=64, seed=2):
    """Latents that are strictly inside P, built by shrinking toward an interior point."""
    Z, lb, ub = problem
    layer = _layer(problem)
    with torch.no_grad():
        x = layer(_outside(problem, n=n, seed=seed))
        centre = layer(torch.zeros(1, Z.shape[1], dtype=torch.float64))
        # 0.9 of the way from a strictly interior point puts every face's slack well
        # inside, so nothing here is merely touching a boundary.
        return centre + 0.9 * (x - centre)


# ── Reference ─────────────────────────────────────────────────────────────────

def _pattern(layer, x):
    """The full per-block selection pattern, from the layer itself.

    Reimplementing the loop here was a mistake: the replica drifted from the real one and
    the test reported failures the layer did not have. Asking the layer is the only way the
    two cannot disagree.
    """
    with torch.no_grad():
        _, pat = layer(x, return_pattern=True)
    return tuple((int(j), bool(v), bool(k)) for j, v, k in pat)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_identity_on_the_polytope_is_bit_exact(problem):
    layer = _layer(problem)
    x = _inside(problem)
    with torch.no_grad():
        y = layer(x)
    assert torch.equal(y, x), (
        "relu(dj) must make every block an exact no-op inside P; "
        f"max drift {float((y - x).abs().max()):.3e}")


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e2, 1e4])
def test_feasible_for_any_latent(problem, scale):
    Z, lb, ub = problem
    layer = _layer(problem)
    with torch.no_grad():
        y = layer(_outside(problem, scale=scale))
    v = y.numpy() @ Z.T
    assert (v - ub).max() < 1e-9, f"over ub by {(v - ub).max():.3e}"
    assert (lb - v).max() < 1e-9, f"under lb by {(lb - v).max():.3e}"


def test_endpoint_lands_strictly_inside_not_on_the_face(problem):
    """The margin is the whole reason float32 output is usable.

    An exact move lands on the face, and the cast to float32 then rounds the endpoint
    outside. Measured on S. cerevisiae: without the margin only 14.1% of float32 states
    were strictly inside, with it 100%.
    """
    layer = _layer(problem)
    bare = _layer(problem, interior_ulps=0)
    x = _outside(problem, scale=1e2).float()
    with torch.no_grad():
        assert float(layer.violation(layer(x)).max()) < 0.0
        assert layer(x).dtype == torch.float32
        # and the bare layer is the one that needs the slack it does not have
        assert float(bare.violation(bare(x)).max()) >= float(layer.violation(layer(x)).max())


def test_projection_is_a_steady_state(problem):
    """Re-projecting changes nothing measurable.

    Not asserted bit-exactly. The interior margin is applied to the faces a block actually
    clears, so a face that merely drifts into violation at the float64 noise floor from the
    other moves gets no slack, and a second pass nudges it by ~1e-13. That is the same
    residual the layer reaches on real S. cerevisiae states and is orders below the
    narrowest reaction interval.
    """
    layer = _layer(problem)
    with torch.no_grad():
        y = layer(_outside(problem, scale=1e2))
        again = layer(y)
    assert torch.allclose(again, y, rtol=0, atol=1e-10), (
        f"max drift {float((again - y).abs().max()):.3e}")


def test_affine_within_a_fixed_selection_pattern(problem):
    Z, lb, ub = problem
    layer = _layer(problem)
    rng = np.random.default_rng(5)
    compared = 0
    for _ in range(200):
        a = torch.tensor(rng.normal(size=(1, Z.shape[1])), dtype=torch.float64)
        b = a + torch.tensor(rng.normal(scale=1e-4, size=a.shape))
        mid = 0.5 * a + 0.5 * b
        if not (_pattern(layer, a) == _pattern(layer, b) == _pattern(layer, mid)):
            continue
        compared += 1
        with torch.no_grad():
            err = (layer(mid) - 0.5 * layer(a) - 0.5 * layer(b)).abs().max()
        assert float(err) < 1e-14, f"not affine within one pattern: {float(err):.3e}"
    assert compared > 10, f"only {compared} pairs shared a pattern; test proved nothing"


def test_gradients_match_central_differences(problem):
    Z, lb, ub = problem
    layer = _layer(problem, depth=8)
    x = _outside(problem, n=1, scale=3.0, seed=6).requires_grad_(True)
    w = torch.tensor(np.random.default_rng(7).normal(size=(1, Z.shape[1])))
    (layer(x) * w).sum().backward()
    g = x.grad.clone()
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 0.0, "gradient vanished; the layer would not train"

    h = 1e-6
    for i in range(Z.shape[1]):
        e = torch.zeros_like(x)
        e[0, i] = h
        with torch.no_grad():
            plus = (layer(x.detach() + e) * w).sum()
            minus = (layer(x.detach() - e) * w).sum()
        fd = float((plus - minus) / (2 * h))
        assert abs(fd - float(g[0, i])) < 1e-8, (
            f"coord {i}: analytic {float(g[0, i]):.12g} vs finite diff {fd:.12g}")


def test_gradcheck_in_float64(problem):
    layer = _layer(problem, depth=6)
    x = _outside(problem, n=2, scale=2.0, seed=8).requires_grad_(True)
    assert torch.autograd.gradcheck(layer.forward, (x,), eps=1e-6, atol=1e-7)


def test_parallel_normals_do_not_stall_or_blow_up(chained):
    """The restart is what rescues the antiparallel sign flip; the pure no-op cannot.

    On S. cerevisiae the no-op form froze 25% of validation samples at 0.79 flux units of
    residual, no matter the depth, on reactions whose Z rows are identical.
    """
    Z, lb, ub = chained
    X = _outside(chained, n=128, scale=20.0, seed=9)
    restart = FluxProjection(Z, lb, ub, depth=DEPTH)
    noop = FluxProjection(Z, lb, ub, depth=DEPTH, on_rank_deficient="noop")
    with torch.no_grad():
        yr, yn = restart(X), noop(X)
    assert torch.isfinite(yr).all() and torch.isfinite(yn).all()
    with torch.no_grad():
        assert float(restart.violation(yr).max()) < 1e-9
        assert float(restart.violation(yr).max()) <= float(noop.violation(yn).max()) + 1e-12


def test_batch_invariance(problem):
    """A sample is projected the same alone as in a batch.

    Not bit-exact, and cannot be: ``x @ Ah.T`` dispatches to a different BLAS kernel at
    batch 1 than at batch 16, and the reassociated sum differs in the last bits. The
    tolerance is there for that and nothing else -- a genuine batch-dependent bug would
    move a facet selection and show up far above it.
    """
    layer = _layer(problem)
    X = _outside(problem, n=16, scale=10.0, seed=10)
    with torch.no_grad():
        batched = layer(X)
        alone = torch.cat([layer(X[i:i + 1]) for i in range(len(X))])
    assert torch.allclose(batched, alone, rtol=0, atol=1e-9), (
        f"max spread {float((batched - alone).abs().max()):.3e}")


def test_feasibility_improves_with_depth_then_stops(problem):
    """Feasibility is monotone in depth, and ``active_set_sizes`` predicts where it lands.

    The helper is what sizes ``depth`` in the configs, so what has to hold is that its
    answer is honest: at that depth the layer is feasible to the tolerance the helper was
    asked for, and more depth cannot undo it. Bit-identity beyond that depth is *not*
    claimed -- the extra blocks still polish a residual sitting at the float64 noise floor.
    """
    TOL = 1e-9
    X = _outside(problem, n=64, scale=1e2, seed=11)
    Z, lb, ub = problem
    need = int(active_set_sizes(Z, lb, ub, X.numpy(), tol=TOL).max())
    worst = []
    for depth in (1, 2, 4, need, need + 8):
        layer = _layer(problem, depth=depth)
        with torch.no_grad():
            worst.append(float(layer.violation(layer(X)).max()))
    assert all(b <= a + 1e-9 for a, b in zip(worst, worst[1:])), f"not monotone: {worst}"
    assert worst[-2] < TOL, (
        f"max(active_set_sizes)={need} moves left {worst[-2]:.3e} of violation, so the "
        f"helper under-reports the depth the configs need")
    assert worst[-1] < TOL


def test_min_slack_holds_fluxes_off_their_bounds(problem):
    """An exactly-feasible state may sit *on* a bound; min_slack keeps it clear.

    Roughly half these reactions have a lower bound of 0, so sitting on it means zero flux,
    which degenerates the EMU forward pass. The inset is capped per reaction so it can never
    empty a narrow box.
    """
    Z, lb, ub = problem
    X = _outside(problem, n=64, scale=50.0, seed=14)
    bare = _layer(problem)
    held = _layer(problem, min_slack=1e-3)
    with torch.no_grad():
        vb = bare(X).numpy() @ Z.T
        vh = held(X).numpy() @ Z.T
    assert (vh - lb).min() >= 1e-3 - 1e-9, "min_slack did not hold the lower bounds off"
    assert (ub - vh).min() >= 1e-3 - 1e-9, "min_slack did not hold the upper bounds off"
    # and it is still feasible against the true bounds
    assert (vh - ub).max() < 1e-9 and (lb - vh).max() < 1e-9
    # the bare layer is the one allowed to touch a bound
    assert np.minimum(vb - lb, ub - vb).min() <= np.minimum(vh - lb, ub - vh).min()


def test_min_slack_cannot_empty_a_narrow_box(dense):
    """A box narrower than 2*min_slack keeps a valid, if tiny, interval."""
    Z, lb, ub = dense
    narrow_ub = ub.copy()
    narrow_ub[3] = lb[3] + 1e-9          # far narrower than the slack asked for
    layer = FluxProjection(Z, lb, narrow_ub, depth=DEPTH, min_slack=1.0)
    with torch.no_grad():
        v = layer(_outside(dense, scale=10.0)).numpy() @ Z.T
    assert np.isfinite(v).all()
    assert (v[:, 3] >= lb[3] - 1e-9).all() and (v[:, 3] <= narrow_ub[3] + 1e-9).all()


def test_violation_matches_numpy(problem):
    Z, lb, ub = problem
    layer = _layer(problem)
    X = _outside(problem, n=32, scale=4.0, seed=12)
    v = X.numpy() @ Z.T
    want = np.maximum(v - ub, lb - v).max(axis=1)
    assert np.allclose(layer.violation(X).numpy(), want, atol=1e-12)


def test_active_set_sizes_is_zero_inside_and_bounded_outside(problem):
    Z, lb, ub = problem
    inside = _inside(problem).numpy()
    assert (active_set_sizes(Z, lb, ub, inside) == 0).all(), (
        "a feasible sample needs no moves; a nonzero count means the bounds and the basis "
        "disagree and any depth read off this is meaningless")
    sizes = active_set_sizes(Z, lb, ub, _outside(problem, scale=1e2, seed=13).numpy())
    assert (sizes > 0).any() and (sizes > 0).all()


def test_layer_holds_no_state(problem):
    """Nothing may enter the state dict: the .sfx exporter checks regenerated keys exactly."""
    layer = _layer(problem)
    assert layer.state_dict() == {}
    assert list(layer.parameters()) == []


# ── Rejected inputs ───────────────────────────────────────────────────────────

def test_rank_deficient_basis_is_refused(dense):
    Z, lb, ub = dense
    bad = np.hstack([Z, Z[:, :1]])
    with pytest.raises(ValueError, match="full column rank"):
        FluxProjection(bad, lb, ub)


def test_empty_polytope_is_refused(dense):
    Z, lb, ub = dense
    flipped = ub.copy()
    flipped[2] = lb[2] - 1.0
    with pytest.raises(ValueError, match="ub < lb"):
        FluxProjection(Z, lb, flipped)


def test_pinned_reaction_outside_its_own_box_is_refused(dense):
    """A zero row in Z is a reaction pinned to flux_offset; the latent cannot move it.

    Dropping it is correct, but only once its constant is known to be inside its box --
    otherwise the polytope is empty and every prediction would be silently flattened onto
    a facet instead of raising.
    """
    Z, lb, ub = dense
    Zp = Z.copy()
    Zp[4] = 0.0
    FluxProjection(Zp, lb, ub)                         # 0 is inside [-3, 2]: fine
    shifted_lb, shifted_ub = lb.copy(), ub.copy()
    shifted_lb[4], shifted_ub[4] = 1.0, 2.0            # now 0 is outside
    with pytest.raises(ValueError, match="do not contain 0"):
        FluxProjection(Zp, shifted_lb, shifted_ub)


def test_zero_rows_are_dropped_not_left_infinite(dense):
    Z, lb, ub = dense
    Zp = Z.copy()
    Zp[4] = 0.0
    layer = FluxProjection(Zp, lb, ub, depth=DEPTH)
    assert layer.n_live == Z.shape[0] - 1
    with torch.no_grad():
        y = layer(_outside(dense, scale=1e2))
    assert torch.isfinite(y).all(), (
        "an empty Z row makes d infinite and argmax locks onto it every block")


def test_bad_mode_and_depth_are_refused(dense):
    Z, lb, ub = dense
    with pytest.raises(ValueError, match="on_rank_deficient"):
        FluxProjection(Z, lb, ub, on_rank_deficient="clip")
    with pytest.raises(ValueError, match="depth must be at least 1"):
        FluxProjection(Z, lb, ub, depth=0)
