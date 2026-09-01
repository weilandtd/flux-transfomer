"""A latent that satisfies ``A x = b`` and ``lb < x < ub`` by construction.

The decoder reconstructs fluxes as ``v = y @ Z.T + offset``, which satisfies ``S v = 0``
but nothing else. Measured on I. orientalis, every one of 1000 validation states landed
outside the reaction bounds -- 35 reactions apiece -- and an EMU forward pass on such a
state returns a largest "fraction" of 9.9e+39, so inference has to repair it. Every repair
operator measured trades accuracy for feasibility, the worst (a single ratio-test ray)
costing per-reaction R2 0.592 -> -0.005. A map that cannot leave the polytope removes the
choice.

Write ``x = x_p + N y`` with ``N`` a null-space basis of ``A``. The box pulls back to
``P = {y : lb - x_p <= N y <= ub - x_p}``, a polytope, so per-coordinate squashing does not
apply -- the coordinates are coupled. What does apply is the Minkowski gauge of ``P``:

    g(y) = max_i max(dw_i / u_i, dw_i / l_i),   dw = N y / (ub - lb)

positively homogeneous of degree one, equal to 1 exactly on the boundary and to the
fraction of the way there inside. Scaling any direction by ``1/g`` lands on the boundary,
so a radial law mapping R -> (0, 1) turns an unconstrained head into a strictly interior
point.

Two radial laws are provided because the choice is not free. Training flux samples sit
close to the boundary -- median ``g`` 0.952 on E. coli core, 0.989 on I. orientalis, with
47.8% of I. orientalis samples above 0.99 -- and the two laws differ by orders of magnitude
in how much gradient survives there:

``mobius``   ``y = z / (1 + g(z))``, so ``g(y) = t/(1+t)`` and ``dg/dt = (1-g)^2``.
             Median gradient 2.3e-03 (E. coli core), 1.2e-04 (I. orientalis).
``sigmoid``  head emits a direction and a scalar; ``rho = sigmoid(s)``,
             ``y = rho * d / g(d)``, so ``g(y) = rho`` exactly and ``drho/ds = rho(1-rho)``.
             Median gradient 4.6e-02 and 1.1e-02 -- 20x and 90x more, and 1e4x at g=0.9999.
"""

import numpy as np
import torch
from torch import nn

#: Guard for the gauge of a direction that is numerically zero. Only ``d = 0`` gives
#: ``g(d) = 0``, and there the direction carries no information to normalise.
_GAUGE_FLOOR = 1e-30

#: Interior margin, in units of the output dtype's epsilon. The algebra puts every state
#: strictly inside, but ``sigmoid`` saturates to exactly 1.0 once ``|s|`` passes ~37, and
#: the reconstruction's own rounding can carry a state the last ulp onto a facet. This
#: keeps the guarantee unconditional at the cost of a reachability ceiling: float32 cannot
#: represent a gauge above ``1 - 1e-4``, which on I. orientalis is its top ~1% of targets.
#: Strict interiority is a float64 property -- see ``forward`` for why float32 gives only
#: ``lb <= x <= ub`` on a polytope with near-pinned coordinates.
_MARGIN_ULPS = 1024

#: Below this Chebyshev radius (as a fraction of the box width) the polytope is too thin
#: for the gauge to be numerically comfortable, and the capping coordinates are reported.
_THIN_POLYTOPE = 1e-3


def chebyshev_center(N, lb, ub, offset=None):
    """Largest inscribed box-shaped ball inside ``P``, solved in basis coordinates.

    Parameterising by ``y`` rather than ``x`` matters: the decoder can only reach
    ``offset + range(N)``, and on I. orientalis that is 83 of the 119 null-space
    dimensions once the pinned directions are split out. A centre found over the whole
    null space would not be representable, and the gauge would be cast from a point the
    model can never emit.

    The radius is in units of each coordinate's own interval, so no reaction's scale
    dominates the objective. A strictly interior point is what the gauge needs -- on the
    boundary it is undefined -- and ``pinv(A) @ b`` does not supply one: it lands on the
    boundary for most metabolic models.

    Args:
        N: null-space basis, shape (n, m).
        lb: lower bounds, shape (n,).
        ub: upper bounds, shape (n,).
        offset: point the basis is anchored at, shape (n,); defaults to zeros.

    Returns ``(x_p, r)``: the centre and its radius as a fraction of the box width.
    """
    from scipy.optimize import linprog

    N = np.asarray(N, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    offset = np.zeros(N.shape[0]) if offset is None else np.asarray(offset, np.float64)
    m = N.shape[1]
    s = ub - lb

    # Variables are [y, r]; maximise r by minimising -r.
    cost = np.zeros(m + 1)
    cost[-1] = -1.0
    ineq = np.vstack([np.hstack([-N, s[:, None]]),      # lb + r*s <= offset + N y
                      np.hstack([N, s[:, None]])])      # offset + N y <= ub - r*s
    res = linprog(
        cost,
        A_ub=ineq, b_ub=np.concatenate([offset - lb, ub - offset]),
        bounds=[(None, None)] * m + [(0, None)],
        method="highs",
    )
    if not res.success:
        raise ValueError(f"Chebyshev LP did not solve (status {res.status}): {res.message}")
    return offset + N @ res.x[:m], float(res.x[-1])


def orthonormalize(N, n_targets=0):
    """Gram-Schmidt ``N``, taking the first *n_targets* columns first.

    Off by default. A targeted basis keeps one identity column per measured reaction, and
    rotating those away is not free -- it took ``DNA_c`` from validation R2 +0.987 to
    -5.92. Ordering the targeted block first means each target keeps its own leading
    coordinate while the basis still becomes orthonormal, so two near-parallel targets
    (``RNA_c`` and ``DNA_c`` sit at ``|G_ij|`` 0.9892) show up as a small diagonal entry
    rather than as a Gram condition of 5.5e3 spread over every coordinate.

    Args:
        N: null-space basis, shape (n, m).
        n_targets: leading columns to orthonormalise first, in their own order.

    Returns the orthonormalised basis, same shape.
    """
    Q, R = np.linalg.qr(np.asarray(N, dtype=np.float64))
    # QR already processes columns left to right, so a targeted block placed first keeps
    # its ownership; the argument documents the requirement and guards a reordering.
    if n_targets and np.abs(np.diag(R)[:n_targets]).min() < 1e-10:
        raise ValueError(
            f"the first {n_targets} basis columns are numerically rank-deficient, so "
            f"orthonormalising them destroys the targeted block rather than preserving it"
        )
    return Q


def facet_report(N, x_p, lb, ub, X=None):
    """Conditioning diagnostics for a gauge latent, printed into the training log.

    ``g`` near 1 is the thing that decides whether this parameterisation can be trained:
    a target sitting on a facet needs an unbounded head output to reach, and the radial
    gradient there vanishes. Reporting it against the *targets* bounds what is achievable;
    reporting it against predictions later says what the model actually does.

    Args:
        N: null-space basis, shape (n, m).
        x_p: interior point, shape (n,).
        lb: lower bounds, shape (n,).
        ub: upper bounds, shape (n,).
        X: optional feasible states to profile, shape (n_samples, n).

    Returns a dict of the reported quantities.
    """
    N = np.asarray(N, dtype=np.float64)
    G = N.T @ N
    out = {
        "m": N.shape[1],
        "gram_cond": float(np.linalg.cond(G)),
        "interior_radius": float(np.minimum((x_p - lb) / (ub - lb),
                                            (ub - x_p) / (ub - lb)).min()),
    }
    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        dx = X - x_p
        u, ell = ub - x_p, lb - x_p
        ratio = np.maximum(np.where(u > 0, dx / np.where(u > 0, u, 1.0), -np.inf),
                           np.where(ell < 0, dx / np.where(ell < 0, ell, -1.0), -np.inf))
        g = ratio.max(axis=1)
        Y = (X - x_p) @ (N @ np.linalg.inv(G))
        span = Y.max(axis=0) - Y.min(axis=0)
        out.update(
            g_median=float(np.median(g)), g_p90=float(np.percentile(g, 90)),
            g_max=float(g.max()), frac_g_above_099=float((g > 0.99).mean()),
            mobius_grad_median=float(np.median((1 - g) ** 2)),
            sigmoid_grad_median=float(np.median(g * (1 - g))),
            y_span_orders=float(np.log10(span.max() / max(span.min(), 1e-300))),
            n_binding_facets=len(np.unique(ratio.argmax(axis=1))),
        )
    return out


class BoxNullspace(nn.Module):
    """``x = x_p + N y`` with ``A x = b`` and ``lb < x < ub`` guaranteed by construction.

    See the module docstring for the geometry and for why ``radial`` matters. The head
    width this layer expects is :attr:`latent_dim`, which is ``m`` for ``mobius`` and
    ``m + 1`` for ``sigmoid`` (the extra output is the radial coordinate).

    ``N`` is not required to be orthonormal -- only full column rank -- so the inverse
    goes through the dual ``(N^T N)^-1 N^T`` rather than ``N^T``. Using the projection
    instead is the trap that left a 74% reconstruction error elsewhere in this codebase.
    """

    def __init__(self, N, x_p, lb, ub, radial="sigmoid", tau=0.0):
        """Args:
            N: null-space basis of the equality system, shape (n, m), full column rank.
            x_p: strictly interior point satisfying the equality system, shape (n,).
            lb: lower bounds, shape (n,); must satisfy ``lb < x_p``.
            ub: upper bounds, shape (n,); must satisfy ``x_p < ub``.
            radial: ``'sigmoid'`` (recommended) or ``'mobius'``.
            tau: LogSumExp softening temperature for the gauge; 0 uses a hard max. LSE
                overestimates the max, so a softened state stays feasible, just pulled
                further inside. It is bounded by ``max + tau*log(n)``, so with n in the
                hundreds tau must be small (~1e-2) or the whole prediction collapses.
        """
        super().__init__()
        if radial not in ("sigmoid", "mobius"):
            raise ValueError(f"radial must be 'sigmoid' or 'mobius', not {radial!r}")
        N = np.asarray(N, dtype=np.float64)
        x_p = np.asarray(x_p, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        slack = np.minimum(x_p - lb, ub - x_p)
        if slack.min() <= 0:
            worst = int(np.argmin(slack))
            raise ValueError(
                f"x_p is not strictly inside the box: coordinate {worst} has slack "
                f"{slack.min():.3g}. The gauge is undefined on the boundary, so every "
                f"direction from there would have zero length."
            )
        if np.linalg.matrix_rank(N) < N.shape[1]:
            raise ValueError(
                f"N has {N.shape[1]} columns but rank "
                f"{np.linalg.matrix_rank(N)}; the gauge needs full column rank for the "
                f"map to be injective."
            )

        self.radial = radial
        self.tau = tau
        self.m = N.shape[1]
        s = ub - lb
        w_p = (x_p - lb) / s                       # strictly inside (0, 1)^n

        self.register_buffer("N", torch.as_tensor(N))
        self.register_buffer("Nn", torch.as_tensor(N / s[:, None]))   # unit-scaled facets
        self.register_buffer("N_dual", torch.as_tensor(N @ np.linalg.inv(N.T @ N)))
        self.register_buffer("x_p", torch.as_tensor(x_p))
        self.register_buffer("lb", torch.as_tensor(lb))
        self.register_buffer("ub", torch.as_tensor(ub))
        self.register_buffer("u", torch.as_tensor(1.0 - w_p))         # > 0
        self.register_buffer("l", torch.as_tensor(-w_p))              # < 0

    @property
    def latent_dim(self):
        """Head width this layer consumes: ``m``, plus one for the sigmoid radius."""
        return self.m + 1 if self.radial == "sigmoid" else self.m

    def gauge(self, y):
        """Minkowski functional of the polytope: 0 at ``x_p``, 1 on the boundary.

        Args:
            y: latent coordinates, shape (..., m).
        """
        dw = y @ self.Nn.T.to(y.dtype)
        ratio = torch.maximum(dw / self.u.to(y.dtype), dw / self.l.to(y.dtype))
        if self.tau <= 0:
            return ratio.amax(-1)
        return self.tau * torch.logsumexp(ratio / self.tau, -1)

    def forward(self, z):
        """Map an unconstrained head output to a point strictly inside the polytope.

        Args:
            z: raw network output, shape (B, latent_dim).
        """
        # Buffers are float64 and the gauge is a ratio against slacks that can be small,
        # so the geometry is worked out in double and cast back at the end.
        work = torch.promote_types(z.dtype, torch.float64)
        zz = z.to(work)
        if self.radial == "mobius":
            y = zz / (1.0 + self.gauge(zz).clamp_min(0.0)).unsqueeze(-1)
        else:
            d, s = zz[..., :-1], zz[..., -1]
            # g is homogeneous of degree one, so d/g(d) is the boundary point along d and
            # rho scales it. g(y) is then exactly rho, which is what makes the radial
            # gradient rho(1-rho) instead of vanishing quadratically at the facets.
            rho = torch.sigmoid(s).unsqueeze(-1)
            y = rho * d / self.gauge(d).clamp_min(_GAUGE_FLOOR).unsqueeze(-1)
        y = y * self._interior_shrink(y, z.dtype)
        x = (self.x_p.to(work) + y @ self.N.T.to(work)).to(z.dtype)
        # The shrink leaves each coordinate a gap of margin * (its own slack at x_p), so a
        # coordinate whose slack is already below the output dtype's resolution rounds onto
        # its bound. I. orientalis has such coordinates -- POOL_pro__L_m sits at 0.9999957
        # of [0, 1], because pro__L is essentially all mitochondrial in every feasible
        # state -- so in float32 the guarantee is lb <= x <= ub, not strict. That is what
        # the EMU forward pass needs; the clamp makes it exact rather than nearly so.
        return torch.clamp(x, self.lb.to(z.dtype), self.ub.to(z.dtype))

    def _interior_shrink(self, y, out_dtype):
        """Factor pulling a state off a facet it only reached through rounding.

        Exactly 1 -- and so invisible to the gradient -- whenever the state already has
        the margin, which is every state the model produces in normal training. It earns
        its keep in saturation: ``sigmoid`` returns exactly 1.0 for ``|s|`` past ~37, and
        without this the state lands on the boundary and rounds outside, which is the one
        thing this layer exists to prevent.

        Args:
            y: latent coordinates, shape (..., m).
            out_dtype: dtype the result will be cast to; sets the margin.
        """
        margin = _MARGIN_ULPS * torch.finfo(out_dtype).eps
        g = self.gauge(y).clamp_min(_GAUGE_FLOOR)
        return ((1.0 - margin) / g).clamp(max=1.0).unsqueeze(-1)

    def inverse(self, x):
        """Recover the head output that produces *x*, for encoding targets.

        ``1 - g`` is formed from the raw slack ``ub - x`` rather than by subtracting the
        gauge from one. The two are algebraically identical and numerically are not: once
        ``g > 1 - 1e-7`` the subtraction is pure roundoff in float32, which covers 2.4% of
        I. orientalis training samples.

        Args:
            x: feasible states strictly inside the box, shape (B, n).
        """
        y = (x - self.x_p.to(x.dtype)) @ self.N_dual.to(x.dtype)
        dw = y @ self.Nn.T.to(x.dtype)
        one_minus_r = torch.where(
            dw > 0,
            (self.ub.to(x.dtype) - x) / (self.ub.to(x.dtype) - self.x_p.to(x.dtype)),
            (x - self.lb.to(x.dtype)) / (self.x_p.to(x.dtype) - self.lb.to(x.dtype)),
        )
        one_minus_g = torch.where(dw == 0, torch.ones_like(one_minus_r),
                                  one_minus_r).amin(-1).clamp_min(_GAUGE_FLOOR)
        if self.radial == "mobius":
            return y / one_minus_g.unsqueeze(-1)
        rho = (1.0 - one_minus_g).clamp(_GAUGE_FLOOR, 1.0 - _GAUGE_FLOOR)
        s = torch.log(rho) - torch.log(one_minus_g)
        return torch.cat([y, s.unsqueeze(-1)], dim=-1)


def whiten_by_data(N, x_p, samples):
    """Rescale the basis so one latent unit is one standard deviation of the data.

    ``N -> N @ L`` with ``L L^T = cov(y)``, so the training coordinates become isotropic.
    It removes the last of the direction anisotropy (I. orientalis: 3.8e3 -> 2.2 once the
    basepoint is already the data mean) but, like any column mix, it does **not** preserve
    a targeted identity block -- the same trade that made `orthonormalize` fragile. Keep
    them as separate switches so the two effects stay separable.

    Args:
        N: null-space basis, shape (n, m).
        x_p: basepoint the coordinates are measured from, shape (n,).
        samples: feasible states defining the scale, shape (n_samples, n).
    """
    N = np.asarray(N, dtype=np.float64)
    y = (np.asarray(samples, dtype=np.float64) - x_p) @ (N @ np.linalg.inv(N.T @ N))
    eigvals, eigvecs = np.linalg.eigh(np.cov(y, rowvar=False))
    root = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    return N @ root


def prepare_box_nullspace(N, A, lb, ub, *, offset=None, radial="sigmoid", tau=0.0,
                          n_targets=0, orthonormal=False, center="chebyshev",
                          whiten=False, x_p=None, samples=None, verbose=True):
    """Build a :class:`BoxNullspace` from a basis and the constraint data.

    Args:
        N: null-space basis of *A*, shape (n, m).
        A: equality constraint matrix, shape (k, n).
        lb: lower bounds, shape (n,).
        ub: upper bounds, shape (n,).
        offset: point the basis is anchored at (the decoder's ``flux_offset``).
        radial: radial law, ``'sigmoid'`` or ``'mobius'``.
        tau: LogSumExp softening temperature; 0 for a hard max.
        n_targets: leading columns holding a targeted identity block.
        orthonormal: orthonormalise *N* first. Off by default -- it reshuffles the
            targeted block, and the gauge needs only full column rank.
        center: where to cast the gauge from -- ``'chebyshev'`` (furthest from every
            facet) or ``'mean'`` (the training data's own centre). Maximising distance to
            the constraints is not the same as sitting where the data is: on I. orientalis
            the Chebyshev centre is 2.9x further from the data cloud, which leaves every
            sample in one narrow cone (cosine 0.914 between random pairs) and a direction
            anisotropy of 4e8. The mean drops that to 3.8e3 with the cone gone (-0.04),
            and it leaves *N* untouched, so a targeted block survives.
        whiten: rescale *N* by the data covariance; see :func:`whiten_by_data`.
        x_p: explicit basepoint, overriding *center*.
        samples: feasible states to profile in the report, shape (n_samples, n).
        verbose: print the conditioning report.
    """
    N = np.asarray(N, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    residual = np.abs(A @ N).max()
    if residual > 1e-8:
        raise ValueError(
            f"N is not a null-space basis of A: max|A N| = {residual:.3g}"
        )
    if orthonormal:
        N = orthonormalize(N, n_targets)

    r = None
    if x_p is not None:
        x_p = np.asarray(x_p, dtype=np.float64)
    elif center == "mean":
        if samples is None:
            raise ValueError("center='mean' needs samples to take the mean of")
        x_p = np.asarray(samples, dtype=np.float64).mean(axis=0)
        slack = np.minimum(x_p - lb, ub - x_p)
        if slack.min() <= 0:
            raise ValueError(
                f"the training mean is not strictly inside the bounds (min slack "
                f"{slack.min():.3g}); it cannot serve as the gauge's basepoint."
            )
    else:
        x_p, r = chebyshev_center(N, lb, ub, offset=offset)
        if r <= 0:
            raise ValueError(
                f"the feasible set has no interior (Chebyshev radius {r:.3g}); it is "
                f"empty or lower-dimensional than the bounds claim, so no gauge exists."
            )
        if r < _THIN_POLYTOPE:
            capping = np.flatnonzero(
                np.isclose(np.minimum((x_p - lb) / (ub - lb), (ub - x_p) / (ub - lb)), r,
                           rtol=1e-6)
            )
            print(f"WARNING: the feasible set is thin -- Chebyshev radius {r:.3g} of the "
                  f"box width, capped simultaneously by {len(capping)} coordinate(s): "
                  f"{capping[:8].tolist()}. The gauge is numerically fragile here.")

    if whiten:
        if samples is None:
            raise ValueError("whiten=True needs samples to take the covariance of")
        N = whiten_by_data(N, x_p, samples)

    layer = BoxNullspace(N, x_p, lb, ub, radial=radial, tau=tau)
    if verbose:
        rep = facet_report(N, x_p, lb, ub, X=samples)
        print(f"BoxNullspace: radial={radial} latent_dim={layer.latent_dim} "
              f"m={rep['m']} cond(N'N)={rep['gram_cond']:.3g} "
              f"interior_radius={rep['interior_radius']:.3g}"
              + (f" chebyshev_r={r:.3g}" if r is not None else ""))
        if samples is not None:
            print(f"  target gauge: median {rep['g_median']:.4f} p90 {rep['g_p90']:.4f} "
                  f"max {rep['g_max']:.6f}, {rep['frac_g_above_099']:.1%} above 0.99")
            print(f"  radial gradient at the median target: mobius "
                  f"{rep['mobius_grad_median']:.3g}, sigmoid "
                  f"{rep['sigmoid_grad_median']:.3g}")
            print(f"  y ranges span {rep['y_span_orders']:.1f} orders; "
                  f"{rep['n_binding_facets']} facet(s) ever bind")
    return layer
