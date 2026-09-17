"""Euclidean projection of a latent vector onto a polytope, in a fixed number of moves.

Projects onto

    P = { x in R^k : lb <= Z x <= ub }

Stack the bounds into half-spaces ``A = [Z; -Z]``, ``b = [ub; -lb]``, so every constraint
reads ``a_j . x <= b_j`` and, with rows normalised,

    d_j = (a_j . x - b_j) / ||a_j||

is the signed distance from ``x`` to face ``j`` -- positive when violated.

Each of ``depth`` blocks clears the largest remaining violation. It moves along the part of
that face's normal orthogonal to the normals already used, so the move cannot undo an
earlier one, and ``d`` is carried forward by a rank-1 update rather than recomputed. Once
the selected faces are all cleared the total displacement lies in the span of their
normals, which is the stationarity condition of the projection -- so the result is the
exact projection whenever the greedily selected faces are the true active set. Selection is
greedy, with no multiplier-sign test and no facet dropping, so that is not guaranteed in
general; feasibility is, and it is the property the caller needs.

The loop is unrolled to exactly ``depth`` blocks with no early exit, no branching and no
solve. Two expressions stand in for the branches an early exit would need, and must not be
"simplified":

- ``relu(dj)``: inside ``P`` every ``dj <= 0``, so ``t = 0`` and the block is an exact
  no-op. This is the early exit, and what makes the layer bit-exactly the identity on ``P``.
- ``nq / (nq**2 + eps)``: when the chosen normal is parallel to one already used, ``nq -> 0``
  so ``t -> 0`` and the block is a no-op. Writing ``1/nq`` behind a guard breaks this.

``tol_rank`` is the load-bearing knob. A block clearing face *j* moves by ``dj / nq``, so
``nq`` -- the cosine between that face's normal and the direction actually available -- is
how much a block can amplify the violation it is fixing. ``on_rank_deficient`` decides what
happens below the threshold: ``'restart'`` discards the spent directions for that sample and
takes one plain unit move, which both unsticks the case where a later move pushes a
coordinate back out through the *opposite* face of the same reaction (whose normal is
antiparallel to one already used, so ``nq`` stays 0 and the facet is otherwise re-selected
forever) and keeps the step bounded. ``'noop'`` leaves the block idle. Both are branch-free:
the choice is made once in ``__init__``, and inside a block it is a ``torch.where``.

An exact move lands *on* a face, where rounding the result to float32 can put it back
outside. ``interior_ulps`` overshoots by a margin in units of the output dtype's eps. The
margin is applied only where the face was violated -- gated on ``dj > 0`` rather than folded
into ``b`` -- so a point already in ``P`` is untouched, and it is capped at a fraction of the
reaction's own interval so both faces together cannot empty a narrow box.

The recursion is local: it is exact and nearly free on a state near ``P``, but from far
outside the residual violation can grow with depth. Use it to refine a trained model's
output, not as a layer inside training. Autograd needs no custom backward -- every op is
differentiable and the ``argmax`` index is correctly a constant, since within one selection
pattern the map is affine. Size ``depth`` with :func:`active_set_sizes` on the states the
caller will actually project; an idle block costs compute and nothing else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODES = ("restart", "noop")

#: Overshoot each move by this many units of the output dtype's eps, so the endpoint lands
#: inside the face rather than on it. Matches the margin the removed gauge latent used.
_INTERIOR_ULPS = 1024

#: A move may consume at most this fraction of a reaction's interval as interior margin,
#: so that both faces together cannot empty a narrow box.
_MARGIN_FRACTION = 0.25

#: A row of Z this small relative to the largest is empty: the reaction is pinned to its
#: offset and the latent cannot move it. Deliberately not tol_rank -- that one bounds the
#: step amplification and wants to be ~0.1, which would discard real facets.
_ZERO_ROW_REL = 1e-10


class FluxProjection(nn.Module):
    """Projects a latent vector onto ``{x : lb <= Z x <= ub}`` in ``depth`` orthogonal moves.

    Holds non-persistent buffers only -- no trainable parameters -- so it contributes nothing
    to the state dict and is rebuilt from the null-space data at load time. That keeps the
    ``.sfx`` exporter's regenerated-key check untouched, which is where the removed gauge
    latent left a landmine by registering persistent buffers it never declared.
    """

    def __init__(self, Z, lb, ub, depth=16, eps=1e-12, tol_rank=0.1,
                 on_rank_deficient="restart", interior_ulps=_INTERIOR_ULPS):
        """Build the half-space form of the polytope.

        Args:
            Z: (n, k) basis mapping latent to the constrained space. Full column rank.
            lb: (n,) lower bounds on ``Z x``.
            ub: (n,) upper bounds on ``Z x``.
            depth: number of moves; the loop is unrolled to exactly this many and never
                exits early, since an idle block is exactly a no-op. Set it from
                ``max(active_set_sizes(...))`` on a representative batch at the training
                noise level -- it grows with how far outside the model predicts, so it is
                larger early in training. Over-provisioning costs only compute.
            eps: guard in the step size; keeps a degenerate move finite without a branch.
            tol_rank: below this the Gram-Schmidt remainder counts as rank-deficient and
                the move restarts instead. It is the one knob that bounds the step: a move
                clearing face *j* has length ``dj / nq``, so ``nq`` is what a block can
                amplify by, and letting it reach 1e-7 lets one block throw the state 1e7
                times further than the violation it was fixing. Harmless once predictions
                are nearly feasible and ``dj`` is tiny; divergent during training, where
                ``dj`` is O(1). Keep it near 0.1 unless you have measured otherwise.
            on_rank_deficient: ``'restart'`` or ``'noop'``; see the module docstring.
            interior_ulps: interior margin per move, in units of the output dtype's eps.
                0 lands exactly on the face.
        """
        super().__init__()
        if on_rank_deficient not in _MODES:
            raise ValueError(
                f"on_rank_deficient must be one of {_MODES}, got {on_rank_deficient!r}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        Z = np.asarray(Z, dtype=np.float64)
        lb = np.asarray(lb, dtype=np.float64).ravel()
        ub = np.asarray(ub, dtype=np.float64).ravel()
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2-D (n, k), got shape {Z.shape}")
        if lb.shape != (Z.shape[0],) or ub.shape != (Z.shape[0],):
            raise ValueError(
                f"lb/ub must both be ({Z.shape[0]},) to match Z, "
                f"got {lb.shape} and {ub.shape}")
        if np.linalg.matrix_rank(Z) < Z.shape[1]:
            raise ValueError(
                f"Z has {Z.shape[1]} columns but rank {np.linalg.matrix_rank(Z)}; the "
                "projection needs full column rank for the polytope to be full-dimensional.")
        if (ub < lb).any():
            i = int(np.argmin(ub - lb))
            raise ValueError(
                f"ub < lb at coordinate {i}: lb {lb[i]:.6g} > ub {ub[i]:.6g}, "
                "so the polytope is empty.")

        self.depth = int(depth)
        self.eps = float(eps)
        self.tol_rank = float(tol_rank)
        self.on_rank_deficient = on_rank_deficient
        self.interior_ulps = float(interior_ulps)
        self.n, self.k = Z.shape

        # split_constant_directions() pins the reactions no training sample moves along and
        # folds them into flux_offset, leaving their Z rows empty. Such a row constrains
        # nothing reachable: d_j = (0 - b_j)/0. Left in, it is inf and argmax locks onto it
        # every step, so the layer emits garbage on the first move. Drop them -- but only
        # after checking the constant they were folded to is actually inside its own box,
        # since nothing downstream can fix that and it would fail silently.
        row_norm = np.linalg.norm(Z, axis=1)
        live = row_norm > _ZERO_ROW_REL * max(row_norm.max(), 1.0)
        pinned_bad = ~live & ((lb > 0.0) | (ub < 0.0))
        if pinned_bad.any():
            i = int(np.argmax(pinned_bad))
            raise ValueError(
                f"reaction {i} has an empty row in Z -- it is pinned to its offset -- but "
                f"its shifted bounds [{lb[i]:.6g}, {ub[i]:.6g}] do not contain 0, so the "
                "polytope is empty. The flux bounds and flux_offset disagree.")
        self.n_live = int(live.sum())

        Zl, lbl, ubl = Z[live], lb[live], ub[live]
        idx = np.flatnonzero(live)

        # Every constraint becomes a_j . x <= b_j, then rows are normalised so d is a
        # distance rather than a scaled residual.
        A = np.vstack([Zl, -Zl])                       # (2*n_live, k)
        b = np.concatenate([ubl, -lbl])                # (2*n_live,)
        An = np.linalg.norm(A, axis=1)

        # Interior margin per face, in the same normalised units as d. Relative to the
        # bound's own magnitude, capped so both faces cannot consume a narrow interval.
        ref = np.maximum(np.abs(b), 1.0) / An
        cap = _MARGIN_FRACTION * np.concatenate([ubl - lbl, ubl - lbl]) / An

        self.register_buffer('Z', torch.tensor(Zl), persistent=False)
        self.register_buffer('lb', torch.tensor(lbl), persistent=False)
        self.register_buffer('ub', torch.tensor(ubl), persistent=False)
        self.register_buffer('Ah', torch.tensor(A / An[:, None]), persistent=False)
        self.register_buffer('bn', torch.tensor(b / An), persistent=False)
        self.register_buffer('margin_ref', torch.tensor(ref), persistent=False)
        self.register_buffer('margin_cap', torch.tensor(cap), persistent=False)
        self.register_buffer('live_idx', torch.tensor(idx), persistent=False)

    def forward(self, x, return_pattern=False):
        """Project each row of *x* onto the polytope.

        Args:
            x: (B, k) latent vectors.
            return_pattern: also return the selection pattern -- one
                ``(j, violated, full_rank)`` triple of (B,) tensors per block. The map is
                affine only within a fixed pattern across *all* blocks, so this is what
                tells you whether two inputs are in the same affine cell.

        Returns:
            (B, k) in the dtype of *x*; bit-exactly *x* for rows already in ``P``. With
            *return_pattern*, ``(x', pattern)``.
        """
        work = torch.promote_types(x.dtype, torch.float64)
        out_dtype = x.dtype
        xx = x.to(work)
        Ah = self.Ah.to(work)

        # The margin is sized against the dtype the caller will read back, not the dtype the
        # geometry runs in -- rounding on the way out is what it exists to survive.
        ulp = self.interior_ulps * torch.finfo(out_dtype).eps
        margin = torch.minimum(ulp * self.margin_ref.to(work), self.margin_cap.to(work))

        d = xx @ Ah.T - self.bn.to(work)              # (B, 2n) >0 = violated
        U = []                                        # orthonormal directions spent so far
        pattern = []

        for _ in range(self.depth):                   # unrolled, fixed count, no early exit
            j = d.argmax(-1)                          # (B,)  longest remaining violation
            dj = d.gather(1, j[:, None])              # (B, 1)
            a = Ah[j]                                 # (B, k) its unit normal

            # Modified Gram-Schmidt: one direction at a time, which is why U stays a Python
            # list and is never assigned into a preallocated tensor that requires grad.
            p = a
            for u_prev in U:
                p = p - (p * u_prev).sum(-1, keepdim=True) * u_prev
            nq = p.norm(dim=-1, keepdim=True)         # (B, 1) part not already spent
            keep = nq > self.tol_rank                 # (B, 1)

            if self.on_rank_deficient == "restart":
                # Spending the history is what unsticks the antiparallel sign flip. Rows of
                # Ah are unit, so the plain move has nq = 1 and clears the facet exactly.
                U = [u_prev * keep for u_prev in U]
                u = torch.where(keep, p / nq.clamp_min(self.eps), a)
                nq = torch.where(keep, nq, torch.ones_like(nq))
                # u is a unit direction either way here, so it is always worth keeping. It
                # must be: the next block has to stay orthogonal to the move just made, and
                # masking it to zero would let that move be silently undone.
                spent = u
            else:
                u = p / nq.clamp_min(self.eps)
                spent = u * keep           # a no-op move spent nothing

            # relu(dj) is the no-op guard; the margin rides on (dj > 0) rather than on dj so
            # a feasible row keeps t exactly 0 and the block stays bit-exactly the identity.
            over = F.relu(dj) + margin[j][:, None] * (dj > 0).to(work)
            t = over * nq / (nq ** 2 + self.eps)      # (B, 1) -- see module docstring
            xx = xx - t * u
            U.append(spent)
            d = d - t * (u @ Ah.T)                    # (B, 2n) rank-1 update, one matvec
            if return_pattern:
                pattern.append((j, dj.squeeze(-1) > 0, keep.squeeze(-1)))

        if return_pattern:
            return xx.to(out_dtype), tuple(pattern)
        return xx.to(out_dtype)

    def violation(self, x):
        """Largest bound violation of ``Z x``, in bound units, against the *true* bounds.

        Negative means strictly inside, with that much slack on the tightest face. Reactions
        with an empty row in ``Z`` are excluded -- the latent cannot move them.

        Args:
            x: (B, k) latent vectors.

        Returns:
            (B,) ``max_j max(Z x - ub, lb - Z x)``.
        """
        work = torch.promote_types(x.dtype, torch.float64)
        v = x.to(work) @ self.Z.to(work).T
        return torch.maximum(v - self.ub.to(work), self.lb.to(work) - v).amax(-1)


def active_set_sizes(Z, lb, ub, X, depth=64, tol=1e-9, on_rank_deficient="restart",
                     tol_rank=0.1):
    """How many moves each sample actually needs. Offline helper for choosing ``depth``.

    Same recursion as :meth:`FluxProjection.forward`, but a real loop with ``break`` when
    ``max(d) <= tol``, in numpy and float64, and with no interior margin. Never used in
    training.

    Args:
        Z: (n, k) basis.
        lb: (n,) lower bounds on ``Z x``.
        ub: (n,) upper bounds on ``Z x``.
        X: (B, k) latent vectors to measure -- model *predictions* at the training noise
            level, not training targets. Targets are feasible by construction, so the count
            on them is 0 and says nothing about the depth the layer needs.
        depth: cap on the loop; a sample needing more is reported as not converged.
        tol: feasibility threshold on the normalised distance. Do not set this to 0 --
            the trailing moves then only polish a residual already at the float64 noise
            floor, so the count balloons well past the depth the layer actually needs.
        on_rank_deficient: matches the layer being sized.
        tol_rank: matches the layer being sized.

    Returns:
        (B,) int array. Non-negative is the number of moves that sample needed. Negative
        means it did not reach feasibility, with magnitude the number of moves attempted --
        under ``'noop'`` that is the antiparallel stall, which no extra depth will fix.
    """
    if on_rank_deficient not in _MODES:
        raise ValueError(
            f"on_rank_deficient must be one of {_MODES}, got {on_rank_deficient!r}")

    Z = np.asarray(Z, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))

    row_norm = np.linalg.norm(Z, axis=1)
    live = row_norm > _ZERO_ROW_REL * max(row_norm.max(), 1.0)
    Z, lb, ub = Z[live], lb[live], ub[live]

    A = np.vstack([Z, -Z])
    b = np.concatenate([ub, -lb])
    An = np.linalg.norm(A, axis=1)
    Ah, bn = A / An[:, None], b / An

    sizes = np.empty(X.shape[0], dtype=np.int64)
    for i, x in enumerate(X):
        d = Ah @ x - bn
        U = []
        moves = 0
        converged = False
        for _ in range(depth):
            if d.max() <= tol:
                converged = True
                break
            j = int(np.argmax(d))
            p = Ah[j].copy()
            for u_prev in U:
                p -= (p @ u_prev) * u_prev
            nq = np.linalg.norm(p)
            if nq <= tol_rank:
                if on_rank_deficient == "noop":
                    moves += 1                        # frozen; depth will not help
                    break
                U, p, nq = [], Ah[j].copy(), 1.0
            u = p / nq
            t = d[j] / nq
            x = x - t * u
            U.append(u)
            d = d - t * (Ah @ u)
            moves += 1
        else:
            converged = d.max() <= tol
        sizes[i] = moves if converged else -moves
    return sizes
