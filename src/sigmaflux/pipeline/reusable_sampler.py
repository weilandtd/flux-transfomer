"""A cobra hit-and-run sampler that is built once and reused across bound changes.

``cobra.sampling.sample()`` builds a brand new ``OptGPSampler``/``ACHRSampler`` on every
call: a deep ``model.copy()`` plus ``constraint_matrices()`` and a nullspace SVD, none of
which depend on reaction bounds. ``ReusableSampler`` builds that once and, for every
subsequent bounds change, mutates only the sampler's cached ``Problem`` arrays at the
columns for the changed reaction(s) (via cobra's own ``fwd_idx``/``rev_idx``), then
regenerates warmup points and resets ``center``/``prev`` for the new feasible region.

Warmup regeneration (``generate_fva_warmup()``, two LP solves per reaction) is real,
inherent cost -- it cannot be skipped for a state something is about to sample from, since
the running point must stay on the model's equality-constraint manifold, and there's no
cheap way to patch that without re-solving. ``set_bounds(..., warmup=False)`` skips it only
for restoring bounds nothing will sample from before the next change replaces them (see
``temporary_bounds``) -- the one case where the regen would be computed and immediately
discarded.

Seeding contract: ``sample(n, seed=...)`` reproduces the independently-seeded-per-call
behaviour the pipeline relies on -- a distinct seed per knockout/stratum, each giving a
result as if drawn from a freshly constructed sampler. Reseeding the RNG alone isn't enough
for that: cobra's samplers keep chain-progress state (``center``/``prev``/``n_samples``)
that persists across ``.sample()`` calls, so a seeded call also resets that state. Passing
``seed=None`` instead lets the chain continue from wherever it left off -- cheaper and
statistically defensible, but not bit-reproducible call to call -- and isn't used by the
current pipeline.

Only reaction-bounds changes are supported. Fixing a reaction to a *nonzero* value changes
the shape of the sampler's equality constraints/nullspace, which can't be patched by array
mutation; that's detected and falls back to a full rebuild (see ``_rebuild``).
"""

import warnings
from contextlib import contextmanager

import numpy as np
from cobra.sampling import ACHRSampler, OptGPSampler

_METHODS = ("optgp", "achr")


class ReusableSampler:
    """Wraps one cobra hit-and-run sampler, built once, with in-place bounds updates."""

    def __init__(self, model, method, thinning, processes=1, seed=None):
        if method not in _METHODS:
            raise ValueError(f"ReusableSampler supports {_METHODS}, got {method!r}")
        self._method = method
        self._processes = processes
        self._thinning = thinning
        self._base_seed = seed
        self._sampler = self._build(model)
        self._col = self._build_column_index()

    def _build(self, model):
        if self._method == "optgp":
            return OptGPSampler(model, processes=self._processes,
                                 thinning=self._thinning, seed=self._base_seed)
        return ACHRSampler(model, thinning=self._thinning, seed=self._base_seed)

    def _build_column_index(self):
        fwd, rev = self._sampler.fwd_idx, self._sampler.rev_idx
        return {r.id: (int(fwd[i]), int(rev[i]))
                for i, r in enumerate(self._sampler.model.reactions)}

    @property
    def model(self):
        """The sampler's own internal model copy -- mutate/inspect bounds through this."""
        return self._sampler.model

    def set_bounds(self, bounds, warmup=True):
        """Apply new (lb, ub) bounds for the given reactions, in place.

        Args:
            bounds: mapping of reaction id to (lower_bound, upper_bound).
            warmup: regenerate warmup points (and reset center/prev) for the new bounds.
                Default True -- the only safe choice whenever something is about to sample
                from the resulting state. Pass False only when nothing will sample before
                the next bounds change (e.g. restoring old bounds in ``temporary_bounds``'s
                exit) -- skipping it there is free, since a full regen would immediately be
                superseded by the next bounds change's own regen anyway.
        """
        problem = self._sampler.problem
        nonzero_fixed = []
        for rxn_id, (lb, ub) in bounds.items():
            rxn = self._sampler.model.reactions.get_by_id(rxn_id)
            rxn.bounds = (lb, ub)  # cobra's own setter -- correct fwd/rev variable split
            fwd_idx, rev_idx = self._col[rxn_id]
            for idx, var in ((fwd_idx, rxn.forward_variable), (rev_idx, rxn.reverse_variable)):
                problem.variable_bounds[0, idx] = var.lb
                problem.variable_bounds[1, idx] = var.ub
                fixed = (var.ub - var.lb) < self._sampler.feasibility_tol
                problem.variable_fixed[idx] = fixed
                if fixed and var.lb != 0:
                    nonzero_fixed.append(rxn_id)

        if nonzero_fixed:
            self._rebuild(nonzero_fixed)
            return

        if not warmup:
            return

        self._sampler.generate_fva_warmup()
        self._sampler.center = self._sampler.warmup.mean(axis=0)
        if self._method == "achr":
            self._sampler.prev = self._sampler.center.copy()

    @contextmanager
    def temporary_bounds(self, bounds):
        """Apply *bounds*, yield, then restore whatever bounds those reactions had before.

        The restore-on-exit skips warmup regeneration (``warmup=False``): nothing samples
        in the restored state before the caller's next bounds change replaces it anyway, so
        regenerating here would just be discarded unread.

        Args:
            bounds: mapping of reaction id to (lower_bound, upper_bound).
        """
        old = {rxn_id: self._sampler.model.reactions.get_by_id(rxn_id).bounds
               for rxn_id in bounds}
        self.set_bounds(bounds)
        try:
            yield self
        finally:
            self.set_bounds(old, warmup=False)

    def sample(self, n, seed=None):
        """Draw *n* samples. See the module docstring for the seeding contract."""
        if seed is not None:
            # Reproduce the old per-call contract: a seeded call behaves like sampling
            # from a freshly constructed sampler at the current bounds, not a
            # continuation of this instance's running chain -- so reset the chain's own
            # progress state, not just the RNG.
            center = self._sampler.warmup.mean(axis=0)
            self._sampler.center = center
            self._sampler.n_samples = 0
            if self._method == "optgp":
                self._sampler._seed = seed % np.iinfo(np.int32).max
            else:
                self._sampler.prev = center.copy()
                np.random.seed(seed % np.iinfo(np.int32).max)
        return self._sampler.sample(n)

    def _rebuild(self, nonzero_fixed_ids):
        warnings.warn(
            f"ReusableSampler: reaction(s) {nonzero_fixed_ids} fixed to a nonzero bound -- "
            "this changes the sampler's equality-constraint structure, which cannot be "
            "patched in place. Falling back to a full sampler rebuild for this call "
            "(not expected under current sigmaflux usage: knockouts fix to zero and "
            "stratified bins narrow within the existing FVA range).",
            RuntimeWarning, stacklevel=3,
        )
        self._sampler = self._build(self._sampler.model)
        self._col = self._build_column_index()
