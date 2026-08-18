"""Map cobra (net flux, reversibility) <-> x3cflux free parameters.

Four transformations compose here and any one of them silently wrong invalidates
the whole comparison, so this is deliberately one small module with a
two-directional round-trip test.

1. **Reaction identity.** The x3cflux reaction list is not our 203: splits add
   sub-reactions, symmetry adds halves, void pools add drains.
2. **Scale.** ``u_r = scale[r] * v_r``; -1 for the flipped EX_ uptakes, and 1/2 for
   FECOOR_m / NADPHBAL_c, whose coefficients were doubled to make them integral.
3. **Split fan-out / fan-in.** A parent's flux populates each sub-reaction at its
   cobra ratio; going back, the parent is read from the reference sub-flux and the
   others are *checked*, not ignored.
4. **Exchange convention.** Ours is ``vf = max(v,0) + r/2``; 13CFLUX's is
   ``vf = max(v.n,0) + v.x``. Hence ``v.x = r/2`` -- derived, not documented, so the
   labeling agreement check has to confirm it. A symmetry half carries a quarter:
   each half has net v/2 and exchange r/4, and
   0.5*(max(v,0) + r/2) = max(v/2,0) + r/4.

Only reactions with scale == 1 ever carry exchange: the flipped EX_ uptakes have
ub <= 0 so cobra's ``reversibility`` (lb < 0 < ub) is False, and the 2x-rescaled
pair is irreversible too. ``_check_exchange_scaling`` asserts that rather than
trusting it.

This module is deliberately free of x3cflux itself -- the parameter space is
passed in -- so the mapping can be unit-tested without the optional dependency.
"""

import numpy as np


class FluxMap:
    def __init__(self, facts, conv, parameter_space):
        """Precompute the reaction/parameter index maps used by both directions.

        Args:
            facts: ``model_facts.json`` dict.
            conv: :class:`~sigmaflux.x3cflux.convert.Converted` for the model.
            parameter_space: x3cflux simulator's ``parameter_space``.
        """
        self.facts = facts
        self.conv = conv
        self.ps = parameter_space
        self.reaction_ids = [r["id"] for r in facts["reactions"]]
        self.param_names = list(parameter_space.parameter_names)
        self.free_names = list(parameter_space.free_parameter_names)
        self._pindex = {n: i for i, n in enumerate(self.param_names)}
        self._free_slots = [self._pindex[n] for n in self.free_names]
        self._void_rows = self._void_balance()
        self._check_exchange_scaling()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_exchange_scaling(self):
        bad = [r["id"] for r in self.facts["reactions"]
               if r["reversible"] and abs(self.conv.scale[r["id"]]) != 1]
        if bad:
            raise AssertionError(
                f"reversible reactions with |scale| != 1 would need the exchange "
                f"rescaled too: {bad}")

    def _void_balance(self):
        """void drain name -> {producing x3cflux reaction: coefficient}.

        A drain's flux is fixed by its pool's balance, so it is not independent;
        it still has to be filled in to build a complete parameter vector.
        """
        rows = {}
        for r in self.conv.network_data.reactions:
            for m in r.product_names:
                if m.startswith("__void_"):
                    terms = rows.setdefault(f"{m}__drain", {})
                    terms[r.name] = terms.get(r.name, 0.0) + 1.0
        return rows

    def subreactions(self, rid):
        """[(x3cflux reaction, share)] for one cobra reaction."""
        return self.conv.parents.get(rid, [(rid, 1.0)])

    # ── cobra -> x3cflux ──────────────────────────────────────────────────────

    def full_vector(self, v, r):
        """Complete x3cflux parameter vector (``parameter_names`` order).

        Args:
            v: cobra net fluxes.
            r: cobra reversibilities.
        """
        v = np.asarray(v, float)
        r = np.asarray(r, float)
        net, xch = {}, {}
        for i, rid in enumerate(self.reaction_ids):
            s = self.conv.scale[rid]
            subs = self.subreactions(rid)
            is_sym = len(subs) > 1 and subs[0][0].endswith("__sym0")
            for (sub, share) in subs:
                net[sub] = s * v[i] * share
                # Exchange only lives on scale == 1 reactions (asserted above).
                # A symmetry half takes a quarter of r, a plain reaction a half.
                xch[sub] = (0.25 if is_sym else 0.5) * r[i] \
                    if self.conv.reversible.get(sub, False) else 0.0
        for drain, terms in self._void_rows.items():
            net[drain] = sum(c * net[src] for src, c in terms.items())
            xch[drain] = 0.0

        self._check_present(net)
        full = np.zeros(len(self.param_names))
        for name, value in net.items():
            full[self._pindex[f"{name}.n"]] = value
        for name, value in xch.items():
            full[self._pindex[f"{name}.x"]] = value
        return full

    def _check_present(self, net):
        """Fail on reactions x3cflux pruned, naming all of them and the likely cause.

        x3cflux silently deletes a reaction whose flux its constraints pin to zero,
        and everything downstream of it; the pool actually at fault is usually not
        the one that happens to be indexed first, so this reports all of them
        rather than a bare KeyError.
        """
        missing = [name for name in net if f"{name}.n" not in self._pindex]
        if not missing:
            return
        orphans = sorted({m for r in self.conv.network_data.reactions
                          for m in r.educt_names} -
                         {m for r in self.conv.network_data.reactions
                          for m in r.product_names})
        raise KeyError(
            f"x3cflux pruned {len(missing)} of {len(net)} reactions, so they have no "
            f"flux parameter: {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}. This is a pinned flux, not a "
            f"conversion error -- look for a pool with no producer, which pins its "
            f"consumer to zero and, through the split ties, everything tied to it. "
            f"Pools consumed but never produced: {orphans[:20]}"
            f"{' ...' if len(orphans) > 20 else ''}")

    def to_x3c(self, v, r):
        """Free parameter vector for cobra fluxes ``v`` and reversibilities ``r``.

        Args:
            v: cobra net fluxes.
            r: cobra reversibilities.
        """
        return self.full_vector(v, r)[self._free_slots]

    # ── x3cflux -> cobra ──────────────────────────────────────────────────────

    def from_x3c(self, theta_free, check_tol=1e-6):
        """(v, r) for a free parameter vector, verifying split consistency.

        Args:
            theta_free: x3cflux free parameter vector.
            check_tol: relative spread tolerated across a split reaction's sub-fluxes.
        """
        theta_free = np.asarray(theta_free, float)
        if theta_free.size != self.ps.num_free_parameters:
            raise ValueError(f"expected {self.ps.num_free_parameters} free parameters, "
                             f"got {theta_free.size}")
        full = np.asarray(self.ps.compute_parameters(theta_free), float).ravel()

        v = np.zeros(len(self.reaction_ids))
        r = np.zeros(len(self.reaction_ids))
        problems = []
        for i, rid in enumerate(self.reaction_ids):
            s = self.conv.scale[rid]
            subs = self.subreactions(rid)
            is_sym = len(subs) > 1 and subs[0][0].endswith("__sym0")
            estimates = [full[self._pindex[f"{sub}.n"]] / (s * share)
                         for sub, share in subs]
            v[i] = estimates[0]
            spread = max(estimates) - min(estimates)
            scale_ref = max(1.0, abs(estimates[0]))
            if spread / scale_ref > check_tol:
                problems.append((rid, spread, estimates))
            if self.conv.reversible.get(subs[0][0], False):
                r[i] = full[self._pindex[f"{subs[0][0]}.x"]] / (0.25 if is_sym else 0.5)
        if problems:
            raise AssertionError(
                f"{len(problems)} split reactions inconsistent across their "
                f"sub-fluxes, e.g. {problems[:3]}")
        return v, r


def cobra_stoichiometry(facts):
    """Dense cobra S and its reaction/metabolite order."""
    rxns = [r["id"] for r in facts["reactions"]]
    mets = [m["id"] for m in facts["metabolites"]]
    mi = {m: i for i, m in enumerate(mets)}
    S = np.zeros((len(mets), len(rxns)))
    for j, r in enumerate(facts["reactions"]):
        for m, c in r["stoichiometry"].items():
            S[mi[m], j] = c
    return S, mets, rxns
