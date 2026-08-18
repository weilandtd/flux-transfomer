"""Fit an MFA problem with x3cflux/Ipopt, as a drop-in for ``fit_mfa_model``.

The backend is an object built once per run and reused across samples, the way
:class:`~sigmaflux.core.model.MFAModel` is: the network conversion and the
constraint set depend only on the model, so only the measurements are rebuilt per
sample. Nothing is written to disk and nothing is serialised through files -- the
caller hands in arrays and gets arrays back.

Scoring is deliberately *not* done here. :meth:`X3CFluxBackend.fit` returns the
fitted cobra ``(v, r)``, and the benchmark driver pushes that through our own EMU
forward model to compute the residual, exactly as it does for the native fit. So
both arms are scored by the same code against the same noise, and the comparison
cannot drift; the price is that it is only meaningful while the two forward models
agree, which is what the labeling agreement check measures.

``run_optimization`` throws away cyipopt's ``info``, so the handful of lines it
wraps are reimplemented here to keep the status and iteration count.
"""

import time

import numpy as np

from .build_config import build_configurations
from .check_network import assert_network
from .convert import convert
from .fluxmap import FluxMap

#: Ipopt options. ``hessian_approximation`` is dropped when a Gauss-Newton Hessian
#: is supplied, since "exact" is then the default.
IPOPT_DEFAULTS = {
    "tol": 1e-6,
    "acceptable_tol": 1e-3,
    "hessian_approximation": "limited-memory",
    "jac_d_constant": "yes",
    "bound_relax_factor": 0.0,
    "mu_strategy": "adaptive",
    "print_level": 0,
}


def flux_bounds(facts, conv):
    """cobra FVA bounds, expressed over the x3cflux reactions.

    Ipopt otherwise walks into configurations that zero out a branch point, where
    x3cflux reports "the current labeling system is not numerically solvable" and
    then segfaults. Its own advice is to constrain the fluxes of branching
    pathways, and the prepared model already carries FVA-tightened bounds (70
    reactions have lb > 0), so those are the natural choice.

    This is a genuine asymmetry with our L-BFGS-B fit, which imposes no cobra
    bounds at all -- it optimises an unbounded null-space coordinate and cannot
    take them, since L-BFGS-B has no general linear constraints. A smaller search
    space flatters x3cflux, so it has to be stated wherever the two are compared.
    """
    bounds = {}
    for r in facts["reactions"]:
        rid, lb, ub = r["id"], r["lower_bound"], r["upper_bound"]
        for sub, share in conv.parents.get(rid, [(rid, 1.0)]):
            f = conv.scale[rid] * share
            lo, hi = sorted((f * lb, f * ub))
            bounds[f"{sub}.n"] = (lo, hi)
    return bounds


def _gauss_newton_problem(x):
    """Build the Ipopt problem class supplying a Gauss-Newton Hessian.

    Defined inside a function because it subclasses an x3cflux type, and this
    module must stay importable while x3cflux is only optionally installed.

    MFA is a weighted nonlinear least-squares problem and x3cflux already exposes
    the machinery -- ``compute_linearized_hessian`` returns ``J^T W J`` -- so
    approximating the Hessian with limited-memory BFGS throws away structure that
    is there for free.

    Two details:

    * the loss is ``sum(r^2)``, whose Hessian is ``2 J^T W J``, while
      ``compute_linearized_hessian`` returns ``J^T W J``. Verified by finite
      differencing the analytic gradient: the returned matrix is exactly half the
      true Hessian. Hence the factor 2.
    * every constraint here is linear (the stoichiometry is eliminated into the
      free parameters, and the bounds are linear inequalities), so the multipliers
      contribute nothing to the Hessian of the Lagrangian and ``lagrange`` is
      ignored.
    """

    class GaussNewtonProblem(x.IpoptOptimizationProblem):
        def hessianstructure(self):
            return np.tril_indices(
                self.simulator.parameter_space.num_free_parameters)

        def hessian(self, x_, lagrange, obj_factor):
            h = np.asarray(self.simulator.compute_linearized_hessian(x_), float)
            return obj_factor * 2.0 * h[np.tril_indices(h.shape[0])]

    return GaussNewtonProblem


def optimize(simulator, start, max_iter=5000, bounds=None, gauss_newton=True,
             **kwargs):
    """Ipopt from one start, keeping the status cyipopt reports."""
    import cyipopt
    import x3cflux as x
    import x3cflux.utilities  # noqa: F401  -- provides x.utilities

    ineq = simulator.parameter_space.inequality_system
    matrix, bound = ineq.matrix, ineq.bound
    if bounds:
        lhs, rhs = x.utilities.get_inequalities_from_bounds(simulator, bounds)
        matrix = np.vstack((matrix, lhs))
        bound = np.concatenate((bound, rhs))
    cls = _gauss_newton_problem(x) if gauss_newton else x.IpoptOptimizationProblem
    problem = cyipopt.Problem(
        n=matrix.shape[1], m=matrix.shape[0],
        problem_obj=cls(simulator, matrix), cu=bound)
    options = dict(IPOPT_DEFAULTS, max_iter=max_iter, **kwargs)
    if gauss_newton:
        options.pop("hessian_approximation", None)
    for key, value in options.items():
        problem.add_option(key, value)

    started = time.time()
    theta, info = problem.solve(np.asarray(start, float))
    status_msg = info["status_msg"]
    return {
        "theta": np.asarray(theta, float),
        "ssr": float(info["obj_val"]),
        "status": int(info["status"]),
        "status_msg": (status_msg.decode() if isinstance(status_msg, bytes)
                       else str(status_msg)),
        "seconds": time.time() - started,
    }


class X3CFluxBackend:
    """A converted x3cflux model, reusable across samples.

    Build once per benchmark run -- conversion and the constraint set depend only
    on the model -- then call :meth:`fit` per sample. ``labeling_index`` and
    ``measured_flux_ids`` fix the row order of the measurement vectors handed to
    :meth:`fit`, and must be the same objects the benchmark scores against.
    """

    def __init__(self, facts, labeling_index, measured_flux_ids, noise_std,
                 symmetry=True, use_flux_bounds=True, verbose=True):
        self.facts = facts
        self.labeling_index = [tuple(k) for k in labeling_index]
        self.measured_flux_ids = list(measured_flux_ids)
        self.noise_std = noise_std
        self.conv = convert(facts, symmetry=symmetry)
        # x3cflux ships no validator and a violated invariant is a SIGSEGV inside
        # create_simulator_from_data, so this must run before any simulator build.
        assert_network(self.conv.network_data, verbose=verbose)
        self.bounds = flux_bounds(facts, self.conv) if use_flux_bounds else None

    @property
    def reaction_ids(self):
        return [r["id"] for r in self.facts["reactions"]]

    def simulator_for(self, noisy_labeling, noisy_flux, flux_sigma):
        """``(simulator, fluxmap)`` for one sample's measurements.

        The simulator is x3cflux's own object and already carries the network, the
        configuration and the simulation -- ``compute_measurements``,
        ``compute_loss`` and ``parameter_space`` are read off it directly rather
        than proxied. Only :class:`~sigmaflux.x3cflux.fluxmap.FluxMap` is ours,
        because the cobra reaction identities are the one thing x3cflux cannot know.
        """
        import x3cflux as x

        configs = build_configurations(
            self.facts, self.conv, self.labeling_index, self.measured_flux_ids,
            noisy_labeling, noisy_flux, flux_sigma, self.noise_std)
        simulator = x.create_simulator_from_data(self.conv.network_data, configs)
        return simulator, FluxMap(self.facts, self.conv, simulator.parameter_space)

    def fit(self, simulator, fluxmap, v_init, r_init, max_iter=5000,
            gauss_newton=True, isolate=True):
        """Fit from one start; returns ``(v, r, info)`` in cobra coordinates.

        ``info`` carries Ipopt's status, its own SSR and the wall time. The residual
        that decides acceptance is computed by the caller through our forward model,
        not from ``info['ssr']``.

        With *isolate*, the solve runs in a forked child (see :func:`run_isolated`)
        and returns ``(None, None, {'crashed': reason})`` if it dies.
        """
        def solve():
            result = optimize(simulator, fluxmap.to_x3c(v_init, r_init),
                              max_iter=max_iter, bounds=self.bounds,
                              gauss_newton=gauss_newton)
            # Map back inside the child: theta is meaningless without the parameter
            # space, and (v, r) are small enough to hand back cheaply.
            v, r = fluxmap.from_x3c(result.pop("theta"))
            return v, r, result

        if not isolate:
            return solve()
        outcome = run_isolated(solve)
        if outcome is None:
            return None, None, {"crashed": "child died"}
        return outcome


def run_isolated(call, timeout=7200):
    """Run *call* in a forked child; return its result, or None if the child died.

    Ipopt segfaults from some starting points -- a real SIGSEGV inside x3cflux, not
    an exception and not memory pressure (observed at 318 MB RSS). In one process
    per sample that takes the whole sample down and it vanishes from the results,
    which is precisely the selection bias this benchmark exists to measure: the
    samples that crash are unlikely to be a random subset. So a crash costs one
    start, and the caller records it.

    Fork rather than a fresh interpreter: the child inherits the built simulator
    through copy-on-write, so nothing has to be serialised or written to disk, and
    only the small ``(v, r, info)`` result travels back through the pipe.
    """
    import multiprocessing

    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def target():
        try:
            sender.send(("ok", call()))
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            sender.send(("error", repr(exc)))
        finally:
            sender.close()

    child = context.Process(target=target)
    child.start()
    sender.close()
    try:
        ready = receiver.poll(timeout)
        payload = receiver.recv() if ready else None
    except EOFError:                                   # child died before sending
        payload = None
    finally:
        receiver.close()
        child.join(timeout=30)
        if child.is_alive():
            child.terminate()
            child.join()

    if payload is None or payload[0] != "ok":
        return None
    return payload[1]
