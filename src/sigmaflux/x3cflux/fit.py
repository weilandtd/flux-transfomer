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

#: What Ipopt reads as "no bound" on a variable.
_IPOPT_INF = 2e19

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

    Without them Ipopt can walk into a branch-point configuration and segfault.
    This is a genuine asymmetry with the L-BFGS-B fit, which takes no cobra bounds
    at all, so a smaller search space flatters x3cflux wherever the two are compared.

    Args:
        facts: ``model_facts.json`` dict.
        conv: :class:`~sigmaflux.x3cflux.convert.Converted` for the model.
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

    Defined inside a function since it subclasses an x3cflux type and this module
    must stay importable without x3cflux installed. The factor of 2 corrects
    ``compute_linearized_hessian``, which returns ``J^T W J`` while the loss's
    Hessian is ``2 J^T W J``; ``lagrange`` is ignored because every constraint here
    is linear.

    Args:
        x: the imported ``x3cflux`` module, supplying the base problem class.
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
    """Ipopt from one start, keeping the status cyipopt reports.

    Args:
        simulator: x3cflux simulator built from the model's configurations.
        start: initial free parameter vector.
        max_iter: Ipopt iteration cap.
        bounds: extra ``{parameter: (lo, hi)}`` flux bounds, if any. Bounds on free
            parameters become Ipopt *variable* bounds, which every iterate respects;
            the rest can only be inequality rows, which iterates may cross.
        gauss_newton: supply an exact Gauss-Newton Hessian instead of L-BFGS.
        **kwargs: extra Ipopt options, overriding :data:`IPOPT_DEFAULTS`.
    """
    import cyipopt
    import x3cflux as x
    import x3cflux.utilities  # noqa: F401  -- provides x.utilities

    ineq = simulator.parameter_space.inequality_system
    matrix, bound = ineq.matrix, ineq.bound

    # These bounds exist to keep Ipopt out of configurations the simulator cannot
    # evaluate -- it aborts inside x3cflux rather than raising. A general inequality
    # row does not do that: Ipopt satisfies those only at the solution and is free to
    # cross them on the way. A *variable* bound holds at every iterate, so whatever can
    # be expressed as one is, and only bounds on dependent parameters stay as rows.
    free_index = {n: i for i, n in enumerate(simulator.parameter_space.free_parameter_names)}
    lower = np.full(len(free_index), -_IPOPT_INF)
    upper = np.full(len(free_index), _IPOPT_INF)
    dependent = {}
    for name, (lo, hi) in (bounds or {}).items():
        slot = free_index.get(name)
        if slot is None:
            dependent[name] = (lo, hi)
        else:
            lower[slot], upper[slot] = max(lower[slot], lo), min(upper[slot], hi)
    if dependent:
        lhs, rhs = x.utilities.get_inequalities_from_bounds(simulator, dependent)
        matrix = np.vstack((matrix, lhs))
        bound = np.concatenate((bound, rhs))
    cls = _gauss_newton_problem(x) if gauss_newton else x.IpoptOptimizationProblem
    problem = cyipopt.Problem(
        n=matrix.shape[1], m=matrix.shape[0],
        problem_obj=cls(simulator, matrix), lb=lower, ub=upper, cu=bound)
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

    Build once per benchmark run, then call :meth:`fit` per sample.
    ``labeling_index`` and ``measured_flux_ids`` fix the row order of the
    measurement vectors and must match what the benchmark scores against.
    """

    def __init__(self, facts, labeling_index, measured_flux_ids, noise_std,
                 symmetry=True, use_flux_bounds=True, verbose=True):
        """Convert the model once, up front, for reuse across samples.

        Args:
            facts: ``model_facts.json`` dict.
            labeling_index: ``(tracer, metabolite, 'M+k')`` keys, fixed row order.
            measured_flux_ids: cobra reaction ids with a flux measurement.
            noise_std: MID measurement sigma, applied to every isotopomer.
            symmetry: emulate symmetric pools with half-flux orientation pairs.
            use_flux_bounds: constrain Ipopt with cobra FVA bounds.
            verbose: print the network conversion/check summary.
        """
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

        The simulator is x3cflux's own object, read directly rather than proxied;
        only :class:`~sigmaflux.x3cflux.fluxmap.FluxMap` is ours, since cobra
        reaction identities are the one thing x3cflux cannot know.

        Args:
            noisy_labeling: measured MID values, aligned to ``self.labeling_index``.
            noisy_flux: measured flux values, aligned to ``self.measured_flux_ids``.
            flux_sigma: measurement sigma per entry of ``self.measured_flux_ids``.
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

        ``info['ssr']`` is Ipopt's own residual, not what decides acceptance -- the
        caller recomputes that through our forward model. With *isolate* the solve
        runs in a forked child and returns ``(None, None, {'crashed': reason})`` if
        it dies.

        Args:
            simulator: x3cflux simulator from :meth:`simulator_for`.
            fluxmap: :class:`~sigmaflux.x3cflux.fluxmap.FluxMap` from :meth:`simulator_for`.
            v_init: initial cobra net fluxes.
            r_init: initial cobra reversibilities.
            max_iter: Ipopt iteration cap.
            gauss_newton: supply an exact Gauss-Newton Hessian instead of L-BFGS.
            isolate: run the solve in a forked child, so a crash costs one start.
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

    Ipopt can SIGSEGV from some starting points; isolating each start in a fork
    means a crash costs one start instead of the whole sample, and the child
    inherits the built simulator via copy-on-write so nothing has to be serialised.

    Args:
        call: zero-argument callable to run in the child.
        timeout: seconds to wait for the child before giving up on it.
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
