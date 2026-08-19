"""
Full-flux-space MFA fitting via Ipopt, driven by the in-house EMU forward model.

Optimizes the full flux vector ``v`` directly. ``S·v = 0`` and the FVA-tightened
reaction bounds (``MFAModel.get_flux_bounds``) are passed to Ipopt as explicit
hard constraints; reversibility ``r`` is box-bounded.

``cyipopt`` is imported lazily, inside :func:`fit_mfa_model_ipopt` and
:func:`require`, never at module import time. cyipopt links ``libspral``, which
needs the conda env's ``libstdc++`` (``CXXABI_1.3.15``); if torch is imported
first, it loads the system ``libstdc++`` instead, and a later ``import cyipopt``
fails. :mod:`sigmaflux.pipeline.mfa_entry` imports cyipopt at process entry, ahead
of the torch-pulling driver, for any CLI invocation with ``--backend
native-ipopt``. Do not hoist the cyipopt import to module level.
"""
import logging

import numpy as np
import scipy.sparse as sp
import torch

from .mfa import _mfa_residual, _prepare_mfa_batches
from .utils import time_function

logger = logging.getLogger(__name__)

#: Ipopt option names, passed through cyipopt.Problem.add_option.
IPOPT_DEFAULTS = {
    "hessian_approximation": "limited-memory",
    "jac_c_constant": "yes",
    "mu_strategy": "adaptive",
    "print_level": 0,
}


def available():
    """Whether the optional cyipopt dependency can be imported."""
    import importlib.util
    return importlib.util.find_spec("cyipopt") is not None


def require():
    """Import cyipopt now, raising with install instructions if it's missing."""
    try:
        import cyipopt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the native-ipopt backend needs the optional cyipopt dependency, "
            "which pip cannot build (it needs Ipopt headers at build time): "
            "conda env create -f environment.yml"
        ) from exc


class _MFAIpoptProblem:
    """cyipopt problem_obj for :func:`fit_mfa_model_ipopt`.

    objective/gradient are computed via torch.autograd on ``_mfa_residual``.
    constraints/jacobian implement ``S·v = 0``.
    """

    def __init__(self, model, experiment_batches, batch_data, mask_f,
                inv_n_L, inv_n_F, n, dtype, device):
        self.model = model
        self.experiment_batches = experiment_batches
        self.batch_data = batch_data
        self.mask_f = mask_f
        self.inv_n_L = inv_n_L
        self.inv_n_F = inv_n_F
        self.n = n
        self.dtype = dtype
        self.device = device
        self.n_iter = 0

        S_np = model.stoichiometry.detach().cpu().numpy()
        coo = sp.coo_matrix(S_np)
        self._jac_rows = coo.row.copy()
        self._jac_cols = coo.col.copy()          # columns index v only (0..n-1)
        self._jac_data = coo.data.astype(float)
        self._S_np = S_np

        self._cache_x = None
        self._cache_loss = None
        self._cache_grad = None

    def _eval(self, x):
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache_loss, self._cache_grad
        x_t = torch.tensor(x, dtype=self.dtype, device=self.device, requires_grad=True)
        with torch.enable_grad():
            v, r = x_t[: self.n], x_t[self.n:]
            loss = _mfa_residual(self.model, v, r, self.experiment_batches,
                                 self.batch_data, self.mask_f,
                                 self.inv_n_L, self.inv_n_F)
        grad, = torch.autograd.grad(loss, x_t)
        self._cache_x = x.copy()
        self._cache_loss = float(loss.item())
        self._cache_grad = grad.detach().cpu().numpy()
        return self._cache_loss, self._cache_grad

    def objective(self, x):
        loss, _ = self._eval(x)
        return loss

    def gradient(self, x):
        _, grad = self._eval(x)
        return grad

    def constraints(self, x):
        return self._S_np @ x[: self.n]

    def jacobianstructure(self):
        return self._jac_rows, self._jac_cols

    def jacobian(self, x):
        return self._jac_data

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        self.n_iter = iter_count


@time_function
def fit_mfa_model_ipopt(model,
                        simultanous_experiments=3,
                        max_iter=1000,
                        tol=1e-6,
                        acceptable_tol=1e-3,
                        checkpoint_file=None,
                        labeling_checkpoint_file=None,
                        disp=0,
                        ipopt_options=None):
    """Fit an MFA model in full flux space via Ipopt.

    S·v = 0 is passed to Ipopt as an explicit linear equality constraint. v is
    box-bounded from FVA (``MFAModel.get_flux_bounds``); r is box-bounded from
    the reversibility mask. Warm-starts from
    ``model._fluxes``/``model._reversibility`` directly.

    Args:
        model: MFAModel to fit in place.
        simultanous_experiments: number of experiments forward-simulated per batch.
        max_iter: maximum Ipopt iterations.
        tol: Ipopt's overall NLP error tolerance.
        acceptable_tol: relaxed tolerance Ipopt accepts after acceptable_iter
            iterations of no progress (cyipopt/Ipopt default acceptable_iter).
        checkpoint_file: path to save model checkpoints to, if given.
        labeling_checkpoint_file: path to save predicted labeling to, if given.
        disp: print Ipopt's own iteration log when truthy.
        ipopt_options: extra options passed to ``cyipopt.Problem.add_option``,
            overriding :data:`IPOPT_DEFAULTS`.

    Returns:
        dict with loss, converged, n_iter, ssr_labeling, ssr_fluxes, n_labeling,
        n_fluxes.
    """
    require()
    import cyipopt

    n = model.num_reactions
    dtype, device = model.dtype, model.device
    mask_f = model._reversibility_mask

    (experiment_batches, batch_data, n_L_total, n_F_total,
     inv_n_L, inv_n_F) = _prepare_mfa_batches(model, simultanous_experiments)

    x0 = torch.cat([model._fluxes.detach(), model._reversibility.detach()]).cpu().numpy()

    lb_v, ub_v = model.get_flux_bounds()
    rev_bool = mask_f.bool()
    lb_r = torch.where(rev_bool,
                       torch.full((n,), model.min_rev_flux, dtype=dtype, device=device),
                       torch.zeros(n, dtype=dtype, device=device))
    ub_r = torch.where(rev_bool,
                       torch.full((n,), float("inf"), dtype=dtype, device=device),
                       torch.zeros(n, dtype=dtype, device=device))
    # Ipopt's sentinel for "unbounded" is +/-1e19 (nlp_lower/upper_bound_inf).
    BIG = 1e19
    lb = np.clip(torch.cat([lb_v, lb_r]).cpu().numpy(), -BIG, BIG)
    ub = np.clip(torch.cat([ub_v, ub_r]).cpu().numpy(), -BIG, BIG)

    problem_obj = _MFAIpoptProblem(model, experiment_batches, batch_data, mask_f,
                                   inv_n_L, inv_n_F, n, dtype, device)
    m = model.stoichiometry.shape[0]
    nlp = cyipopt.Problem(
        n=2 * n, m=m, problem_obj=problem_obj,
        lb=lb, ub=ub, cl=np.zeros(m), cu=np.zeros(m),
    )
    options = dict(IPOPT_DEFAULTS)
    options.update(max_iter=max_iter, tol=tol, acceptable_tol=acceptable_tol,
                   print_level=(5 if disp else 0))
    options.update(ipopt_options or {})
    for key, value in options.items():
        nlp.add_option(key, value)

    logger.info(
        f"MFA Ipopt: {n} reactions (full flux space), {m} mass-balance constraints, "
        f"{len(model.experiments)} experiments, max_iter={max_iter}, tol={tol}"
    )

    x_opt, info = nlp.solve(x0)

    v_opt = torch.tensor(x_opt[:n], dtype=dtype, device=device)
    r_opt = torch.tensor(x_opt[n:], dtype=dtype, device=device)
    with torch.no_grad():
        model._fluxes.copy_(v_opt)
        model._reversibility.copy_(r_opt)

    status = int(info["status"])
    converged = status in (0, 1)     # 0=Solve_Succeeded, 1=Solved_To_Acceptable_Level
    n_iter = problem_obj.n_iter

    sv_max = float((model.stoichiometry @ model._fluxes).abs().max().item())
    logger.info(
        f"MFA Ipopt done: status={status} ({info.get('status_msg')}), "
        f"iter={n_iter}, |S·v|_max={sv_max:.2e}"
    )

    if checkpoint_file is not None:
        model.save_checkpoint(file=checkpoint_file,
                              meta_data={"chi2": float(info["obj_val"]), "converged": converged})

    if labeling_checkpoint_file is not None:
        with torch.no_grad():
            pred_labeling = model.forward()
        torch.save(pred_labeling.cpu(), labeling_checkpoint_file)

    with torch.no_grad():
        rchi2_L, rchi2_F = _mfa_residual(model, v_opt, r_opt, experiment_batches,
                                         batch_data, mask_f, inv_n_L, inv_n_F,
                                         return_blocks=True)

    return {
        "loss":         float(info["obj_val"]),    # raw SSR at the optimum
        "converged":    bool(converged),
        "n_iter":       int(n_iter),
        "ssr_labeling": float(rchi2_L),
        "ssr_fluxes":   float(rchi2_F),
        "n_labeling":   n_L_total,
        "n_fluxes":     n_F_total,
    }
