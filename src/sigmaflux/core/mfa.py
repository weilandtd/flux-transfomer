"""
Functions for MFA
"""
from sigmaflux.expansion.emu import OriginSet, EMU
from sigmaflux.core.data import get_model_labeling_data, get_model_flux_data
import numpy as np
import torch
import logging
import time
from scipy.optimize import minimize as scipy_minimize, Bounds
from sigmaflux.core.utils import time_function
# TODO MAKE A CENTRAL DEFINITION
EPSILON = 1e-9

logger = logging.getLogger(__name__)


@time_function
def fit_mfa_model(model,
                  simultanous_experiments=3,
                  max_iter=1000,
                  tol=1e-3,
                  checkpoint_file=None,
                  labeling_checkpoint_file=None,
                  disp=0):
    """Fit an MFA model using chi-squared minimization with null-space reparameterization.

    The mass-balance constraint S·v = 0 is satisfied automatically by parameterizing
    net fluxes as v_net = T @ p, where T = null(S) (right null space of S).
    Exchange fluxes (reversibility) r are bounded below by min_rev_flux.
    Optimization uses scipy L-BFGS-B with autograd-based gradient.

    Objective (reduced χ² per data block, so the few flux measurements are not drowned
    out by the many labeling measurements):
        (1/n_L) Σ (L_sim − L_meas)² / σ²_L  +  (1/n_F) Σ (v_sim − v_meas)² / σ²_v

    Parameters
    ----------
    model : MFAModel
    simultanous_experiments : int
    max_iter : int
    tol : float
        Projected-gradient tolerance (gtol) for L-BFGS-B convergence.
    checkpoint_file, labeling_checkpoint_file : str or None
    disp : int
        Verbosity: 0 = silent, >1 = per-iteration output.

    Returns
    -------
    dict
        {"loss": float, "converged": bool, "n_iter": int,
         "ssr_labeling": float, "ssr_fluxes": float,
         "n_labeling": int, "n_fluxes": int}
        where ``loss`` is the reduced χ² (mean_ssr_labeling + mean_ssr_fluxes) and
        ``ssr_labeling`` / ``ssr_fluxes`` are the per-block reduced χ² (χ²_L/n_L, χ²_F/n_F).
    """
    n = model.num_reactions
    dtype, device = model.dtype, model.device
    mask_f = model._reversibility_mask  # float tensor, 1.0 for reversible reactions

    # ── Null-space basis of S: v_net = T @ p satisfies S @ v_net = 0 ─────────
    T_np = gen_SVD_kernel(model.stoichiometry.cpu().numpy())  # [n, k]
    T_t = torch.tensor(T_np, dtype=dtype, device=device)
    k = T_t.shape[1]

    # ── Experiment batches + measurement data ─────────────────────────────────
    experiment_batches = []
    for i in range(0, len(model.experiments), simultanous_experiments):
        experiment_batches.append(range(i, min(i + simultanous_experiments, len(model.experiments))))

    labeling_scaler = torch.ones(len(model.experiments), device=device)
    batch_data = {}
    for exp_ix in experiment_batches:
        index_labeling, L_mean, _, L_std = get_model_labeling_data(
            model, labeling_scaler, with_std=True, exp_ix=exp_ix)
        index_fluxes, v_mean, v_std = get_model_flux_data(
            model, with_std=True, exp_ix=exp_ix)
        batch_data[tuple(exp_ix)] = (index_labeling, L_mean, L_std,
                                     index_fluxes, v_mean, v_std)

    # Per-block measurement counts. The labeling block has many MID measurements
    # (~hundreds) and the flux block only a handful; a plain summed χ² lets labeling
    # drown out the flux constraints. Reduced χ² per block (χ²_L/n_L + χ²_F/n_F) gives
    # each data type equal footing so the few measured fluxes actually constrain the fit.
    n_L_total = int(sum(bd[0].numel() for bd in batch_data.values()))
    n_F_total = int(sum(bd[3].numel() for bd in batch_data.values()))
    # Precompute inverse counts once so the per-iteration objective is branch-free.
    inv_n_L = 1.0 / max(n_L_total, 1)
    inv_n_F = (1.0 / n_F_total) if n_F_total > 0 else 0.0

    # ── Initial point: project current fluxes onto null(S), keep r ───────────
    # Columns of T are orthonormal, so T^T is the pseudo-inverse.
    p0 = T_t.T @ model._fluxes.detach()       # [k]
    x0 = torch.cat([p0, model._reversibility.detach()])  # [k + n]

    # ── Bounds: p unconstrained, r bounded below by min_rev_flux ─────────────
    rev_bool = mask_f.bool()
    lb_p = torch.full((k,), float('-inf'), dtype=dtype, device=device)
    ub_p = torch.full((k,), float('inf'),  dtype=dtype, device=device)
    lb_r = torch.where(rev_bool,
                       torch.full((n,), model.min_rev_flux, dtype=dtype, device=device),
                       torch.zeros(n, dtype=dtype, device=device))
    ub_r = torch.where(rev_bool,
                       torch.full((n,), float('inf'), dtype=dtype, device=device),
                       torch.zeros(n, dtype=dtype, device=device))

    # ── Chi-squared objective (differentiable w.r.t. x via autograd) ─────────
    def objective(x: torch.Tensor, return_blocks: bool = False) -> torch.Tensor:
        p, r = x[:k], x[k:]
        v_net = T_t @ p  # S @ v_net = 0 automatically
        rev = torch.where((r * mask_f) <= model.min_rev_flux,
                          torch.full_like(r, model.min_rev_flux),
                          r * mask_f)
        vf = 0.5 * (v_net.abs() + v_net + rev)
        vb = 0.5 * (v_net.abs() - v_net + rev)
        vf = vf - vb * (1.0 - mask_f)
        vb = vb * mask_f
        rev_fluxes = torch.cat([vf, vb])

        total_L = x.new_zeros(())
        total_F = x.new_zeros(())
        for exp_ix in experiment_batches:
            index_labeling, L_mean, L_std, index_fluxes, v_mean, v_std = batch_data[tuple(exp_ix)]
            output = model.forward(exp_ix=exp_ix, reversible_fluxes=rev_fluxes)
            sim_L = torch.take(output, index_labeling)
            total_L = total_L + ((sim_L - L_mean) / L_std).square().sum()
            if index_fluxes.numel() > 0:
                total_F = total_F + ((v_net[index_fluxes] - v_mean) / v_std).square().sum()
        # Reduced χ² per block so the flux constraints aren't drowned out by the
        # far more numerous labeling measurements (inverse counts precomputed).
        if return_blocks:
            return total_L * inv_n_L, total_F * inv_n_F
        return total_L * inv_n_L + total_F * inv_n_F

    # ── Gradient via autograd ─────────────────────────────────────────────────
    # Track the most recent objective/gradient + evaluation count so the per-iteration
    # callback can report progress without recomputing the forward pass.
    progress = {"nfev": 0, "loss": float("nan"), "gnorm": float("nan")}

    def f_and_grad(x_np: np.ndarray):
        x = torch.tensor(x_np, dtype=dtype, device=device).requires_grad_(True)
        with torch.enable_grad():
            loss = objective(x)
        grad, = torch.autograd.grad(loss, x)
        grad_np = grad.detach().cpu().numpy()
        progress["nfev"] += 1
        progress["loss"] = loss.item()
        progress["gnorm"] = float(np.max(np.abs(grad_np)))
        return loss.item(), grad_np

    bounds_scipy = Bounds(
        torch.cat([lb_p, lb_r]).cpu().numpy(),
        torch.cat([ub_p, ub_r]).cpu().numpy(),
    )

    logger.info(
        f"MFA L-BFGS-B: {n} reactions, {k} free fluxes (null-space dim), "
        f"{len(model.experiments)} experiments, max_iter={max_iter}, tol={tol}"
    )

    # ── Per-iteration progress (verbose) ──────────────────────────────────────
    callback = None
    if disp:
        t_start = time.time()
        timer = {"iter": 0, "t_last": t_start}
        print(f"[MFA] L-BFGS-B start: {k} free fluxes + {n} reversibility, "
              f"max_iter={max_iter}, gtol={tol}", flush=True)

        def callback(xk):
            timer["iter"] += 1
            now = time.time()
            print(f"[MFA] iter {timer['iter']:4d}  χ²={progress['loss']:.6g}  "
                  f"|g|inf={progress['gnorm']:.3e}  nfev={progress['nfev']}  "
                  f"dt={now - timer['t_last']:.2f}s  total={now - t_start:.1f}s",
                  flush=True)
            timer["t_last"] = now

    # ── Optimize ──────────────────────────────────────────────────────────────
    result = scipy_minimize(
        f_and_grad, x0.cpu().numpy(), method='L-BFGS-B',
        jac=True, bounds=bounds_scipy, callback=callback,
        options=dict(maxiter=max_iter, gtol=tol),
    )

    # ── Write back: recover v_net from p, update model parameters ────────────
    x_opt = torch.tensor(result.x, dtype=dtype, device=device)
    p_opt = x_opt[:k]
    r_opt = x_opt[k:]
    v_net_opt = T_t @ p_opt
    with torch.no_grad():
        model._fluxes.copy_(v_net_opt.detach())
        model._reversibility.copy_(r_opt.detach())

    sv_max = float((model.stoichiometry @ model._fluxes).abs().max().item())
    logger.info(
        f"MFA done: converged={result.success}, iter={result.nit}, "
        f"χ²={float(result.fun):.5f}, |S·v|_max={sv_max:.2e}"
    )

    if checkpoint_file is not None:
        model.save_checkpoint(file=checkpoint_file,
                              meta_data={"chi2": float(result.fun), "converged": result.success})

    if labeling_checkpoint_file is not None:
        with torch.no_grad():
            pred_labeling = model.forward()
        torch.save(pred_labeling.cpu(), labeling_checkpoint_file)

    # Re-evaluate the two reduced-χ² blocks at the optimum so callers can apply a
    # per-block acceptance test (e.g. effective-df multi-start gating).
    with torch.no_grad():
        rchi2_L, rchi2_F = objective(x_opt, return_blocks=True)

    return {
        "loss":         float(result.fun),    # mean_ssr_labeling + mean_ssr_fluxes
        "converged":    bool(result.success),
        "n_iter":       int(result.nit),
        "ssr_labeling": float(rchi2_L),       # reduced χ²_L  (mean ssr labeling)
        "ssr_fluxes":   float(rchi2_F),       # reduced χ²_F  (mean ssr fluxes)
        "n_labeling":   n_L_total,
        "n_fluxes":     n_F_total,
    }


@time_function
def fit_mfa_model_penalized(model, loss_fn, flux_loss_fn, optimizer,
                            lam1=1e-4, lam2=1e-5, epochs=100,
                            simultanous_experiments=3,
                            labeling_scaler=None, flux_scaler=1, checkpoint=10,
                            checkpoint_file=None,
                            labeling_checkpoint_file=None,
                            optimizer_checkpoint=None,
                            lr_scheduler=None, with_std=True,
                            early_stop_tol=None):
    """Legacy penalized-Lagrangian MFA optimizer (soft constraints, L1/L2 reg).

    Retained for reproducing analyses published before :func:`fit_mfa_model` was
    rewritten as a null-space chi-squared fit. New work should use
    :func:`fit_mfa_model`.
    """
    from tqdm import trange
    from sigmaflux.core.scaling import Scaling

    with_std = True
    if labeling_scaler is None:
        labeling_scaler = torch.ones(len(model.experiments), device=model.device)
    if isinstance(labeling_scaler, float):
        labeling_scaler = labeling_scaler * torch.ones(len(model.experiments), device=model.device)

    Scaling(labeling_scaler)
    log = []

    experiment_batches = []
    for i in range(0, len(model.experiments), simultanous_experiments):
        experiment_batches.append(range(i, min(i + simultanous_experiments, len(model.experiments))))

    batch_data = {}
    for exp_ix in experiment_batches:
        index_labeling, labeling_mean, labeling_scaler, labeling_std = get_model_labeling_data(model, labeling_scaler, with_std, exp_ix=exp_ix)
        index_fluxes, flux_mean, flux_std = get_model_flux_data(model, with_std, exp_ix=exp_ix)
        batch_labeling_scaling = Scaling(labeling_scaler)
        batch_data[tuple(exp_ix)] = (index_labeling, labeling_mean, labeling_scaler, labeling_std, index_fluxes,
                                     flux_mean, flux_std, batch_labeling_scaling)

    logger.info(f"Penalized MFA: LR:{optimizer.param_groups[0]['lr']}, L1:{lam1}, L2:{lam2}, Epochs:{epochs}")

    if checkpoint_file is not None:
        model.save_checkpoint(file=checkpoint_file)

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    with trange(epochs) as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            if epoch % checkpoint == 0:
                epoch_outputs = []

            for exp_ix in experiment_batches:
                (index_labeling, labeling_mean, labeling_scaler, labeling_std,
                 index_fluxes, flux_mean, flux_std, batch_labeling_scaling) = batch_data[tuple(exp_ix)]

                _state = {}

                def _closure():
                    optimizer.zero_grad()
                    output = model.forward(exp_ix=exp_ix)
                    _state['output'] = output.detach().cpu()

                    simulated_labeling = torch.take(output, index_labeling)
                    simulated_labeling.retain_grad()

                    labeling_loss = loss_fn(
                        batch_labeling_scaling.scale(simulated_labeling),
                        batch_labeling_scaling.scale(labeling_mean),
                    )

                    mass_balance_constraints = (model.stoichiometry @ model.fluxes).square()
                    positive_fluxes_constraints = torch.relu(-1 * model.reversible_fluxes)
                    l1_reg = lam1 * model.reversible_fluxes.abs().mean()
                    l2_reg = lam2 * model.reversible_fluxes.square().mean()

                    if not index_fluxes.nelement() == 0:
                        measured_fluxes = torch.take(model.fluxes, index_fluxes)
                        flux_loss = (flux_loss_fn(
                            measured_fluxes / flux_mean.squeeze(),
                            (flux_mean / flux_mean).squeeze(),
                        ) * flux_scaler)
                        msrmt_labeling_loss = labeling_loss + flux_loss
                    else:
                        msrmt_labeling_loss = labeling_loss
                        flux_loss = labeling_loss * 0.0

                    lam = model._lambda0
                    mu = model._mu0
                    loss = (msrmt_labeling_loss + l1_reg + l2_reg
                            + lam.T @ mass_balance_constraints
                            + mu.T @ positive_fluxes_constraints)

                    loss.backward()
                    _state['loss'] = loss
                    _state['labeling_loss'] = labeling_loss
                    _state['flux_loss'] = flux_loss
                    _state['l1'] = l1_reg
                    _state['l2'] = l2_reg
                    return loss

                if is_lbfgs:
                    optimizer.step(_closure)
                else:
                    _closure()
                    optimizer.step()

                loss = _state['loss']
                labeling_loss = _state['labeling_loss']
                flux_loss = _state['flux_loss']
                l1_reg = _state['l1']
                l2_reg = _state['l2']

                if epoch % checkpoint == 0:
                    epoch_outputs.append(_state['output'])

                epoch_loss += loss.item()

            if lr_scheduler is not None:
                lr_scheduler.step()

            log.append(epoch_loss / len(experiment_batches))
            pbar.set_postfix(loss=epoch_loss / len(experiment_batches), epoch=epoch)

            if early_stop_tol is not None and len(log) > 1:
                if abs(log[-1] - log[-2]) < early_stop_tol:
                    logger.info(f"Early stop at epoch {epoch}: Δloss={abs(log[-1] - log[-2]):.2e} < {early_stop_tol:.2e}")
                    break

            logger.info(
                f"Optimization step {epoch} :total loss {log[-1]:.5f}, "
                f"labeling loss {labeling_loss:.5f}, flux loss {flux_loss:.5f}, "
                f"l1 {l1_reg:.5f}, l2 {l2_reg:.5f}"
            )

            if epoch % checkpoint == 0:
                if checkpoint_file is not None:
                    meta_data = {
                        "loss": loss.item(), "epoch": epoch,
                        "labeling_loss": labeling_loss.item(), "flux_loss": flux_loss.item(),
                        "l1": l1_reg.item(), "l2": l2_reg.item(),
                    }
                    model.save_checkpoint(file=checkpoint_file, meta_data=meta_data)

                if labeling_checkpoint_file is not None:
                    aggregated_output = torch.cat(epoch_outputs, dim=0)
                    try:
                        labeling_data = torch.load(labeling_checkpoint_file)
                    except Exception:
                        labeling_data = []
                    labeling_data.append(aggregated_output)
                    torch.save(labeling_data, labeling_checkpoint_file)

                if optimizer_checkpoint is not None:
                    torch.save(optimizer.state_dict(), optimizer_checkpoint)

    logger.info(f"Penalized MFA complete: loss={log[-1]:.5f}")
    return log


#: Deprecated alias kept for one release; use :func:`fit_mfa_model_penalized`.
_fit_mfa_model_penalized = fit_mfa_model_penalized


def filter_conv_inputs(emu_inputs):
    return [b for b in emu_inputs if isinstance(b, OriginSet) and len(b._values) > 1]

def filter_conv_inputs_with_indices(emu_inputs): 
    emus = []
    emu_indices = []
    for i, b in enumerate(emu_inputs): 
        if isinstance(b, OriginSet) and len(b._values) > 1:
            emus.append(b)
            emu_indices.append(i)
    return emus, emu_indices

def calc_emu_input(emu_states, emu_inputs, mid_inputs, ):
    this_emu_inputs = [0.0, ] * ()
    for ix, the_input in enumerate(emu_inputs):
        if isinstance(the_input, OriginSet) and len(OriginSet) > 1:
            this_emu_inputs[ix] = convolution(emu_states, the_input)
        else:
            this_emu_inputs[ix] = extract_input(mid_inputs, the_input)

    return emu_inputs


# Invariant: every component of a multi-element OriginSet must be a core node
# (present in core_met_ix for its subnet size).  This is guaranteed by
# EMUExpansion._prune_orphaned_edges(); a KeyError here means the prune step
# was skipped or a new code path bypasses it.
def convolution(core_met_ix, emu_states, origin_set, device, dtype):
    """
    emu_states matrix is a dictionary of # of subnets torch tensors with # of core emu x # of mids
    """
    emu_size = origin_set.size

    if len(origin_set) > 2:
        emus = origin_set._values 
        ## first convolute every two emus, then convolute the results from that convolution and so on until we have convoluted all 8 elements together 
        ## Both reductions step in pairs, so an ODD count leaves a last element
        ## with no partner: emus[i+1] / all_convolutions[i+1] then overrun and
        ## raise IndexError. Only sets whose size is a power of two survived.
        ## Convolution is associative, so the unpaired element is simply carried
        ## into the next round untouched.
        all_convolutions = []
        for i in range(0, len(emus) - 1, 2): 
            all_convolutions.append(convolution(core_met_ix, emu_states, OriginSet([emus[i], emus[i+1]]), device, dtype))
        if len(emus) % 2:
            lone = emus[-1]
            lone_size = lone.size
            all_convolutions.append(
                emu_states[lone_size][core_met_ix[lone_size][lone]])
        
        
        while len(all_convolutions) > 1: 
            new_convolutions = []
            for i in range(0, len(all_convolutions) - 1, 2): 
                cur_set_size = all_convolutions[i].shape[0] + all_convolutions[i+1].shape[0] - 2
                t1 = torch.zeros((cur_set_size + 1),device=device, dtype = dtype)
                t2 = torch.zeros((cur_set_size + 1),device=device, dtype = dtype)
                t1[:all_convolutions[i].shape[0]]= all_convolutions[i]
                t2[:all_convolutions[i+1].shape[0]] = all_convolutions[i+1]
                
                conv = torch.tensor([sum([t1[alpha - beta] * t2[beta] for beta in range(alpha + 1)])
                    for alpha in range(cur_set_size + 1)], device=device, dtype=dtype)
                new_convolutions.append(conv)
            if len(all_convolutions) % 2:
                new_convolutions.append(all_convolutions[-1])
            all_convolutions = new_convolutions 

        return all_convolutions[0]
    met_emu_size_1 = origin_set._values[0].size
    met_emu_size_2 = origin_set._values[1].size

    metid_1 = core_met_ix[met_emu_size_1][origin_set._values[0]]
    metid_2 = core_met_ix[met_emu_size_2][origin_set._values[1]]
    t1 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)
    t2 = torch.zeros((emu_size + 1,), device=device, dtype = dtype)
    t1[:met_emu_size_1+1] = emu_states[met_emu_size_1][metid_1]
    t2[:met_emu_size_2+1] = emu_states[met_emu_size_2][metid_2]

    conv = [sum([t1[alpha - beta]
                 * t2[beta] for beta in range(alpha + 1)])
            for alpha in range(emu_size + 1)]

    return torch.stack(conv, axis=0)

#def conv(x, y, emu_size, device):
#    return torch.tensor([sum([x[alpha - beta] * y[beta] for beta in range(alpha + 1)]) for alpha in range(emu_size + 1)],dtype = torch.float32, device=device)

def conv(x, y, emu_size, device, dtype):
    """Mass-isotopomer convolution ``result[n, a] = sum_b x[n, a-b] * y[b]``.

    Written as one matmul against the lower-triangular Toeplitz matrix of *y*
    (``M[j, a] = y[a-j]`` for ``a >= j``), rather than a Python loop over ``a`` that
    rebuilds an ``arange`` and gathers columns on every pass. The tensors here are
    tiny — ``emu_size`` is at most a handful — so loop overhead dominated the
    arithmetic: this was the largest remaining self-time in an MFA fit at 2.3 s of
    11.5 s, and the rewrite is 2-5x faster on the shapes the profile shows.

    Not bit-identical to the loop: a single matmul accumulates in a different order,
    which moves results by ~1e-15 in float64 (a few ulp). The fit is unaffected in
    any meaningful sense, but exact reproduction against older runs will not hold.
    """
    k = emu_size + 1
    y_flat = y.reshape(-1)[:k]
    shift = (torch.arange(k, device=device).unsqueeze(0)
             - torch.arange(k, device=device).unsqueeze(1))
    toeplitz = torch.where(shift >= 0, y_flat[shift.clamp(min=0)],
                           torch.zeros((), device=device, dtype=dtype))
    return x[:, :k] @ toeplitz

# can this be done more efficiently? 
def convolution_derivative(core_met_ix, derivative_states, emu_states, origin_set, device, dtype):
    """
    emu_states matrix [M,K]
    """
    emu_size = origin_set.size
    if len(origin_set) > 2:
        emus = origin_set._values
        ## Same pairwise assumption as convolution(): an odd count overruns.
        ## The unpaired element is carried forward with its own derivative and
        ## its own single-element set, so the reduction stays consistent.
        all_convolutions = []
        all_sets = []
        for i in range(0, len(emus) - 1, 2): 
            all_convolutions.append(convolution_derivative(core_met_ix, derivative_states, emu_states, OriginSet([emus[i], emus[i+1]]), device, dtype))
            all_sets.append(OriginSet([emus[i], emus[i+1]]))
        if len(emus) % 2:
            lone = emus[-1]
            lone_size = lone.size
            lone_ix = core_met_ix[lone_size][lone]
            all_convolutions.append(derivative_states[lone_size][lone_ix])
            all_sets.append(OriginSet([lone]))
    
        while len(all_convolutions) > 1: 
            new_sets = []
            new_convolutions = []
            for i in range(0, len(all_convolutions) - 1, 2): 
                origin_set_1 = all_sets[i]
                origin_set_2 = all_sets[i+1]
                cur_set_size = all_convolutions[i].shape[1] + all_convolutions[i+1].shape[1] - 2

                origin1_derivative = torch.zeros((all_convolutions[0].shape[0], cur_set_size + 1,),device=device, dtype = dtype)
                origin2_derivative = torch.zeros((all_convolutions[0].shape[0], cur_set_size + 1,),device=device, dtype = dtype)
                origin1_derivative[:,:all_convolutions[i].shape[1]] = all_convolutions[i]
                origin2_derivative[:,:all_convolutions[i+1].shape[1]] = all_convolutions[i+1]

                origin1_emu = torch.zeros((cur_set_size + 1,),device=device, dtype = dtype)
                origin2_emu = torch.zeros((cur_set_size + 1,),device=device, dtype = dtype)
                origin1_emu[:all_convolutions[i].shape[1]] = convolution(core_met_ix, emu_states, origin_set_1, device, dtype)
                origin2_emu[:all_convolutions[i+1].shape[1]] = convolution(core_met_ix, emu_states, origin_set_2, device, dtype)

                conv_derivative_1 = conv(origin1_derivative , origin2_emu.unsqueeze(-1), cur_set_size, device, dtype)
                conv_derivative_2 = conv(origin2_derivative , origin1_emu.unsqueeze(-1), cur_set_size, device, dtype)   
                
                new_sets.append(OriginSet(origin_set_1._values + origin_set_2._values))
                new_convolutions.append(conv_derivative_1 + conv_derivative_2)
            if len(all_convolutions) % 2:
                new_convolutions.append(all_convolutions[-1])
                new_sets.append(all_sets[-1])
            all_convolutions = new_convolutions 
            all_sets = new_sets

        return all_convolutions[0]

   
    met_emu_size_1 = origin_set._values[0].size
    met_emu_size_2 = origin_set._values[1].size

    metid_1 = core_met_ix[met_emu_size_1][origin_set._values[0]]
    metid_2 = core_met_ix[met_emu_size_2][origin_set._values[1]]
    
    met1_derivative = torch.zeros((derivative_states[met_emu_size_1].shape[1], emu_size + 1,),device=device, dtype = dtype)
    met2_derivative = torch.zeros((derivative_states[met_emu_size_2].shape[1], emu_size + 1,),device=device, dtype = dtype)
    
    met1_derivative[:,:met_emu_size_1+1] = derivative_states[met_emu_size_1][metid_1]
    met2_derivative[:,:met_emu_size_2+1] = derivative_states[met_emu_size_2][metid_2]

    met_emu_1 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)
    met_emu_2 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)

    met_emu_1[:met_emu_size_1+1] = emu_states[met_emu_size_1][metid_1]
    met_emu_2[:met_emu_size_2+1] = emu_states[met_emu_size_2][metid_2]

    
    # for each reaction 
    #for i in range(met1_derivative.shape[0]): 
        # convolve the MIDs 
    conv_derivative_1 = conv(met1_derivative , met_emu_2.unsqueeze(-1), emu_size, device, dtype) 
     
    conv_derivative_2 = conv(met2_derivative , met_emu_1.unsqueeze(-1), emu_size, device, dtype) 

    return conv_derivative_1 + conv_derivative_2


def update_derivative_states(derivative_states, dX, emu_size,core_met_ix): 
    for i, met in enumerate(core_met_ix[emu_size]):
        derivative_states[emu_size][core_met_ix[emu_size][met]] = dX[:,i]


def update_states(emu_states, X, emu_size, core_met_ix):
    for i, met in enumerate(core_met_ix[emu_size]):
        # set for each emu size and each metabolite, the labels for each MID 
        emu_states[emu_size][core_met_ix[emu_size][met]] = X[i]

    
def extract_input(mid_inputs, origin):
    return mid_inputs[origin.ix, :]


def get_labeling(metabolite_hash, emu_states, emu_met_filter, core_met_ix, device, dtype=torch.float64):
    """
    emu_states: list(torch.tensor)
    emu_met_filter: list(torch.tensor)
    """
    labeling = torch.zeros((len(emu_states)+1, len(metabolite_hash),), dtype=dtype, device=device) 
    for subnet in emu_met_filter: 
        for met in emu_met_filter[subnet]: 
            if met in core_met_ix[subnet]: 
                labeling[:(subnet+1), metabolite_hash[met.id]] = emu_states[subnet][core_met_ix[subnet][met]]
    #return sum([es * emf for es, emf in zip(emu_states, emu_met_filter)])
    return labeling


def get_derivative_labeling(metabolite_hash, derivative_states, emu_met_filter, core_met_ix,device, dtype): 
    derivative_labeling = torch.zeros((len(derivative_states)+1, len(metabolite_hash), derivative_states[1].shape[1]),device=device, dtype=dtype)
    for subnet in emu_met_filter:
        for met in emu_met_filter[subnet]:
            if met in core_met_ix[subnet]:
                derivative_labeling[:(subnet+1), metabolite_hash[met.id], :] = derivative_states[subnet][core_met_ix[subnet][met]].T
    return derivative_labeling
    # return sum([torch.einsum('enp, pe -> pen', es, emf) for es, emf in zip(derivative_states, emu_met_filter)])



def build_emu_met_filter(model, metabolites, device=None):
    """

    """
    #emu_met_filter = [0.0 * torch.ones(model.max_mid_size, model.max_emu_mets, dtype=torch.float32, device=device)
    #                  for _ in range(model.max_emu_size)]
    emu_met_filter ={i+1: [] for i in range(model.max_emu_size)}
    for met in metabolites:
        for i in range(model.expansion.emu_network.max_size, 0, -1):
            subnet = model.expansion.emu_network.subnetworks[i]
            # I want to modify this code to check if emu._values[0] has a size equal to subnet size
            # if so, then I want to add the metabolite to the emu_met_filter
            # selects the emus in the subnet 
            emu_mets = []
            for emu in subnet.nodes(): 
                if isinstance(emu, EMU): 
                    emu_mets.append(emu)
                else: 
                    # Walk thtugh all values off the 
                    for emu_values in emu._values:
                        if emu_values.size == i: 
                            emu_mets.append(emu_values)
            found_met = False 
            for j, emu_met in enumerate(emu_mets): 
                if emu_met.id == met.id: 
                    emu_met_filter[i].append(emu_mets[j])
                    found_met = True
                    break
            if found_met: 
                break
            
    return emu_met_filter


def gen_SVD_kernel(S):
    U, D, V = np.linalg.svd(S)
    T = V[np.max(np.where(D > EPSILON)[0]) + 1:, :].T
    return T
