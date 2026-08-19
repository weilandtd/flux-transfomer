"""
MFA accuracy benchmark against the training data.

Evaluates how accurately traditional MFA recovers fluxes when given inputs that
mirror a real experiment:
  - Ground-truth fluxes from the training HDF5
  - 13C MID restricted to the metabolites in features.csv (all metabolite_ids, not
    filtered by the included flag — equivalent to the mapping CSV)
  - Measured flux constraints (cfg.training.measured_fluxes) with added noise

Each benchmark run:
  1. Draws a ground-truth flux from the training flux HDF5.
  2. Simulates per-tracer 13C MID via an MFAModel forward pass (+ Gaussian noise).
  3. Adds measured-flux soft constraints (with flux_noise_std noise).
  4. Fits MFA from a random feasible flux — another training sample, chosen without
     reference to the ground truth, because a real fit has no answer to start near.
     Restarts until the fit passes the standard ¹³C-MFA goodness-of-fit test
     (Antoniewicz, Kelleher & Stephanopoulos 2006): SSR ≤ χ²_{1-α}(n_meas - n_params).
     That is the criterion a real experiment can apply; comparing against the residual
     at the true flux cannot, and is looser besides — the truth spends no parameters,
     so a converged fit beats it by (n-p)/n on average.
  5. Writes predicted fluxes and labeling in the same schema as
     ``sigmaflux-benchmark``, so the two are directly comparable.

Usage (I. orientalis):
    sigmaflux-benchmark-mfa \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 10 --max-iter 500

``--diagnose`` fits a single sample and prints step-by-step convergence and
recovery diagnostics instead of writing HDF5 — use it to tune the noise and
multi-start hyperparameters before committing to the full array benchmark:

    sigmaflux-benchmark-mfa --diagnose \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --start-ix 0 --max-iter 1000
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ..core.data import (
    FluxData,
    MassIsotopomerData,
    get_model_flux_data,
    get_model_labeling_data,
)
from ..core.experiment import Experiment
from ..core.mfa import fit_mfa_model
from ._benchmark import (
    build_mfa_context,
    forward_labeling,
    labeling_chi2_scale,
    load_benchmark_inputs,
    noisy_measurements,
    pin_threads,
    resolve_split_source,
    select_samples,
    write_predictions,
)
from ._paths import add_root_argument, load_config, resolve_path


def build_parser():
    p = argparse.ArgumentParser(
        prog="sigmaflux-benchmark-mfa",
        description="MFA accuracy benchmark vs training data",
    )
    p.add_argument("--config", required=True, help="Organism YAML config")
    p.add_argument("--data-dir", required=True,
                   help="Training run directory containing model_prepared_for_mfa.json "
                        "and HDF5 data files")
    p.add_argument("--output-dir", default=None,
                   help="Results directory (default: <data-dir>/mfa_benchmark)")
    p.add_argument("--checkpoint", default=None,
                   help="Trained AI .pt whose held-out validation split defines the "
                        "benchmark samples (auto-detected from data-dir if omitted)")
    p.add_argument("--split", choices=["val", "train", "all"], default="val",
                   help="Which AI-training split to benchmark on (default: val — the set "
                        "withheld from AI training)")
    p.add_argument("--n-benchmarks", type=int, default=10,
                   help="Number of samples to evaluate, as an offset count into the split")
    p.add_argument("--start-ix", type=int, default=0,
                   help="Offset of the first sample within the split (for SLURM array "
                        "parallelism); indexes the sorted split, not raw HDF5 columns")
    p.add_argument("--backend", choices=["native", "native-ipopt", "x3cflux"], default="native",
                   help="Which optimiser fits the flux. 'native' is the in-house "
                        "null-space L-BFGS-B (S·v=0 satisfied by construction, no "
                        "flux bounds). 'native-ipopt' is the same in-house EMU "
                        "forward model fit in full flux space via Ipopt, with "
                        "S·v=0 and FVA bounds passed as hard constraints. 'x3cflux' "
                        "runs 13CFLUX/Ipopt on its own simulator. native-ipopt and "
                        "x3cflux are optional dependencies.")
    p.add_argument("--x3cflux-no-isolate", action="store_true",
                   help="Run x3cflux fits in-process. Faster, but Ipopt segfaults from "
                        "some starts and would take the whole task down; by default "
                        "each start runs in a forked child so a crash costs one start "
                        "and is recorded as crashed_starts.")
    p.add_argument("--x3cflux-no-bounds", action="store_true",
                   help="Drop the cobra FVA bounds from the x3cflux fit. They are on "
                        "by default because Ipopt otherwise zeroes a branch point and "
                        "x3cflux segfaults. This flag affects only the x3cflux backend; "
                        "'native' takes no flux bounds at all, and 'native-ipopt' "
                        "always applies them.")
    p.add_argument("--max-iter", type=int, default=5000,
                   help="L-BFGS-B max iterations. A converged T-cell fit needs ~2200, "
                        "so the previous default of 500 silently returned unconverged "
                        "fits with converged=False")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Gradient tolerance (gtol) for convergence")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=None,
                   help="Relative noise std on measured flux observations "
                        "(default: cfg.training.flux_noise_std, else 0.05). "
                        "0 = no flux noise (measured fluxes pinned to GT)")
    p.add_argument("--max-starts", type=int, default=10,
                   help="N_max multi-start restarts per sample. Restart from a new "
                        "feasible init until the fit reaches the global optimum; if none "
                        "does, the best of the N_max is kept (never NaN) and its distance "
                        "from optimal is recorded as ssr_ratio.")
    p.add_argument("--accept-rule", choices=["chi2", "ratio"], default="chi2",
                   help="How a fit is judged acceptable, which is also when the "
                        "multi-start loop stops. 'chi2' is the standard 13C-MFA "
                        "goodness-of-fit test (Antoniewicz et al. 2006): SSR <= "
                        "χ²_{1-α}(n_measurements - n_parameters), using no ground truth. "
                        "'ratio' is the older SSR <= --accept-ratio · SSR_true, which "
                        "needs an answer no real experiment has.")
    p.add_argument("--chi2-alpha", type=float, default=0.05,
                   help="Significance level for the χ² goodness-of-fit test (default "
                        "0.05, i.e. reject fits above the 95th percentile of χ²(dof))")
    p.add_argument("--chi2-scale", type=float, default=None,
                   help="Variance of the standardised residual under the noise model "
                        "(default: measured from the MID renormalisation). The declared "
                        "σ is optimistic because renormalising each group to sum to one "
                        "shrinks the residual, and an unscaled test accepts ~100%% of "
                        "fits. Pass 1.0 to take σ at face value.")
    p.add_argument("--accept-ratio", type=float, default=1.0,
                   help="Only with --accept-rule ratio: accept when SSR <= this multiple "
                        "of the SSR the TRUE flux achieves against the same noise. Note "
                        "the truth spends no parameters, so a converged fit beats it by "
                        "(n-p)/n on average — this bound is looser than the χ² test and "
                        "stops the restart loop early.")
    p.add_argument("--accept-z", type=float, default=1.645,
                   help="z-score for the legacy reduced-χ² bound M ≤ 1 + z·sqrt(2/h_eff). "
                        "Reported for comparison only; it divides by the measurement "
                        "count rather than the degrees of freedom and ignores the σ "
                        "deflation, so it accepted nearly everything.")
    p.add_argument("--verbose", action="store_true",
                   help="Print the optimiser trace (iteration, chi-square, gradient norm, "
                        "elapsed) for every fit. Without it a long fit is silent until it "
                        "finishes, so there is no way to tell a slow fit from a stuck one.")
    p.add_argument("--diagnose", action="store_true",
                   help="Fit a single sample (the one at --start-ix) with step-by-step "
                        "diagnostics and recovery metrics, writing no HDF5. Ignores "
                        "--max-starts / --accept-z: exactly one fit is run.")
    p.add_argument("--seed", type=int, default=42)
    add_root_argument(p)
    return p


def measured_flux_ids(measured_rxn_ids, gt_flux_series):
    """The measured reactions this sample actually has ground truth for, in order.

    Args:
        measured_rxn_ids: reaction ids configured as measured fluxes.
        gt_flux_series: this sample's ground-truth flux values, indexed by reaction id.
    """
    return [r for r in measured_rxn_ids if r in gt_flux_series.index]


def make_measured_flux_data(mfa_model, rxn_ids, gt_flux_series, flux_noise_std, noisy_values):
    """Wrap already-noised flux observations as FluxData.

    The χ² variance is ``(flux_noise_std·|true|)²``, since fluxes are real
    magnitudes.

    Args:
        mfa_model: model the FluxData observations are built against.
        rxn_ids: reaction ids of the measured fluxes, in the order of ``noisy_values``.
        gt_flux_series: ground-truth flux values, used to scale the observation std.
        flux_noise_std: relative Gaussian noise std used to size each std.
        noisy_values: already-noised measured flux values.
    """
    rows = [{"reaction": rxn_id, "mean": float(meas),
             "std": max(flux_noise_std * abs(float(gt_flux_series[rxn_id])), 1e-9)}
            for rxn_id, meas in zip(rxn_ids, noisy_values)]
    return FluxData(mfa_model, pd.DataFrame(rows)) if rows else None


def make_init_fluxes(all_flux_df, all_rev_df, gt_flux, sample_ix, reaction_ids, rng):
    """Feasible init: a different training sample, drawn at random.

    Deliberately uses no knowledge of the ground truth -- a real MFA fit has no
    answer to start near, so seeding close to it would measure how well the
    optimiser polishes a handed solution rather than whether it can find one. Every
    training sample is a feasible point of the same polytope, so picking one
    guarantees S·v = 0 and in-bounds.

    Args:
        all_flux_df: full training-HDF5 flux table to draw the init sample from.
        all_rev_df: full training-HDF5 reversibility table, aligned to ``all_flux_df``.
        gt_flux: this sample's ground truth, used only to report how far the
            start landed from the answer.
        sample_ix: this sample's own column, excluded from the draw.
        reaction_ids: reaction order to align the drawn columns to.
        rng: random generator the draw is made from.

    Returns ``(v_init, r_init, info)`` with ``info = {j, dev}``.
    """
    n_samples = all_flux_df.shape[1]
    j = int(rng.integers(0, n_samples))
    while j == sample_ix and n_samples > 1:
        j = int(rng.integers(0, n_samples))

    v_init = all_flux_df.iloc[:, j].reindex(reaction_ids).values.astype(np.float64)
    r_init = all_rev_df.iloc[:, j].reindex(reaction_ids).values.astype(np.float64)

    v_gt = gt_flux.reindex(reaction_ids).values.astype(np.float64)
    gt_norm = float(np.linalg.norm(v_gt))
    dev = float(np.linalg.norm(v_init - v_gt) / gt_norm) if gt_norm > 0 else 0.0
    return v_init, r_init, {"j": j, "dev": dev}


def fitted_parameter_count(model):
    """Free parameters the fit spends: null-space net-flux coordinates + exchanges.

    ``fit_mfa_model`` optimises ``v_net = T·p`` with ``T = null(S)`` plus one
    reversibility per reversible reaction; this ``p`` is what the χ² test subtracts
    from the measurement count, and applies to both the native and x3cflux backends.
    """
    from ..core.mfa import gen_SVD_kernel

    k = gen_SVD_kernel(model.stoichiometry.cpu().numpy()).shape[1]
    return int(k + int(model._reversibility_mask.sum().item()))


def chi2_acceptance(n_L, n_F, n_params, scale_labeling=1.0, alpha=0.05):
    """Upper bound on the raw SSR for a statistically acceptable fit.

    The standard ¹³C-MFA goodness-of-fit test (Antoniewicz, Kelleher & Stephanopoulos
    2006): at a global optimum with known measurement σ, the weighted SSR follows
    χ² on ``dof = n_measurements - n_params``; a fit is accepted when its SSR falls
    below ``χ²_{1-α}(dof)``. The flux block needs no correction, so the two blocks
    are blended by size and the returned bound carries that scale factor.

    Args:
        n_L: number of labeling measurements.
        n_F: number of flux measurements.
        n_params: number of free parameters spent by the fit.
        scale_labeling: labeling block's standardised residual variance under the
            actual noise model (see
            :func:`~sigmaflux.pipeline._benchmark.labeling_chi2_scale`).
        alpha: significance level of the goodness-of-fit test.
    """
    dof = max(n_L + n_F - n_params, 1)
    scale = (scale_labeling * n_L + 1.0 * n_F) / max(n_L + n_F, 1)
    return float(scale * stats.chi2.ppf(1.0 - alpha, dof)), dof, scale


def acceptance_threshold(n_L, n_F, z=1.645):
    """Upper bound on M = ½·(χ²_L/n_L) + ½·(χ²_F/n_F), the equal-weight per-block reduced χ².

    Legacy statistic: rejects above ``1 + z·sqrt(2/h_eff)`` where ``h_eff`` is M's
    moment-matched effective degrees of freedom.

    Args:
        n_L: number of labeling measurements.
        n_F: number of flux measurements.
        z: z-score for the acceptance bound.
    """
    if n_F > 0:
        h_eff = 4.0 * n_L * n_F / (n_L + n_F)
    else:
        h_eff = float(n_L)
    return 1.0 + z * np.sqrt(2.0 / h_eff)


def build_noisy_experiments(experiments, noisy_series, noise_std, mfa_model, flux_data):
    """Build one Experiment per tracer from an already-noised labeling series.

    Args:
        experiments: template experiments, one per tracer, supplying the tracer objects.
        noisy_series: already-noised MIDs, restricted to the observable metabolites
            so the sum-to-one renormalisation was applied over exactly the entries
            used here.
        noise_std: labeling noise std recorded on each measurement row.
        mfa_model: model the resulting MassIsotopomerData is built against.
        flux_data: measured-flux soft constraints, attached to the first Experiment
            only (see the loop below).
    """
    noisy_exps = []
    for i, experiment in enumerate(experiments):
        tracer = experiment.tracers[0]
        tracer_str = "|".join(str(t) for t in experiment.tracers)
        tracer_data = noisy_series.loc[tracer_str]  # (Metabolite, Isotopomer) Series

        met_frac_sum = defaultdict(float)
        rows = []
        for (met_id, iso_str), frac in tracer_data.items():
            noisy_frac = float(frac)
            rows.append({"id": met_id, "mid": int(iso_str[2:]), "mean": noisy_frac,
                         "std": noise_std})
            met_frac_sum[met_id] += noisy_frac

        # M+0 per metabolite: needed by MassIsotopomerData to infer EMU size
        m0_rows = [{"id": m, "mid": 0, "mean": max(0.0, 1.0 - s), "std": noise_std}
                   for m, s in met_frac_sum.items()]
        noisy_mid = MassIsotopomerData(model=mfa_model, table=pd.DataFrame(m0_rows + rows))
        # A measured flux is a property of the culture, not of any one tracer, so it
        # only goes on the first Experiment — get_model_flux_data's own dedup guard
        # only catches a flux repeated within a single fit_mfa_model batch, not
        # across batches, so the single attachment point has to be here.
        noisy_exps.append(Experiment(noisy_mid, [tracer], fluxes=flux_data if i == 0 else None))
    return noisy_exps


# ── Diagnostic metrics (--diagnose only) ──────────────────────────────────────

def _chi2_split(model):
    """Objective contributions at the model's current flux state.

    Mirrors fit_mfa_model's objective: labeling χ² = Σ((sim_L - L_mean)/L_std)²,
    flux χ² = Σ((v - v_mean)/v_std)². Returns (labeling_chi2, flux_chi2).
    """
    scaler = torch.ones(len(model.experiments), device=model.device)
    idx_L, L_mean, _, L_std = get_model_labeling_data(model, scaler, with_std=True)
    idx_v, v_mean, v_std = get_model_flux_data(model, with_std=True)
    with torch.no_grad():
        output = model.forward()
        sim_L = torch.take(output, idx_L)
        lab_chi2 = float((((sim_L - L_mean) / L_std) ** 2).sum().item())
        flux_chi2 = 0.0
        if idx_v.numel() > 0:
            v = model.fluxes[idx_v]
            flux_chi2 = float((((v - v_mean) / v_std) ** 2).sum().item())
    return lab_chi2, flux_chi2


def _block_counts(model):
    """(n_labeling, n_fluxes) currently entering the objective."""
    scaler = torch.ones(len(model.experiments), device=model.device)
    idx_L, *_ = get_model_labeling_data(model, scaler, with_std=True)
    idx_v, *_ = get_model_flux_data(model, with_std=True)
    return int(idx_L.numel()), int(idx_v.numel())


def native_shaped_result(mfa_model, n_iter, converged):
    """``fit_mfa_model``'s return shape, read off the model's current state.

    Lets an external backend's fit (in its own coordinates) flow through the rest of
    this driver unchanged: the fitted flux is written into the model and the
    objective re-derived here, so acceptance is always decided by *our* forward
    model against *our* noise, never the optimiser's own report.

    Args:
        mfa_model: model already holding the external backend's fitted flux state.
        n_iter: iteration count reported by the external backend.
        converged: convergence flag reported by the external backend.
    """
    lab_chi2, flux_chi2 = _chi2_split(mfa_model)
    n_L, n_F = _block_counts(mfa_model)
    reduced_L = lab_chi2 / max(n_L, 1)
    reduced_F = flux_chi2 / max(n_F, 1)
    return {"loss": reduced_L + reduced_F, "converged": converged, "n_iter": n_iter,
            "ssr_labeling": reduced_L, "ssr_fluxes": reduced_F,
            "n_labeling": n_L, "n_fluxes": n_F}


def _r2(pred, true):
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _agreement(pred, true):
    """(R², Pearson r, RMSE) of pred vs true.

    Args:
        pred: predicted values.
        true: ground-truth values, aligned to ``pred``.
    """
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    corr = float(np.corrcoef(pred, true)[0, 1])
    return _r2(pred, true), corr, rmse


def _labeling_agreement(pred, gt, extra_mask=None):
    """Labeling (R², r, RMSE) scored only where GT != 0, since unpopulated M+n
    isotopomers carry no information and would inflate R²/RMSE.

    Args:
        pred: predicted labeling vector.
        gt: ground-truth labeling vector, aligned to ``pred``.
        extra_mask: additional boolean mask ANDed with the GT-nonzero mask.
    """
    mask = gt > 1e-12
    if extra_mask is not None:
        mask = mask & extra_mask
    if not mask.any():
        return (float('nan'),) * 3
    return _agreement(pred[mask], gt[mask])


def _report_init(mfa_model, init_info, gt_flux, gt_flux_arr, gt_labeling_vec,
                 labeling_to, flux_data_obs):
    """Pre-fit diagnostics: is the init feasible, and where does the objective sit?

    Args:
        mfa_model: model already reset to the init flux/reversibility state.
        init_info: ``{j, dev}`` from :func:`make_init_fluxes`, for the printed report.
        gt_flux: ground-truth flux values, for the measured-constraint table.
        gt_flux_arr: ground-truth flux values as an array aligned to ``reaction_ids``.
        gt_labeling_vec: ground-truth labeling vector to compare the init against.
        labeling_to: index tensor gathering the model's output into labeling order.
        flux_data_obs: measured-flux soft constraints, or ``None``.
    """
    if flux_data_obs is not None:
        frows = [{"reaction": f.reaction.id, "true": float(gt_flux[f.reaction.id]),
                  "mean": f._mean, "std": f._std,
                  "rel_std": f._std / max(abs(f._mean), 1e-12)} for f in flux_data_obs]
        print("\n[debug] measured flux constraints (relative std)")
        print(pd.DataFrame(frows).to_string(index=False,
                                            float_format=lambda x: f"{x:.4g}"))

    with torch.no_grad():
        sv_init = (mfa_model.stoichiometry @ mfa_model.fluxes).abs().max().item()
    idx_v, v_mean, v_std = get_model_flux_data(mfa_model, with_std=True)
    same = bool(torch.allclose(v_std, v_mean)) if idx_v.numel() > 0 else False
    # Regression guard: the data.py bug overwrote v_std with v_mean.
    assert not (idx_v.numel() > 0 and same), \
        "v_std == v_mean — flux-std bug reintroduced (data.py)!"

    lab_chi2, flux_chi2 = _chi2_split(mfa_model)
    init_flux = mfa_model.fluxes.detach().cpu().numpy()
    with torch.no_grad():
        init_lab = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()
    f_r2, f_corr, f_rmse = _agreement(init_flux, gt_flux_arr)
    l_r2, l_corr, l_rmse = _labeling_agreement(init_lab, gt_labeling_vec)

    print("\n[debug] feasible init: agreement & objective split")
    print(f"  init via sample : j={init_info['j']}  "
          f"dev from truth={init_info['dev']:.4f} (not used to build it)")
    print(f"  ||S·v_init||_max: {sv_init:.2e}  (a feasible sample → should be ~0)")
    print(f"  v_std==v_mean   : {same if idx_v.numel() > 0 else 'n/a'} (must be False)")
    print(f"  init flux       : R²={f_r2:.4f} r={f_corr:.4f} RMSE={f_rmse:.4f}")
    print(f"  init labeling   : R²={l_r2:.4f} r={l_corr:.4f} RMSE={l_rmse:.4f}  (GT-nonzero)")
    print(f"  init labeling χ²: {lab_chi2:.4f}")
    print(f"  init flux χ²    : {flux_chi2:.4f}")


def _report_fit(mfa_model, res, gt_flux, gt_labeling_vec, labeling_to, labeling_index,
                reaction_ids, measured_rxn_ids, mapped_met_ids, flux_data_obs, accept_z):
    """Post-fit diagnostics: convergence, flux/labeling recovery, mass balance.

    Args:
        mfa_model: model holding the fitted flux/reversibility state.
        res: fit result dict, as returned by :func:`native_shaped_result` or
            ``fit_mfa_model``.
        gt_flux: ground-truth flux values to compare the fit against.
        gt_labeling_vec: ground-truth labeling vector to compare the fit against.
        labeling_to: index tensor gathering the model's output into labeling order.
        labeling_index: ordered (tracer, metabolite, isotopomer) labeling slots.
        reaction_ids: reaction order the flux vectors are aligned to.
        measured_rxn_ids: reaction ids configured as measured fluxes.
        mapped_met_ids: metabolite ids the measurement platform can see.
        flux_data_obs: measured-flux soft constraints, or ``None``.
        accept_z: z-score for the legacy reduced-χ² bound, for the printed report.
    """
    pred_flux = mfa_model.fluxes.detach().cpu().numpy()
    gt_vals = gt_flux.reindex(reaction_ids).values

    mae = float(np.mean(np.abs(pred_flux - gt_vals)))
    r2_flux, corr, rmse = _agreement(pred_flux, gt_vals)

    # Measured fluxes are soft-constrained → should be near-perfect; residual error
    # should isolate to the unmeasured / non-identifiable DOFs.
    measured_set = {r for r in measured_rxn_ids if r in reaction_ids}
    meas_mask = np.array([r in measured_set for r in reaction_ids])

    def _block(mask):
        # R² and RMSE only: a handful of measured reactions makes Pearson r noise.
        if not mask.any():
            return float('nan'), float('nan')
        return (_r2(pred_flux[mask], gt_vals[mask]),
                float(np.sqrt(np.mean((pred_flux[mask] - gt_vals[mask]) ** 2))))

    r2_meas, rmse_meas = _block(meas_mask)
    r2_unmeas, rmse_unmeas = _block(~meas_mask)

    with torch.no_grad():
        sv = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().cpu().numpy()
        pred_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()

    # Labeling recovery over the observed subset (what the fit actually sees) and over
    # all atom-pattern metabolites (the full H5 columns).
    r2_lab, lab_corr, lab_rmse = _labeling_agreement(pred_labeling_vec, gt_labeling_vec)
    obs_mask = np.array([m in mapped_met_ids for (_, m, _) in labeling_index])
    r2_lab_obs, lab_corr_obs, lab_rmse_obs = _labeling_agreement(
        pred_labeling_vec, gt_labeling_vec, extra_mask=obs_mask)

    lab_chi2, flux_chi2 = _chi2_split(mfa_model)
    print("\n[debug] post-fit objective split (raw χ², per block)")
    print(f"  labeling χ²     : {lab_chi2:.4f}")
    print(f"  flux χ²         : {flux_chi2:.4f}")
    if flux_data_obs is not None:
        # Are the measured-flux soft constraints actually satisfied? Each should
        # sit within ~1σ of GT once the flux block carries proper weight.
        rid_to_pred = dict(zip(reaction_ids, pred_flux))
        print("  measured-flux recovery (should be |err|/σ ≈ 1):")
        print(f"    {'reaction':16s} {'GT':>9s} {'pred':>9s} {'|err|':>8s} {'|err|/σ':>8s}")
        for f in flux_data_obs:
            rid = f.reaction.id
            gt_v, pr_v = float(gt_flux[rid]), float(rid_to_pred[rid])
            err = abs(pr_v - gt_v)
            print(f"    {rid:16s} {gt_v:9.3g} {pr_v:9.3g} {err:8.3g} "
                  f"{err / max(f._std, 1e-12):8.2f}")

    mean_reduced_chi2 = 0.5 * (res["ssr_labeling"] + res["ssr_fluxes"])
    legacy_bound = acceptance_threshold(res["n_labeling"], res["n_fluxes"], accept_z)

    print("\n── Results ──────────────────────────────────────────────────────")
    print(f"  Converged         : {res['converged']}")
    print(f"  Iterations        : {res['n_iter']}")
    print(f"  Final χ²          : {res['loss']:.6f}")
    print(f"  mean reduced χ²   : {mean_reduced_chi2:.4f}  (legacy bound ≤ {legacy_bound:.4f})")
    print("")
    print("  Flux vs ground truth")
    print(f"    MAE             : {mae:.4f}")
    print(f"    RMSE            : {rmse:.4f}")
    print(f"    Pearson r       : {corr:.4f}")
    print(f"    R²              : {r2_flux:.4f}")
    print(f"    R² (measured)   : {r2_meas:.4f}  RMSE {rmse_meas:.4f}")
    print(f"    R² (unmeasured) : {r2_unmeas:.4f}  RMSE {rmse_unmeas:.4f}")
    print("")
    print("  Mass balance  ||Sv||")
    print(f"    max |Sv_i|      : {float(np.max(np.abs(sv))):.2e}")
    print(f"    RMSE Sv         : {float(np.sqrt(np.mean(sv ** 2))):.2e}")
    print("")
    print("  Labeling recovery (pred vs GT, GT-nonzero components)")
    print(f"    R² (observed)   : {r2_lab_obs:.4f}  r {lab_corr_obs:.4f}  RMSE {lab_rmse_obs:.4f}")
    print(f"    R² (all H5)     : {r2_lab:.4f}  r {lab_corr:.4f}  RMSE {lab_rmse:.4f}")


def main(args):
    run_started = time.perf_counter()
    pin_threads()
    cfg, root = load_config(args.config, args.root)

    def _p(rel):
        return resolve_path(rel, root)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = (args.flux_noise_std if args.flux_noise_std is not None
                      else cfg.get("training", {}).get("flux_noise_std", 0.05))

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    output_dir = _p(args.output_dir) or os.path.join(data_dir, "mfa_benchmark")
    if not args.diagnose:
        os.makedirs(output_dir, exist_ok=True)

    cobra_model, mapped_met_ids, _, all_flux_df, all_rev_df = load_benchmark_inputs(
        data_dir, cfg, root)

    split_source = resolve_split_source(_p(args.checkpoint), data_dir, args.split)
    sel, tag = select_samples(args.split, split_source, all_flux_df.shape[1], cfg,
                              args.start_ix, 1 if args.diagnose else args.n_benchmarks)
    n_runs = len(sel)
    if n_runs == 0:
        print("  No samples for this offset range — nothing to do.")
        return 0

    flux_df = all_flux_df.iloc[:, sel]
    rev_df = all_rev_df.iloc[:, sel]

    if measured_rxn_ids:
        present = [r for r in measured_rxn_ids if r in flux_df.index]
        print(f"  {len(present)}/{len(measured_rxn_ids)} measured flux reactions found")
    print(f"  MID noise std       : {args.noise_std}")
    print(f"  Flux noise std (rel): {flux_noise_std}")

    # x3cflux does not use our backward EMU network -- the expensive half, and the
    # one that drops dAs.pkl/dBs.pkl into the working directory. native-ipopt does.
    mfa_model, reaction_ids, experiments, labeling_to, labeling_index = build_mfa_context(
        cobra_model, tracer_cfg, flux_df.iloc[:, 0], "MFA_benchmark",
        with_backward=(args.backend != "x3cflux"),
    )
    backend = None      # built on the first sample, once the fitted index is known
    chi2_bound = chi2_dof = chi2_scale = None   # built on the first sample too
    # The fixed cost every array task pays before it fits anything. Reported because
    # the launcher's sizing formula subtracts it from the per-task budget, and it is
    # the one term a per-sample timing cannot show.
    print(f"\n  Setup complete in {time.perf_counter() - run_started:.0f}s "
          f"(model load + EMU network)")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    pred_lab_data, pred_flux_data, sample_indices = [], [], []
    diagnostics = []
    pred_base = os.path.join(output_dir, f"pred_{tag}")

    for run_ix in range(n_runs):
        global_ix = int(sel[run_ix])   # real HDF5 column index of this sample
        # Wall clock for the whole sample: setup, every start including the ones that
        # crash, and the scoring. This is what sizes the SLURM array, so it has to be
        # the cost a task actually pays. x3cflux's own per-start ``info["seconds"]``
        # goes deliberately unused -- it excludes the per-sample simulator build and
        # has no native counterpart, so the two arms would not be measuring the same
        # thing.
        sample_started = time.perf_counter()
        gt_flux = flux_df.iloc[:, run_ix]
        gt_rev = rev_df.iloc[:, run_ix]
        print(f"\n── Run {run_ix + 1}/{n_runs}  (sample {global_ix}) ──────────────────")

        # ── Step 1: GT labeling via per-tracer forward pass ──────────────────
        gt_labeling_vec = forward_labeling(
            mfa_model, gt_flux.reindex(reaction_ids).values,
            gt_rev.reindex(reaction_ids).values, labeling_to,
        )
        if args.diagnose:
            with torch.no_grad():
                sv_gt = (mfa_model.stoichiometry @ mfa_model.fluxes).abs().max().item()
            print("\n[debug] GT feasibility & labeling")
            print(f"  ||S·v_gt||_max  : {sv_gt:.2e}  (should be ~0)")
            print(f"  GT labeling     : min={np.nanmin(gt_labeling_vec):.4f} "
                  f"max={np.nanmax(gt_labeling_vec):.4f} "
                  f"NaN={bool(np.isnan(gt_labeling_vec).any())}")

        # ── Step 2: build noisy per-tracer MID + measured flux constraints ────
        # Restrict to the observable metabolites BEFORE drawing, so the sum-to-one
        # renormalisation runs over exactly the entries that get fitted. The draw is
        # keyed on (seed, global_ix), so the AI benchmark sees this same realisation.
        gt_series = pd.Series(gt_labeling_vec, index=labeling_index)
        if mapped_met_ids:
            keep = gt_series.index.get_level_values("Metabolite").isin(mapped_met_ids)
            gt_series = gt_series[keep]
        flux_ids = measured_flux_ids(measured_rxn_ids, gt_flux)
        noisy_series, noisy_flux = noisy_measurements(
            gt_series, gt_flux.reindex(flux_ids).to_numpy(float),
            args.noise_std, flux_noise_std, args.seed, global_ix,
        )
        flux_data_obs = make_measured_flux_data(
            mfa_model, flux_ids, gt_flux, flux_noise_std, noisy_flux
        )
        noisy_exps = build_noisy_experiments(
            experiments, noisy_series, args.noise_std, mfa_model, flux_data_obs
        )
        n_obs = sum(len(e.mid_data) for e in noisy_exps)
        print(f"  Noisy MID: {n_obs} entries across {len(noisy_exps)} tracers")

        # ── x3cflux: convert once, then rebuild only the measurements per sample ──
        simulator = fluxmap = None
        if args.backend == "x3cflux":
            from ..x3cflux.facts import model_facts
            from ..x3cflux.fit import X3CFluxBackend
            if backend is None:
                backend = X3CFluxBackend(
                    model_facts(cobra_model, mfa_model, mapped_met_ids, cfg),
                    noisy_series.index, flux_ids, args.noise_std,
                    use_flux_bounds=not args.x3cflux_no_bounds,
                )
            # Same sigmas make_measured_flux_data gives the native arm.
            flux_sigma = np.maximum(
                flux_noise_std * np.abs(gt_flux.reindex(flux_ids).to_numpy(float)), 1e-9)
            simulator, fluxmap = backend.simulator_for(
                noisy_series.to_numpy(float), noisy_flux, flux_sigma)

        # ── Step 3: multi-start fit until the fit is statistically acceptable ──
        # The measurements above are fixed for this sample; only the starting point
        # changes across restarts. Each start IS a different random feasible training
        # sample, taken verbatim: ground truth is never used to construct it (see
        # make_init_fluxes). Restart until the fit passes the acceptance test, or
        # max_starts is exhausted — L-BFGS-B otherwise sticks in the first local minimum.
        mfa_model.experiments = noisy_exps

        # The acceptance bound is a property of the measurement layout and the parameter
        # count, neither of which varies across samples, so it is built once and reused.
        if chi2_bound is None:
            n_L, n_F = _block_counts(mfa_model)
            n_params = fitted_parameter_count(mfa_model)
            scale_L = (args.chi2_scale if args.chi2_scale is not None
                       else labeling_chi2_scale(noisy_series, args.noise_std,
                                                np.random.default_rng(args.seed)))
            chi2_bound, chi2_dof, chi2_scale = chi2_acceptance(
                n_L, n_F, n_params, scale_L, args.chi2_alpha)
            print(f"  Goodness-of-fit test: {n_L}+{n_F} measurements − {n_params} "
                  f"parameters = {chi2_dof} dof, σ² scale {chi2_scale:.3f} → "
                  f"accept SSR ≤ {chi2_bound:.1f} (α={args.chi2_alpha})")

        # The residual the TRUE flux achieves against this very noise. Not zero: the
        # truth reproduces the noiseless MIDs, so its residual IS the noise energy,
        # Σ(noise/σ)².
        #
        # The truth is a feasible point of this very optimisation — `reversible_fluxes`
        # floors reversibility at min_rev_flux and the GT labeling above was simulated
        # through that same floor — so a minimiser that scores worse than it has
        # demonstrably not found the minimum. That is the whole argument: no linearity,
        # no Gaussian assumption, no degrees-of-freedom count. A converged fit will
        # usually score somewhat better than the truth, but ssr_ratio > 1 is already
        # proof the optimiser stopped short, whatever its reduced χ² claims.
        forward_labeling(mfa_model, gt_flux.reindex(reaction_ids).values,
                         gt_rev.reindex(reaction_ids).values, labeling_to)
        lab_true, flux_true = _chi2_split(mfa_model)
        ssr_true = lab_true + flux_true
        print(f"  SSR at the true optimum: {ssr_true:.2f} "
              f"(labeling {lab_true:.2f} + flux {flux_true:.2f})")

        accepted = False
        best = None
        crashed_starts = 0
        for start in range(1 if args.diagnose else args.max_starts):
            v_init, rev_init, init_info = make_init_fluxes(
                all_flux_df, all_rev_df, gt_flux, global_ix, reaction_ids, rng
            )
            mfa_model.reset_initial_fluxes(pd.Series(v_init, index=reaction_ids))
            # reset_initial_fluxes() re-seeds reversibility randomly; use the drawn
            # sample's own reversibility, which is feasible by construction.
            with torch.no_grad():
                mfa_model._reversibility.copy_(
                    torch.tensor(rev_init, dtype=mfa_model.dtype, device=mfa_model.device)
                )
            if args.diagnose:
                _report_init(mfa_model, init_info, gt_flux,
                             gt_flux.reindex(reaction_ids).values.astype(np.float64),
                             gt_labeling_vec, labeling_to, flux_data_obs)
                print(f"\nFitting (max_iter={args.max_iter}, tol={args.tol}) ...")

            if backend is None:
                if args.backend == "native-ipopt":
                    from ..core.mfa_ipopt import fit_mfa_model_ipopt
                    res = fit_mfa_model_ipopt(mfa_model, max_iter=args.max_iter, tol=args.tol,
                                              disp=1 if (args.diagnose or args.verbose) else 0)
                else:
                    res = fit_mfa_model(mfa_model, max_iter=args.max_iter, tol=args.tol,
                                        disp=1 if (args.diagnose or args.verbose) else 0)
            else:
                v_fit, r_fit, info = backend.fit(
                    simulator, fluxmap, v_init, rev_init, max_iter=args.max_iter,
                    isolate=not args.x3cflux_no_isolate)
                if v_fit is None:
                    # A crashed start is recorded rather than retried silently: the
                    # samples that crash are unlikely to be a random subset.
                    crashed_starts += 1
                    print(f"  start {start + 1}/{args.max_starts}: CRASHED "
                          f"({info.get('crashed')})")
                    continue
                # Put the fitted flux into the model, so everything downstream --
                # scoring, acceptance, the snapshot, the HDF5 -- is the native path.
                forward_labeling(mfa_model, v_fit, r_fit, labeling_to)
                res = native_shaped_result(
                    mfa_model, info.get("n_iter", 0), info.get("status", 1) in (0, 1))
            # Mean reduced chi-square, the legacy accept statistic. Reported only for
            # comparison: it is measured against the declared sigma, which overstates
            # the true noise, so it reads low and passes fits that never converged.
            mean_reduced_chi2 = 0.5 * (res["ssr_labeling"] + res["ssr_fluxes"])
            if args.diagnose:
                _report_fit(mfa_model, res, gt_flux, gt_labeling_vec, labeling_to,
                            labeling_index, reaction_ids, measured_rxn_ids,
                            mapped_met_ids, flux_data_obs, args.accept_z)
                return 0
            lab_fit, flux_fit = _chi2_split(mfa_model)
            ssr_fit = lab_fit + flux_fit
            ratio = ssr_fit / ssr_true if ssr_true > 0 else np.inf
            if best is None or ssr_fit < best["ssr_fit"]:
                # Snapshot the state here: a later restart may score worse, and the model
                # only ever holds the most recent fit.
                with torch.no_grad():
                    best_lab = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()
                    best_flux = mfa_model.fluxes.detach().cpu().numpy().copy()
                best = {"ssr_fit": ssr_fit, "ratio": ratio, "start": start + 1,
                        "converged": bool(res["converged"]), "n_iter": int(res["n_iter"]),
                        "labeling": best_lab, "fluxes": best_flux}
            print(f"  start {start + 1}/{args.max_starts}: SSR={ssr_fit:.2f} "
                  f"(χ² accept≤{chi2_bound:.1f}, dof={chi2_dof}) ratio={ratio:.3f} "
                  f"reduced_chi2={mean_reduced_chi2:.4f} conv={res['converged']} "
                  f"iter={res['n_iter']}")
            if args.accept_rule == "chi2":
                accepted = ssr_fit <= chi2_bound
            else:
                accepted = ssr_fit <= args.accept_ratio * ssr_true
            if accepted:
                break

        # ── Step 4: record the best solution found, plus how far it is from optimal ──
        #
        # Always keep the best fit rather than NaN-ing a sample that missed the bar. A
        # NaN silently removes the hardest samples from the benchmark, which flatters
        # whatever produced them; ssr_ratio is written alongside instead, so downstream
        # can filter on "did this reach the global optimum" and still see what it drops.
        sample_seconds = time.perf_counter() - sample_started
        if best is not None and not np.isnan(best["ssr_fit"]):
            pred_labeling_vec = best["labeling"]
            pred_flux_vec = best["fluxes"]
            mark = ("✓ statistically acceptable" if accepted
                    else "✗ rejected by the goodness-of-fit test")
            print(f"  {mark} after {best['start']} start(s): "
                  f"SSR {best['ssr_fit']:.2f} vs true {ssr_true:.2f} "
                  f"(ratio {best['ratio']:.3f})  [{sample_seconds:.1f}s]")
        else:  # the optimiser never returned a usable state for this sample
            pred_labeling_vec = np.full(len(labeling_index), np.nan)
            pred_flux_vec = np.full(len(reaction_ids), np.nan)
            print(f"  ✗ no usable fit → NaN  [{sample_seconds:.1f}s]")
        diagnostics.append({
            "sample": global_ix,
            "ssr_fit": float(best["ssr_fit"]) if best else np.nan,
            "ssr_true": float(ssr_true),
            "ssr_ratio": float(best["ratio"]) if best else np.nan,
            "reached_optimum": bool(accepted),
            "starts_used": int(best["start"]) if best else args.max_starts,
            "converged": bool(best["converged"]) if best else False,
            "n_iter": int(best["n_iter"]) if best else 0,
            # Always written, 0 for the native arm, so both backends share one schema.
            "crashed_starts": int(crashed_starts),
            "backend": args.backend,
            # What the array is sized from; see submit_mfa_benchmark.sh's PILOT mode.
            "seconds": round(sample_seconds, 2),
            # The goodness-of-fit test that decided `reached_optimum`, so a reader can
            # re-apply or re-tune it without re-running the benchmark.
            "accept_rule": args.accept_rule,
            "chi2_dof": int(chi2_dof) if chi2_dof else np.nan,
            "chi2_bound": float(chi2_bound) if chi2_bound else np.nan,
        })

        pred_lab_data.append(pred_labeling_vec)
        pred_flux_data.append(pred_flux_vec)
        sample_indices.append(global_ix)

        # Convergence diagnostics travel with the predictions: without them a reader
        # cannot tell an optimal fit from one that merely stopped.
        pd.DataFrame(diagnostics).to_csv(f"{pred_base}_convergence.csv", index=False)
        write_predictions(pred_base, pred_lab_data, labeling_index, pred_flux_data,
                          reaction_ids, sample_indices)
        print(f"  Saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–"
          f"{args.start_ix + n_runs - 1}).")
    print(f"  → {pred_base}_labeling.h5")
    print(f"  → {pred_base}_flux.h5")
    return 0


def cli(argv=None):
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
