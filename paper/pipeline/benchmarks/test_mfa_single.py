"""
Single-run MFA calibration test.

Fits one MFA sample and prints convergence diagnostics so hyperparameters can
be tuned before committing to the full array benchmark. The data-construction and
fitting logic mirror run_mfa_benchmark.py exactly (per-tracer experiments,
reversibility from the training data, relative flux noise, near-GT init); the only
additions are step-by-step debug instrumentation and R² recovery metrics.

Usage:
    python -u paper/pipeline/benchmarks/test_mfa_single.py \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --sample-ix 0 --noise-std 0.05 --flux-noise-std 0 --debug
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cobra.io.json import load_json_model

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
torch.set_num_threads(4)


from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.ai.data import load_data
from sigmaflux.core.data import get_model_flux_data, get_model_labeling_data
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from sigmaflux.pipeline.generate_training_data import (
    build_experiments,
    build_labeling_index,
    build_targets,
)
# Reuse the benchmark's data-construction helpers verbatim so the single-run test
# and the array benchmark share identical logic.
from run_mfa_benchmark import (_make_measured_flux_data, _build_noisy_experiments,
                               _make_init_fluxes)


def parse_args():
    p = argparse.ArgumentParser(description="Single-run MFA hyperparameter test")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sample-ix", type=int, default=0,
                   help="Which sample from the HDF5 to use as ground truth (the real "
                        "sample we simulate AND the basis for the near-GT init)")
    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--max-iter", type=int, default=1000,
                   help="L-BFGS-B max iterations")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Projected-gradient tolerance (gtol) for L-BFGS-B convergence")
    # ── Data noise / experiment ────────────────────────────────────────────────
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=None,
                   help="Relative noise std on measured fluxes (default: cfg.training."
                        "flux_noise_std). 0 = no flux noise (measured fluxes pinned to GT)")
    p.add_argument("--init-noise", type=float, default=0.20,
                   help="Relative noise for the near-GT init: v_gt * (1 + N(0, init_noise))")
    p.add_argument("--debug", action="store_true",
                   help="Print per-step diagnostics to localise misfit")
    p.add_argument("--seed", type=int, default=42)
    add_root_argument(p)
    return p.parse_args()


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


def _r2(pred, true):
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _agreement(pred, true):
    """(R², Pearson r, RMSE) of pred vs true."""
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    corr = float(np.corrcoef(pred, true)[0, 1])
    return _r2(pred, true), corr, rmse


def _labeling_agreement(pred, gt, extra_mask=None):
    """Labeling (R², r, RMSE) over GT-nonzero components only.

    The labeling vector is mostly exact zeros (unpopulated M+n isotopomers); those
    carry no information and inflate R²/RMSE, so we score only where GT != 0.
    """
    mask = gt > 1e-12
    if extra_mask is not None:
        mask = mask & extra_mask
    if not mask.any():
        return (float('nan'),) * 3
    return _agreement(pred[mask], gt[mask])


def main():
    args = parse_args()

    cfg, root = load_config(args.config, args.root)

    def _p(rel):
        return resolve_path(rel, root)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = (args.flux_noise_std if args.flux_noise_std is not None
                      else cfg.get("training", {}).get("flux_noise_std", 0.05))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)

    # ── Load model + metabolite mapping ───────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)

    features_path = os.path.join(data_dir, "features.csv")
    mapped_met_ids = set(pd.read_csv(features_path)["metabolite_id"].dropna())

    # ── Load ground-truth flux + reversibility for this sample ────────────────
    print("Loading training data ...")
    _, all_flux_df, all_rev_df = load_data(data_dir)
    n_total = all_flux_df.shape[1]
    print(f"  {n_total} samples, {all_flux_df.shape[0]} reactions")

    gt_flux = all_flux_df.iloc[:, args.sample_ix]
    gt_rev = all_rev_df.iloc[:, args.sample_ix]
    print(f"  Ground-truth sample : {args.sample_ix}")
    print(f"  MID noise std       : {args.noise_std}")
    print(f"  Flux noise std (rel): {flux_noise_std}")

    # ── Build MFAModel + per-tracer experiments (same as run_mfa_benchmark) ───
    mfa_model = MFAModel(cobra_model, "MFA_test", initial_flux_distribution=gt_flux)
    reaction_ids = [r.id for r in mfa_model.reactions]
    dtype, device = mfa_model.dtype, mfa_model.device

    targets, measured_metabolites, atom_patterns = build_targets(mfa_model)
    experiments = build_experiments(mfa_model, tracer_cfg)
    print("Preparing EMU network (per-tracer experiments) ...")
    mfa_model.prepare(experiments, targets=targets, with_backward=True)
    print("  done.")

    labeling_to, labeling_index = build_labeling_index(
        mfa_model, experiments, measured_metabolites, atom_patterns
    )

    # ── Step 1: GT labeling via forward pass (fluxes + reversibility set) ─────
    with torch.no_grad():
        mfa_model._fluxes.copy_(
            torch.tensor(gt_flux.reindex(reaction_ids).values, dtype=dtype, device=device)
        )
        mfa_model._reversibility.copy_(
            torch.tensor(gt_rev.reindex(reaction_ids).values, dtype=dtype, device=device)
        )
        gt_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()

    if args.debug:
        with torch.no_grad():
            sv_gt = (mfa_model.stoichiometry @ mfa_model.fluxes).abs().max().item()
        print("\n[debug] Step 0/1 — GT feasibility & labeling")
        print(f"  ||S·v_gt||_max  : {sv_gt:.2e}  (should be ~0)")
        print(f"  GT labeling     : min={np.nanmin(gt_labeling_vec):.4f} "
              f"max={np.nanmax(gt_labeling_vec):.4f} "
              f"NaN={bool(np.isnan(gt_labeling_vec).any())}")

    # ── Step 2: noisy per-tracer MID + measured flux constraints ──────────────
    gt_series = pd.Series(gt_labeling_vec, index=labeling_index)
    flux_data_obs = _make_measured_flux_data(
        mfa_model, measured_rxn_ids, gt_flux, flux_noise_std, rng
    )
    noisy_exps = _build_noisy_experiments(
        experiments, gt_series, args.noise_std, rng, mfa_model, mapped_met_ids, flux_data_obs
    )
    n_obs = sum(len(e.mid_data) for e in noisy_exps)
    print(f"  Noisy MID: {n_obs} entries across {len(noisy_exps)} tracers")

    if args.debug and flux_data_obs is not None:
        frows = [{"reaction": f.reaction.id, "true": float(gt_flux[f.reaction.id]),
                  "mean": f._mean, "std": f._std,
                  "rel_std": f._std / max(abs(f._mean), 1e-12)} for f in flux_data_obs]
        print("\n[debug] Step 2 — measured flux constraints (relative std)")
        print(pd.DataFrame(frows).to_string(index=False,
                                             float_format=lambda x: f"{x:.4g}"))

    # ── Step 3: feasible near-GT init and fit ────────────────────────────────
    # Interpolate GT toward a real training sample (convex combo → stays feasible),
    # instead of projecting a perturbed GT (which can invert fluxes).
    mfa_model.experiments = noisy_exps
    gt_arr = gt_flux.reindex(reaction_ids).values.astype(np.float64)
    v_init, rev_init, init_info = _make_init_fluxes(
        all_flux_df, all_rev_df, gt_flux, gt_rev, args.sample_ix,
        args.init_noise, reaction_ids, rng
    )
    mfa_model.reset_initial_fluxes(pd.Series(v_init, index=reaction_ids))
    # reset_initial_fluxes() re-seeds reversibility randomly; use the feasible
    # interpolated reversibility instead.
    with torch.no_grad():
        mfa_model._reversibility.copy_(torch.tensor(rev_init, dtype=dtype, device=device))

    if args.debug:
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
        f_r2, f_corr, f_rmse = _agreement(init_flux, gt_arr)
        l_r2, l_corr, l_rmse = _labeling_agreement(init_lab, gt_labeling_vec)
        print(f"\n[debug] Step 3 — feasible init: agreement & objective split")
        print(f"  init via sample : j={init_info['j']}  dev={init_info['dev']:.4f}  "
              f"α={init_info['alpha']:.4f}  realized_dev={init_info['realized_dev']:.4f} "
              f"(target {args.init_noise})")
        print(f"  ||S·v_init||_max: {sv_init:.2e}  (convex combo → should be ~0)")
        print(f"  v_std==v_mean   : {same if idx_v.numel() > 0 else 'n/a'} (must be False)")
        print(f"  init flux       : R²={f_r2:.4f} r={f_corr:.4f} RMSE={f_rmse:.4f}")
        print(f"  init labeling   : R²={l_r2:.4f} r={l_corr:.4f} RMSE={l_rmse:.4f}  (GT-nonzero)")
        print(f"  init labeling χ²: {lab_chi2:.4f}")
        print(f"  init flux χ²    : {flux_chi2:.4f}")

    print(f"\nFitting (max_iter={args.max_iter}, tol={args.tol}) ...")
    loss_log = fit_mfa_model(mfa_model, max_iter=args.max_iter, tol=args.tol,
                             disp=1 if args.debug else 0)

    # ── Step 4: diagnostics ───────────────────────────────────────────────────
    pred_flux = mfa_model.fluxes.detach().cpu().numpy()
    gt_vals = gt_flux.reindex(reaction_ids).values

    mae  = float(np.mean(np.abs(pred_flux - gt_vals)))
    rmse = float(np.sqrt(np.mean((pred_flux - gt_vals) ** 2)))
    corr = float(np.corrcoef(pred_flux, gt_vals)[0, 1])
    r2_flux = _r2(pred_flux, gt_vals)

    # Measured fluxes are soft-constrained → should be near-perfect; residual error
    # should isolate to the unmeasured / non-identifiable DOFs.
    measured_set = {r for r in measured_rxn_ids if r in reaction_ids}
    meas_mask = np.array([r in measured_set for r in reaction_ids])
    r2_meas = _r2(pred_flux[meas_mask], gt_vals[meas_mask]) if meas_mask.any() else float('nan')
    r2_unmeas = (_r2(pred_flux[~meas_mask], gt_vals[~meas_mask])
                 if (~meas_mask).any() else float('nan'))
    rmse_meas = (float(np.sqrt(np.mean((pred_flux[meas_mask] - gt_vals[meas_mask]) ** 2)))
                 if meas_mask.any() else float('nan'))
    rmse_unmeas = (float(np.sqrt(np.mean((pred_flux[~meas_mask] - gt_vals[~meas_mask]) ** 2)))
                   if (~meas_mask).any() else float('nan'))

    # Mass balance  Sv = 0
    with torch.no_grad():
        sv = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().cpu().numpy()
    sv_max  = float(np.max(np.abs(sv)))
    sv_rmse = float(np.sqrt(np.mean(sv ** 2)))

    # Labeling recovery: predicted vs GT (clean) labeling, extracted with the SAME
    # labeling_to index used to write the H5 (aligned by construction). Reported both
    # over the observed subset (what the fit actually sees: mapped_met_ids) and over
    # all atom-pattern metabolites (the full H5 columns).
    with torch.no_grad():
        pred_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()
    r2_lab, lab_corr, lab_rmse = _labeling_agreement(pred_labeling_vec, gt_labeling_vec)
    obs_mask = np.array([m in mapped_met_ids for (_, m, _) in labeling_index])
    r2_lab_obs, lab_corr_obs, lab_rmse_obs = _labeling_agreement(
        pred_labeling_vec, gt_labeling_vec, extra_mask=obs_mask)

    chi2_final, converged, n_iter = (loss_log["loss"], loss_log["converged"],
                                     loss_log["n_iter"])

    if args.debug:
        lab_chi2_post, flux_chi2_post = _chi2_split(mfa_model)
        print(f"\n[debug] Step 4 — post-fit objective split (raw χ², per block)")
        print(f"  labeling χ²     : {lab_chi2_post:.4f}")
        print(f"  flux χ²         : {flux_chi2_post:.4f}")
        if flux_data_obs is not None:
            # Are the measured-flux soft constraints actually satisfied? Each should
            # sit within ~1σ of GT once the flux block carries proper weight (Option A).
            rid_to_pred = dict(zip(reaction_ids, pred_flux))
            print("  measured-flux recovery (should be |err|/σ ≈ 1):")
            print(f"    {'reaction':16s} {'GT':>9s} {'pred':>9s} {'|err|':>8s} {'|err|/σ':>8s}")
            for f in flux_data_obs:
                rid = f.reaction.id
                gt_v, pr_v = float(gt_flux[rid]), float(rid_to_pred[rid])
                err = abs(pr_v - gt_v)
                print(f"    {rid:16s} {gt_v:9.3g} {pr_v:9.3g} {err:8.3g} "
                      f"{err / max(f._std, 1e-12):8.2f}")

    print(f"\n── Results ──────────────────────────────────────────────────────")
    print(f"  Converged         : {converged}")
    print(f"  Iterations        : {n_iter}")
    print(f"  Final χ²          : {chi2_final:.6f}")
    print(f"")
    print(f"  Flux vs ground truth")
    print(f"    MAE             : {mae:.4f}")
    print(f"    RMSE            : {rmse:.4f}")
    print(f"    Pearson r       : {corr:.4f}")
    print(f"    R²              : {r2_flux:.4f}")
    print(f"    R² (measured)   : {r2_meas:.4f}  RMSE {rmse_meas:.4f}")
    print(f"    R² (unmeasured) : {r2_unmeas:.4f}  RMSE {rmse_unmeas:.4f}")
    print(f"")
    print(f"  Mass balance  ||Sv||")
    print(f"    max |Sv_i|      : {sv_max:.2e}")
    print(f"    RMSE Sv         : {sv_rmse:.2e}")
    print(f"")
    print(f"  Labeling recovery (pred vs GT, GT-nonzero components)")
    print(f"    R² (observed)   : {r2_lab_obs:.4f}  r {lab_corr_obs:.4f}  RMSE {lab_rmse_obs:.4f}")
    print(f"    R² (all H5)     : {r2_lab:.4f}  r {lab_corr:.4f}  RMSE {lab_rmse:.4f}")


if __name__ == "__main__":
    main()
