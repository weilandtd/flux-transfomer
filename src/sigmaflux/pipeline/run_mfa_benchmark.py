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
  4. Fits MFA from a feasible near-GT initialization, restarting until the fit
     clears the reduced-χ² acceptance bound.
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
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

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
    load_benchmark_inputs,
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
    p.add_argument("--max-iter", type=int, default=500,
                   help="L-BFGS-B max iterations")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Gradient tolerance (gtol) for convergence")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=None,
                   help="Relative noise std on measured flux observations "
                        "(default: cfg.training.flux_noise_std, else 0.05). "
                        "0 = no flux noise (measured fluxes pinned to GT)")
    p.add_argument("--init-noise", type=float, default=0.20,
                   help="Target relative L2 deviation ||v_init - v_gt|| / ||v_gt|| of the "
                        "feasible near-GT initialization")
    p.add_argument("--max-starts", type=int, default=10,
                   help="N_max multi-start restarts per sample. Restart from a new "
                        "feasible init until the reduced-χ² acceptance bound is met; "
                        "if none of the N_max starts is acceptable, write NaN.")
    p.add_argument("--accept-z", type=float, default=1.645,
                   help="z-score for the acceptance bound M ≤ 1 + z·sqrt(2/h_eff) "
                        "(1.645 = one-sided 95%%).")
    p.add_argument("--diagnose", action="store_true",
                   help="Fit a single sample (the one at --start-ix) with step-by-step "
                        "diagnostics and recovery metrics, writing no HDF5. Ignores "
                        "--max-starts / --accept-z: exactly one fit is run.")
    p.add_argument("--seed", type=int, default=42)
    add_root_argument(p)
    return p


def make_measured_flux_data(mfa_model, measured_rxn_ids, gt_flux_series, flux_noise_std, rng):
    # Relative flux noise: measured = true·(1 + N(0, flux_noise_std)); the variance
    # used in the χ² objective is (flux_noise_std·|true|)² (fluxes are real magnitudes).
    rows = []
    for rxn_id in measured_rxn_ids:
        if rxn_id not in gt_flux_series.index:
            continue
        true_val = float(gt_flux_series[rxn_id])
        meas = true_val * (1.0 + rng.normal(0.0, flux_noise_std))
        rows.append({"reaction": rxn_id, "mean": meas,
                     "std": max(flux_noise_std * abs(true_val), 1e-9)})
    return FluxData(mfa_model, pd.DataFrame(rows)) if rows else None


def make_init_fluxes(all_flux_df, all_rev_df, gt_flux, gt_rev, sample_ix,
                     init_noise, reaction_ids, rng):
    """Feasible near-GT init by interpolating toward a real training sample.

    The steady-state flux space is a convex polytope and every training sample is
    feasible, so a convex combination of GT and another sample is feasible too — unlike
    a null-space projection of a perturbed GT, which can invert fluxes and leave the
    polytope. Pick a random sample j != sample_ix, then interpolate GT → sample j with a
    fraction α set so the global relative deviation ||v_init - v_gt|| / ||v_gt|| equals
    init_noise (relative L2, dominated by the large/meaningful fluxes — NOT a per-reaction
    ratio that tiny fluxes would blow up).

    Returns (v_init, r_init, info) where info = {j, dev, alpha, realized_dev}.
    """
    v_gt = gt_flux.reindex(reaction_ids).values.astype(np.float64)
    r_gt = gt_rev.reindex(reaction_ids).values.astype(np.float64)

    n_samples = all_flux_df.shape[1]
    j = int(rng.integers(0, n_samples))
    while j == sample_ix and n_samples > 1:
        j = int(rng.integers(0, n_samples))
    v_j = all_flux_df.iloc[:, j].reindex(reaction_ids).values.astype(np.float64)
    r_j = all_rev_df.iloc[:, j].reindex(reaction_ids).values.astype(np.float64)

    gt_norm = float(np.linalg.norm(v_gt))
    dev = float(np.linalg.norm(v_j - v_gt) / gt_norm) if gt_norm > 0 else 0.0
    alpha = min(1.0, init_noise / dev) if dev > 0 else 0.0

    v_init = v_gt + alpha * (v_j - v_gt)          # convex combo → S·v_init = 0
    r_init = r_gt + alpha * (r_j - r_gt)          # convex combo of nonneg → ≥ 0
    return v_init, r_init, {"j": j, "dev": dev, "alpha": alpha,
                            "realized_dev": alpha * dev}


def acceptance_threshold(n_L, n_F, z=1.645):
    """Upper bound on M = ½·(χ²_L/n_L) + ½·(χ²_F/n_F), the equal-weight per-block reduced χ².

    Each reduced χ² block has mean ≈1 and variance ≈2/n_b. Matching the first two moments
    of M to a single scaled χ² gives an effective df h_eff = 1 / Σ_b w_b²/n_b; with equal
    weights w_L=w_F=½ this is h_eff = 4·n_L·n_F/(n_L+n_F) (and h_eff = n_L when n_F == 0).
    M then behaves like a reduced χ² on h_eff df, so we reject above
    χ²_0.95(h_eff)/h_eff ≈ 1 + z·sqrt(2/h_eff).
    """
    if n_F > 0:
        h_eff = 4.0 * n_L * n_F / (n_L + n_F)
    else:
        h_eff = float(n_L)
    return 1.0 + z * np.sqrt(2.0 / h_eff)


def build_noisy_experiments(experiments, gt_series, noise_std, rng, mfa_model,
                            mapped_met_ids, flux_data):
    """Build one noisy Experiment per tracer from GT labeling series."""
    noisy_exps = []
    for experiment in experiments:
        tracer = experiment.tracers[0]
        tracer_str = "|".join(str(t) for t in experiment.tracers)
        tracer_data = gt_series.loc[tracer_str]  # (Metabolite, Isotopomer) Series

        met_frac_sum = defaultdict(float)
        rows = []
        for (met_id, iso_str), frac in tracer_data.items():
            if mapped_met_ids and met_id not in mapped_met_ids:
                continue
            noise = np.clip(rng.normal(0.0, noise_std), -noise_std, noise_std)
            noisy_frac = max(0.0, float(frac) + noise)
            rows.append({"id": met_id, "mid": int(iso_str[2:]), "mean": noisy_frac,
                         "std": noise_std})
            met_frac_sum[met_id] += noisy_frac

        # M+0 per metabolite: needed by MassIsotopomerData to infer EMU size
        m0_rows = [{"id": m, "mid": 0, "mean": max(0.0, 1.0 - s), "std": noise_std}
                   for m, s in met_frac_sum.items()]
        noisy_mid = MassIsotopomerData(model=mfa_model, table=pd.DataFrame(m0_rows + rows))
        noisy_exps.append(Experiment(noisy_mid, [tracer], fluxes=flux_data))
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


def _report_init(mfa_model, init_info, init_noise, gt_flux, gt_flux_arr, gt_labeling_vec,
                 labeling_to, flux_data_obs):
    """Pre-fit diagnostics: is the init feasible, and where does the objective sit?"""
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
    print(f"  init via sample : j={init_info['j']}  dev={init_info['dev']:.4f}  "
          f"α={init_info['alpha']:.4f}  realized_dev={init_info['realized_dev']:.4f} "
          f"(target {init_noise})")
    print(f"  ||S·v_init||_max: {sv_init:.2e}  (convex combo → should be ~0)")
    print(f"  v_std==v_mean   : {same if idx_v.numel() > 0 else 'n/a'} (must be False)")
    print(f"  init flux       : R²={f_r2:.4f} r={f_corr:.4f} RMSE={f_rmse:.4f}")
    print(f"  init labeling   : R²={l_r2:.4f} r={l_corr:.4f} RMSE={l_rmse:.4f}  (GT-nonzero)")
    print(f"  init labeling χ²: {lab_chi2:.4f}")
    print(f"  init flux χ²    : {flux_chi2:.4f}")


def _report_fit(mfa_model, res, gt_flux, gt_labeling_vec, labeling_to, labeling_index,
                reaction_ids, measured_rxn_ids, mapped_met_ids, flux_data_obs, accept_z):
    """Post-fit diagnostics: convergence, flux/labeling recovery, mass balance."""
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

    M = 0.5 * (res["ssr_labeling"] + res["ssr_fluxes"])
    M_accept = acceptance_threshold(res["n_labeling"], res["n_fluxes"], accept_z)

    print("\n── Results ──────────────────────────────────────────────────────")
    print(f"  Converged         : {res['converged']}")
    print(f"  Iterations        : {res['n_iter']}")
    print(f"  Final χ²          : {res['loss']:.6f}")
    print(f"  M (½·reduced χ²)  : {M:.4f}  (accept ≤ {M_accept:.4f})")
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

    cobra_model, mapped_met_ids, _, all_flux_df, all_rev_df = load_benchmark_inputs(data_dir)

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

    mfa_model, reaction_ids, experiments, labeling_to, labeling_index = build_mfa_context(
        cobra_model, tracer_cfg, flux_df.iloc[:, 0], "MFA_benchmark", with_backward=True,
    )

    # ── Benchmark loop ────────────────────────────────────────────────────────
    pred_lab_data, pred_flux_data, sample_indices = [], [], []
    pred_base = os.path.join(output_dir, f"pred_{tag}")

    for run_ix in range(n_runs):
        global_ix = int(sel[run_ix])   # real HDF5 column index of this sample
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
        gt_series = pd.Series(gt_labeling_vec, index=labeling_index)
        flux_data_obs = make_measured_flux_data(
            mfa_model, measured_rxn_ids, gt_flux, flux_noise_std, rng
        )
        noisy_exps = build_noisy_experiments(
            experiments, gt_series, args.noise_std, rng, mfa_model, mapped_met_ids,
            flux_data_obs
        )
        n_obs = sum(len(e.mid_data) for e in noisy_exps)
        print(f"  Noisy MID: {n_obs} entries across {len(noisy_exps)} tracers")

        # ── Step 3: multi-start fit until the reduced-χ² bound is met ─────────
        # The measurements above are fixed for this sample; only the starting point
        # changes across restarts. Each start interpolates GT toward a *different*
        # random feasible training sample (convex combo → stays feasible, never inverts
        # fluxes). Restart until M = ½·(ssr_L + ssr_F) clears the effective-df bound, or
        # max_starts is exhausted — L-BFGS-B otherwise sticks in the first local minimum.
        mfa_model.experiments = noisy_exps
        accepted = False
        for start in range(1 if args.diagnose else args.max_starts):
            v_init, rev_init, init_info = make_init_fluxes(
                all_flux_df, all_rev_df, gt_flux, gt_rev, global_ix,
                args.init_noise, reaction_ids, rng
            )
            mfa_model.reset_initial_fluxes(pd.Series(v_init, index=reaction_ids))
            # reset_initial_fluxes() re-seeds reversibility randomly; use the feasible
            # interpolated reversibility instead.
            with torch.no_grad():
                mfa_model._reversibility.copy_(
                    torch.tensor(rev_init, dtype=mfa_model.dtype, device=mfa_model.device)
                )
            if args.diagnose:
                _report_init(mfa_model, init_info, args.init_noise, gt_flux,
                             gt_flux.reindex(reaction_ids).values.astype(np.float64),
                             gt_labeling_vec, labeling_to, flux_data_obs)
                print(f"\nFitting (max_iter={args.max_iter}, tol={args.tol}) ...")

            res = fit_mfa_model(mfa_model, max_iter=args.max_iter, tol=args.tol,
                                disp=1 if args.diagnose else 0)
            M = 0.5 * (res["ssr_labeling"] + res["ssr_fluxes"])      # = res["loss"] / 2
            M_accept = acceptance_threshold(res["n_labeling"], res["n_fluxes"], args.accept_z)
            if args.diagnose:
                _report_fit(mfa_model, res, gt_flux, gt_labeling_vec, labeling_to,
                            labeling_index, reaction_ids, measured_rxn_ids,
                            mapped_met_ids, flux_data_obs, args.accept_z)
                return 0
            print(f"  start {start + 1}/{args.max_starts}: M={M:.4f} (accept≤{M_accept:.4f}) "
                  f"Loss={res['loss']:.5f} conv={res['converged']} iter={res['n_iter']}")
            if M <= M_accept:
                accepted = True
                break

        # ── Step 4: record predicted labeling/fluxes (NaN if no start accepted) ──
        if accepted:  # model holds the accepted solution (fit writes x_opt back)
            with torch.no_grad():
                pred_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()
            pred_flux_vec = mfa_model.fluxes.detach().cpu().numpy()
            print(f"  ✓ accepted after {start + 1} start(s)")
        else:  # not a real solution — write NaN so it is excluded from the benchmark
            pred_labeling_vec = np.full(len(labeling_index), np.nan)
            pred_flux_vec = np.full(len(reaction_ids), np.nan)
            print(f"  ✗ no acceptable fit in {args.max_starts} starts → NaN")

        pred_lab_data.append(pred_labeling_vec)
        pred_flux_data.append(pred_flux_vec)
        sample_indices.append(global_ix)

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
