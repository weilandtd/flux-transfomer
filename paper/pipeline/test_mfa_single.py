"""
Single-run MFA calibration test.

Fits one MFA sample and prints convergence diagnostics so hyperparameters can
be tuned before committing to the full array benchmark.

Usage:
    python -u paper/pipeline/test_mfa_single.py \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --sample-ix 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from cobra.io.json import load_json_model

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.ai.data import load_data
from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


def parse_args():
    p = argparse.ArgumentParser(description="Single-run MFA hyperparameter test")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sample-ix", type=int, default=0,
                   help="Which sample from the HDF5 to use as ground truth")
    p.add_argument("--init-ix", type=int, default=None,
                   help="Sample index used as initial flux guess (default: random)")
    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--max-iter", type=int, default=1000,
                   help="L-BFGS-B max iterations")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Projected-gradient tolerance (gtol) for L-BFGS-B convergence")
    # ── Data noise / experiment ────────────────────────────────────────────────
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = cfg.get("training", {}).get("flux_noise_std", 0.05)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)

    features_path = os.path.join(data_dir, "features.csv")
    mapped_met_ids = set(pd.read_csv(features_path)["metabolite_id"].dropna())

    print("Loading training data ...")
    _, all_flux_df, _ = load_data(data_dir)
    n_total = all_flux_df.shape[1]
    print(f"  {n_total} samples, {all_flux_df.shape[0]} reactions")

    gt_flux = all_flux_df.iloc[:, args.sample_ix]
    print(f"  Ground-truth sample : {args.sample_ix}")

    init_ix = args.init_ix if args.init_ix is not None else int(rng.integers(0, n_total))
    init_flux = all_flux_df.iloc[:, init_ix]
    print(f"  Initial-guess sample: {init_ix}")

    # ── Build MFAModel ────────────────────────────────────────────────────────
    mfa_model = MFAModel(cobra_model, "MFA_test",
                         initial_flux_distribution=gt_flux)

    targets = set()
    for met in mfa_model.metabolites:
        pat = met.annotation.get("atom_pattern", "")
        if pat:
            targets.add(EMU(met, list(range(1, len(pat) + 1))))

    tracers = [
        Tracer(mfa_model.metabolites.get_by_id(t["tracer_id"]),
               t["enrichment"], t["labeling"])
        for t in tracer_cfg
    ]

    placeholder_exp = Experiment(MassIsotopomerData(), tracers, fluxes=None)
    print("Preparing EMU network ...")
    mfa_model.prepare([placeholder_exp], targets=targets, with_backward=True)
    print("  done.")

    # ── Simulate ground-truth MID ─────────────────────────────────────────────
    mfa_model.reset_initial_fluxes(gt_flux)
    with torch.no_grad():
        labeling = mfa_model()

    rows = []
    for subnet_size, emus in mfa_model.emu_met_filter.items():
        for emu in emus:
            if emu.id not in mapped_met_ids:
                continue
            met_col = mfa_model._metabolite_hash[emu.id]
            fracs = labeling[:subnet_size + 1, met_col].detach().cpu().numpy()
            for mid_ix, frac in enumerate(fracs):
                noisy = float(frac) + rng.normal(0.0, args.noise_std)
                rows.append({"id": emu.id, "mid": mid_ix,
                             "mean": max(0.0, noisy),
                             "std": max(args.noise_std, 1e-6)})
    mid_df = pd.DataFrame(rows)
    print(f"  MID observations: {len(mid_df)} rows ({len(mid_df['id'].unique())} metabolites)")

    flux_rows = []
    for rxn_id in measured_rxn_ids:
        if rxn_id not in gt_flux.index:
            continue
        mean_val = float(gt_flux[rxn_id]) + rng.normal(0.0, flux_noise_std)
        flux_rows.append({"reaction": rxn_id, "mean": mean_val, "std": flux_noise_std})
    flux_data = FluxData(mfa_model, pd.DataFrame(flux_rows)) if flux_rows else None

    # ── Fit ───────────────────────────────────────────────────────────────────
    synthetic_mid = MassIsotopomerData(model=mfa_model, table=mid_df)
    synthetic_exp = Experiment(synthetic_mid, tracers, fluxes=flux_data)
    mfa_model.experiments = [synthetic_exp]
    mfa_model.reset_initial_fluxes(init_flux)

    print(f"\nFitting (max_iter={args.max_iter}, tol={args.tol}) ...")
    loss_log = fit_mfa_model(
        mfa_model,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    pred_flux = mfa_model.fluxes.detach().cpu().numpy()
    reaction_ids = [r.id for r in mfa_model.reactions]
    gt_vals = gt_flux.reindex(reaction_ids).values

    # Flux agreement
    mae  = float(np.mean(np.abs(pred_flux - gt_vals)))
    rmse = float(np.sqrt(np.mean((pred_flux - gt_vals) ** 2)))
    corr = float(np.corrcoef(pred_flux, gt_vals)[0, 1])
    ss_res = float(np.sum((pred_flux - gt_vals) ** 2))
    ss_tot = float(np.sum((gt_vals - gt_vals.mean()) ** 2))
    r2_flux = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Mass balance  Sv = 0
    with torch.no_grad():
        sv = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().cpu().numpy()
    sv_max  = float(np.max(np.abs(sv)))
    sv_rmse = float(np.sqrt(np.mean(sv ** 2)))

    # Labeling agreement: predicted MID vs noisy observations
    with torch.no_grad():
        pred_labeling = mfa_model()
    pred_rows = []
    for subnet_size, emus in mfa_model.emu_met_filter.items():
        for emu in emus:
            if emu.id not in mapped_met_ids:
                continue
            met_col = mfa_model._metabolite_hash[emu.id]
            fracs = pred_labeling[:subnet_size + 1, met_col].detach().cpu().numpy()
            for mid_ix, frac in enumerate(fracs):
                pred_rows.append({"id": emu.id, "mid": mid_ix, "pred": float(frac)})
    pred_mid_df = pd.DataFrame(pred_rows)
    obs_mid_df  = mid_df.rename(columns={"mean": "obs"})
    cmp = pred_mid_df.merge(obs_mid_df[["id", "mid", "obs"]], on=["id", "mid"])
    lab_rmse = float(np.sqrt(np.mean((cmp["pred"].values - cmp["obs"].values) ** 2)))
    lab_corr = float(np.corrcoef(cmp["pred"].values, cmp["obs"].values)[0, 1])
    lab_ss_res = float(np.sum((cmp["pred"].values - cmp["obs"].values) ** 2))
    lab_ss_tot = float(np.sum((cmp["obs"].values - cmp["obs"].values.mean()) ** 2))
    r2_lab = 1.0 - lab_ss_res / lab_ss_tot if lab_ss_tot > 0 else float('nan')

    chi2_final, converged, n_iter = loss_log[0], loss_log[1], loss_log[2]

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
    print(f"")
    print(f"  Mass balance  ||Sv||")
    print(f"    max |Sv_i|      : {sv_max:.2e}")
    print(f"    RMSE Sv         : {sv_rmse:.2e}")
    print(f"")
    print(f"  Labeling agreement (pred vs noisy obs)")
    print(f"    RMSE            : {lab_rmse:.4f}")
    print(f"    Pearson r       : {lab_corr:.4f}")
    print(f"    R²              : {r2_lab:.4f}")


if __name__ == "__main__":
    main()
