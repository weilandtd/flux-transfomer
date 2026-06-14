"""
Synthetic MFA accuracy benchmark.

Evaluates how accurately traditional MFA recovers fluxes from simulated 13C
labeling data with known ground truth.  Each benchmark run:
  1. Draws a ground-truth flux from COBRA sampling (or a pre-computed CSV).
  2. Simulates 13C MID via MFAModel forward pass (+ optional Gaussian noise).
  3. Fits MFA from a pFBA initialization.
  4. Compares predicted vs. ground-truth fluxes.

Results are directly comparable to AI model accuracy metrics.

Usage (I. orientalis, 10 runs):
    python -u paper/pipeline/run_mfa_benchmark.py \\
        --config paper/configs/i_orientalis.yaml \\
        --output-dir paper/.results/mfa_benchmark_io_YYYYMMDD/ \\
        --n-benchmarks 10 --epochs 1000
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from cobra.flux_analysis import pfba
from cobra.io.json import load_json_model
from cobra.sampling import sample
from scipy.stats import pearsonr
from torch.nn import MSELoss

_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

torch.set_num_threads(_n_threads)

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic MFA accuracy benchmark")
    p.add_argument("--config", required=True, help="Organism YAML config (relative to repo root)")
    p.add_argument("--output-dir", required=True, help="Results directory (relative to repo root)")
    p.add_argument("--n-benchmarks", type=int, default=10,
                   help="Number of ground-truth flux / fit pairs to evaluate")
    p.add_argument("--flux-samples", default=None,
                   help="Pre-computed flux samples CSV (skips COBRA sampling if provided)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flux-scaler", type=float, default=200.0)
    p.add_argument("--lambda0", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _apply_media_bounds(cobra_model, media_cfg):
    for rxn in cobra_model.reactions:
        if rxn.id.startswith("EX_"):
            rxn.lower_bound = 0.0
            rxn.upper_bound = 0.0
    for rxn_id, bounds in media_cfg.items():
        try:
            rxn = cobra_model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = bounds["lb"]
            rxn.upper_bound = bounds["ub"]
        except KeyError:
            print(f"  Warning: media reaction {rxn_id} not in model, skipping")


def _build_all_targets(mfa_model):
    """Return (targets, measured_metabolites) for all metabolites with atom_pattern annotations."""
    targets = set()
    measured_metabolites = []
    for met in mfa_model.metabolites:
        atom_pattern = met.annotation.get("atom_pattern", "")
        if not atom_pattern:
            continue
        targets.add(EMU(met, list(range(1, len(atom_pattern) + 1))))
        measured_metabolites.append(met)
    if not targets:
        raise ValueError("No EMU targets found — check model atom_pattern annotations.")
    print(f"  EMU targets: {len(targets)} from {len(measured_metabolites)} metabolites")
    return targets, measured_metabolites


def _labeling_tensor_to_mid_df(mfa_model, labeling, noise_std, rng):
    """
    Convert a forward-pass labeling tensor to a [id, mid, mean, std] DataFrame
    suitable for MassIsotopomerData.

    labeling shape: (max_mid_size, n_model_metabolites)
    emu_met_filter: {subnet_size: [EMU objects]} — populated by prepare()
    """
    rows = []
    for subnet_size, emus in mfa_model.emu_met_filter.items():
        for emu in emus:
            met_col = mfa_model._metabolite_hash[emu.id]
            # M+0 through M+subnet_size
            fracs = labeling[:subnet_size + 1, met_col].detach().cpu().numpy()
            for mid_ix, frac in enumerate(fracs):
                noisy = float(frac) + rng.normal(0.0, noise_std)
                rows.append({
                    "id": emu.id,
                    "mid": mid_ix,
                    "mean": max(0.0, noisy),
                    "std": max(noise_std, 1e-6),
                })
    return pd.DataFrame(rows)


def _sample_ground_truth_fluxes(cobra_model, n, seed, flux_samples_path=None):
    """Return a DataFrame with n feasible flux distributions (rows=samples, cols=reactions)."""
    if flux_samples_path is not None:
        path = _p(flux_samples_path)
        print(f"  Loading flux samples from {path}")
        df = pd.read_csv(path, index_col=0)
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    print(f"  Sampling {n} ground-truth flux distributions via COBRA (seed={seed}) ...")
    flux_sample = sample(cobra_model, n, seed=seed, processes=1)
    return flux_sample.reset_index(drop=True)


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _p(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model and apply media bounds ─────────────────────────────────────
    print(f"Loading model: {cfg['model_file']}")
    cobra_model = load_json_model(_p(cfg["model_file"]))
    print(f"  {len(cobra_model.reactions)} reactions, {len(cobra_model.metabolites)} metabolites")
    _apply_media_bounds(cobra_model, cfg["sampling"]["media"])

    # ── Sample ground-truth flux distributions ────────────────────────────────
    gt_flux_df = _sample_ground_truth_fluxes(
        cobra_model, args.n_benchmarks, seed=args.seed, flux_samples_path=args.flux_samples
    )
    n_runs = len(gt_flux_df)
    print(f"  Running {n_runs} benchmark iterations")

    # ── Create MFAModel from first ground-truth flux ──────────────────────────
    first_gt = gt_flux_df.iloc[0]
    mfa_model = MFAModel(
        cobra_model,
        "MFA_benchmark",
        initial_flux_distribution=first_gt,
        lambda0=args.lambda0,
    )

    # ── Build EMU targets and prepare model once ──────────────────────────────
    targets, measured_metabolites = _build_all_targets(mfa_model)

    tracers = [
        Tracer(
            mfa_model.metabolites.get_by_id(t["tracer_id"]),
            t["enrichment"],
            t["labeling"],
        )
        for t in tracer_cfg
    ]

    # Placeholder experiment: empty MID (targets supplied explicitly so EMU
    # network is fully expanded even without measurement entries)
    placeholder_exp = Experiment(MassIsotopomerData(), tracers, fluxes=None)
    print("Preparing EMU network (first call, builds dA/dB — may take several minutes) ...")
    mfa_model.prepare([placeholder_exp], targets=targets, with_backward=True)
    print("  EMU network ready.")

    # pFBA solution used as the fixed initialization for every benchmark fit
    print("Computing pFBA initialization ...")
    pfba_sol = pfba(cobra_model)
    pfba_fluxes = pfba_sol.fluxes
    print("  pFBA done.")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    reaction_ids = [r.id for r in mfa_model.reactions]
    gt_matrix = np.zeros((n_runs, len(reaction_ids)))
    pred_matrix = np.zeros((n_runs, len(reaction_ids)))
    all_loss_logs = []

    for run_ix in range(n_runs):
        gt_flux = gt_flux_df.iloc[run_ix]
        print(f"\n── Run {run_ix + 1}/{n_runs} ───────────────────────────────────────")

        # ── Step 1: simulate MID from ground-truth fluxes ────────────────────
        mfa_model.reset_initial_fluxes(gt_flux)
        with torch.no_grad():
            labeling = mfa_model()   # (max_mid_size, n_metabolites)

        # Store ground truth (after setting the fluxes)
        gt_matrix[run_ix] = mfa_model.fluxes.detach().cpu().numpy()

        # ── Step 2: extract noisy synthetic MID table ─────────────────────────
        synthetic_table = _labeling_tensor_to_mid_df(mfa_model, labeling, args.noise_std, rng)

        # ── Step 3: build new experiment with synthetic MID, swap into model ──
        synthetic_mid = MassIsotopomerData(model=mfa_model, table=synthetic_table)
        synthetic_exp = Experiment(synthetic_mid, tracers, fluxes=None)
        mfa_model.experiments = [synthetic_exp]

        # ── Step 4: reset to pFBA initialization ─────────────────────────────
        mfa_model.reset_initial_fluxes(pfba_fluxes)

        # ── Step 5: fit ───────────────────────────────────────────────────────
        tag = f"run{run_ix:03d}_{timestamp}"
        ckpt = os.path.join(output_dir, f"ckpt_{tag}.pt")
        optimizer = torch.optim.Adam(mfa_model.parameters(), lr=args.lr)
        loss_log = fit_mfa_model(
            mfa_model,
            MSELoss(),
            MSELoss(),
            optimizer,
            epochs=args.epochs,
            flux_scaler=args.flux_scaler,
            checkpoint_file=ckpt,
            labeling_checkpoint_file=os.path.join(output_dir, f"ckpt_{tag}_labeling.pt"),
            optimizer_checkpoint=os.path.join(output_dir, f"ckpt_{tag}_optim.pt"),
        )
        all_loss_logs.append(loss_log)
        pred_matrix[run_ix] = mfa_model.fluxes.detach().cpu().numpy()

        print(f"  Loss: {loss_log[0]:.5f} (start)  →  {loss_log[-1]:.5f} (final)")

    # ── Compute per-reaction metrics across all runs ──────────────────────────
    print("\nComputing per-reaction metrics ...")
    metrics_rows = []
    for j, rxn_id in enumerate(reaction_ids):
        gt_vals = gt_matrix[:, j]
        pred_vals = pred_matrix[:, j]
        mae = float(np.mean(np.abs(gt_vals - pred_vals)))
        rmse = float(np.sqrt(np.mean((gt_vals - pred_vals) ** 2)))
        mean_abs_gt = float(np.mean(np.abs(gt_vals)))
        if n_runs >= 2 and np.std(gt_vals) > 1e-10 and np.std(pred_vals) > 1e-10:
            r, _ = pearsonr(gt_vals, pred_vals)
        else:
            r = float("nan")
        metrics_rows.append({
            "reaction": rxn_id,
            "pearson_r": r,
            "mae": mae,
            "rmse": rmse,
            "mean_abs_gt_flux": mean_abs_gt,
        })

    metrics_df = pd.DataFrame(metrics_rows)

    # ── Build flux comparison table ───────────────────────────────────────────
    flux_cols = {}
    for run_ix in range(n_runs):
        flux_cols[f"gt_{run_ix}"] = gt_matrix[run_ix]
        flux_cols[f"pred_{run_ix}"] = pred_matrix[run_ix]
    flux_df = pd.DataFrame({"reaction": reaction_ids, **flux_cols})

    # ── Build loss table ──────────────────────────────────────────────────────
    max_epochs = max(len(ll) for ll in all_loss_logs)
    loss_dict = {"epoch": list(range(max_epochs))}
    for run_ix, ll in enumerate(all_loss_logs):
        padded = ll + [float("nan")] * (max_epochs - len(ll))
        loss_dict[f"run_{run_ix}"] = padded
    loss_df = pd.DataFrame(loss_dict)

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, f"benchmark_metrics_{timestamp}.csv")
    fluxes_path = os.path.join(output_dir, f"benchmark_fluxes_{timestamp}.csv")
    loss_path = os.path.join(output_dir, f"benchmark_loss_{timestamp}.csv")

    metrics_df.to_csv(metrics_path, index=False)
    flux_df.to_csv(fluxes_path, index=False)
    loss_df.to_csv(loss_path, index=False)

    print(f"\nBenchmark complete ({n_runs} runs).")
    print(f"  Metrics  → {metrics_path}")
    print(f"  Fluxes   → {fluxes_path}")
    print(f"  Loss     → {loss_path}")

    # Print summary statistics
    finite_r = metrics_df["pearson_r"].dropna()
    print(f"\nAggregate Pearson r:  mean={finite_r.mean():.3f}, median={finite_r.median():.3f}")
    print(f"Aggregate MAE:        mean={metrics_df['mae'].mean():.4f}")
    print(f"Aggregate RMSE:       mean={metrics_df['rmse'].mean():.4f}")


if __name__ == "__main__":
    main()
