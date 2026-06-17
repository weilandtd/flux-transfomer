"""
MFA accuracy benchmark using real training data.

Evaluates how accurately traditional MFA recovers fluxes when given the same
inputs that mirror a real experiment:
  - Ground-truth fluxes from the training HDF5 data
  - 13C MID restricted to metabolites in features.csv (all metabolite_ids, not
    filtered by the included flag — equivalent to the mapping CSV)
  - Measured flux constraints (cfg.training.measured_fluxes) with added noise

Each benchmark run:
  1. Draws a ground-truth flux from the training flux HDF5.
  2. Simulates 13C MID via MFAModel forward pass (+ optional Gaussian noise),
     then restricts to metabolites present in features.csv.
  3. Adds measured-flux soft constraints (with flux_noise_std noise).
  4. Fits MFA from a pFBA initialization.
  5. Compares predicted vs. ground-truth fluxes.

Results are directly comparable to AI model accuracy metrics.

Usage (I. orientalis):
    python -u paper/pipeline/run_mfa_benchmark.py \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 10 --max-iter 500
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from cobra.io.json import load_json_model

_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

torch.set_num_threads(_n_threads)
torch.set_num_interop_threads(1)

from sigmaflux.ai.data import load_data
from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


def parse_args():
    p = argparse.ArgumentParser(description="MFA accuracy benchmark vs training data")
    p.add_argument("--config", required=True,
                   help="Organism YAML config")
    p.add_argument("--data-dir", required=True,
                   help="Training run directory containing model_prepared_for_mfa.json "
                        "and HDF5 data files")
    p.add_argument("--output-dir", default=None,
                   help="Results directory (default: <data-dir>/mfa_benchmark)")
    p.add_argument("--n-benchmarks", type=int, default=10,
                   help="Number of ground-truth flux / fit pairs to evaluate")
    p.add_argument("--start-ix", type=int, default=0,
                   help="Index of first sample (for SLURM array parallelism)")
    p.add_argument("--max-iter", type=int, default=500,
                   help="L-BFGS-B max iterations")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Gradient tolerance (gtol) for convergence")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=0.05,
                   help="Gaussian noise std added to measured flux observations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _build_all_targets(mfa_model):
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


def _labeling_tensor_to_mid_df(mfa_model, labeling, noise_std, rng, mapped_met_ids=None):
    """Convert forward-pass labeling tensor to [id, mid, mean, std] DataFrame.

    If mapped_met_ids is provided, only metabolites in that set are included.
    """
    rows = []
    for subnet_size, emus in mfa_model.emu_met_filter.items():
        for emu in emus:
            if mapped_met_ids is not None and emu.id not in mapped_met_ids:
                continue
            met_col = mfa_model._metabolite_hash[emu.id]
            fracs = labeling[:subnet_size + 1, met_col].detach().cpu().numpy()
            for mid_ix, frac in enumerate(fracs):
                noise = np.clip(rng.normal(0.0, noise_std), -noise_std, noise_std)
                rows.append({
                    "id": emu.id,
                    "mid": mid_ix,
                    "mean": max(0.0, float(frac) + noise),
                    "std": max(noise_std, 1e-6),
                })
    return pd.DataFrame(rows)


def _make_measured_flux_data(mfa_model, measured_rxn_ids, gt_flux_series, flux_noise_std, rng):
    """Build a FluxData object for the measured reactions with noisy ground-truth values."""
    rows = []
    for rxn_id in measured_rxn_ids:
        if rxn_id not in gt_flux_series.index:
            continue
        true_val = float(gt_flux_series[rxn_id])
        bound = flux_noise_std * abs(true_val)
        noise = np.clip(rng.normal(0.0, bound if bound > 0 else 1e-12), -bound, bound)
        rows.append({"reaction": rxn_id, "mean": true_val + noise, "std": max(bound, 1e-6)})
    if not rows:
        return None
    return FluxData(mfa_model, pd.DataFrame(rows))


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = args.flux_noise_std

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    output_dir = _p(args.output_dir) if args.output_dir else os.path.join(data_dir, "mfa_benchmark")
    os.makedirs(output_dir, exist_ok=True)
    end_ix = args.start_ix + args.n_benchmarks - 1
    tag = f"samples_{args.start_ix:05d}_{end_ix:05d}"

    # ── Load atom-pattern annotated model ─────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)
    print(f"  {len(cobra_model.reactions)} reactions, {len(cobra_model.metabolites)} metabolites")

    # ── Load features.csv → restrict MIDs to metabolites in training data ─────
    features_path = os.path.join(data_dir, "features.csv")
    feat_df = pd.read_csv(features_path)
    mapped_met_ids = set(feat_df["metabolite_id"].dropna())
    print(f"Loading features: {features_path}")
    print(f"  {len(mapped_met_ids)} metabolite IDs from training data")

    # ── Load ground-truth fluxes from training HDF5 ───────────────────────────
    print(f"Loading training data from: {data_dir}")
    _, all_flux_df, all_reversibility_df = load_data(data_dir)
    n_total = all_flux_df.shape[1]
    flux_df = all_flux_df.iloc[:, args.start_ix:args.start_ix + args.n_benchmarks]
    reversibility_df = all_reversibility_df.iloc[:, args.start_ix:args.start_ix + args.n_benchmarks]
    n_runs = flux_df.shape[1]
    print(f"  {n_runs} flux samples (indices {args.start_ix}–{args.start_ix + n_runs - 1} "
          f"of {n_total} total, {flux_df.shape[0]} reactions)")

    if measured_rxn_ids:
        missing = [r for r in measured_rxn_ids if r not in flux_df.index]
        if missing:
            print(f"  Warning: {len(missing)} measured_fluxes not in flux_df: {missing}")
        present = [r for r in measured_rxn_ids if r in flux_df.index]
        print(f"  {len(present)}/{len(measured_rxn_ids)} measured flux reactions found in data")

    # ── Create MFAModel from first ground-truth flux ──────────────────────────
    first_gt = flux_df.iloc[:, 0]
    mfa_model = MFAModel(
        cobra_model,
        "MFA_benchmark",
        initial_flux_distribution=first_gt,
    )

    # ── Build EMU targets and prepare model once ──────────────────────────────
    targets, _ = _build_all_targets(mfa_model)

    tracers = [
        Tracer(
            mfa_model.metabolites.get_by_id(t["tracer_id"]),
            t["enrichment"],
            t["labeling"],
        )
        for t in tracer_cfg
    ]

    placeholder_exp = Experiment(MassIsotopomerData(), tracers, fluxes=None)
    print("Preparing EMU network (first call, builds dA/dB — may take several minutes) ...")
    mfa_model.prepare([placeholder_exp], targets=targets, with_backward=True)
    print("  EMU network ready.")


    # ── Benchmark loop ────────────────────────────────────────────────────────
    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_cols = {"reaction": reaction_ids}
    rev_cols = {"reaction": reaction_ids}
    labeling_df_out = None

    fluxes_path = os.path.join(output_dir, f"fluxes_{tag}.csv")
    rev_path = os.path.join(output_dir, f"reversibility_{tag}.csv")
    labeling_path = os.path.join(output_dir, f"labeling_{tag}.csv")

    for run_ix in range(n_runs):
        global_ix = args.start_ix + run_ix
        gt_flux = flux_df.iloc[:, run_ix]
        gt_rev = reversibility_df.iloc[:, run_ix]
        print(f"\n── Run {run_ix + 1}/{n_runs}  (sample {global_ix}) ──────────────────")

        # ── Step 1: simulate MID from ground-truth fluxes ────────────────────
        mfa_model.reset_initial_fluxes(gt_flux)
        with torch.no_grad():
            labeling = mfa_model()

        flux_cols[f"gt_{global_ix}"] = mfa_model.fluxes.detach().cpu().numpy()
        rev_cols[f"gt_{global_ix}"] = gt_rev.reindex(reaction_ids).values

        # ── Step 2: build MID table restricted to mapped metabolites ──────────
        mid_df = _labeling_tensor_to_mid_df(
            mfa_model, labeling, args.noise_std, rng, mapped_met_ids=mapped_met_ids
        )
        print(f"  MID observations: {len(mid_df)} rows ({len(mid_df['id'].unique())} metabolites)")

        synthetic_mid = MassIsotopomerData(model=mfa_model, table=mid_df)

        # ── Step 3: measured flux constraints ─────────────────────────────────
        flux_data = _make_measured_flux_data(
            mfa_model, measured_rxn_ids, gt_flux, flux_noise_std, rng
        )

        # ── Step 4: build experiment and reset to random initial sample ───────
        synthetic_exp = Experiment(synthetic_mid, tracers, fluxes=flux_data)
        mfa_model.experiments = [synthetic_exp]
        init_ix = int(rng.integers(0, n_total))
        mfa_model.reset_initial_fluxes(all_flux_df.iloc[:, init_ix])

        # ── Step 5: fit ───────────────────────────────────────────────────────
        loss_log = fit_mfa_model(mfa_model, max_iter=args.max_iter, tol=args.tol)
        print(f"  χ²={loss_log[0]:.5f}, converged={loss_log[1]}, iter={loss_log[2]}")

        # ── Step 6: collect predictions and flush to disk ─────────────────────
        flux_cols[f"pred_{global_ix}"] = mfa_model.fluxes.detach().cpu().numpy()
        rev_cols[f"pred_{global_ix}"] = mfa_model.reversibility.detach().cpu().numpy()

        with torch.no_grad():
            pred_labeling = mfa_model()
        pred_mid_df = _labeling_tensor_to_mid_df(
            mfa_model, pred_labeling, 0.0, rng, mapped_met_ids=mapped_met_ids
        )
        pred_mid_df = pred_mid_df.rename(columns={"mean": f"pred_{global_ix}"})[
            ["id", "mid", f"pred_{global_ix}"]
        ]
        labeling_df_out = (
            pred_mid_df if labeling_df_out is None
            else labeling_df_out.merge(pred_mid_df, on=["id", "mid"], how="outer")
        )

        pd.DataFrame(flux_cols).to_csv(fluxes_path, index=False)
        pd.DataFrame(rev_cols).to_csv(rev_path, index=False)
        labeling_df_out.to_csv(labeling_path, index=False)
        print(f"  Saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–{end_ix}).")
    print(f"  Fluxes        → {fluxes_path}")
    print(f"  Reversibility → {rev_path}")
    print(f"  Labeling      → {labeling_path}")


if __name__ == "__main__":
    main()
