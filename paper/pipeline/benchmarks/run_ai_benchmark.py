"""
SigmaFlux AI accuracy benchmark using real training data.

Mirrors run_mfa_benchmark.py but uses the trained AI model instead of
traditional MFA to predict fluxes. Outputs are written to H5 files in the
same schema as the MFA benchmark so plot_test_performance.py can load both
identically. Ground-truth labeling is computed via the EMU forward pass with
GT fluxes from training H5, using the same per-tracer experiment structure
as generate_training_data.py.

Each benchmark run:
  1. Draws a ground-truth flux / labeling pair from the training HDF5.
  2. Adds Gaussian noise to the simulated MID and measured fluxes.
  3. Runs SigmaFlux.predict() to obtain predicted fluxes.
  4. Uses MFAModel.forward() on both predicted and GT fluxes to obtain
     labeling vectors via torch.take (per-tracer, same format as training H5).
  5. Writes pred and GT labeling + flux + reversibility to H5.

Usage (I. orientalis):
    python -u paper/pipeline/benchmarks/run_ai_benchmark.py \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 10
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from cobra.io.json import load_json_model

_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))

_REPO_ROOT = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))

torch.set_num_threads(_n_threads)

from sigmaflux.ai.data import find_checkpoint, load_data, split_sample_indices
from sigmaflux.ai.inference import load_predictor
from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment
from sigmaflux.core.model import MFAModel
from sigmaflux.pipeline.generate_training_data import (
    build_experiments,
    build_labeling_index,
    build_targets,
)


def parse_args():
    p = argparse.ArgumentParser(description="SigmaFlux AI accuracy benchmark vs training data")
    p.add_argument("--config", required=True, help="Organism YAML config")
    p.add_argument("--data-dir", required=True,
                   help="Training run directory containing model checkpoint and HDF5 data")
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained .pt model file (auto-detected from data-dir if omitted)")
    p.add_argument("--output-dir", default=None,
                   help="Results directory (default: <data-dir>/ai_benchmark)")
    p.add_argument("--split", choices=["val", "train", "all"], default="val",
                   help="Which AI-training split to benchmark on (default: val — the set "
                        "withheld from AI training)")
    p.add_argument("--n-benchmarks", type=int, default=100,
                   help="Number of samples to evaluate, as an offset count into the split")
    p.add_argument("--start-ix", type=int, default=0,
                   help="Offset of the first sample within the split (for SLURM array "
                        "parallelism); indexes the sorted split, not raw HDF5 columns")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=0.05,
                   help="Gaussian noise std added to measured flux observations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _write_h5(base, labeling_data, labeling_index, flux_data, reaction_ids, sample_indices):
    (pd.DataFrame(labeling_data, index=sample_indices, columns=labeling_index)
       .T.to_hdf(f"{base}_labeling.h5", key="labeling", mode="w"))
    (pd.DataFrame(flux_data, index=sample_indices, columns=reaction_ids)
       .T.to_hdf(f"{base}_flux.h5", key="fluxes", mode="w"))


def _build_df_L(labeling_df, cols, noise_std, rng, inv_tracer):
    """Convert the selected columns of training labeling_df to df_L for SigmaFlux.predict()."""
    batch = labeling_df.iloc[:, cols].copy()
    batch.columns = list(range(len(cols)))
    long = batch.stack().reset_index()
    long.columns = ["Tracer", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]
    long["Tracer Name"] = long["Tracer"].map(inv_tracer)
    unknown = long["Tracer Name"].isna().sum()
    if unknown:
        raise ValueError(f"{unknown} tokens have unknown tracer IDs — check _tracer_name_to_id mapping")
    noise = np.clip(rng.normal(0.0, noise_std, size=len(long)), -noise_std, noise_std)
    long["Fraction"] = (long["Fraction"].values + noise).clip(min=0.0)
    return long[["Tracer Name", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]]


def _build_df_Vm(training_flux_df, measured_rxn_ids, cols, flux_noise_std, rng):
    """Build df_Vm with noisy measured flux values for the selected columns."""
    present = [r for r in measured_rxn_ids if r in training_flux_df.index]
    if not present:
        return None
    batch = training_flux_df.loc[present].iloc[:, cols].copy()
    batch.columns = list(range(len(cols)))
    long = batch.stack().reset_index()
    long.columns = ["Flux Name", "Sample", "Value"]
    true_vals = long["Value"].values.copy()
    bounds = flux_noise_std * np.abs(true_vals)
    safe_std = np.where(bounds > 0, bounds, 1e-12)
    noise = np.clip(rng.normal(0.0, safe_std, size=len(long)), -bounds, bounds)
    long["Value"] = true_vals + noise
    return long[["Flux Name", "Sample", "Value"]]


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    checkpoint = _p(args.checkpoint) if args.checkpoint else find_checkpoint(data_dir)
    output_dir = _p(args.output_dir) if args.output_dir else os.path.join(data_dir, "ai_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load AI model (ensemble when <data-dir>/ensemble exists, else single) ──
    print(f"Loading SigmaFlux predictor (checkpoint hint: {checkpoint})")
    sf = load_predictor(_p(args.config), checkpoint, data_dir=data_dir)
    inv_tracer = {v: k for k, v in sf._tracer_name_to_id.items()}
    print(f"  Flux outputs : {len(sf.flux_names)}")
    print(f"  Tracers      : {sf.tracer_names}")

    # ── Load atom-pattern annotated model ─────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)

    features_path = os.path.join(data_dir, "features.csv")
    mapped_met_ids = set(pd.read_csv(features_path)["metabolite_id"].dropna())

    # ── Load ground-truth fluxes and labeling from training HDF5 ─────────────
    print(f"Loading training data from: {data_dir}")
    training_labeling_df, training_flux_df, _ = load_data(data_dir)
    n_total = training_flux_df.shape[1]

    # Restrict to the AI model's split (default: the withheld validation set) so AI and
    # MFA are scored on the same held-out samples; the checkpoint's stored val_indices are
    # the source of truth. --start-ix / --n-benchmarks index into this split, not raw columns.
    # For an ensemble, the shared val split lives in member 0's checkpoint, not the
    # (possibly stale) top-level .pt that `find_checkpoint` returned.
    split_ckpt = getattr(sf, "split_source", checkpoint)
    split_cols = split_sample_indices(
        args.split, split_ckpt, n_total,
        cfg.get("training", {}).get("val_fraction"), cfg.get("seed"),
    )
    sel = split_cols[args.start_ix:args.start_ix + args.n_benchmarks]
    n_runs = len(sel)
    print(f"  split={args.split}: {len(split_cols)} samples; this job evaluates "
          f"offsets {args.start_ix}–{args.start_ix + n_runs - 1} → columns "
          f"{sel[0] if n_runs else '-'}..{sel[-1] if n_runs else '-'} of {n_total} total")
    if n_runs == 0:
        print("  No samples for this offset range — nothing to do.")
        return

    end_ix = args.start_ix + n_runs - 1
    tag = f"{args.start_ix:05d}_{end_ix:05d}"

    # ── Build MFAModel and prepare per-tracer experiments ────────────────────
    first_gt = training_flux_df.iloc[:, int(sel[0])]
    mfa_model = MFAModel(cobra_model, "AI_benchmark", initial_flux_distribution=first_gt)
    reaction_ids = [r.id for r in mfa_model.reactions]

    targets, measured_metabolites, atom_patterns = build_targets(mfa_model)
    experiments = build_experiments(mfa_model, tracer_cfg)
    print("Preparing EMU network (per-tracer experiments; may take several minutes) ...")
    mfa_model.prepare(experiments, targets=targets, with_backward=False)
    print("  EMU network ready.")

    labeling_to, labeling_index = build_labeling_index(
        mfa_model, experiments, measured_metabolites, atom_patterns
    )
    dtype, device = mfa_model.dtype, mfa_model.device

    # ── Run AI predictions for the full chunk at once ─────────────────────────
    print(f"\nBuilding input tensors for {n_runs} samples ...")
    df_L = _build_df_L(training_labeling_df, sel, args.noise_std, rng, inv_tracer)
    df_Vm = _build_df_Vm(training_flux_df, measured_rxn_ids, sel, args.flux_noise_std, rng)

    print(f"Running SigmaFlux.predict() on {n_runs} samples ...")
    result = sf.predict(df_L, df_Vm=df_Vm)

    n_flux_out = len(sf.flux_names)
    n_rev_out  = len(sf.reversibility_names)
    result["_pos"] = result.groupby("Sample", sort=False).cumcount()
    flux_result = result[result["_pos"] < n_flux_out].drop(columns="_pos")
    rev_result  = result[result["_pos"] >= n_flux_out].drop(columns="_pos")
    result.drop(columns="_pos", inplace=True)

    flux_pivoted = (
        flux_result
        .pivot(index="Sample", columns="Flux Name", values="Value")
        .reindex(columns=reaction_ids)
    )
    rev_pivoted = (
        rev_result
        .pivot(index="Sample", columns="Flux Name", values="Value")
        .reindex(columns=reaction_ids)
    )
    print(f"  Flux predictions : {flux_pivoted.shape}")
    print(f"  Rev  predictions : {rev_pivoted.shape}")

    # ── Per-sample: EMU labeling forward pass, accumulate in memory ──────────
    pred_lab_data, pred_flux_data = [], []
    sample_indices = []
    CHECKPOINT_EVERY = 500

    for run_ix in range(n_runs):
        global_ix = int(sel[run_ix])   # real HDF5 column index of this validation sample

        pred_flux_arr = np.nan_to_num(flux_pivoted.loc[run_ix].values, nan=0.0)
        pred_rev_arr  = np.nan_to_num(rev_pivoted.loc[run_ix].values, nan=1.0)

        with torch.no_grad():
            mfa_model._fluxes.copy_(torch.tensor(pred_flux_arr, dtype=dtype, device=device))
            mfa_model._reversibility.copy_(torch.tensor(pred_rev_arr, dtype=dtype, device=device))
            pred_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()

        pred_lab_data.append(pred_labeling_vec)
        pred_flux_data.append(pred_flux_arr)
        sample_indices.append(global_ix)

        if (run_ix + 1) % 100 == 0:
            print(f"  {run_ix + 1}/{n_runs} samples processed")

        if run_ix == 0 or (run_ix + 1) % CHECKPOINT_EVERY == 0 or run_ix == n_runs - 1:
            pred_base = os.path.join(output_dir, f"pred_{tag}")
            _write_h5(pred_base, pred_lab_data, labeling_index, pred_flux_data, reaction_ids,
                      sample_indices)
            print(f"  Checkpoint saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–{end_ix}).")
    print(f"  → {output_dir}/pred_{tag}_labeling.h5")
    print(f"  → {output_dir}/pred_{tag}_flux.h5")


if __name__ == "__main__":
    main()
