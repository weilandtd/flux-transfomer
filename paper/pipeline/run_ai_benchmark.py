"""
SigmaFlux AI accuracy benchmark using real training data.

Mirrors run_mfa_benchmark.py but uses the trained AI model instead of
traditional MFA to predict fluxes. Outputs are saved in the same CSV
schema (reaction, gt_N, pred_N) so plot_test_performance.py can load
both benchmarks identically.

Each benchmark run:
  1. Draws a ground-truth flux / labeling pair from the training HDF5.
  2. Adds Gaussian noise to the simulated MID and measured fluxes.
  3. Runs SigmaFlux.predict() to obtain predicted fluxes.
  4. Uses MFAModel.forward() on those predicted fluxes to obtain
     predicted labeling (for labeling_samples_*.csv).
  5. Compares predicted vs. ground-truth fluxes.

Results are directly comparable to run_mfa_benchmark.py output.

Usage (I. orientalis):
    python -u paper/pipeline/run_ai_benchmark.py \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 10
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
from sigmaflux.ai.inference import SigmaFlux
from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


def parse_args():
    p = argparse.ArgumentParser(description="SigmaFlux AI accuracy benchmark vs training data")
    p.add_argument("--config", required=True, help="Organism YAML config")
    p.add_argument("--data-dir", required=True,
                   help="Training run directory containing model checkpoint and HDF5 data")
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained .pt model file (auto-detected from data-dir if omitted)")
    p.add_argument("--output-dir", default=None,
                   help="Results directory (default: <data-dir>/ai_benchmark)")
    p.add_argument("--n-benchmarks", type=int, default=100,
                   help="Number of ground-truth flux / prediction pairs to evaluate")
    p.add_argument("--start-ix", type=int, default=0,
                   help="Index of first sample (for SLURM array parallelism)")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=0.05,
                   help="Gaussian noise std added to measured flux observations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _find_checkpoint(data_dir):
    for fname in os.listdir(data_dir):
        if fname.endswith(".pt"):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No .pt checkpoint found in {data_dir}")


def _build_all_targets(mfa_model):
    targets = set()
    for met in mfa_model.metabolites:
        pat = met.annotation.get("atom_pattern", "")
        if pat:
            targets.add(EMU(met, list(range(1, len(pat) + 1))))
    if not targets:
        raise ValueError("No EMU targets found — check model atom_pattern annotations.")
    return targets


def _labeling_tensor_to_mid_df(mfa_model, labeling, mapped_met_ids=None):
    """Convert forward-pass labeling tensor to [id, mid, mean] DataFrame (no noise)."""
    rows = []
    for subnet_size, emus in mfa_model.emu_met_filter.items():
        for emu in emus:
            if mapped_met_ids is not None and emu.id not in mapped_met_ids:
                continue
            met_col = mfa_model._metabolite_hash[emu.id]
            fracs = labeling[:subnet_size + 1, met_col].detach().cpu().numpy()
            for mid_ix, frac in enumerate(fracs):
                rows.append({"id": emu.id, "mid": mid_ix, "mean": float(frac)})
    return pd.DataFrame(rows)


def _build_df_L(labeling_df, start_ix, n_runs, noise_std, rng, inv_tracer):
    """Convert a slice of the training labeling_df to df_L format for SigmaFlux.predict()."""
    batch = labeling_df.iloc[:, start_ix:start_ix + n_runs].copy()
    batch.columns = list(range(n_runs))
    long = batch.stack().reset_index()
    long.columns = ["Tracer", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]
    long["Tracer Name"] = long["Tracer"].map(inv_tracer)
    unknown = long["Tracer Name"].isna().sum()
    if unknown:
        raise ValueError(f"{unknown} tokens have unknown tracer IDs — check _tracer_name_to_id mapping")
    noise = np.clip(rng.normal(0.0, noise_std, size=len(long)), -noise_std, noise_std)
    long["Fraction"] = (long["Fraction"].values + noise).clip(min=0.0)
    return long[["Tracer Name", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]]


def _build_df_Vm(training_flux_df, measured_rxn_ids, start_ix, n_runs, flux_noise_std, rng):
    """Build df_Vm with noisy measured flux values for a batch of samples."""
    present = [r for r in measured_rxn_ids if r in training_flux_df.index]
    if not present:
        return None
    batch = training_flux_df.loc[present].iloc[:, start_ix:start_ix + n_runs].copy()
    batch.columns = list(range(n_runs))
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
    flux_noise_std = args.flux_noise_std

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    checkpoint = _p(args.checkpoint) if args.checkpoint else _find_checkpoint(data_dir)
    output_dir = _p(args.output_dir) if args.output_dir else os.path.join(data_dir, "ai_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    end_ix = args.start_ix + args.n_benchmarks - 1
    tag = f"samples_{args.start_ix:05d}_{end_ix:05d}"

    # ── Load AI model ─────────────────────────────────────────────────────────
    print(f"Loading SigmaFlux checkpoint: {checkpoint}")
    sf = SigmaFlux(config=_p(args.config), checkpoint=checkpoint)
    inv_tracer = {v: k for k, v in sf._tracer_name_to_id.items()}
    n_flux_out = len(sf.flux_names)
    print(f"  Flux outputs : {n_flux_out}")
    print(f"  Tracers      : {sf.tracer_names}")

    # ── Load atom-pattern annotated model ─────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)

    # ── Load features.csv → restrict MIDs to metabolites in training data ─────
    features_path = os.path.join(data_dir, "features.csv")
    feat_df = pd.read_csv(features_path)
    mapped_met_ids = set(feat_df["metabolite_id"].dropna())

    # ── Load ground-truth fluxes and labeling from training HDF5 ─────────────
    print(f"Loading training data from: {data_dir}")
    training_labeling_df, training_flux_df, _ = load_data(data_dir)
    n_total = training_flux_df.shape[1]
    n_runs = min(args.n_benchmarks, n_total - args.start_ix)
    print(f"  {n_runs} samples (indices {args.start_ix}–{args.start_ix + n_runs - 1} "
          f"of {n_total} total)")

    # ── Build MFAModel for forward-pass labeling ──────────────────────────────
    first_gt = training_flux_df.iloc[:, args.start_ix]
    mfa_model = MFAModel(cobra_model, "AI_benchmark", initial_flux_distribution=first_gt)
    reaction_ids = [r.id for r in mfa_model.reactions]

    targets = _build_all_targets(mfa_model)
    tracers = [
        Tracer(mfa_model.metabolites.get_by_id(t["tracer_id"]), t["enrichment"], t["labeling"])
        for t in tracer_cfg
    ]
    placeholder_exp = Experiment(MassIsotopomerData(), tracers, fluxes=None)
    print("Preparing EMU network (may take several minutes) ...")
    mfa_model.prepare([placeholder_exp], targets=targets, with_backward=False)
    print("  EMU network ready.")

    # ── Run AI predictions for the full chunk at once ─────────────────────────
    print(f"\nBuilding input tensors for {n_runs} samples ...")
    df_L = _build_df_L(training_labeling_df, args.start_ix, n_runs, args.noise_std, rng, inv_tracer)
    df_Vm = _build_df_Vm(training_flux_df, measured_rxn_ids, args.start_ix, n_runs, flux_noise_std, rng)

    print(f"Running SigmaFlux.predict() on {n_runs} samples ...")
    result = sf.predict(df_L, df_Vm=df_Vm)

    # Split result into flux and reversibility by per-sample position.
    # _format_output emits flux_names rows first, then reversibility_names rows.
    n_rev_out = len(sf.reversibility_names)
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
    flux_cols = {"reaction": reaction_ids}
    labeling_rows = []   # list of (id, mid, global_ix, value) DataFrames

    fluxes_path = os.path.join(output_dir, f"fluxes_{tag}.csv")
    labeling_path = os.path.join(output_dir, f"labeling_{tag}.csv")
    CHECKPOINT_EVERY = 500

    for run_ix in range(n_runs):
        global_ix = args.start_ix + run_ix

        gt_vals = training_flux_df.iloc[:, global_ix].reindex(reaction_ids).values
        pred_flux_arr = np.nan_to_num(flux_pivoted.iloc[run_ix].values, nan=0.0)
        pred_rev_arr  = np.nan_to_num(rev_pivoted.iloc[run_ix].values, nan=1.0)

        flux_cols[f"gt_{global_ix}"] = gt_vals
        flux_cols[f"pred_{global_ix}"] = pred_flux_arr

        # EMU forward labeling: set both net fluxes AND reversibility from AI
        # predictions. Using copy_() avoids reset_initial_fluxes() which would
        # randomly reinitialize reversibility and corrupt the labeling.
        dtype, device = mfa_model.dtype, mfa_model.device
        with torch.no_grad():
            mfa_model._fluxes.copy_(torch.tensor(pred_flux_arr, dtype=dtype, device=device))
            mfa_model._reversibility.copy_(torch.tensor(pred_rev_arr, dtype=dtype, device=device))
            pred_labeling = mfa_model()

        mid_df = _labeling_tensor_to_mid_df(mfa_model, pred_labeling, mapped_met_ids)
        mid_df["sample"] = global_ix
        labeling_rows.append(mid_df[["id", "mid", "sample", "mean"]])

        if (run_ix + 1) % 100 == 0:
            ss_res = float(np.sum((pred_flux_arr - gt_vals) ** 2))
            ss_tot = float(np.sum((gt_vals - gt_vals.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  {run_ix + 1}/{n_runs}  R²={r2:.4f}")

        # Periodic checkpoint to recover partial results if job is killed
        if (run_ix + 1) % CHECKPOINT_EVERY == 0 or run_ix == n_runs - 1:
            pd.DataFrame(flux_cols).to_csv(fluxes_path, index=False)
            lab = pd.concat(labeling_rows, ignore_index=True)
            lab_wide = lab.pivot_table(index=["id", "mid"], columns="sample",
                                       values="mean", aggfunc="first")
            lab_wide.columns = [f"pred_{c}" for c in lab_wide.columns]
            lab_wide.reset_index().to_csv(labeling_path, index=False)
            print(f"  Checkpoint saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–{end_ix}).")
    print(f"  Fluxes   → {fluxes_path}")
    print(f"  Labeling → {labeling_path}")


if __name__ == "__main__":
    main()
