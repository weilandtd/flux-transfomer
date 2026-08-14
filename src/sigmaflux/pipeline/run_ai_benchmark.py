"""
SigmaFlux AI accuracy benchmark against the training data.

Mirrors ``sigmaflux-benchmark-mfa`` but predicts with the trained model instead of
fitting traditional MFA, and writes the same HDF5 schema so the two are loaded and
scored identically downstream.

Each benchmark run:
  1. Draws a ground-truth flux / labeling pair from the training HDF5.
  2. Adds Gaussian noise to the simulated MID and the measured fluxes.
  3. Runs ``SigmaFlux.predict()`` to obtain predicted fluxes and reversibilities.
  4. Runs the EMU forward pass on the prediction to obtain its labeling vector,
     in the same per-tracer layout as the training HDF5.
  5. Writes predicted labeling + flux to HDF5.

Usage (I. orientalis):
    sigmaflux-benchmark \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 2000
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from ..ai.data import find_checkpoint
from ..ai.inference import load_predictor
from ._benchmark import (
    build_mfa_context,
    forward_labeling,
    load_benchmark_inputs,
    noisy_measurements,
    pin_threads,
    select_samples,
    write_predictions,
)
from ._paths import add_root_argument, load_config, resolve_path

#: Samples between HDF5 checkpoints. Prediction is a single batched call, so the
#: per-sample work here is only the forward pass — write far less often than MFA.
CHECKPOINT_EVERY = 500


def build_parser():
    p = argparse.ArgumentParser(
        prog="sigmaflux-benchmark",
        description="SigmaFlux AI accuracy benchmark vs training data",
    )
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
    add_root_argument(p)
    return p


def build_df_L(labeling_df, cols, noise_std, seed, inv_tracer):
    """Convert the selected columns of the training labeling to df_L for ``predict()``.

    Noise comes from the shared per-sample generator, so the MFA benchmark scoring the
    same sample sees the identical realisation rather than an independent draw.
    """
    noisy_cols = {}
    for run_ix, col in enumerate(cols):
        gt = labeling_df.iloc[:, int(col)]
        gt.index.names = ["Tracer", "Metabolite", "Isotopomer"]
        noisy_cols[run_ix], _ = noisy_measurements(gt, np.empty(0), noise_std, 0.0,
                                                   seed, int(col))
    batch = pd.DataFrame(noisy_cols)
    long = batch.stack().reset_index()
    long.columns = ["Tracer", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]
    long["Tracer Name"] = long["Tracer"].map(inv_tracer)
    unknown = long["Tracer Name"].isna().sum()
    if unknown:
        raise ValueError(f"{unknown} tokens have unknown tracer IDs — check _tracer_name_to_id mapping")
    return long[["Tracer Name", "Metabolite ID", "Isotopomer", "Sample", "Fraction"]]


def build_df_Vm(training_flux_df, measured_rxn_ids, cols, noise_std, flux_noise_std,
                labeling_df, seed):
    """Build df_Vm with noisy measured flux values for the selected columns.

    The labeling block is drawn first and discarded here purely to advance the shared
    generator in the same order the MFA benchmark uses, so both land on the identical
    flux realisation for a given sample.
    """
    present = [r for r in measured_rxn_ids if r in training_flux_df.index]
    if not present:
        return None
    rows = []
    for run_ix, col in enumerate(cols):
        gt_lab = labeling_df.iloc[:, int(col)]
        gt_lab.index.names = ["Tracer", "Metabolite", "Isotopomer"]
        true_vals = training_flux_df.loc[present].iloc[:, int(col)].to_numpy(float)
        _, noisy = noisy_measurements(gt_lab, true_vals, noise_std, flux_noise_std,
                                      seed, int(col))
        rows.append(pd.DataFrame({"Flux Name": present, "Sample": run_ix, "Value": noisy}))
    return pd.concat(rows, ignore_index=True)


def main(args):
    pin_threads()
    cfg, root = load_config(args.config, args.root)

    def _p(rel):
        return resolve_path(rel, root)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    checkpoint = _p(args.checkpoint) if args.checkpoint else find_checkpoint(data_dir)
    output_dir = _p(args.output_dir) or os.path.join(data_dir, "ai_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load AI model (ensemble when <data-dir>/ensemble exists, else single) ──
    print(f"Loading SigmaFlux predictor (checkpoint hint: {checkpoint})")
    sf = load_predictor(args.config, checkpoint, data_dir=data_dir)
    inv_tracer = {v: k for k, v in sf._tracer_name_to_id.items()}
    print(f"  Flux outputs : {len(sf.flux_names)}")
    print(f"  Tracers      : {sf.tracer_names}")

    cobra_model, _, training_labeling_df, training_flux_df, _ = load_benchmark_inputs(data_dir)

    # For an ensemble the shared val split lives in member 0's checkpoint, not the
    # (possibly stale) top-level .pt that `find_checkpoint` returned.
    sel, tag = select_samples(args.split, getattr(sf, "split_source", checkpoint),
                              training_flux_df.shape[1], cfg, args.start_ix,
                              args.n_benchmarks)
    n_runs = len(sel)
    if n_runs == 0:
        print("  No samples for this offset range — nothing to do.")
        return 0

    mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, tracer_cfg, training_flux_df.iloc[:, int(sel[0])],
        "AI_benchmark", with_backward=False,
    )

    # ── Run AI predictions for the full chunk at once ─────────────────────────
    print(f"\nBuilding input tensors for {n_runs} samples ...")
    df_L = build_df_L(training_labeling_df, sel, args.noise_std, args.seed, inv_tracer)
    df_Vm = build_df_Vm(training_flux_df, measured_rxn_ids, sel, args.noise_std,
                        args.flux_noise_std, training_labeling_df, args.seed)

    print(f"Running SigmaFlux.predict() on {n_runs} samples ...")
    result = sf.predict(df_L, df_Vm=df_Vm)

    # predict() stacks fluxes then reversibilities per sample; split them by position.
    result["_pos"] = result.groupby("Sample", sort=False).cumcount()
    flux_result = result[result["_pos"] < len(sf.flux_names)].drop(columns="_pos")
    rev_result = result[result["_pos"] >= len(sf.flux_names)].drop(columns="_pos")

    flux_pivoted = (flux_result.pivot(index="Sample", columns="Flux Name", values="Value")
                               .reindex(columns=reaction_ids))
    rev_pivoted = (rev_result.pivot(index="Sample", columns="Flux Name", values="Value")
                             .reindex(columns=reaction_ids))
    print(f"  Flux predictions : {flux_pivoted.shape}")
    print(f"  Rev  predictions : {rev_pivoted.shape}")

    # ── Per-sample: EMU labeling forward pass, accumulate in memory ──────────
    pred_lab_data, pred_flux_data, sample_indices = [], [], []
    pred_base = os.path.join(output_dir, f"pred_{tag}")

    for run_ix in range(n_runs):
        pred_flux_arr = np.nan_to_num(flux_pivoted.loc[run_ix].values, nan=0.0)
        pred_rev_arr = np.nan_to_num(rev_pivoted.loc[run_ix].values, nan=1.0)

        pred_lab_data.append(
            forward_labeling(mfa_model, pred_flux_arr, pred_rev_arr, labeling_to)
        )
        pred_flux_data.append(pred_flux_arr)
        sample_indices.append(int(sel[run_ix]))   # real HDF5 column index

        if (run_ix + 1) % 100 == 0:
            print(f"  {run_ix + 1}/{n_runs} samples processed")

        if run_ix == 0 or (run_ix + 1) % CHECKPOINT_EVERY == 0 or run_ix == n_runs - 1:
            write_predictions(pred_base, pred_lab_data, labeling_index, pred_flux_data,
                              reaction_ids, sample_indices)
            print(f"  Checkpoint saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–"
          f"{args.start_ix + n_runs - 1}).")
    print(f"  → {pred_base}_labeling.h5")
    print(f"  → {pred_base}_flux.h5")
    return 0


def cli(argv=None):
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
