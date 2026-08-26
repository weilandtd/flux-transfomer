"""
SigmaFlux AI accuracy benchmark against the training data.

Mirrors ``sigmaflux-benchmark-mfa`` but predicts with the trained model instead of
fitting traditional MFA, and writes the same HDF5 schema so the two are loaded and
scored identically downstream.

Each benchmark run:
  1. Draws a ground-truth flux / labeling pair from the training HDF5.
  2. Adds Gaussian noise to the simulated MID and the measured fluxes.
  3. Runs ``<member>.predict()`` to obtain predicted fluxes and reversibilities.
  4. Runs the EMU forward pass on the prediction to obtain its labeling vector,
     in the same per-tracer layout as the training HDF5.
  5. Writes predicted labeling + flux to HDF5.

When the checkpoint is an ensemble, one job predicts with every member and writes
each member's own *raw* prediction into the same ``pred_{tag}_{labeling,flux}.h5``
pair MFA and a single checkpoint also write — columns become a ``(sample, member)``
MultiIndex instead of one column per sample. No ensemble-median combination and no
null-space re-projection happens here: each member's own output already satisfies
``S v = 0`` by construction, and only a median across members breaks that. Combining
members (and re-projecting if a plain median is wanted) is left to the evaluation
side (``plot_test_performance.load_benchmark``), which medians per (sample, reaction)
across whatever members a file holds — so a different aggregation can be tried
without rerunning inference.

Usage (I. orientalis):
    sigmaflux-benchmark \\
        --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260614 \\
        --n-benchmarks 2000
"""

import argparse
import json
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
    metabolite_subset,
    noisy_measurements,
    pin_threads,
    select_samples,
    write_predictions,
)
from ._paths import add_root_argument, load_config, resolve_path

#: Rows (samples × members) between HDF5 checkpoints. Each checkpoint rewrites every
#: row accumulated so far, so this trades write overhead against how much of an
#: ensemble a killed job loses; 2000 keeps a checkpoint to roughly once per member
#: at the usual 2000-sample benchmark size without rewriting tens of thousands of
#: rows dozens of times over.
CHECKPOINT_EVERY = 2000


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
    p.add_argument("--subsample", type=int, default=None,
                   help="Draw this many samples out of the full split before applying "
                        "--start-ix/--n-benchmarks (seeded at 0, independent of --seed, "
                        "so a sensitivity study's conditions all score the identical "
                        "subset). Default: use the whole split, as today. Ignored when "
                        "--samples is given.")
    p.add_argument("--samples", default=None,
                   help="JSON file holding a list of explicit absolute HDF5 sample "
                        "columns to use as the pool (e.g. a sensitivity study's "
                        "solved-sample pool), bypassing --subsample entirely so every "
                        "condition and method shares exactly this pool. Default: derive "
                        "the pool from --split/--subsample, as today.")
    p.add_argument("--keep-met-frac", type=float, default=None,
                   help="Restrict the mapped metabolites to this fraction (nested "
                        "subset — see sigmaflux.pipeline._benchmark.metabolite_subset), "
                        "simulating a measurement platform that resolves only some of "
                        "them. Dropped metabolites are omitted from df_L entirely "
                        "(predict() then zero-fills their tokens). Default: use every "
                        "mapped metabolite, as today.")
    p.add_argument("--keep-met-seed", type=int, default=0,
                   help="Seed for --keep-met-frac's metabolite draw (default: 0, "
                        "matching metabolite_subset's own default). Vary this to draw "
                        "an independent replicate of the same fraction; must match the "
                        "MFA arm's --keep-met-seed for the two to see the same metabolites.")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="Gaussian noise std added to simulated MID fractions")
    p.add_argument("--flux-noise-std", type=float, default=0.05,
                   help="Gaussian noise std added to measured flux observations")
    p.add_argument("--seed", type=int, default=42)
    add_root_argument(p)
    return p


def build_df_L(labeling_df, cols, noise_std, seed, inv_tracer, keep_met_ids=None):
    """Convert the selected training-labeling columns to df_L for ``predict()``.

    Uses the shared per-sample RNG, so the MFA benchmark scoring the same sample sees
    the identical noise realisation.

    Args:
        labeling_df: training-HDF5 labeling table, columns are samples.
        cols: HDF5 column indices of the selected samples.
        noise_std: Gaussian noise std added to the MID fractions.
        seed: base seed for the per-sample RNG stream.
        inv_tracer: maps internal tracer token to the predictor's tracer name.
        keep_met_ids: optional metabolite ids to restrict to (dropping the rest
            before noising, matching run_mfa_benchmark's ordering so the two arms'
            RNG streams stay in lockstep).
    """
    noisy_cols = {}
    for run_ix, col in enumerate(cols):
        gt = labeling_df.iloc[:, int(col)]
        gt.index.names = ["Tracer", "Metabolite", "Isotopomer"]
        if keep_met_ids is not None:
            gt = gt[gt.index.get_level_values("Metabolite").isin(keep_met_ids)]
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
                labeling_df, seed, keep_met_ids=None):
    """Build df_Vm with noisy measured flux values for the selected columns.

    Draws (and discards) the labeling noise first to keep the shared RNG's draw
    order in sync with the MFA benchmark, so both land on the identical flux noise.

    Args:
        training_flux_df: training-HDF5 flux table, columns are samples.
        measured_rxn_ids: reaction ids treated as measured fluxes.
        cols: HDF5 column indices of the selected samples.
        noise_std: Gaussian noise std for the (discarded) labeling draw.
        flux_noise_std: relative Gaussian noise std added to the flux values.
        labeling_df: training-HDF5 labeling table, for the shared RNG draw order.
        seed: base seed for the per-sample RNG stream.
        keep_met_ids: optional metabolite ids the discarded labeling draw is
            restricted to, matching ``build_df_L`` so the discarded draw's size
            (and hence the flux noise that follows it) stays in sync.
    """
    present = [r for r in measured_rxn_ids if r in training_flux_df.index]
    if not present:
        return None
    rows = []
    for run_ix, col in enumerate(cols):
        gt_lab = labeling_df.iloc[:, int(col)]
        gt_lab.index.names = ["Tracer", "Metabolite", "Isotopomer"]
        if keep_met_ids is not None:
            gt_lab = gt_lab[gt_lab.index.get_level_values("Metabolite").isin(keep_met_ids)]
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

    # An ensemble scores each member on its own — no combined predict() call, no
    # projection (see module docstring). A single checkpoint predicts once, unchanged.
    ensemble_members = getattr(sf, "members", None)
    members = list(enumerate(ensemble_members)) if ensemble_members is not None else [(None, sf)]
    # The shared val split lives in member 0's checkpoint for an ensemble (see
    # EnsembleSigmaFlux._adopt), not the (possibly stale) top-level .pt that
    # find_checkpoint returned — sf.split_source already resolved that.
    split_source = getattr(sf, "split_source", checkpoint)
    print(f"  Members      : {len(members)}")

    cobra_model, mapped_met_ids, training_labeling_df, training_flux_df, _ = \
        load_benchmark_inputs(data_dir, cfg, root)
    keep_met_ids = None
    if args.keep_met_frac is not None and mapped_met_ids:
        keep_met_ids = metabolite_subset(mapped_met_ids, args.keep_met_frac,
                                         seed=args.keep_met_seed)
        print(f"  Restricted to {len(keep_met_ids)} metabolites "
              f"({args.keep_met_frac:.0%} of the mapped set, seed={args.keep_met_seed})")

    sample_ids = None
    if args.samples:
        with open(_p(args.samples)) as fh:
            sample_ids = json.load(fh)

    sel, tag = select_samples(args.split, split_source, training_flux_df.shape[1],
                              cfg, args.start_ix, args.n_benchmarks,
                              subsample=args.subsample, sample_ids=sample_ids)
    n_runs = len(sel)
    if n_runs == 0:
        print("  No samples for this offset range — nothing to do.")
        return 0

    # Built once and reused for every member: it depends only on the model/tracers,
    # not on which member is predicting, and EMU network prep is the expensive part.
    mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, tracer_cfg, training_flux_df.iloc[:, int(sel[0])],
        "AI_benchmark", with_backward=False,
    )

    print(f"\nBuilding input tensors for {n_runs} samples ...")
    df_L = build_df_L(training_labeling_df, sel, args.noise_std, args.seed, inv_tracer,
                      keep_met_ids=keep_met_ids)
    df_Vm = build_df_Vm(training_flux_df, measured_rxn_ids, sel, args.noise_std,
                        args.flux_noise_std, training_labeling_df, args.seed,
                        keep_met_ids=keep_met_ids)

    # Every member's rows accumulate into these same three lists, so one checkpoint
    # write and one final file cover the whole ensemble — the row key is a plain
    # sample index for a single checkpoint, or a (sample, member) pair for an
    # ensemble, which write_predictions turns into a (sample, member) column
    # MultiIndex instead of one column per sample.
    pred_base = os.path.join(output_dir, f"pred_{tag}")
    pred_lab_data, pred_flux_data, row_keys = [], [], []
    total_rows = n_runs * len(members)
    done = 0

    for midx, member in members:
        who = f"member {midx}" if midx is not None else "single model"
        print(f"\nRunning {who}.predict() on {n_runs} samples ...")
        result = member.predict(df_L, df_Vm=df_Vm)

        # predict() stacks fluxes then reversibilities per sample; split by position.
        result["_pos"] = result.groupby("Sample", sort=False).cumcount()
        flux_result = result[result["_pos"] < len(sf.flux_names)].drop(columns="_pos")
        rev_result = result[result["_pos"] >= len(sf.flux_names)].drop(columns="_pos")

        flux_pivoted = (flux_result.pivot(index="Sample", columns="Flux Name", values="Value")
                                   .reindex(columns=reaction_ids))
        rev_pivoted = (rev_result.pivot(index="Sample", columns="Flux Name", values="Value")
                                 .reindex(columns=reaction_ids))
        print(f"  Flux predictions : {flux_pivoted.shape}")
        print(f"  Rev  predictions : {rev_pivoted.shape}")

        # ── Per-sample: EMU labeling forward pass, accumulate in memory ──────
        for run_ix in range(n_runs):
            pred_flux_arr = np.nan_to_num(flux_pivoted.loc[run_ix].values, nan=0.0)
            pred_rev_arr = np.nan_to_num(rev_pivoted.loc[run_ix].values, nan=1.0)

            pred_lab_data.append(
                forward_labeling(mfa_model, pred_flux_arr, pred_rev_arr, labeling_to)
            )
            pred_flux_data.append(pred_flux_arr)
            sample_ix = int(sel[run_ix])   # real HDF5 column index
            row_keys.append((sample_ix, midx) if midx is not None else sample_ix)
            done += 1

            if done % 100 == 0:
                print(f"  {done}/{total_rows} rows processed")

            if done == 1 or done % CHECKPOINT_EVERY == 0 or done == total_rows:
                write_predictions(pred_base, pred_lab_data, labeling_index, pred_flux_data,
                                  reaction_ids, row_keys)
                print(f"  Checkpoint saved ({done}/{total_rows})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–"
          f"{args.start_ix + n_runs - 1}, {len(members)} member(s)).")
    print(f"  → {pred_base}_labeling.h5")
    print(f"  → {pred_base}_flux.h5")
    return 0


def cli(argv=None):
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
