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
  2. Simulates per-tracer 13C MID via MFAModel forward pass (+ Gaussian noise),
     restricted to metabolites present in features.csv.
  3. Adds measured-flux soft constraints (with flux_noise_std noise).
  4. Fits MFA from a pFBA initialization.
  5. Compares predicted vs. ground-truth fluxes and labeling.

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
from collections import defaultdict
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.set_num_threads(_n_threads)

from sigmaflux.ai.data import find_checkpoint, load_data, split_sample_indices
from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from generate_training_data import build_experiments, build_labeling_index, build_targets


def parse_args():
    p = argparse.ArgumentParser(description="MFA accuracy benchmark vs training data")
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
    p.add_argument("--flux-noise-std", type=float, default=0.05,
                   help="Relative noise std on measured flux observations")
    p.add_argument("--init-noise", type=float, default=0.20,
                   help="Near-GT init perturbation: v_gt·(1 + N(0, init_noise)), "
                        "applied to both net fluxes and reversibility")
    p.add_argument("--max-starts", type=int, default=10,
                   help="N_max multi-start restarts per sample. Restart from a new "
                        "feasible init until the reduced-χ² acceptance bound is met; "
                        "if none of the N_max starts is acceptable, write NaN.")
    p.add_argument("--accept-z", type=float, default=1.645,
                   help="z-score for the acceptance bound M ≤ 1 + z·sqrt(2/h_eff) "
                        "(1.645 = one-sided 95%%).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _make_measured_flux_data(mfa_model, measured_rxn_ids, gt_flux_series, flux_noise_std, rng):
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


def _make_init_fluxes(all_flux_df, all_rev_df, gt_flux, gt_rev, sample_ix,
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


def _write_h5(base, labeling_data, labeling_index, flux_data, reaction_ids, sample_indices):
    (pd.DataFrame(labeling_data, index=sample_indices, columns=labeling_index)
       .T.to_hdf(f"{base}_labeling.h5", key="labeling", mode="w"))
    (pd.DataFrame(flux_data, index=sample_indices, columns=reaction_ids)
       .T.to_hdf(f"{base}_flux.h5", key="fluxes", mode="w"))


def _build_noisy_experiments(experiments, gt_series, noise_std, rng, mfa_model,
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


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])

    rng = np.random.default_rng(args.seed + args.start_ix)
    torch.manual_seed(args.seed)

    data_dir = _p(args.data_dir)
    output_dir = _p(args.output_dir) if args.output_dir else os.path.join(data_dir, "mfa_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load atom-pattern annotated model ─────────────────────────────────────
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)
    print(f"  {len(cobra_model.reactions)} reactions, {len(cobra_model.metabolites)} metabolites")

    features_path = os.path.join(data_dir, "features.csv")
    mapped_met_ids = set(pd.read_csv(features_path)["metabolite_id"].dropna())
    print(f"  {len(mapped_met_ids)} metabolite IDs from features.csv")

    # ── Load ground-truth fluxes from training HDF5 ───────────────────────────
    print(f"Loading training data from: {data_dir}")
    _, all_flux_df, all_rev_df = load_data(data_dir)
    n_total = all_flux_df.shape[1]

    # Restrict the benchmark to the AI model's split (default: the withheld validation
    # set) so MFA and AI are scored on the same held-out samples. --start-ix / --n-benchmarks
    # now index into this split (sorted real column indices), not the raw HDF5 columns.
    checkpoint = _p(args.checkpoint) if args.checkpoint else None
    if checkpoint is None and args.split != "all":
        try:
            checkpoint = find_checkpoint(data_dir)
        except FileNotFoundError:
            checkpoint = None  # fall back to recomputing the split from config
    split_cols = split_sample_indices(
        args.split, checkpoint, n_total,
        cfg.get("training", {}).get("val_fraction"), cfg.get("seed"),
    )
    sel = split_cols[args.start_ix:args.start_ix + args.n_benchmarks]
    n_runs = len(sel)
    print(f"  split={args.split}: {len(split_cols)} samples; this task evaluates "
          f"offsets {args.start_ix}–{args.start_ix + n_runs - 1} → columns "
          f"{sel[0] if n_runs else '-'}..{sel[-1] if n_runs else '-'} of {n_total} total")
    if n_runs == 0:
        print("  No samples for this offset range — nothing to do.")
        return

    end_ix = args.start_ix + n_runs - 1
    tag = f"{args.start_ix:05d}_{end_ix:05d}"
    flux_df = all_flux_df.iloc[:, sel]
    rev_df  = all_rev_df.iloc[:,  sel]

    if measured_rxn_ids:
        present = [r for r in measured_rxn_ids if r in flux_df.index]
        print(f"  {len(present)}/{len(measured_rxn_ids)} measured flux reactions found")

    # ── Build MFAModel and prepare per-tracer experiments ────────────────────
    first_gt = flux_df.iloc[:, 0]
    mfa_model = MFAModel(cobra_model, "MFA_benchmark", initial_flux_distribution=first_gt)
    reaction_ids = [r.id for r in mfa_model.reactions]

    targets, measured_metabolites, atom_patterns = build_targets(mfa_model)
    experiments = build_experiments(mfa_model, tracer_cfg)
    print("Preparing EMU network (per-tracer experiments; may take several minutes) ...")
    mfa_model.prepare(experiments, targets=targets, with_backward=True)
    print("  EMU network ready.")

    labeling_to, labeling_index = build_labeling_index(
        mfa_model, experiments, measured_metabolites, atom_patterns
    )
    dtype, device = mfa_model.dtype, mfa_model.device

    # ── Benchmark loop ────────────────────────────────────────────────────────
    pred_lab_data, pred_flux_data = [], []
    sample_indices = []

    for run_ix in range(n_runs):
        global_ix = int(sel[run_ix])   # real HDF5 column index of this validation sample
        gt_flux = flux_df.iloc[:, run_ix]
        gt_rev  = rev_df.iloc[:,  run_ix]
        print(f"\n── Run {run_ix + 1}/{n_runs}  (sample {global_ix}) ──────────────────")

        # ── Step 1: GT labeling via per-tracer forward pass ──────────────────
        with torch.no_grad():
            mfa_model._fluxes.copy_(
                torch.tensor(gt_flux.reindex(reaction_ids).values, dtype=dtype, device=device)
            )
            mfa_model._reversibility.copy_(
                torch.tensor(gt_rev.reindex(reaction_ids).values, dtype=dtype, device=device)
            )
            gt_labeling_vec = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()

        # ── Step 2: build noisy per-tracer MID + measured flux constraints ────
        gt_series = pd.Series(gt_labeling_vec, index=labeling_index)
        flux_data_obs = _make_measured_flux_data(
            mfa_model, measured_rxn_ids, gt_flux, args.flux_noise_std, rng
        )
        noisy_exps = _build_noisy_experiments(
            experiments, gt_series, args.noise_std, rng, mfa_model, mapped_met_ids, flux_data_obs
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
        for start in range(args.max_starts):
            v_init, rev_init, _ = _make_init_fluxes(
                all_flux_df, all_rev_df, gt_flux, gt_rev, global_ix,
                args.init_noise, reaction_ids, rng
            )
            mfa_model.reset_initial_fluxes(pd.Series(v_init, index=reaction_ids))
            # reset_initial_fluxes() re-seeds reversibility randomly; use the feasible
            # interpolated reversibility instead.
            with torch.no_grad():
                mfa_model._reversibility.copy_(torch.tensor(rev_init, dtype=dtype, device=device))
            res = fit_mfa_model(mfa_model, max_iter=args.max_iter, tol=args.tol)
            M = 0.5 * (res["ssr_labeling"] + res["ssr_fluxes"])      # = res["loss"] / 2
            M_accept = acceptance_threshold(res["n_labeling"], res["n_fluxes"], args.accept_z)
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

        pred_base = os.path.join(output_dir, f"pred_{tag}")
        _write_h5(pred_base, pred_lab_data, labeling_index, pred_flux_data, reaction_ids,
                  sample_indices)
        print(f"  Saved ({run_ix + 1}/{n_runs})")

    print(f"\nBenchmark complete ({n_runs} runs, samples {args.start_ix}–{end_ix}).")
    print(f"  → {output_dir}/pred_{tag}_labeling.h5")
    print(f"  → {output_dir}/pred_{tag}_flux.h5")


if __name__ == "__main__":
    main()
