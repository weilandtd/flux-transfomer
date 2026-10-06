"""
Effective parameter count of the MFA benchmark's χ² test, one organism per call.

The structural count (``fitted_parameter_count``) charges one parameter per null-space
direction and per reversible reaction, whether or not the measurements can see it. This
computes γ = Σ λ/(λ+1), the number of parameter directions the scored measurements
actually determine (``effective_parameter_count``), and records it next to the
benchmark's convergence files, where ``read_convergence`` re-judges every fit with it.

It also reports the check the count has to pass: the SSR a fit gains over the ground
truth, divided by the variance scale, which for a fit at its optimum is ≈ the number of
parameters it spends.

Output: <run-dir>/mfa_benchmark/effective_dof.json

Usage:
    python paper/scripts/mfa_effective_dof.py \\
        --run-dir paper/.results/s_cerevisiae_20260921 \\
        --config paper/configs/s_cerevisiae.yaml
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy import stats

from sigmaflux.pipeline._benchmark import build_mfa_context, load_benchmark_inputs, pin_threads
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.pipeline.run_mfa_benchmark import (
    EFFECTIVE_DOF_FILE, effective_parameter_count, fitted_parameter_count, measured_flux_ids)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Training run directory")
    p.add_argument("--config", required=True, help="Organism YAML config")
    add_root_argument(p)
    p.add_argument("--n-states", type=int, default=30,
                   help="Sampled flux states the Fisher information is averaged over")
    p.add_argument("--eps", type=float, default=1e-2,
                   help="Finite-difference step, fraction of each coordinate's std")
    p.add_argument("--noise-std", type=float, default=0.01, help="MID noise std")
    p.add_argument("--flux-noise-std", type=float, default=None,
                   help="Relative measured-flux noise std (default: from the config, "
                        "else 0.05)")
    p.add_argument("--seed", type=int, default=0, help="Seed for the state draw")
    return p.parse_args()


def main():
    args = parse_args()
    pin_threads()
    started = time.perf_counter()

    cfg, root = load_config(args.config, args.root)
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = (args.flux_noise_std if args.flux_noise_std is not None
                      else cfg.get("training", {}).get("flux_noise_std", 0.05))
    data_dir = pathlib.Path(resolve_path(args.run_dir, root))
    bench_dir = data_dir / "mfa_benchmark"

    # The benchmark's own test, as recorded: raw rows, before any re-judging.
    conv = pd.concat([pd.read_csv(f) for f in sorted(bench_dir.glob("pred_*_convergence.csv"))],
                     ignore_index=True)
    alpha = 0.05
    scale = float((conv["chi2_bound"] / stats.chi2.ppf(1 - alpha, conv["chi2_dof"])).median())

    cobra_model, mapped_met_ids, _, flux_df, rev_df = load_benchmark_inputs(
        str(data_dir), cfg, root)
    mfa_model, _, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"], flux_df.iloc[:, 0], "MFA_effective_dof",
        with_backward=False)
    scored = labeling_index.get_level_values("Metabolite").isin(mapped_met_ids)
    flux_ids = measured_flux_ids(measured_rxn_ids, flux_df.iloc[:, 0])
    n_L, n_F = int(scored.sum()), len(flux_ids)
    scale_L = (scale * (n_L + n_F) - n_F) / n_L
    print(f"Setup in {time.perf_counter() - started:.0f}s: {n_L} labeling + {n_F} flux "
          f"measurements, labeling scale {scale_L:.3f}")

    p_eff, lam = effective_parameter_count(
        mfa_model, labeling_to, scored, flux_df, rev_df, flux_ids, args.noise_std,
        flux_noise_std, scale_L, n_states=args.n_states, eps=args.eps, seed=args.seed)
    p_structural = fitted_parameter_count(mfa_model, mapped_met_ids)
    dof = n_L + n_F - p_eff

    # A fit at its optimum spends its parameters on the noise: E[SSR_true - SSR_fit] ≈ scale·p.
    best = conv.sort_values("ssr_fit").drop_duplicates("sample")
    gain = (best["ssr_true"] - best["ssr_fit"]) / scale
    record = dict(
        p_eff=p_eff, p_structural=p_structural, eigenvalues=lam.tolist(),
        n_L=n_L, n_F=n_F, n_measurements=n_L + n_F, alpha=alpha, scale=scale,
        chi2_dof=dof, chi2_bound=scale * stats.chi2.ppf(1 - alpha, dof),
        n_states=args.n_states, eps=args.eps,
        fit_gain_mean=float(gain.mean()), fit_gain_median=float(gain.median()),
    )
    path = bench_dir / EFFECTIVE_DOF_FILE
    path.write_text(json.dumps(record, indent=2))
    print(f"p_eff = {p_eff:.1f} (structural {p_structural}); "
          f"fit gain over truth / scale: mean {gain.mean():.1f}, median {gain.median():.1f}")
    print(f"χ² cutoff: dof {dof:.1f} → SSR ≤ {record['chi2_bound']:.1f} "
          f"(was ≤ {conv['chi2_bound'].median():.1f})")
    print(f"Saved: {path}  ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
