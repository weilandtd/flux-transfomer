"""
SSR ("local optima") landscape for the MFA benchmark.

For one solved and one unsolved exemplar sample per organism, scores every
ground-truth flux vector in the full training population as a CANDIDATE SOLUTION
against that exemplar's fixed noisy measurements -- a forward-simulate-and-score
pass, no optimization involved. This tests directly whether unsolved samples fail
because the chi-square loss surface has multiple disconnected low-SSR basins (a
restart can land in a false optimum and never find the true one) or a single basin
a restart should reliably find.

Reuses the exact primitives run_mfa_benchmark.py fits with, minus the optimizer:
``forward_labeling`` sets a candidate's flux/reversibility state and simulates its
labeling in one forward pass; ``_chi2_split`` scores the model's current state
against whatever target currently sits in ``mfa_model.experiments``. Per candidate
this costs 3 forward passes total -- one ``forward_labeling`` call (shared) plus
one ``_chi2_split`` call per exemplar target -- scoring BOTH exemplars in a single
pass over the population rather than sweeping it twice. There is no batching
anywhere in MFAModel (its flux/reversibility state is a single 1-D tensor, and the
forward pass solves a per-subnet linear system that depends on it), so this is an
unavoidable, CPU-only, embarrassingly-parallel-across-candidates Python loop.

Heavy -- run via paper/submit/submit_mfa_local_optima_landscape.sh (a SLURM array
sharding the candidate population across tasks), not interactively, for the full
~10,000-candidate population. Each task computes one contiguous chunk of
candidates against BOTH exemplars.

Exemplars are chosen deterministically from --seed alone (uniformly random from
the solved/unsolved sample lists in mfa_benchmark/pred_*_convergence.csv), so every
array task recomputes the identical pair without any shared state. Each exemplar's
exact noisy measurement set is reconstructed with the same (seed, sample_ix) keys
run_mfa_benchmark.py used originally (see noisy_measurements/benchmark_rng), so a
--seed here must match the seed the benchmark actually ran with (default 42, the
benchmark's own default).

Output, under --out-dir (default <run-dir>/mfa_local_optima/):
  landscape_<start>_<end>.csv   one row per candidate in this task's chunk --
                                candidate_sample, ssr_vs_solved_exemplar,
                                ssr_vs_unsolved_exemplar
  landscape_exemplars.json      written once, by the task starting at offset 0 --
                                {solved_sample, unsolved_sample, chi2_bound, chi2_dof,
                                 ssr_solved_exemplar_self, ssr_unsolved_exemplar_self}

paper/scripts/plot_mfa_local_optima_landscape.py consumes both, plus
mfa_failure_tsne_<key>.csv (the full-population t-SNE from mfa_failure_modes.py),
to plot the SSR field on the shared embedding.

Usage:
    python paper/scripts/mfa_local_optima_landscape.py \\
        --run-dir paper/.results/m_musculus_t_cell_20260822 \\
        --config paper/configs/m_musculus_t_cell.yaml \\
        --start-ix 0 --n-candidates 500
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from sigmaflux.pipeline._benchmark import (
    build_mfa_context, forward_labeling, labeling_chi2_scale, load_benchmark_inputs,
    noisy_measurements, pin_threads)
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.pipeline.run_mfa_benchmark import (
    _block_counts, _chi2_split, build_noisy_experiments, chi2_acceptance,
    fitted_parameter_count, make_measured_flux_data, measured_flux_ids)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Training run directory")
    p.add_argument("--config", required=True, help="Organism YAML config")
    add_root_argument(p)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <run-dir>/mfa_local_optima)")
    p.add_argument("--start-ix", type=int, default=0,
                   help="First candidate column index -- this task's chunk offset "
                        "(default: 0)")
    p.add_argument("--n-candidates", type=int, default=100,
                   help="Number of candidates this task scores (default: 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for exemplar selection and noisy-measurement "
                        "reconstruction -- must match the seed the original MFA "
                        "benchmark ran with (default: 42)")
    p.add_argument("--noise-std", type=float, default=0.01,
                   help="MID (labeling) noise std, relative (default: 0.01)")
    p.add_argument("--flux-noise-std", type=float, default=None,
                   help="Measured-flux noise std, relative (default: from the "
                        "organism config, else 0.05)")
    p.add_argument("--chi2-alpha", type=float, default=0.05,
                   help="Significance level of the goodness-of-fit test (default: 0.05)")
    return p.parse_args()


def pick_exemplars(bench_dir, seed):
    """One solved + one unsolved sample index, uniformly random.

    Deterministic from ``seed`` alone -- every array task recomputes the identical
    pair without any shared state.
    """
    records = sorted(bench_dir.glob("pred_*_convergence.csv"))
    if not records:
        raise SystemExit(f"No convergence records in {bench_dir} -- "
                         f"run the MFA benchmark for this organism first")
    conv = pd.concat([pd.read_csv(f) for f in records], ignore_index=True)
    conv = conv.drop_duplicates(subset="sample", keep="last")
    solved = conv.loc[conv["reached_optimum"], "sample"].to_numpy()
    unsolved = conv.loc[~conv["reached_optimum"], "sample"].to_numpy()
    if len(solved) == 0 or len(unsolved) == 0:
        raise SystemExit(f"Need at least one solved and one unsolved sample in "
                         f"{bench_dir}, got {len(solved)} solved / {len(unsolved)} unsolved")
    rng = np.random.default_rng(seed)
    return int(rng.choice(solved)), int(rng.choice(unsolved))


def build_target(mfa_model, template_experiments, reaction_ids, labeling_to, labeling_index,
                 mapped_met_ids, all_flux_df, all_rev_df, measured_rxn_ids, global_ix,
                 noise_std, flux_noise_std, seed):
    """One exemplar's exact noisy measurement set, reconstructed with no fit run.

    Mirrors run_mfa_benchmark.py's per-sample target construction (Step 1-2 of its
    benchmark loop) verbatim: the same (seed, global_ix) keys the noise RNG
    identically, so this reproduces the exact noise realization that sample's
    original benchmark run used. ``template_experiments`` supplies only tracer
    identity (build_noisy_experiments never reads its measurement values) and is
    the same for every sample, so it is passed explicitly rather than read off
    ``mfa_model.experiments`` -- which this function's own caller reassigns to the
    result for each exemplar in turn, and would otherwise make this function's
    behavior depend on call order.
    """
    gt_flux = all_flux_df.iloc[:, global_ix]
    gt_rev = all_rev_df.iloc[:, global_ix]
    gt_labeling_vec = forward_labeling(
        mfa_model, gt_flux.reindex(reaction_ids).values,
        gt_rev.reindex(reaction_ids).values, labeling_to,
    )
    gt_series = pd.Series(gt_labeling_vec, index=labeling_index)
    if mapped_met_ids:
        keep = gt_series.index.get_level_values("Metabolite").isin(mapped_met_ids)
        gt_series = gt_series[keep]
    flux_ids = measured_flux_ids(measured_rxn_ids, gt_flux)
    noisy_series, noisy_flux = noisy_measurements(
        gt_series, gt_flux.reindex(flux_ids).to_numpy(float),
        noise_std, flux_noise_std, seed, global_ix,
    )
    flux_data_obs = make_measured_flux_data(
        mfa_model, flux_ids, gt_flux, flux_noise_std, noisy_flux)
    noisy_exps = build_noisy_experiments(
        experiments=template_experiments, noisy_series=noisy_series,
        noise_std=noise_std, mfa_model=mfa_model, flux_data=flux_data_obs,
    )
    return dict(global_ix=global_ix, noisy_exps=noisy_exps, noisy_series=noisy_series)


def score_candidate(mfa_model, reaction_ids, labeling_to, flux_series, rev_series, targets):
    """SSR of one candidate flux/reversibility state against every exemplar target.

    One forward_labeling call sets the candidate's state (the expensive step);
    _chi2_split is then called once per target, reusing that same state.
    """
    forward_labeling(mfa_model, flux_series.reindex(reaction_ids).values,
                     rev_series.reindex(reaction_ids).values, labeling_to)
    ssr = {}
    for tag, target in targets.items():
        mfa_model.experiments = target["noisy_exps"]
        lab_chi2, flux_chi2 = _chi2_split(mfa_model)
        ssr[tag] = lab_chi2 + flux_chi2
    return ssr


def main():
    args = parse_args()
    pin_threads()
    started = time.perf_counter()

    cfg, root = load_config(args.config, args.root)

    def _p(rel):
        return resolve_path(rel, root)

    tracer_cfg = cfg["labeling"]["tracers"]
    measured_rxn_ids = cfg.get("training", {}).get("measured_fluxes", [])
    flux_noise_std = (args.flux_noise_std if args.flux_noise_std is not None
                      else cfg.get("training", {}).get("flux_noise_std", 0.05))

    data_dir = _p(args.run_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(data_dir) / "mfa_local_optima"
    out_dir.mkdir(parents=True, exist_ok=True)

    cobra_model, mapped_met_ids, _, all_flux_df, all_rev_df = load_benchmark_inputs(
        data_dir, cfg, root)

    # with_backward=False: this script never fits anything, so the backward
    # (dA/dB) matrices fit_mfa_model needs for gradients are pure setup cost here.
    mfa_model, reaction_ids, experiments, labeling_to, labeling_index = build_mfa_context(
        cobra_model, tracer_cfg, all_flux_df.iloc[:, 0], "MFA_local_optima",
        with_backward=False,
    )
    print(f"Setup complete in {time.perf_counter() - started:.0f}s (model load + EMU network)")

    bench_dir = pathlib.Path(data_dir) / "mfa_benchmark"
    solved_ix, unsolved_ix = pick_exemplars(bench_dir, args.seed)
    print(f"Exemplars (seed={args.seed}): solved={solved_ix}  unsolved={unsolved_ix}")

    targets = {}
    for tag, global_ix in [("solved", solved_ix), ("unsolved", unsolved_ix)]:
        targets[tag] = build_target(
            mfa_model, experiments, reaction_ids, labeling_to, labeling_index, mapped_met_ids,
            all_flux_df, all_rev_df, measured_rxn_ids, global_ix,
            args.noise_std, flux_noise_std, args.seed,
        )

    # Goodness-of-fit bound: measurement layout (n_L, n_F, n_params) is a property
    # of the config, not any one sample's noise draw -- confirmed constant across
    # every benchmarked sample for a given organism -- so it is computed once, from
    # whichever exemplar's target happens to be loaded.
    mfa_model.experiments = targets["solved"]["noisy_exps"]
    n_L, n_F = _block_counts(mfa_model)
    n_params = fitted_parameter_count(mfa_model)
    scale_L = labeling_chi2_scale(targets["solved"]["noisy_series"], args.noise_std,
                                  np.random.default_rng(args.seed))
    chi2_bound, chi2_dof, _ = chi2_acceptance(n_L, n_F, n_params, scale_L, args.chi2_alpha)
    print(f"Goodness-of-fit: {n_L}+{n_F} measurements - {n_params} params = "
          f"{chi2_dof} dof -> accept SSR <= {chi2_bound:.1f}")

    # Self-check: each exemplar scored against its own target should land close to
    # that sample's own recorded ssr_true/ssr_fit in pred_*_convergence.csv --
    # confirms the noisy-target reconstruction above is exact.
    self_ssr = {}
    for tag, target in targets.items():
        ssr = score_candidate(mfa_model, reaction_ids, labeling_to,
                              all_flux_df.iloc[:, target["global_ix"]],
                              all_rev_df.iloc[:, target["global_ix"]], {tag: target})
        self_ssr[tag] = ssr[tag]
    print(f"  SSR at own exemplar (self-check): "
          f"solved={self_ssr['solved']:.2f}  unsolved={self_ssr['unsolved']:.2f}")

    if args.start_ix == 0:
        exemplars_path = out_dir / "landscape_exemplars.json"
        exemplars_path.write_text(json.dumps(dict(
            solved_sample=solved_ix, unsolved_sample=unsolved_ix,
            chi2_bound=chi2_bound, chi2_dof=chi2_dof,
            ssr_solved_exemplar_self=self_ssr["solved"],
            ssr_unsolved_exemplar_self=self_ssr["unsolved"],
        ), indent=2))
        print(f"Saved: {exemplars_path}")

    # ── Candidate sweep ──────────────────────────────────────────────────────
    n_total = all_flux_df.shape[1]
    end_ix = min(args.start_ix + args.n_candidates, n_total)
    candidates = range(args.start_ix, end_ix)
    print(f"Scoring candidates {args.start_ix}..{end_ix - 1} of {n_total} "
          f"against both exemplars")

    rows = []
    sweep_started = time.perf_counter()
    for i, c in enumerate(candidates):
        row = {"candidate_sample": c}
        try:
            ssr = score_candidate(mfa_model, reaction_ids, labeling_to,
                                  all_flux_df.iloc[:, c], all_rev_df.iloc[:, c], targets)
            row["ssr_vs_solved_exemplar"] = ssr["solved"]
            row["ssr_vs_unsolved_exemplar"] = ssr["unsolved"]
        except Exception as exc:   # a pathological candidate shouldn't lose the task
            print(f"  candidate {c}: {exc!r} -- recording NaN")
            row["ssr_vs_solved_exemplar"] = np.nan
            row["ssr_vs_unsolved_exemplar"] = np.nan
        rows.append(row)
        if (i + 1) % 100 == 0 or (i + 1) == len(candidates):
            elapsed = time.perf_counter() - sweep_started
            per_candidate = elapsed / (i + 1)
            print(f"  {i + 1}/{len(candidates)}  ({per_candidate:.3f}s/candidate, "
                  f"~{per_candidate * len(candidates):.0f}s total for this task)")

    tag = f"{args.start_ix:05d}_{end_ix - 1:05d}"
    out_path = out_dir / f"landscape_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
