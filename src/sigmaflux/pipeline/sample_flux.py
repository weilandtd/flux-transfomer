"""
Prepare and sample an MFA flux problem for SigmaFlux AI data generation.

Prepares a COBRA sampling problem from an annotated model, writes sampled
fluxes, and saves the exact model that those fluxes belong to.

Example:
    sigmaflux-sample-flux --config paper/configs/i_orientalis.yaml
"""

import argparse
import datetime
import os
import warnings
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import yaml
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.json import load_json_model
from cobra.sampling import sample

from ..core.utils import add_dilution_inputs, media_dilution_targets

try:
    from cobra.sampling import HopsySampler
except Exception:
    # Not just ImportError: cobra's own hopsy module uses `hopsy.Proposal` as a default
    # argument in a class body, so importing it without the hopsy package raises NameError
    # rather than ImportError. Degrade to "unavailable" instead of taking the process down.
    HopsySampler = None
from optlang.symbolics import Zero

from sigmaflux.pipeline._paths import (
    add_root_argument,
    default_output_dir,
    load_config,
)
from sigmaflux.pipeline._paths import resolve_path as _resolve_path

# Mapping from YAML-friendly names to hopsy proposal class names.
_HOPSY_PROPOSALS = {
    "uniform_coordinate_hit_and_run": "UniformCoordinateHitAndRunProposal",  # default
    "uniform_hit_and_run": "UniformHitAndRunProposal",
    "gaussian_coordinate_hit_and_run": "GaussianCoordinateHitAndRunProposal",
    "gaussian_hit_and_run": "GaussianHitAndRunProposal",
    "billiard_walk": "BilliardWalkProposal",
    "billiard_adaptive_metropolis": "BilliardAdaptiveMetropolisProposal",
    "adaptive_metropolis": "AdaptiveMetropolisProposal",
    "dikin_walk": "DikinWalkProposal",
    "ball_walk": "BallWalkProposal",
}


def resolve_path(path, root=None):
    """Resolve a config path against the config's project root."""
    if os.path.isabs(path) or root is None:
        return path
    return _resolve_path(path, root)


def output_dir_for(cfg, output_dir_override=None):
    return default_output_dir(cfg, output_dir_override)


def set_bounds(model, reaction_id, lower_bound, upper_bound):
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.bounds = (lower_bound, upper_bound)


def apply_media(model, media_cfg, close_unlisted=True):
    """Constrain the model's boundary fluxes.

    With ``close_unlisted`` (the default) every ``EX_`` reaction is closed first
    and only those named in *media_cfg* are reopened. That is the right model for
    an organism grown in a defined medium: anything not in the medium is simply
    not available.

    It is the wrong model for a whole-body or multi-tissue model, where the
    ``EX_`` reactions are internal physiology — inter-tissue exchange — rather
    than a medium the experimenter composed. There, *media_cfg* lists the fluxes
    that were *measured* (respiration, protein synthesis, TG release, ...) and
    everything else must keep the bounds the model was built with. Closing them
    makes the model infeasible.

    Set ``close_unlisted: false`` alongside ``media:`` in the sampling config for
    that case.
    """
    if close_unlisted:
        for reaction in model.reactions:
            if reaction.id.startswith("EX_"):
                reaction.bounds = (0, 0)

    for rxn_id, bounds in media_cfg.items():
        set_bounds(model, rxn_id, bounds["lb"], bounds["ub"])

    missing = [r for r in media_cfg if r not in model.reactions]
    if missing:
        print(f"WARNING: {len(missing)} media reactions not in the model: {missing[:5]}")

    if close_unlisted:
        print(f"Applied media: {len(media_cfg)} open exchange reactions, all others closed")
    else:
        print(f"Applied media: constrained {len(media_cfg) - len(missing)} measured "
              f"reactions; other exchanges left at their model bounds")


def apply_reaction_bounds(model, bounds_cfg):
    """Override bounds on named non-exchange reactions, before FVA.

    `media` only reaches EX_ reactions, but a GEM's internal directions can be wrong
    for the organism being modelled.  Transport directionality is the usual case:
    microbial models can mostly write a transporter one-way, while cultured mammalian
    cells run amino acid and monocarboxylate carriers in whichever direction the
    gradient dictates, so the same reaction reverses between conditions.
    """
    if not bounds_cfg:
        return
    for reaction_id, bounds in bounds_cfg.items():
        if reaction_id not in model.reactions:
            print(f"  reaction_bounds: '{reaction_id}' not in model, skipped")
            continue
        set_bounds(model, reaction_id, bounds["lb"], bounds["ub"])
    print(f"Applied reaction_bounds overrides: {sorted(bounds_cfg)}")


def constrain_growth(model, growth_cfg):
    set_bounds(model, "BIOMASS", growth_cfg["lb"], growth_cfg["ub"])


def can_consume(reaction, metabolite):
    """Whether *reaction* can consume *metabolite* in some feasible direction.

    Which side of a reaction a metabolite sits on is a labelling convention; what
    decides whether it is consumed is the sign of the flux, and that is set by the
    bounds. A reversible reaction listing the metabolite as a product consumes it
    whenever it runs backwards.
    """
    coef = reaction.metabolites[metabolite]
    return (coef < 0 and reaction.upper_bound > 0) or (coef > 0 and reaction.lower_bound < 0)


def drop_skip_constraints(model, skip_constraints):
    """Remove the listed reactions, draining any pool left with no consumer.

    Removing e.g. BIOMASS strands the pseudo-metabolites it used to consume: they
    can still be produced but nothing takes them away, so the steady-state balance
    forces their producers to zero flux and sampling becomes infeasible.

    Two things this deliberately does *not* do, both of which it used to:

    * It does not decide "dead end" from the stoichiometric sign. Asking whether a
      metabolite is a product everywhere ignores reversibility -- a reversible
      reaction consumes its products on reverse flux -- and would strand pools
      that are perfectly well balanced. ``can_consume`` asks about the bounds
      instead. (The sign test also swept in orphans, since ``all()`` over no
      reactions is vacuously true.)
    * It does not delete the metabolite. ``remove_metabolites`` is non-destructive:
      it strips the pool out of its reactions and leaves them in place, silently
      unbalanced, which is an invisible and unbounded sink. On the T-cell model
      that quietly cost ``MDH_c`` its entire carbon output and stripped citrate
      C3-C6 from ``ACITL_c``. An explicit demand reaction keeps mass balance, keeps
      the atom mapping usable, and shows up in the flux vector where it can be
      inspected and bounded.
    """
    to_remove = [model.reactions.get_by_id(r) for r in skip_constraints if r in model.reactions]
    model.remove_reactions(to_remove, remove_orphans=True)

    stranded = [m for m in model.metabolites
                if m.reactions and not any(can_consume(r, m) for r in m.reactions)]
    demands = [model.add_boundary(m, type="demand") for m in stranded]

    print(f"Dropped {len(to_remove)} skip-constraints reaction(s): {[r.id for r in to_remove]}")
    if demands:
        print(f"  Added {len(demands)} demand reaction(s) for pools left without a "
              f"consumer: {[r.id for r in demands]}")


def constrain_by_fva(model):
    fva = flux_variability_analysis(model, fraction_of_optimum=0)
    epsilon = 1e-6
    zero_reactions = []
    fixed_count = 0

    for reaction_id, row in fva.iterrows():
        reaction = model.reactions.get_by_id(reaction_id)
        if row["minimum"] > -epsilon and row["maximum"] < epsilon:
            zero_reactions.append(reaction)
        else:
            new_lb = max(reaction.lower_bound, row["minimum"] - epsilon)
            new_ub = min(reaction.upper_bound, row["maximum"] + epsilon)
            reaction.bounds = (new_lb, new_ub)
            if abs(row["maximum"] - row["minimum"]) < 1e-9:
                fixed_count += 1

    model.remove_reactions(zero_reactions)
    orphaned_metabolites = [metabolite for metabolite in model.metabolites if len(metabolite.reactions) == 0]
    model.remove_metabolites(orphaned_metabolites)
    print(f"  Fixed (non-zero, min≈max) reactions kept: {fixed_count}")

    return len(zero_reactions), len(orphaned_metabolites), fva


def check_sampling_coverage(samples, fva, key_reactions=None):
    common = [r for r in samples.columns if r in fva.index]
    obs_min = samples[common].min(axis=0)
    obs_max = samples[common].max(axis=0)
    fva_range = fva.loc[common, "maximum"] - fva.loc[common, "minimum"]
    variable_mask = fva_range > 1e-6
    coverage = (obs_max - obs_min)[variable_mask] / fva_range[variable_mask]

    print("\n--- Bound Coverage Analysis ---")
    print(f"Reactions compared: {len(common)} (FVA variable: {variable_mask.sum()})")
    print(f"Coverage (obs range / FVA range):")
    print(f"  Median : {coverage.median():.3f}")
    print(f"  Mean   : {coverage.mean():.3f}")
    print(f"  > 90%  : {(coverage > 0.9).sum()} reactions")
    print(f"  < 10%  : {(coverage < 0.1).sum()} reactions")

    poor = coverage[coverage < 0.1].sort_values()
    if len(poor) > 0:
        print("\nPoorly covered reactions (coverage < 10%):")
        fva_lo = fva.loc[poor.index, "minimum"]
        fva_hi = fva.loc[poor.index, "maximum"]
        for rxn_id in poor.index:
            print(f"  {rxn_id:<40s}  cov={poor[rxn_id]:.3f}  FVA=[{fva_lo[rxn_id]:.4f}, {fva_hi[rxn_id]:.4f}]  obs=[{obs_min[rxn_id]:.4f}, {obs_max[rxn_id]:.4f}]")

    if key_reactions:
        print("\nKey reaction stats:")
        for rxn_id in key_reactions:
            if rxn_id not in samples.columns:
                print(f"  {rxn_id}: not in samples (pruned as zero-flux)")
                continue
            obs_lo, obs_hi = obs_min[rxn_id], obs_max[rxn_id]
            obs_std = samples[rxn_id].std()
            if rxn_id in fva.index:
                fva_lo = fva.loc[rxn_id, "minimum"]
                fva_hi = fva.loc[rxn_id, "maximum"]
                fva_rng = fva_hi - fva_lo
                cov = (obs_hi - obs_lo) / fva_rng if fva_rng > 1e-6 else float("nan")
                print(f"  {rxn_id}:")
                print(f"    FVA bounds : [{fva_lo:.4f}, {fva_hi:.4f}]  width={fva_rng:.4f}")
                print(f"    Obs range  : [{obs_lo:.4f}, {obs_hi:.4f}]  width={obs_hi - obs_lo:.4f}")
                print(f"    Obs std    : {obs_std:.4f}")
                print(f"    Coverage   : {cov:.3f}")


def diagnose_pool_coverage(samples, fva, reactions, n_bins, label="Pool"):
    """
    Per-reaction bin occupancy diagnostic. Reports observed FVA coverage,
    bin fill rate, and boundary bin counts (first/last bin) for each reaction.
    """
    present = [r for r in reactions if r in samples.columns and r in fva.index]
    if not present:
        print(f"[{label}] WARNING: no measured reactions found in samples/fva")
        return

    lo = fva.loc[present, "minimum"].values
    hi = fva.loc[present, "maximum"].values
    width = np.where(hi - lo > 1e-9, hi - lo, 1.0)

    vals = samples[present].to_numpy()
    obs_cov = np.clip((vals.max(axis=0) - vals.min(axis=0)) / width, 0, 1)

    norm = np.clip((vals - lo) / width, 0, 1)
    bin_asgn = (norm * n_bins).astype(int).clip(0, n_bins - 1)

    print(f"\n--- {label}: coverage diagnostic ({len(samples)} samples, {n_bins} bins/rxn) ---")
    print(f"  {'Reaction':<30s}  {'ObsCov':>7s}  {'Filled':>8s}  {'Empty':>6s}  {'MinCnt':>7s}  {'MaxCnt':>7s}")
    for i, rxn in enumerate(present):
        counts = np.bincount(bin_asgn[:, i], minlength=n_bins)
        n_occ = (counts > 0).sum()
        print(f"  {rxn:<30s}  {obs_cov[i]:>7.3f}  {n_occ:>4d}/{n_bins:<3d}  {n_bins - n_occ:>6d}  {counts[0]:>7d}  {counts[-1]:>7d}")


def _ipf_weights(bin_asgn, n_bins, n_iter=30):
    """
    Iterative Proportional Fitting: weights w_i such that for every reaction f
    and bin b the weighted fraction equals 1/n_bins (uniform marginal).
    """
    n = len(bin_asgn)
    w = np.ones(n, dtype=np.float64) / n
    target = 1.0 / n_bins
    for _ in range(n_iter):
        for f in range(bin_asgn.shape[1]):
            bins_f = bin_asgn[:, f]
            bin_sums = np.bincount(bins_f, weights=w, minlength=n_bins)
            scale = np.where(bin_sums[bins_f] > 1e-15, target / bin_sums[bins_f], 0.0)
            w *= scale
        w /= w.sum()
    return w


def draw_quasi_uniform(samples, fva, n_final, n_bins, reactions):
    """
    IPF-weighted draw from `samples` targeting uniform marginals for each reaction.
    Returns a DataFrame of n_final rows drawn without replacement (with replacement
    only if the pool is smaller than n_final).
    """
    present = [r for r in reactions if r in samples.columns and r in fva.index]
    if not present:
        return samples.sample(min(n_final, len(samples)), random_state=0).reset_index(drop=True)

    lo = fva.loc[present, "minimum"].values
    hi = fva.loc[present, "maximum"].values
    width = np.where(hi - lo > 1e-9, hi - lo, 1.0)

    samples_np = samples.to_numpy()
    present_cols = np.array([samples.columns.get_loc(r) for r in present])
    norm = np.clip((samples_np[:, present_cols] - lo) / width, 0, 1)
    bin_asgn = (norm * n_bins).astype(int).clip(0, n_bins - 1)

    weights = _ipf_weights(bin_asgn, n_bins)
    rng = np.random.default_rng(0)
    replace = len(samples_np) < n_final
    chosen = rng.choice(len(samples_np), n_final, replace=replace, p=weights)
    result_np = samples_np[chosen]

    result_asgn = (np.clip((result_np[:, present_cols] - lo) / width, 0, 1) * n_bins).astype(int).clip(0, n_bins - 1)
    print(f"\n--- Quasi-uniform draw ({len(present)} reactions × {n_bins} bins, n={n_final}) ---")
    print(f"  {'Reaction':<30s}  {'Pool':>10s}  {'Final':>10s}  {'Uniformity':>10s}")
    for f_idx, rxn in enumerate(present):
        pool_cov = (np.bincount(bin_asgn[:, f_idx], minlength=n_bins) > 0).sum()
        final_cnt = np.bincount(result_asgn[:, f_idx], minlength=n_bins).astype(float)
        final_cov = (final_cnt > 0).sum()
        p = final_cnt[final_cnt > 0] / final_cnt.sum()
        uniformity = (-np.sum(p * np.log2(p))) / np.log2(final_cov) if final_cov > 1 else 0.0
        print(f"  {rxn:<30s}  {pool_cov:>4d}/{n_bins}     {final_cov:>4d}/{n_bins}     {uniformity:>10.3f}")

    return pd.DataFrame(result_np, columns=samples.columns)


def summarize_variation(samples):
    stats = pd.DataFrame({
        "min": samples.min(axis=0),
        "max": samples.max(axis=0),
        "std": samples.std(axis=0),
    }).sort_values("std", ascending=False)
    print(f"Variable reactions > 1e-8: {(stats['std'] > 1e-8).sum()} / {samples.shape[1]}")
    print("Flux summary (sorted by std):")
    print(stats.to_string(float_format=lambda x: f"{x:10.4f}"))


def screen_nonessential(model):
    """Return IDs of reactions whose knockout leaves the prepared model feasible."""
    nonessential = []
    essential = []
    for rxn in model.reactions:
        with model:
            rxn.bounds = (0, 0)
            if model.optimize().status == "optimal":
                nonessential.append(rxn.id)
            else:
                essential.append(rxn.id)
    print(f"Essentiality screen: {len(nonessential)} non-essential, {len(essential)} essential / {len(model.reactions)} total reactions")
    print("Essential reactions:")
    for rxn_id in essential:
        print(f"  {rxn_id}")
    return nonessential


def _do_sample_n(model, s, n_samples, seed):
    return _do_sample(model, dict(s, n_samples=n_samples), seed)


def stratified_sample(model, s, fva, reactions, n_bins, n_subsamples, seed):
    """
    For each variable measured reaction, run one constrained MCMC chain per bin.
    Returns a DataFrame containing all strata concatenated.
    """
    all_strata = []
    for rxn_id in reactions:
        if rxn_id not in model.reactions or rxn_id not in fva.index:
            continue
        lo = fva.loc[rxn_id, "minimum"]
        hi = fva.loc[rxn_id, "maximum"]
        if hi - lo < 1e-9:
            continue
        bin_width = (hi - lo) / n_bins
        for b in range(n_bins):
            bin_lo = lo + b * bin_width
            bin_hi = lo + (b + 1) * bin_width
            with model:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.bounds = (max(rxn.lower_bound, bin_lo), min(rxn.upper_bound, bin_hi))
                if model.optimize().status != "optimal":
                    print(f"  [stratify] skip {rxn_id} bin {b + 1}/{n_bins} — infeasible")
                    continue
                all_strata.append(_do_sample_n(model, s, n_subsamples, seed + b))
        print(f"  [stratify] {rxn_id}: {n_bins} strata × {n_subsamples} samples")
    return pd.concat(all_strata, ignore_index=True)


def _do_sample(model, s, seed):
    # optgp, not hopsy: HopsySampler ships only on the cobrapy devel branch, so a released
    # cobra cannot honour a hopsy default. Every config in paper/configs sets this explicitly.
    method = s.get("method", "optgp")
    if method == "hopsy":
        if HopsySampler is None:
            warnings.warn(
                "method: hopsy was requested but HopsySampler is unavailable — SAMPLING WITH "
                "optgp INSTEAD. The samples will not be the ones the config asked for. To use "
                "hopsy, install the hopsy package and a cobrapy checkout from the devel branch "
                "(HopsySampler is not in any released cobra).",
                RuntimeWarning,
                stacklevel=2,
            )
            method = "optgp"
            return sample(model, n=s["n_samples"], thinning=s["thinning"],
                          processes=s["processes"], method=method, seed=seed)
        import hopsy as _hopsy
        proposal_key = s.get("hopsy_proposal", "uniform_coordinate_hit_and_run")
        if proposal_key not in _HOPSY_PROPOSALS:
            raise ValueError(f"Unknown hopsy_proposal '{proposal_key}'; valid: {list(_HOPSY_PROPOSALS)}")
        proposal_cls = getattr(_hopsy, _HOPSY_PROPOSALS[proposal_key])
        sol = model.optimize()
        assert sol.status == "optimal", f"Model infeasible before sampling: {sol.status}"
        sampler = HopsySampler(
            model,
            sampler=proposal_cls,
            thinning=s["thinning"],
            processes=s["processes"],
            nproj=s.get("nproj"),
            seed=seed,
            rounding=s.get("rounding", True),
        )
        return sampler.sample(s["n_samples"])
    return sample(
        model,
        n=s["n_samples"],
        thinning=s["thinning"],
        processes=s["processes"],
        method=method,
        seed=seed,
    )


def main(cfg, output_dir_override=None):
    s = cfg["sampling"]
    output_dir = output_dir_for(cfg, output_dir_override)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {cfg['model_file']}")
    model = load_json_model(resolve_path(cfg["model_file"], cfg.get("_root")))
    model.solver.configuration.tolerances.feasibility = 1e-9

    apply_media(model, s["media"], close_unlisted=s.get("close_unlisted", True))

    # Medium components the cell also produces are mixing nodes whose exchange
    # carries no label, so they cannot be diluted by the medium. Give each one an
    # explicit unlabeled input. Done here rather than in the model files because
    # the medium is a property of the experiment, not of the organism.
    diluted = add_dilution_inputs(model, media_dilution_targets(model, s["media"]))
    if diluted:
        print(f"  Added {len(diluted)} unlabeled media dilution input(s): {diluted}")
    apply_reaction_bounds(model, s.get("reaction_bounds"))
    if s.get("growth"):
        constrain_growth(model, s["growth"])

    solution = model.optimize()
    if solution.status != "optimal":
        raise RuntimeError(f"Prepared model is not feasible: {solution.status}")
    print(f"Prepared model status: {solution.status}, objective={solution.objective_value}")

    model.objective = Zero

    n_zero, n_orphan, fva_result = constrain_by_fva(model)
    print(f"Removed {n_zero} zero-flux reactions and {n_orphan} orphaned metabolites; tightened remaining bounds with ±1e-6")

    skip_constraints = cfg.get("skip_constraints", [])
    if skip_constraints:
        drop_skip_constraints(model, skip_constraints)

    measured_rxns = cfg.get("training", {}).get("measured_fluxes", [])
    sub_rxns = s.get("stratified_sampling", {}).get("reactions") or measured_rxns
    if sub_rxns:
        print("\nModel bounds seen by sampler (after FVA tightening):")
        for rxn_id in sub_rxns:
            if rxn_id in model.reactions:
                r = model.reactions.get_by_id(rxn_id)
                fva_lo = fva_result.loc[rxn_id, "minimum"] if rxn_id in fva_result.index else float("nan")
                fva_hi = fva_result.loc[rxn_id, "maximum"] if rxn_id in fva_result.index else float("nan")
                match = abs(r.lower_bound - fva_lo) < 2e-6 and abs(r.upper_bound - fva_hi) < 2e-6
                flag = "" if match else "  *** MISMATCH ***"
                print(f"  {rxn_id:<30s}  bounds=[{r.lower_bound:8.4f}, {r.upper_bound:8.4f}]  FVA=[{fva_lo:8.4f}, {fva_hi:8.4f}]{flag}")
            else:
                print(f"  {rxn_id:<30s}  REMOVED (zero-flux by FVA)")

    prepared_model_path = os.path.join(output_dir, "model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, prepared_model_path)
    print(f"Saved prepared model: {prepared_model_path}")

    if s.get("ko", False) and s.get("growth", {}).get("lb") == 0:
        print("WARNING: ko is enabled but growth lb is 0 — knocked-out models may be trivially feasible at v=0, producing uninformative zero-flux samples")

    strat_cfg = s.get("stratified_sampling", {})
    n_final = s["n_samples"]

    if strat_cfg.get("enabled", False):
        n_bins = strat_cfg.get("n_bins", 5)
        n_subsamples = strat_cfg["n_subsamples"]
        print(f"Stratified sampling: {len(sub_rxns)} reactions × {n_bins} bins × {n_subsamples} samples/stratum...")
        all_strata = [stratified_sample(model, s, fva_result, sub_rxns, n_bins, n_subsamples, cfg["seed"])]
        if s.get("ko", False):
            nonessential = screen_nonessential(model)
            for i, rxn_id in enumerate(nonessential):
                print(f"  KO [{i + 1}/{len(nonessential)}]: {rxn_id}")
                with model:
                    model.reactions.get_by_id(rxn_id).bounds = (0, 0)
                    all_strata.append(stratified_sample(model, s, fva_result, sub_rxns, n_bins, n_subsamples, cfg["seed"] + (i + 1) * 1000))
        samples = pd.concat(all_strata, ignore_index=True)
        print(f"Stratified pool: {len(samples)} samples")
        diagnose_pool_coverage(samples, fva_result, sub_rxns, n_bins, label="Stratified pool")
        samples = draw_quasi_uniform(samples, fva_result, n_final, n_bins, sub_rxns)
    else:
        print(f"Sampling {n_final} flux distributions (WT)...")
        all_samples = [_do_sample(model, s, cfg["seed"])]
        if s.get("ko", False):
            nonessential = screen_nonessential(model)
            for i, rxn_id in enumerate(nonessential):
                print(f"  KO [{i + 1}/{len(nonessential)}]: {rxn_id}  ({n_final} samples)")
                with model:
                    model.reactions.get_by_id(rxn_id).bounds = (0, 0)
                    all_samples.append(_do_sample(model, s, cfg["seed"] + i + 1))
        samples = pd.concat(all_samples, ignore_index=True) if len(all_samples) > 1 else all_samples[0]

    samples_path = os.path.join(output_dir, "flux_samples.csv")
    samples.to_csv(samples_path)
    print(f"Saved flux samples: {samples_path} shape={samples.shape}")
    summarize_variation(samples)
    if fva_result is not None:
        check_sampling_coverage(samples, fva_result, key_reactions=["EX_glc__D_e"])


def build_parser():
    parser = argparse.ArgumentParser(description="Sample fluxes from an annotated MFA model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for generated outputs (overrides date-based default)")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, _ = load_config(args.config, args.root)
    return main(cfg, args.output_dir)


if __name__ == "__main__":
    cli()
