"""
Prepare and sample an MFA flux problem for SigmaFlux AI data generation.

Prepares a COBRA sampling problem from an annotated model, writes sampled
fluxes, and saves the exact model that those fluxes belong to.

Example:
    PYTHONPATH=src python paper/sample_flux.py --config paper/io_config.yaml
"""

import argparse
import datetime
import os

import cobra
import pandas as pd
import yaml
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis
from cobra.io.json import load_json_model
from cobra.sampling import sample

try:
    from cobra.sampling import HopsySampler
except ImportError:
    HopsySampler = None
from optlang.symbolics import Zero

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BIOMASS_COMPONENTS = ('PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c', 'CARB_c')

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


def resolve_path(path):
    """Resolve a config path relative to the repo root if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def output_dir_for(cfg, output_dir_override=None):
    if output_dir_override:
        return output_dir_override
    return f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"


def set_bounds(model, reaction_id, lower_bound, upper_bound):
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.bounds = (lower_bound, upper_bound)


def apply_media(model, media_cfg):
    """Close all EX_ reactions, then open only those listed in media_cfg."""
    for reaction in model.reactions:
        if reaction.id.startswith("EX_"):
            reaction.bounds = (0, 0)
    for rxn_id, bounds in media_cfg.items():
        set_bounds(model, rxn_id, bounds["lb"], bounds["ub"])
    print(f"Applied media: {len(media_cfg)} open exchange reactions, all others closed")


def constrain_growth(model, growth_cfg):
    set_bounds(model, "BIOMASS", growth_cfg["lb"], growth_cfg["ub"])


def constrain_by_fva(model):
    fva = flux_variability_analysis(model, fraction_of_optimum=0)
    epsilon = 1e-6
    zero_reactions = []

    for reaction_id, row in fva.iterrows():
        reaction = model.reactions.get_by_id(reaction_id)
        if row["minimum"] > -epsilon and row["maximum"] < epsilon:
            zero_reactions.append(reaction)
        else:
            reaction.bounds = (row["minimum"] - epsilon, row["maximum"] + epsilon)

    model.remove_reactions(zero_reactions)
    orphaned_metabolites = [metabolite for metabolite in model.metabolites if len(metabolite.reactions) == 0]
    model.remove_metabolites(orphaned_metabolites)

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


def _do_sample(model, s, seed):
    method = s.get("method", "hopsy")
    if method == "hopsy":
        if HopsySampler is None:
            raise ImportError("HopsySampler not available; install dev cobrapy+hopsy or set method: optgp")
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
    model = load_json_model(resolve_path(cfg["model_file"]))
    model.solver.configuration.tolerances.feasibility = 1e-9

    apply_media(model, s["media"])
    constrain_growth(model, s["growth"])

    solution = model.optimize()
    if solution.status != "optimal":
        raise RuntimeError(f"Prepared model is not feasible: {solution.status}")
    print(f"Prepared model status: {solution.status}, objective={solution.objective_value}")

    model.objective = Zero

    # Remove structurally blocked reactions (zero flux capacity) before saving the model
    blocked = find_blocked_reactions(model)
    model.remove_reactions(blocked, remove_orphans=True)
    print(f"Removed {len(blocked)} blocked reactions")

    # Tighten bounds via FVA and remove fixed reactions (degenerate polytope dimensions)
    fva_result = flux_variability_analysis(model, fraction_of_optimum=0.0)
    fixed = []
    for rxn_id, row in fva_result.iterrows():
        if abs(row["maximum"] - row["minimum"]) < 1e-9:
            fixed.append(model.reactions.get_by_id(rxn_id))
        else:
            model.reactions.get_by_id(rxn_id).bounds = (row["minimum"], row["maximum"])
    model.remove_reactions(fixed, remove_orphans=True)
    print(f"Removed {len(fixed)} fixed reactions; tightened remaining bounds")

    prepared_model_path = os.path.join(output_dir, "model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, prepared_model_path)
    print(f"Saved prepared model: {prepared_model_path}")

    if s.get("ko", False) and s["growth"]["lb"] == 0:
        print("WARNING: ko is enabled but growth lb is 0 — knocked-out models may be trivially feasible at v=0, producing uninformative zero-flux samples")

    print(f"Sampling {s['n_samples']} flux distributions (WT)...")
    all_samples = [_do_sample(model, s, cfg["seed"])]

    if s.get("ko", False):
        nonessential = screen_nonessential(model)
        total = (1 + len(nonessential)) * s["n_samples"]
        print(f"Total samples to generate: {total} (WT + {len(nonessential)} KOs × {s['n_samples']})")
        for i, rxn_id in enumerate(nonessential):
            print(f"  KO [{i + 1}/{len(nonessential)}]: {rxn_id}  ({s['n_samples']} samples)")
            with model:
                model.reactions.get_by_id(rxn_id).bounds = (0, 0)
                all_samples.append(_do_sample(model, s, cfg["seed"] + i + 1))
        samples = pd.concat(all_samples, ignore_index=True)
        print(f"Combined samples: {len(samples)} total ({len(nonessential)} KOs + WT)")
    else:
        samples = all_samples[0]

    samples_path = os.path.join(output_dir, "flux_samples.csv")
    samples.to_csv(samples_path)
    print(f"Saved flux samples: {samples_path} shape={samples.shape}")
    summarize_variation(samples)
    if fva_result is not None:
        check_sampling_coverage(samples, fva_result, key_reactions=["EX_glc__D_e"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample fluxes from an annotated MFA model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    parser.add_argument("--output-dir", default=None, help="Directory for generated outputs (overrides date-based default)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.output_dir)
