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
import yaml
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.json import load_json_model
from cobra.sampling import sample
from optlang.symbolics import Zero

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def constrain_by_fva(model, fraction_of_optimum):
    fva = flux_variability_analysis(model, fraction_of_optimum=fraction_of_optimum)
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

    return len(zero_reactions), len(orphaned_metabolites)


def summarize_variation(samples):
    std = samples.std(axis=0)
    print(f"Variable reactions > 1e-8: {(std > 1e-8).sum()} / {samples.shape[1]}")
    print("Top variable reactions:")
    print(std.sort_values(ascending=False).head(20).to_string())


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

    if s.get("fva", {}).get("enabled", False):
        removed_reactions, removed_metabolites = constrain_by_fva(model, s["fva"]["fraction"])
        print(f"Removed {removed_reactions} zero-FVA reactions and {removed_metabolites} orphaned metabolites")

    prepared_model_path = os.path.join(output_dir, "model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, prepared_model_path)
    print(f"Saved prepared model: {prepared_model_path}")

    print(f"Sampling {s['n_samples']} flux distributions...")
    samples = sample(
        model,
        n=s["n_samples"],
        thinning=s["thinning"],
        processes=s["processes"],
        method=s["method"],
        seed=cfg["seed"],
    )

    samples_path = os.path.join(output_dir, "flux_samples.csv")
    samples.to_csv(samples_path)
    print(f"Saved flux samples: {samples_path} shape={samples.shape}")
    summarize_variation(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample fluxes from an annotated MFA model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    parser.add_argument("--output-dir", default=None, help="Directory for generated outputs (overrides date-based default)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.output_dir)
