"""
Build and sample a Maranas yeast flux problem for SigmaFlux AI data generation.

This is the Maranas-model analogue of the mammalian atlas
`sample_flux_problem.py`: it prepares a COBRA sampling problem, writes sampled
fluxes, and saves the exact model that those fluxes belong to.

Example:
    PYTHONPATH=src python paper/sample_flux_maranas.py \
        --model-file models/I_orientalis/imIsor_annotated.json \
        --output-dir training_data_maranas \
        --n-samples 10000 \
        --processes 5
"""

import argparse
import os

import cobra
from cobra import Reaction
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.json import load_json_model
from cobra.sampling import sample
from optlang.symbolics import Zero


DEFAULT_MODEL_FILE = "models/I_orientalis/imIsor_annotated.json"


def add_out_reservoir_reactions(model, bound):
    added = []
    for metabolite in list(model.metabolites):
        if not metabolite.id.endswith("_out"):
            continue

        source = Reaction(f"SRC_{metabolite.id}")
        source.bounds = (0, bound)
        source.add_metabolites({metabolite: 1})

        demand = Reaction(f"DM_{metabolite.id}")
        demand.bounds = (0, bound)
        demand.add_metabolites({metabolite: -1})

        model.add_reactions([source, demand])
        added.extend([source.id, demand.id])

    return added


def set_bounds(model, reaction_id, lower_bound, upper_bound):
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.bounds = (lower_bound, upper_bound)


def constrain_yeast_exchange(model, args):
    set_bounds(model, "EX_glc__D_e", args.glucose_lb, args.glucose_ub)
    set_bounds(model, "GLCt_c_e", args.glucose_lb, args.glucose_ub)
    set_bounds(model, "BIOMASS", args.biomass_lb, args.biomass_ub)
    set_bounds(model, "EX_o2_e", args.o2_lb, args.o2_ub)
    set_bounds(model, "EX_co2_e", args.co2_lb, args.co2_ub)
    set_bounds(model, "EX_etoh_e", args.etoh_lb, args.etoh_ub)
    set_bounds(model, "EX_ac_e", args.ac_lb, args.ac_ub)


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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_file}")
    model = load_json_model(args.model_file)
    model.solver.configuration.tolerances.feasibility = 1e-9

    added = add_out_reservoir_reactions(model, args.reservoir_bound)
    print(f"Added {len(added)} reservoir source/demand reactions for *_out metabolites")

    constrain_yeast_exchange(model, args)

    solution = model.optimize()
    if solution.status != "optimal":
        raise RuntimeError(f"Prepared model is not feasible: {solution.status}")
    print(f"Prepared model status: {solution.status}, objective={solution.objective_value}")

    expression = sum(reaction.forward_variable + reaction.reverse_variable for reaction in model.reactions)
    model.objective = expression
    model.objective_direction = "min"
    parsimonious_solution = model.optimize()
    if parsimonious_solution.status != "optimal":
        raise RuntimeError(f"Parsimony optimization failed: {parsimonious_solution.status}")
    model.add_cons_vars(
        model.problem.Constraint(expression, ub=parsimonious_solution.objective_value * args.parsimony_factor)
    )
    model.objective = Zero
    print(f"Added parsimony constraint: total flux <= {parsimonious_solution.objective_value * args.parsimony_factor:.6g}")

    if args.use_fva:
        removed_reactions, removed_metabolites = constrain_by_fva(model, args.fva_fraction)
        print(f"Removed {removed_reactions} zero-FVA reactions and {removed_metabolites} orphaned metabolites")

    prepared_model_path = os.path.join(args.output_dir, "maranas_model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, prepared_model_path)
    print(f"Saved prepared model: {prepared_model_path}")

    print(f"Sampling {args.n_samples} flux distributions...")
    samples = sample(
        model,
        n=args.n_samples,
        thinning=args.thinning,
        processes=args.processes,
        method=args.method,
        seed=args.seed,
    )

    samples_path = os.path.join(args.output_dir, "flux_samples.csv")
    samples.to_csv(samples_path)
    print(f"Saved flux samples: {samples_path} shape={samples.shape}")
    summarize_variation(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample fluxes for Maranas yeast models.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output-dir", default="training_data_maranas")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--thinning", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--method", choices=["optgp", "achr"], default="optgp")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reservoir-bound", type=float, default=1000.0)
    parser.add_argument("--glucose-lb", type=float, default=1.0)
    parser.add_argument("--glucose-ub", type=float, default=10.0)
    parser.add_argument("--biomass-lb", type=float, default=0.01)
    parser.add_argument("--biomass-ub", type=float, default=7.0)
    parser.add_argument("--o2-lb", type=float, default=0.0)
    parser.add_argument("--o2-ub", type=float, default=100.0)
    parser.add_argument("--co2-lb", type=float, default=0.0)
    parser.add_argument("--co2-ub", type=float, default=100.0)
    parser.add_argument("--etoh-lb", type=float, default=0.0)
    parser.add_argument("--etoh-ub", type=float, default=100.0)
    parser.add_argument("--ac-lb", type=float, default=0.0)
    parser.add_argument("--ac-ub", type=float, default=100.0)

    parser.add_argument("--parsimony-factor", type=float, default=1.5)
    parser.add_argument("--use-fva", action="store_true")
    parser.add_argument("--fva-fraction", type=float, default=0.1)

    main(parser.parse_args())
