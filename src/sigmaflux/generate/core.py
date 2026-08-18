from cobra import Model
from cobra.flux_analysis import flux_variability_analysis
from sigmaflux.generate.utils import read_subnet

def add_reactions_from_tasks(model, tasks, filepath, bound = 20, medium=None, prune_boundaries=True):
    """Build a model from a list of named tasks by pulling in each task's subnet reactions (JSON files under filepath), applying medium bounds, and optionally pruning flux-dead reactions."""
    
    new = Model("model")
    
    new.add_reactions(model.boundary)

    for this_tasks in tasks: 
        subnet_file_path = filepath + '/' + this_tasks + '.json'
        subnets = read_subnet(subnet_file_path)
        for subnet in subnets:
            for r_id in subnet:
                rxn = model.reactions.get_by_id(r_id).copy()
                if rxn.lower_bound != 0: 
                    rxn.lower_bound = -bound 
                if rxn.upper_bound != 0:
                    rxn.upper_bound = bound
                if rxn not in new.reactions:
                    new.add_reactions([rxn,])

    # Assing a medium to the model i.e. metabolites that can be uptaked or secreted
    # by the specific tissue
    if medium is not None:
        # Assing medium to the model close boundaries by default
        for rxn in new.boundary:
            if rxn.id in medium.keys():
                rxn.bounds = medium[rxn.id]
            else:
                rxn.bounds = (0,0)  

    # Remove reactions that cannot carry flux
    if prune_boundaries: 
        fva = flux_variability_analysis(new)
        zero_rxns = [i for i in fva[(fva['minimum'] == 0) & (fva['maximum'] == 0)].index]
        rxns_to_remove = new.reactions.get_by_any(zero_rxns)

        for r in rxns_to_remove:
            print(f"Removing {r.id} {r.reaction}")

        new.remove_reactions(rxns_to_remove)

    return new 