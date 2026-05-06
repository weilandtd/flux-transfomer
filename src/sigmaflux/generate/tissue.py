from cobra import Reaction, Metabolite
from sigmaflux.core.utils import insert_mapping_into_reaction
import re
import csv 
from sigmaflux.generate.read_data import TRACEBASE_SERUM_METABOLITE_MAPPING
import numpy as np 
import pandas as pd 

def restrict_exchange(model, serum_concentration_data_file, tissue_cardiac_output, metabolite_mapping=TRACEBASE_SERUM_METABOLITE_MAPPING, lowest_bound = 0.01, compartment = "a", input = True):
    """ tissue_cardiac_output in ml/min"""
    serum_exchange_df = pd.read_csv(serum_concentration_data_file)

    with open(metabolite_mapping, newline='') as csvfile:
        reader = csv.reader(csvfile)
        metabolite_naming = {rows[0]: rows[1] for rows in reader}

    # Map the metabolite names to the dataframe
    serum_exchange_df['metabolite'] = serum_exchange_df['metabolite'].apply(lambda x: metabolite_naming.get(x, ))
    serum_exchange_df["upper bound"] = (serum_exchange_df['mean'] + serum_exchange_df['std']) * tissue_cardiac_output / 1000
    if input: 
        serum_exchange_df["lower bound"] = 0 
    else: 
        lb = (serum_exchange_df['mean']  - serum_exchange_df['std']) * tissue_cardiac_output / 1000
        serum_exchange_df["lower bound"] = np.clip(lb, 0, None)
    for reaction in model.boundary: 
        if reaction.id.startswith("IN_") and reaction.id.endswith(f"[{compartment}]"):
            reaction.bounds = [0, lowest_bound]

    for metid in serum_exchange_df['metabolite'].dropna().unique(): 
        if metid in model.metabolites: 
            exchange_rxn_name = f"IN_{metid.rsplit('_',1)[0]}[{compartment}]"
            if exchange_rxn_name in model.reactions:
                exchange_rxn = model.reactions.get_by_id(exchange_rxn_name)
                exchange_rxn.bounds = [serum_exchange_df[serum_exchange_df['metabolite'] == metid]["lower bound"].item(), 
                                       serum_exchange_df[serum_exchange_df['metabolite'] == metid]["upper bound"].item()]
            else: 
                print(f"exchange reaction for {metid} not found, looked for {exchange_rxn_name}" )
        else: 
            print(f"metabolite {metid} not found")

    return model 

# add _label to the end of all reaction and metabolite names in the model
def label_model(model, label): 
    model_new = model.copy()
    for reaction in model_new.reactions: 
        reaction.id = reaction.id + label
        if "atom_mapping" in reaction.annotation: 
            atom_map = " " + reaction.annotation['atom_mapping'].lstrip()
            for met in list(reaction.metabolites): 
                if not met.id.endswith("_a") and not met.id.endswith("_v"): 
                    atom_map = atom_map.replace(f" {met.id} ", f" {met.id}{label} ")
            reaction.annotation['atom_mapping'] = atom_map

    for metabolite in model_new.metabolites: 
        if not metabolite.id.endswith("_a") and not metabolite.id.endswith("_v"): 
            metabolite.id = metabolite.id + label
    return model_new


def get_tissue_model_from_model(model, bounds = [0,20]):
    """This function takes a model and creates a new issue model from the given model .

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """

    reactions = []
    metabolites = []


    # For each reaction in the boundary make two reactions, one for arterial and one for venuous
    # remove the existing boundary reactions

    for x in model.boundary: 

        if len(x.metabolites) > 1: 
            raise ValueError("Boundary reaction has more than one metabolite")
        if x.id.startswith("EX_") and x.id.endswith("[e]") : 

            in_arterial_reaction = Reaction((x.id).replace("[e]", "[a]").replace("EX", "IN"))
            ex_arterial_reaction = Reaction(x.id.replace("[e]", "[a]"))

            in_venial_reaction = Reaction(x.id.replace("[e]", "[v]").replace("EX", "IN"))
            ex_venial_reaction = Reaction(x.id.replace("[e]", "[v]"))
        else: 
            in_arterial_reaction = Reaction("IN_" + x.id + "_a")
            ex_arterial_reaction = Reaction("EX_" + x.id + "_a")

            in_venial_reaction = Reaction("IN_" + x.id + "_v")
            ex_venial_reaction = Reaction("EX_" + x.id + "_v")

        metabolite = list(x.metabolites.keys())[0]
        if metabolite.id.endswith("_e"):
            arterial_metabolite = Metabolite(metabolite.id.replace("_e", "_a"))
            venial_metabolite = Metabolite(metabolite.id.replace("_e", "_v"))
        elif metabolite.id.endswith("_c"):
            arterial_metabolite = Metabolite(metabolite.id.replace("_c", "_a"))
            venial_metabolite = Metabolite(metabolite.id.replace("_c", "_v"))
        else: 
            arterial_metabolite = Metabolite(metabolite.id + "_a")
            venial_metabolite = Metabolite(metabolite.id + "_v")
        metabolites.extend([arterial_metabolite, venial_metabolite])

        in_arterial_reaction.add_metabolites({arterial_metabolite:1})
        in_venial_reaction.add_metabolites({venial_metabolite:-1})
        ex_arterial_reaction.add_metabolites({arterial_metabolite:-1, metabolite:1})
        ex_venial_reaction.add_metabolites({venial_metabolite:1, metabolite:-1})

        if "atom_mapping" in x.annotation: 
            annotation = x.annotation["atom_mapping"]
            pattern = re.escape(metabolite.id) + r'\s*(\([^)]+\))'
            match = re.search(pattern, annotation)
            if match:
                atom_map = match.group(1)
                in_arterial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(in_arterial_reaction.reaction, arterial_metabolite.id, atom_map)
                in_venial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(in_venial_reaction.reaction, venial_metabolite.id, atom_map)
                ex_arterial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_arterial_reaction.reaction, arterial_metabolite.id, atom_map)
                ex_venial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_venial_reaction.reaction, venial_metabolite.id, atom_map)
                ex_arterial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_arterial_reaction.annotation['atom_mapping'], metabolite.id, atom_map)
                ex_venial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction( ex_venial_reaction.annotation['atom_mapping'], metabolite.id, atom_map)       

        in_arterial_reaction.bounds = bounds 
        in_venial_reaction.bounds = bounds 
        ex_arterial_reaction.bounds = bounds 
        ex_venial_reaction.bounds =bounds 

        reactions.extend([in_arterial_reaction, ex_arterial_reaction, in_venial_reaction, ex_venial_reaction])

        model.remove_reactions(x)
        
    model.add_reactions(reactions)
    model.add_metabolites(metabolites)
    return model 


def get_organ_model_from_model(model, bounds = [0,20]):
    """This function takes a model and creates a new issue model from the given model .

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """

    reactions = []
    metabolites = []


    # For each reaction in the boundary make two reactions, one for arterial and one for venuous
    # remove the existing boundary reactions

    for x in model.boundary: 

        if len(x.metabolites) > 1: 
            raise ValueError("Boundary reaction has more than one metabolite")
        if x.id.startswith("EX_") and x.id.endswith("[e]") : 

            ex_arterial_reaction = Reaction(x.id.replace("[e]", "[a]"))

            ex_venial_reaction = Reaction(x.id.replace("[e]", "[v]"))
        else: 
            ex_arterial_reaction = Reaction("EX_" + x.id + "_a")

            ex_venial_reaction = Reaction("EX_" + x.id + "_v")

        metabolite = list(x.metabolites.keys())[0]
        if metabolite.id.endswith("_e"):
            arterial_metabolite = Metabolite(metabolite.id.replace("_e", "_a"))
            venial_metabolite = Metabolite(metabolite.id.replace("_e", "_v"))
        elif metabolite.id.endswith("_c"):
            arterial_metabolite = Metabolite(metabolite.id.replace("_c", "_a"))
            venial_metabolite = Metabolite(metabolite.id.replace("_c", "_v"))
        else: 
            arterial_metabolite = Metabolite(metabolite.id + "_a")
            venial_metabolite = Metabolite(metabolite.id + "_v")
        metabolites.extend([arterial_metabolite, venial_metabolite])

        ex_arterial_reaction.add_metabolites({arterial_metabolite:-1, metabolite:1})
        ex_venial_reaction.add_metabolites({venial_metabolite:1, metabolite:-1})

        if "atom_mapping" in x.annotation: 
            annotation = x.annotation["atom_mapping"]
            pattern = re.escape(metabolite.id) + r'\s*(\([^)]+\))'
            match = re.search(pattern, annotation)
            if match:
                atom_map = match.group(1)
                ex_arterial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_arterial_reaction.reaction, arterial_metabolite.id, atom_map)
                ex_venial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_venial_reaction.reaction, venial_metabolite.id, atom_map)
                ex_arterial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(ex_arterial_reaction.annotation['atom_mapping'], metabolite.id, atom_map)
                ex_venial_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction( ex_venial_reaction.annotation['atom_mapping'], metabolite.id, atom_map)       

                ex_arterial_reaction.annotation['atom_mapping'] = " " + ex_arterial_reaction.annotation['atom_mapping'].lstrip() 
                ex_venial_reaction.annotation['atom_mapping'] = " " + ex_venial_reaction.annotation['atom_mapping'].lstrip() 


        ex_arterial_reaction.bounds = bounds 
        ex_venial_reaction.bounds =bounds 

        reactions.extend([ex_arterial_reaction, ex_venial_reaction])

        model.remove_reactions(x)
        
    model.add_reactions(reactions)
    model.add_metabolites(metabolites)
    return model 

# take in two cobra models and merge them together 
def merge_models(model1, model2): 
    model1.add_reactions(model2.reactions)
    model1.add_metabolites(model2.metabolites)
    return model1

def add_heart(model, bounds = [0,20]): 
    heart_reactions = []
    for met in model.metabolites: 
        if met.id.endswith("_a"): 
            vein_met = model.metabolites.get_by_id(met.id.replace("_a", "_v"))
            heart_reaction = Reaction('_'.join(met.id.split("_")[:-1]) + "_heart")
            heart_reaction.add_metabolites({met: 1, vein_met: -1})
            match = None 
            for r in met.reactions: 
                if "atom_mapping" in r.annotation: 
                    annotation = r.annotation["atom_mapping"]
                    pattern = re.escape(met.id) + r'\s*(\([^)]+\))'
                    match = re.search(pattern, annotation)
                    if match: 
                        break 
            if match: 
                atom_map = match.group(1)
                heart_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(heart_reaction.reaction, met.id, atom_map)
                heart_reaction.annotation['atom_mapping'] = insert_mapping_into_reaction(heart_reaction.annotation['atom_mapping'], vein_met.id, atom_map)
            heart_reaction.bounds = bounds
            heart_reactions.extend([heart_reaction])
    model.add_reactions(heart_reactions)
    return model 

# adds an input reaction into the arterial metabolite, atommaps and sets the bound 
def add_input_reaction(model, met, suffix = "IN", bounds = [0,20]): 

    input_metabolite = Metabolite(f"{met.id}_input")
    rxn = Reaction(f"{met.id}_{suffix}")
    input_rxn = Reaction(f"IN_{met.id}_{suffix}")

    rxn.add_metabolites({input_metabolite:-1, met: 1})
    input_rxn.add_metabolites({input_metabolite: 1})

    rxn.bounds = bounds
    input_rxn.bounds = bounds

    match = None
    for r in met.reactions: 
        if "atom_mapping" in r.annotation: 
            annotation = r.annotation["atom_mapping"]
            pattern = re.escape(met.id) + r'\s*(\([^)]+\))'
            match = re.search(pattern, annotation)
            if match: 
                break
    if match:
        atom_map = match.group(1)
        rxn.annotation['atom_mapping'] = insert_mapping_into_reaction(rxn.reaction, met.id, atom_map)
        rxn.annotation['atom_mapping'] = insert_mapping_into_reaction(rxn.annotation['atom_mapping'], input_metabolite.id, atom_map)

        input_rxn.annotation["atom_mapping"] = insert_mapping_into_reaction(input_rxn.reaction, input_metabolite.id, atom_map)
        
    model.add_reactions([rxn, input_rxn])
    model.add_metabolites(input_metabolite)
    return model

def add_output_reaction(model, met, suffix = "IN", bounds = [0,20]): 
    output_metabolite = Metabolite(f"{met.id}_output")
    rxn = Reaction(f"{met.id}_{suffix}")
    output_rxn = Reaction(f"OUT_{met.id}_{suffix}")

    rxn.add_metabolites({output_metabolite:1, met: -1})
    output_rxn.add_metabolites({output_metabolite: -1})
    rxn.bounds = bounds
    output_rxn.bounds = bounds
    match = None
    for r in met.reactions: 
        if "atom_mapping" in r.annotation: 
            annotation = r.annotation["atom_mapping"]
            pattern = re.escape(met.id) + r'\s*(\([^)]+\))'
            match = re.search(pattern, annotation)
            if match: 
                break
    if match: 
        atom_map = match.group(1)
        rxn.annotation['atom_mapping'] = insert_mapping_into_reaction(rxn.reaction, met.id, atom_map)
        rxn.annotation['atom_mapping'] = insert_mapping_into_reaction(rxn.annotation['atom_mapping'] , output_metabolite.id, atom_map)
        output_rxn.annotation['atom_mapping'] = insert_mapping_into_reaction(output_rxn.reaction, output_metabolite.id, atom_map)


    model.add_reactions([rxn, output_rxn])
    model.add_metabolites(output_metabolite)
    return model