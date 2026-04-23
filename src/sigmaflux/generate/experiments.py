"""
Functions for building MFA experiments from tracer data.
"""

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.generate.read_data import get_data_from_dataframe


def build_tissue_labeling_data(mfa_model, 
                                tracer_list_for_exp, 
                                tissue_tracer_data, 
                                tissues,
                                metabolite_mapping,
                                column_mapping,
                                label_tissue=True):
    """
    Build combined labeling data from multiple tissues for a single tracer experiment.
    
    Parameters:
    -----------
    mfa_model : MFAModel
        The MFA model object
    tracer_list_for_exp : list
        List of Tracer objects for this experiment (single or combined tracers)
    tissue_tracer_data : dict
        Dictionary mapping tissue names to their tracer dataframes
    tissues : list
        List of tissue names to process
    metabolite_mapping : str
        Path to metabolite name mapping file
    column_mapping : dict
        Dictionary mapping column names
    label_tissue : bool, optional
        Whether to append tissue name to metabolite IDs (default: True)
    
    Returns:
    --------
    list : List of MassIsotopomerData objects, one per tissue with data
           Returns empty list if no data found
    
    Examples:
    ---------
    >>> data = build_tissue_labeling_data(
    ...     mfa_model, tracer_list, tissue_data, 
    ...     ['liver', 'kidney'], metabolite_mapping, column_mapping
    ... )
    """
    data = []
    
    for tissue in tissues:
        # For combined tracers, we need to get data for each individual tracer metabolite
        # but they all contribute to the same experiment
        tissue_data_list = []
        
        for tracer in tracer_list_for_exp:
            tissue_df = get_data_from_dataframe(
                tissue_tracer_data[tissue], 
                tissue, 
                tracer.metabolite.id, 
                tracer.labeling,
                column_mapping=column_mapping,
                metabolite_mapping=metabolite_mapping,
                label_tissue=label_tissue
            )
            
            # Only add if we have data (non-empty dataframe)
            if not tissue_df.empty:
                this_tissue_data = MassIsotopomerData(mfa_model, tissue_df)
                tissue_data_list.append(this_tissue_data)
        
        # Combine data from all tracers in this tissue (for combined tracers)
        if len(tissue_data_list) > 0:
            combined_tissue_data = sum(tissue_data_list) if len(tissue_data_list) > 1 else tissue_data_list[0]
            data.append(combined_tissue_data)
    
    return data


def build_experiments_from_tracers(mfa_model,
                                    tracers_dict,
                                    tissue_tracer_data,
                                    tissues,
                                    flux_data,
                                    metabolite_mapping,
                                    column_mapping,
                                    label_tissue=True):
    """
    Build a list of Experiment objects from tracer data across multiple tissues.
    
    Parameters:
    -----------
    mfa_model : MFAModel
        The MFA model object
    tracers_dict : dict
        Dictionary mapping tracer_string -> list of Tracer objects
    tissue_tracer_data : dict
        Dictionary mapping tissue names to their tracer dataframes
    tissues : list
        List of tissue names to process
    flux_data : DataFrame
        Flux constraint data
    metabolite_mapping : str
        Path to metabolite name mapping file
    column_mapping : dict
        Dictionary mapping column names
    label_tissue : bool, optional
        Whether to append tissue name to metabolite IDs (default: True)
    
    Returns:
    --------
    tuple : (experiments, experiment_tracer_mapping)
        - experiments: List of Experiment objects
        - experiment_tracer_mapping: List of dicts with experiment metadata
    
    Examples:
    ---------
    >>> experiments, mapping = build_experiments_from_tracers(
    ...     mfa_model, tracers_dict, tissue_data, tissues,
    ...     flux_data, metabolite_mapping, column_mapping
    ... )
    """
    from sigmaflux.core.experiment import Experiment
    from sigmaflux.core.data import FluxData
    
    experiments = []
    experiment_tracer_mapping = []
    
    for exp_idx, (tracer_string, tracer_list_for_exp) in enumerate(tracers_dict.items()):
        
        # Build tissue labeling data for this tracer experiment
        data = build_tissue_labeling_data(
            mfa_model=mfa_model,
            tracer_list_for_exp=tracer_list_for_exp,
            tissue_tracer_data=tissue_tracer_data,
            tissues=tissues,
            metabolite_mapping=metabolite_mapping,
            column_mapping=column_mapping,
            label_tissue=label_tissue
        )
        
        # Skip this experiment if no data was found
        if len(data) == 0:
            print(f"Warning: No data found for tracer(s) {tracer_string}, skipping experiment")
            continue

        # Create flux data object
        fluxes = FluxData(mfa_model, flux_data)

        # Combine all the data for this tracer into a single MassIsotopomerData object
        data = sum(data)

        # Construct the experiment object with the list of tracers (single or combined)
        experiment = Experiment(data, tracer_list_for_exp, fluxes=fluxes)
        experiments.append(experiment)
        
        # Record the experiment to tracer mapping
        experiment_tracer_mapping.append({
            'experiment_index': exp_idx,
            'tracer_string': tracer_string,
            'tracers': [str(t) for t in tracer_list_for_exp]
        })
    
    return experiments, experiment_tracer_mapping
