import pandas as pd
import numpy as np
import csv
import json
from sigmaflux.core.experiment import Tracer
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DEFAULT_METABOLITE_MAPPING = os.path.join(BASE_DIR, "data", "metabolite_name_mapping.csv")
DEFAULT_COLUMN_MAPPING = {'Compound': 'id', 'C_Label': 'mid', 'mouse': 'sample', 'fraction': 'enrichment',
                          'tissue':'tissue', 'infusate':'infusate', 'Labeling mapped':'labeling'}

TRACEBASE_TISSUE_METABOLITE_MAPPING = os.path.join(BASE_DIR, "data", "tracebase_labeling_data", "tracebase_tissue_metabolite_mapping.csv")
TRACEBASE_SERUM_METABOLITE_MAPPING = os.path.join(BASE_DIR, "data", "tracebase_labeling_data", "tracebase_serum_metabolite_mapping.csv")
WB_TRACEBASE_SERUM_METABOLITE_MAPPING = os.path.join(BASE_DIR, "data", "tracebase_labeling_data", "wb_tracebase_serum_metabolite_mapping.csv")

TRACEBASE_COLUMN_MAPPING = {'Measured Compound(s)': 'id', 'Labeled Element:Count': 'mid', 
                            'Sample': 'sample', 'Fraction': 'enrichment',
                            'Tissue':'tissue', 'Tracer mapped':'infusate', 'Labeling mapped':'labeling'}


TRACEBASE_OPTIONS = {"skiprows":3, "sep": "\t", "na_values": ["None", "none", "NaN", "nan", "NAN", "Nan"]}


def parse_tracer_string(tracer_string):
    """
    Parse a tracer string that may contain single or combined tracers.
    
    Parameters:
    -----------
    tracer_string : str
        A single tracer like "glucose-[13C6]" or combined tracers like 
        "histidine-[13C6];lysine-[13C6];methionine-[13C5]"
    
    Returns:
    --------
    list : List of individual tracer names
    
    Examples:
    ---------
    >>> parse_tracer_string("glucose-[13C6]")
    ["glucose-[13C6]"]
    
    >>> parse_tracer_string("histidine-[13C6];lysine-[13C6]")
    ["histidine-[13C6]", "lysine-[13C6]"]
    """
    return [t.strip() for t in tracer_string.split(';')]


def is_combined_tracer(tracer_string):
    """
    Check if a tracer string represents a combined tracer experiment.
    
    Parameters:
    -----------
    tracer_string : str
        A tracer string to check
    
    Returns:
    --------
    bool : True if the tracer string contains multiple tracers (semicolon-separated)
    """
    return ';' in tracer_string


def map_tracer_ids(tracer_string, tracer_dict):
    """
    Map single or combined tracers to their IDs.
    
    Parameters:
    -----------
    tracer_string : str
        Tracer string (single or combined)
    tracer_dict : dict
        Dictionary mapping tracer names to their properties
    
    Returns:
    --------
    str or list : Single tracer_id string or list of tracer_id strings for combined tracers
    """
    if pd.isna(tracer_string):
        return np.nan
    
    individual_tracers = parse_tracer_string(tracer_string)
    mapped_ids = []
    
    for tracer in individual_tracers:
        if tracer in tracer_dict:
            mapped_ids.append(tracer_dict[tracer]["tracer_id"])
        else:
            return np.nan  # If any tracer is not in dict, mark as unmapped
    
    return mapped_ids if len(mapped_ids) > 1 else mapped_ids[0]


def map_tracer_labelings(tracer_string, tracer_dict):
    """
    Map single or combined tracers to their labeling patterns.
    
    Parameters:
    -----------
    tracer_string : str
        Tracer string (single or combined)
    tracer_dict : dict
        Dictionary mapping tracer names to their properties
    
    Returns:
    --------
    list or list of lists : Single labeling pattern or list of labeling patterns for combined tracers
    """
    if pd.isna(tracer_string):
        return np.nan
    
    individual_tracers = parse_tracer_string(tracer_string)
    mapped_labelings = []
    
    for tracer in individual_tracers:
        if tracer in tracer_dict:
            mapped_labelings.append(tracer_dict[tracer]["labeling"])
        else:
            return np.nan  # If any tracer is not in dict, mark as unmapped
    
    return mapped_labelings if len(mapped_labelings) > 1 else mapped_labelings[0]


def map_model_infusion_rate(tracer_string, tracer_dict):
    """
    Map single or combined tracers to their model infusion rates.
    For combined tracers, returns the average infusion rate.
    
    Parameters:
    -----------
    tracer_string : str
        Tracer string (single or combined)
    tracer_dict : dict
        Dictionary mapping tracer names to their properties
    
    Returns:
    --------
    float : Model infusion rate in nmol/min/gBW (default: 1.0 if not specified)
    """
    if pd.isna(tracer_string):
        return np.nan
    
    individual_tracers = parse_tracer_string(tracer_string)
    rates = []
    
    for tracer in individual_tracers:
        if tracer in tracer_dict and "infusion_rate" in tracer_dict[tracer]:
            rates.append(tracer_dict[tracer]["infusion_rate"])
        else:
            return 1.0  # Default to 1.0 if not specified
    
    return sum(rates) / len(rates) if rates else 1.0


#//TODO: Refactor data input - https://trello.com/c/uaA2PY6L/28-refactor-data-input
def get_data_from_dataframe(dataframe, tissue, tracer_metabolite, tracer_labeling,
                          metabolite_mapping=DEFAULT_METABOLITE_MAPPING,
                          column_mapping=DEFAULT_COLUMN_MAPPING,label_tissue = False):
    """
    imports data for specific tissues from ?? get_data_from_tissue

    parameters:
        input_file ([type]): [description]
        tissue ([type]): [description]
        tracer_metabolite ([str]): metabolite_id of the tracer metabolite
        tracer_labeling ([list]): labeling pattern of the tracer
        mapping ([type], optional): [description]. Defaults to DEFAULT_MAPPING.

    Returns:
        [type]: [description]
    """
   
    df = dataframe.rename(columns=column_mapping)

    with open(metabolite_mapping, newline='') as csvfile:
        reader = csv.reader(csvfile)
        if label_tissue: 
            metabolite_naming = {rows[0].lower(): f"{rows[1]}_{tissue}" for rows in reader}
        else: 
            metabolite_naming = {rows[0].lower(): rows[1] for rows in reader}

    # Filter based on the tracer data
    # Handle both single tracer and combined tracer cases
    # For combined tracers, 'infusate' might be a list of tracer IDs
    def match_infusate(infusate_val):
        if isinstance(infusate_val, list):
            return tracer_metabolite in infusate_val
        else:
            return infusate_val == tracer_metabolite
    
    def match_labeling(labeling_val):
        if isinstance(labeling_val, list) and isinstance(labeling_val[0], list):
            # Combined tracer case: labeling_val is a list of labeling patterns
            return tracer_labeling in labeling_val
        else:
            # Single tracer case
            return labeling_val == tracer_labeling
    
    new_df = df.loc[(df['tissue'] == tissue) 
                    & df['infusate'].apply(match_infusate)
                    & df['labeling'].apply(match_labeling)]

    # Check if we have any data after filtering
    if new_df.empty:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['id', 'mid', 'mean', 'std'])
    
    # Check if we really need this or if the whole thing will work without
    # # Subselect columns form data table 
    new_df = new_df.loc[:, column_mapping.values()]
    
    # check is MID are enumerated with ints and curate if needed
    if new_df['mid'].dtype == 'O':
        # Handle both "C:1" and "C:1; N:2" formats
        # TODO: Add support for nitrogen tracing by setting include_nitrogen=True
        def extract_carbon_mid(mid_str, include_nitrogen=False):
            """
            Extract mass isotopomer distribution number from MID string.
            Currently extracts only carbon count. Set include_nitrogen=True to add N count.
            
            Examples:
                "C:1" -> 1
                "C:1; N:2" -> 1 (or 3 if include_nitrogen=True)
            """
            if ';' in mid_str:
                # Multiple elements present (e.g., "C:1; N:2")
                parts = [p.strip() for p in mid_str.split(';')]
                carbon_count = 0
                nitrogen_count = 0
                
                for part in parts:
                    if 'C:' in part or 'C :' in part:
                        carbon_count = int(part.split(':')[1].strip())
                    elif ('N:' in part or 'N :' in part) and include_nitrogen:
                        nitrogen_count = int(part.split(':')[1].strip())
                
                return carbon_count + nitrogen_count
            else:
                # Single element (e.g., "C:1")
                return int(mid_str.split(':')[1].strip())
        
        # Set include_nitrogen=True in the future to track nitrogen labeling
        new_df['mid'] = new_df['mid'].apply(lambda x: extract_carbon_mid(x, include_nitrogen=False))

    new_df["id"] = new_df['id'].apply(lambda x: metabolite_naming.get(x.lower(), ))

    new_df = new_df.dropna(subset=['id'])

    if new_df.empty:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['id', 'mid', 'mean', 'std'])

    if new_df['enrichment'].dtype == 'O':
        new_df['enrichment'] = new_df['enrichment'].astype(float)

    data = new_df.groupby(['id', 'mid'])[['enrichment', ]].mean()
    data['std'] = new_df.groupby(['id', 'mid'])[['enrichment', ]].std()
    data.columns = ['mean', 'std']

    data = data.reset_index()
    return data


def get_fluxes_from_tissue(input_file):
    """AI is creating summary for get_fluxes_from_tissue

    Args:
        input_file ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Raw file
    df = pd.read_csv(input_file)
    # Process the file
    data = df.groupby(['id', ]).mean()[['flux', ]]
    data['std'] = df.groupby(['id', ]).std()['flux']
    data.columns = ['mean', 'std']

    data = data.reset_index()
    return data


def read_database_file(filename, 
                       tracer_dict_file=None, 
                       options=TRACEBASE_OPTIONS, 
                       tracer_column="Tracer(s)",
                       filter_missing_infusion=True,
                       scale_to_model_infusion=True,
                       recompute_m0_after_scaling=False):
    """
    Reads a database file and returns a dataframe
    Handles both single tracers and combined tracers (semicolon-separated)
    
    Parameters:
    -----------
    filename : str
        Path to the data file
    tracer_dict_file : str, optional
        Path to tracer dictionary JSON file
    options : dict
        Options for pd.read_csv
    tracer_column : str
        Column name containing tracer information
    filter_missing_infusion : bool
        If True, remove rows with missing/zero infusion rate data
    scale_to_model_infusion : bool
        If True, scale the Fraction column based on experimental vs model infusion rates
        Model infusion rates are read from tracer_dict_file
    recompute_m0_after_scaling : bool
        If True, recompute M0 (unlabeled fraction) after scaling to ensure fractions sum to 1
        Default is False to preserve original M0 measurements
    """
    df = pd.read_csv(filename, **options)
    
    # Filter out rows with missing infusion rate data
    if filter_missing_infusion and 'Infusion Rate (ul/min/g)' in df.columns and 'Tracer Concentration(s) (mM)' in df.columns:
        # Calculate infusion rate
        infusion_rate = df['Infusion Rate (ul/min/g)'].astype(float) * df['Tracer Concentration(s) (mM)'].astype(float)
        
        # Identify invalid rows
        invalid_infusion = (infusion_rate == 0) | infusion_rate.isna()
        
        if invalid_infusion.any():
            affected_tracers = df[invalid_infusion][tracer_column].unique()
            # Only show basename of file
            import os
            filename_short = os.path.basename(filename)
            tracer_list = ', '.join([f"'{t}'" for t in affected_tracers])
            print(f"  Filtered rows with missing tracer concentration from {filename_short}. Affected tracers: {tracer_list}")
            
            # Remove invalid rows
            df = df[~invalid_infusion].copy()
    
    if tracer_dict_file is None:
        Warning("No tracer dict file provided, tracers will not be mapped")
        return df
    
    # Load tracer database file 
    with open(tracer_dict_file, 'r') as file:
        tracer_dict = json.load(file)

    # Map the tracer metabolites from the tracer dict file using module-level functions
    df['Tracer mapped'] = df[tracer_column].apply(lambda x: map_tracer_ids(x, tracer_dict))
    df['Labeling mapped'] = df[tracer_column].apply(lambda x: map_tracer_labelings(x, tracer_dict))
    df['Model infusion rate (nmol/min/gBW)'] = df[tracer_column].apply(lambda x: map_model_infusion_rate(x, tracer_dict))

    # Clean the dataframe for non-mapped tracers
    df.dropna(subset=['Tracer mapped','Labeling mapped'], inplace=True)
    
    # Scale the fraction by the infusion rate used in the model
    if scale_to_model_infusion and 'Infusion Rate (ul/min/g)' in df.columns and 'Tracer Concentration(s) (mM)' in df.columns and 'Fraction' in df.columns:
        # Calculate experimental infusion rate (nmol/min/gBW)
        df['Experimental infusion rate (nmol/min/gBW)'] = (
            df['Infusion Rate (ul/min/g)'].astype(float) * 
            df['Tracer Concentration(s) (mM)'].astype(float)
        )
        
        # Scale fraction: Fraction_scaled = Fraction_experimental * (R_model / R_experimental)
        df['Fraction'] = df['Fraction'] * df['Model infusion rate (nmol/min/gBW)'] / df['Experimental infusion rate (nmol/min/gBW)']
        
        # Recompute M0 (unlabeled fraction) after scaling to ensure fractions sum to 1
        if recompute_m0_after_scaling and 'Labeled Element:Count' in df.columns:
            # Identify grouping columns for isotopologues of the same compound
            grouping_cols = ['Tracer(s)', 'Measured Compound(s)', 'Sample']
            
            # Check if all grouping columns exist
            if all(col in df.columns for col in grouping_cols):
                # For each group (compound + sample + tracer), recompute M0
                for group_keys, group_df in df.groupby(grouping_cols):
                    # Find M0 rows (C:0 or C:0; N:0)
                    m0_mask = group_df['Labeled Element:Count'].isin(['C:0', 'N:0', 'C:0; N:0'])
                    
                    if m0_mask.any():
                        # Sum all labeled fractions (excluding M0)
                        labeled_sum = group_df.loc[~m0_mask, 'Fraction'].sum()
                        
                        # Recompute M0 as 1 - sum(labeled fractions)
                        new_m0 = 1.0 - labeled_sum
                        
                        # Update M0 in the dataframe
                        m0_indices = group_df[m0_mask].index
                        df.loc[m0_indices, 'Fraction'] = new_m0 / len(m0_indices)  # Split equally if multiple M0 entries

    return df


def get_tracers_from_data(mfa_model, tracer_list, serum_data = None, infusion_enrichment = 1.0, tracer_column="Tracer(s)", id_column = 'id',
                tracer_dict_file=None, metabolite_mapping=DEFAULT_METABOLITE_MAPPING, column_mapping=DEFAULT_COLUMN_MAPPING):
    """
    Gets the tracers from the dataframe
    Handles both single tracers and combined tracers (semicolon-separated)
    
    Parameters:
    -----------
    mfa_model : MFAModel
        The MFA model containing metabolite definitions
    tracer_list : set or list
        Set/list of tracer strings (single or combined)
    serum_data : DataFrame, optional
        Serum enrichment data for tracers
    infusion_enrichment : float
        Default enrichment for infusion (default: 1.0)
    tracer_column : str
        Column name containing tracer information
    id_column : str
        Column name for metabolite IDs
    tracer_dict_file : str
        Path to JSON file mapping tracer names to model IDs
    metabolite_mapping : str
        Path to metabolite name mapping file
    column_mapping : dict
        Dictionary mapping column names
    
    Returns:
    --------
    dict : Dictionary mapping tracer_string -> list of Tracer objects
           For single tracers: {"glucose-[13C6]": [Tracer(...)]}
           For combined tracers: {"his-[13C6];lys-[13C6]": [Tracer(...), Tracer(...)]}
    """

    if tracer_dict_file is None:
        # Return the tracer metabolites from the dataframe w/o procesing
        return tracer_list
    
    # load tracer database file
    with open(tracer_dict_file, 'r') as file:
            tracer_dict = json.load(file)
        
    # Curate from tracer construction dict 
    if serum_data is None:
        tracer_objects = {}
        for tracer_string in tracer_list:
            # Parse the tracer string (handles both single and combined)
            individual_tracers = parse_tracer_string(tracer_string)
            
            tracer_objects[tracer_string] = [
                Tracer(mfa_model.metabolites.get_by_id(tracer_dict[tracer]["tracer_id"]), 
                       infusion_enrichment, 
                       tracer_dict[tracer]["labeling"]) 
                for tracer in individual_tracers
            ]
        return tracer_objects
    else: 
        # Read the tracer infusion data 
        tracer_enrichment = read_tracer_enrichment_data(serum_data, tracer_column=tracer_column, tracer_dict_file=tracer_dict_file, 
                                                        id_column = id_column, metabolite_mapping=metabolite_mapping, column_mapping=column_mapping)
        
        # Build tracer objects - use measured data when available, extrapolate when missing
        tracer_objects = {}
        for tracer_string in tracer_list:
            # Parse the tracer string (handles both single and combined)
            individual_tracers = parse_tracer_string(tracer_string)
            
            # Check which tracers have enrichment data and which are missing
            measured_tracers = [t for t in individual_tracers if t in tracer_enrichment]
            missing_tracers = [t for t in individual_tracers if t not in tracer_enrichment]
            
            if missing_tracers:
                if measured_tracers:
                    # Use average enrichment from measured tracers in this experiment
                    avg_enrichment = sum(tracer_enrichment[t] for t in measured_tracers) / len(measured_tracers)
                    print(f"  Extrapolating enrichment for {len(missing_tracers)} unmeasured tracer(s) in '{tracer_string}' from {len(measured_tracers)} measured (avg: {avg_enrichment:.4f})")
                    
                    # Apply average enrichment to missing tracers
                    for tracer in missing_tracers:
                        tracer_enrichment[tracer] = avg_enrichment
                else:
                    # No measured tracers in this experiment - skip it
                    print(f"  Skipping '{tracer_string}' - no tracers measured")
                    continue
            
            # Create the experiment with all tracers (measured + extrapolated)
            tracer_objects[tracer_string] = [
                Tracer(mfa_model.metabolites.get_by_id(tracer_dict[tracer]["tracer_id"]), 
                       infusion_enrichment,
                       tracer_dict[tracer]["labeling"],
                       serum_enrichment=tracer_enrichment[tracer])
                for tracer in individual_tracers
            ]
        return tracer_objects
        

def read_tracer_enrichment_data(serum_data, tracer_column = "Tracer(s)", id_column = 'id', tracer_dict_file=None, 
                                metabolite_mapping=DEFAULT_METABOLITE_MAPPING, column_mapping=DEFAULT_COLUMN_MAPPING):
    """
    Reads the tracer infusion data and returns the tracer infusions for each tracer 
    """
    if tracer_dict_file is not None: 
        with open(tracer_dict_file, 'r') as file:
            tracer_dict = json.load(file)

    # Map the columns
    serum_data = serum_data.rename(columns=column_mapping)

    # Load metablite mapping file
    with open(metabolite_mapping, newline='') as csvfile:
        reader = csv.reader(csvfile)
        metabolite_naming = {rows[0].lower(): rows[1] for rows in reader}

    # Map the metabolite names to the dataframe
    serum_data[id_column] = serum_data[id_column].apply(lambda x: metabolite_naming.get(x.lower(), ))

    # Extract the number after 'C:' in the 'mid' column, even if there are other elements (e.g., 'C:1,N:2')
    serum_data['number_carbons'] = serum_data['mid'].str.extract(r'C:(\d+)').astype(float).astype('Int64')
    serum_data['number_nitrogens'] = serum_data['mid'].str.extract(r'N:(\d+)').astype(float).astype('Int64')

    enrichments = {} 

    # Compute number of labels
    serum_data['number_labels'] = serum_data['number_carbons'] + serum_data['number_nitrogens'].fillna(0)
    # tracer labels - handle both single labeling [1,1,1] and combined labeling [[1,1,1], [1,1,1,1]]
    def get_tracer_labels(labeling_val):
        if isinstance(labeling_val, list):
            # Check if it's a nested list (combined tracer)
            if len(labeling_val) > 0 and isinstance(labeling_val[0], list):
                # Combined tracer: return sum of first labeling pattern
                return sum(labeling_val[0]) if len(labeling_val[0]) > 0 else 0
            else:
                # Single tracer
                return sum(labeling_val)
        return 0
    
    serum_data['tracer_labels'] = serum_data['labeling'].apply(get_tracer_labels)

    # Process each unique tracer (including combined tracers)
    for tracer_string in serum_data[tracer_column].unique():
        
        # Parse the tracer string to get individual tracers
        individual_tracers = parse_tracer_string(tracer_string)
        
        # For each individual tracer in a potentially combined tracer
        for tracer in individual_tracers:
            if tracer_dict is not None:
                if tracer not in tracer_dict:
                    Warning(f"Tracer {tracer} not found in tracer dictionary")
                    continue
                tracer_metabolite = tracer_dict[tracer]["met_id"]
            else:
                tracer_metabolite = tracer

            # Filter - match the original tracer_string in the data
            serum_enrichment_filter = (serum_data[tracer_column] == tracer_string) & (serum_data[id_column] == tracer_metabolite) 

            if not serum_enrichment_filter.any():
                Warning(f"Tracer {tracer} (from {tracer_string}) not found in the tracer dataframe")
                continue
            
            # Get the filtered data
            filtered_data = serum_data[serum_enrichment_filter]
            
            # Check for valid enrichment and carbon count data
            valid_data = filtered_data.dropna(subset=['enrichment', 'number_carbons'])
            
            if valid_data.empty:
                Warning(f"Tracer {tracer} (from {tracer_string}) has no valid enrichment or carbon count data")
                continue
                
            carbon_enrichment = valid_data['enrichment'] * valid_data['number_carbons']
            enrichment = carbon_enrichment.mean()
            
            # Double-check that enrichment is not NaN
            if pd.isna(enrichment):
                Warning(f"Tracer {tracer} (from {tracer_string}) resulted in NaN enrichment")
                continue

            # Store enrichment for individual tracer (not the combined string)
            enrichments[tracer] = enrichment

    return enrichments


## processes the reactions and points it to the correct reaction in the model 
def get_flux_data_from_dataframe( df, tissue = None): 
    """ 
    df columns = [reaction, mean, std ]

    """

    if tissue is not None: 
        df["reaction"] = df["reaction"] + "_" + tissue
   # TODO: "Check if the reaction is in the model"
    return df