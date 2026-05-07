"""
Data loading and preprocessing utilities for flux prediction.

Provides functions for loading data from HDF5 files, creating tracer mappings,
and preparing datasets for training.
"""

import os
import glob
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from cobra.io.json import load_json_model
from cobra.util.array import create_stoichiometric_matrix


# Default path to biosynthetic bypass reactions config
_DEFAULT_MEASURED_FLUXES_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '../../data/biosynthetic_bypass_reactions.json'
)


def load_measured_fluxes_config(config_path=None):
    """
    Load biosynthetic bypass reactions configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
    
    Returns:
        config: Dictionary with bypass reaction configuration
    """
    if config_path is None:
        config_path = _DEFAULT_MEASURED_FLUXES_CONFIG_PATH
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_measured_fluxes_reaction_ids(mode='respiratory', config_path=None):
    """
    Get list of bypass reaction IDs based on mode.
    
    Args:
        mode: Bypass mode - one of:
            - 'respiratory' or 'legacy': Just VO2/VCO2/RER derived features (default)
            - 'biosynthetic' or 'extended': All biosynthetic fluxes (protein, TG, glycogen)
            - 'full': Respiratory + all biosynthetic fluxes
            - list: Custom list of reaction IDs to extract as bypass features
            - None or False: No bypass features
        config_path: Path to bypass config JSON file
    
    Returns:
        reaction_ids: List of reaction IDs to extract (excludes derived features like RER)
        derived_features: List of derived feature names to compute (e.g., ['VO2', 'VCO2', 'RER'])
        feature_names: List of all feature names in order they appear in bypass data
    """
    if mode is None or mode is False:
        return [], [], []
    
    if isinstance(mode, list):
        # Custom list of reaction IDs
        return mode, [], mode
    
    config = load_measured_fluxes_config(config_path)
    
    reaction_ids = []
    derived_features = []
    feature_names = []
    
    if mode in ('respiratory', 'legacy', 'full'):
        # Respiratory exchange features (VO2, VCO2, RER)
        reaction_ids.extend(['EX_o2_e_lung', 'EX_co2_e_lung'])
        derived_features.extend(['VO2', 'VCO2', 'RER'])
        feature_names.extend(['VO2', 'VCO2', 'RER'])
    
    if mode in ('biosynthetic', 'extended', 'full'):
        # Protein synthesis
        protein_rxns = config['protein_synthesis']['reaction_ids']
        reaction_ids.extend(protein_rxns)
        feature_names.extend(protein_rxns)
        
        # Triglyceride synthesis (DGAT)
        tg_rxns = config['triglyceride_synthesis']['reaction_ids']
        reaction_ids.extend(tg_rxns)
        feature_names.extend(tg_rxns)
        
        # Glycogen synthesis
        glyc_rxns = config['glycogen_synthesis']['reaction_ids']
        reaction_ids.extend(glyc_rxns)
        feature_names.extend(glyc_rxns)
    
    return reaction_ids, derived_features, feature_names


def extract_measured_fluxes(flux_df, train_indices, mode='respiratory', config_path=None, reaction_ids=None):
    """
    Extract bypass features from flux data.

    These are fluxes that will be measured with orthogonal methods and passed
    directly to the transformer, bypassing the labeling data pathway.

    Args:
        flux_df: DataFrame with flux data (reactions x samples)
        train_indices: Sample indices to extract
        mode: Bypass mode (see get_bypass_reaction_ids for options). Ignored when
            reaction_ids is provided.
        config_path: Path to bypass config JSON file. Ignored when reaction_ids
            is provided.
        reaction_ids: Explicit list of reaction IDs to extract. When provided,
            mode and config_path are ignored and no derived features are computed.

    Returns:
        measured_fluxes_data: (num_samples, num_bypass_features) numpy array
        feature_names: List of feature names
    """
    if reaction_ids is not None:
        # Caller supplied an explicit list — skip mode resolution entirely.
        direct_rxn_ids = reaction_ids
        derived_features = []
        feature_names = list(reaction_ids)
    else:
        if mode is None or mode is False:
            return None, []
        direct_rxn_ids_from_mode, derived_features, feature_names = get_measured_fluxes_reaction_ids(mode, config_path)
        if not direct_rxn_ids_from_mode and not derived_features:
            return None, []
        direct_rxn_ids = direct_rxn_ids_from_mode
    
    flux_data_subset = flux_df.values[:, train_indices].T  # (num_samples, num_reactions)
    reaction_list = flux_df.index.tolist()
    
    measured_fluxes_columns = []
    
    # Handle respiratory-derived features (VO2, VCO2, RER)
    if 'VO2' in derived_features:
        o2_idx = reaction_list.index('EX_o2_e_lung') if 'EX_o2_e_lung' in reaction_list else None
        co2_idx = reaction_list.index('EX_co2_e_lung') if 'EX_co2_e_lung' in reaction_list else None
        
        if o2_idx is not None and co2_idx is not None:
            # O2 is negative (uptake), convert to positive VO2
            vo2 = -flux_data_subset[:, o2_idx]
            # CO2 is positive (production)
            vco2 = flux_data_subset[:, co2_idx]
            # Calculate RER
            rer = np.where(vo2 > 1e-6, vco2 / vo2, 0.0)
            
            measured_fluxes_columns.extend([vo2, vco2, rer])
        else:
            print("WARNING: Could not find EX_o2_e_lung or EX_co2_e_lung in flux data")
            print("         Respiratory bypass features will not be included")
            # Remove respiratory features from feature_names
            feature_names = [f for f in feature_names if f not in ['VO2', 'VCO2', 'RER']]
    
    # Handle direct flux extraction (filter out exchange reactions already handled above)
    direct_rxn_ids = [r for r in direct_rxn_ids if r not in ['EX_o2_e_lung', 'EX_co2_e_lung']]
    missing_rxns = []
    
    for rxn_id in direct_rxn_ids:
        if rxn_id in reaction_list:
            rxn_idx = reaction_list.index(rxn_id)
            measured_fluxes_columns.append(flux_data_subset[:, rxn_idx])
        else:
            missing_rxns.append(rxn_id)
            # Remove from feature_names
            if rxn_id in feature_names:
                feature_names.remove(rxn_id)
    
    if missing_rxns:
        print(f"WARNING: {len(missing_rxns)} biosynthetic reactions not found in flux data:")
        for rxn in missing_rxns[:5]:
            print(f"         - {rxn}")
        if len(missing_rxns) > 5:
            print(f"         ... and {len(missing_rxns) - 5} more")
    
    if not measured_fluxes_columns:
        return None, []
    
    measured_fluxes_data = np.column_stack(measured_fluxes_columns)
    return measured_fluxes_data, feature_names


def find_data_batches(base_dir, base_filename, pattern='*_*.h5'):
    """
    Find all batch files matching a base pattern.
    
    Args:
        base_dir: Directory containing data files
        base_filename: Base name of files (e.g., 'wb_model_labeling_data')
        pattern: Glob pattern for batch suffix (default: '*_*.h5')
    
    Returns:
        batch_files: List of (start_idx, end_idx, filepath) tuples sorted by start_idx
    """
    search_pattern = os.path.join(base_dir, f"{base_filename}_{pattern}")
    files = glob.glob(search_pattern)
    
    batch_info = []
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract batch indices from filename (e.g., "0_500" from "data_0_500.h5")
        match = re.search(r'(\d+)_(\d+)\.h5$', filename)
        if match:
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            batch_info.append((start_idx, end_idx, filepath))
    
    # Sort by start index
    batch_info.sort(key=lambda x: x[0])
    
    return batch_info


def load_data_batches(base_dir, base_filename, key='labeling', max_samples=None):
    """
    Load data from multiple batch files.
    
    Args:
        base_dir: Directory containing data files
        base_filename: Base name of files (e.g., 'wb_model_labeling_data')
        key: HDF5 key to load
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        df: Combined DataFrame from all batches
    """
    batch_files = find_data_batches(base_dir, base_filename)
    
    if not batch_files:
        # Try single file without batch suffix
        single_file = os.path.join(base_dir, f"{base_filename}.h5")
        if os.path.exists(single_file):
            return pd.read_hdf(single_file, key=key)
        else:
            raise FileNotFoundError(f"No batch files found for {base_filename} in {base_dir}")
    
    dfs = []
    total_samples = 0
    
    for start_idx, end_idx, filepath in batch_files:
        df = pd.read_hdf(filepath, key=key)
        num_samples = end_idx - start_idx
        
        # Select appropriate samples
        if max_samples is not None:
            remaining = max_samples - total_samples
            if remaining <= 0:
                break
            if num_samples > remaining:
                df = df.iloc[:, :remaining]
                num_samples = remaining
        
        dfs.append(df)
        total_samples += num_samples
        
        if max_samples is not None and total_samples >= max_samples:
            break
    
    combined_df = pd.concat(dfs, axis=1)
    return combined_df


def load_data(base_dir, max_samples=None):
    """
    Load labeling, flux, and reversibility data from HDF5 files.
    
    Automatically detects and loads batch files with patterns like:
    - wb_model_labeling_data_0_500.h5
    - wb_model_flux_data_0_500.h5
    - wb_model_reversibility_data_0_500.h5
    
    Args:
        base_dir: Base directory containing the data files
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        labeling_df: DataFrame with labeling data
        flux_df: DataFrame with flux data
        reversibility_df: DataFrame with reversibility data
    """
    print("Loading data from files...")
    
    labeling_df = load_data_batches(base_dir, 'wb_model_labeling_data', 'labeling', max_samples)
    flux_df = load_data_batches(base_dir, 'wb_model_flux_data', 'fluxes', max_samples)
    reversibility_df = load_data_batches(base_dir, 'wb_model_reversibility_data', 
                                         'reversibility', max_samples)
    
    print(f"Labeling data shape: {labeling_df.shape}")
    print(f"Flux data shape: {flux_df.shape}")
    print(f"Reversibility data shape: {reversibility_df.shape}")
    
    return labeling_df, flux_df, reversibility_df


def create_tracer_mapping(labeling_df):
    """
    Create mapping from tracer names to multi-hot encoded vectors.
    
    Supports single and multi-tracer experiments (e.g., "tracer1|tracer2").
    
    Args:
        labeling_df: DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer)
    
    Returns:
        tracer_multi_hot: Array of multi-hot vectors (num_rows, num_base_tracers)
        base_tracer_names: List of unique base tracer names
    """
    print("\nCreating tracer mapping...")
    
    tracers = labeling_df.index.get_level_values('Tracer')
    unique_tracers = tracers.unique().tolist()
    
    # Extract base tracers
    base_tracers_set = set()
    for tracer_str in unique_tracers:
        individual_tracers = [t.strip() for t in tracer_str.split('|')]
        base_tracers_set.update(individual_tracers)
    
    base_tracer_names = sorted(list(base_tracers_set))
    num_base_tracers = len(base_tracer_names)
    base_tracer_to_idx = {tracer: i for i, tracer in enumerate(base_tracer_names)}
    
    # Create multi-hot encoding
    tracer_to_multi_hot = {}
    for tracer_str in unique_tracers:
        multi_hot = np.zeros(num_base_tracers, dtype=np.float32)
        individual_tracers = [t.strip() for t in tracer_str.split('|')]
        for tracer in individual_tracers:
            if tracer in base_tracer_to_idx:
                multi_hot[base_tracer_to_idx[tracer]] = 1.0
        tracer_to_multi_hot[tracer_str] = multi_hot
    
    tracer_multi_hot = np.array([tracer_to_multi_hot[tracer] for tracer in tracers])
    
    print(f"Number of unique tracer combinations: {len(unique_tracers)}")
    print(f"Number of base tracers: {num_base_tracers}")
    print(f"Base tracers: {base_tracer_names}")
    
    return tracer_multi_hot, base_tracer_names


def load_stoichiometric_matrix(model_path):
    """
    Load stoichiometric matrix from a COBRA model JSON file.
    
    Args:
        model_path: Path to model JSON file
    
    Returns:
        S: Stoichiometric matrix (numpy array)
        model: COBRA model object
    """
    print(f"\nLoading model from {model_path}")
    
    model = load_json_model(model_path)
    S = create_stoichiometric_matrix(model, array_type='dense')
    
    print(f"Stoichiometric matrix shape: {S.shape}")
    print(f"Number of metabolites: {S.shape[0]}")
    print(f"Number of reactions: {S.shape[1]}")
    
    return S, model


def prepare_dataset(labeling_df, flux_df, reversibility_df, tracer_multi_hot,
                   train_indices=None, normalize=True, measured_fluxes=None,
                   norm_stats_input=None):
    """
    Prepare PyTorch dataset from DataFrames.

    Args:
        labeling_df: DataFrame with labeling data
        flux_df: DataFrame with flux data
        reversibility_df: DataFrame with reversibility data
        tracer_multi_hot: Array of multi-hot tracer encodings
        train_indices: Indices to use (None for all)
        normalize: Whether to normalize data
        measured_fluxes: List of reaction IDs whose fluxes are directly measured
                        and should be appended to the model input alongside labeling
                        data.  Pass None (default) or [] to disable.
        norm_stats_input: Optional pre-computed normalization statistics to apply.
                         Should contain 'labeling_mean', 'labeling_std', and
                         optionally 'measured_fluxes_mean', 'measured_fluxes_std',
                         'measured_fluxes_feature_names'.

    Returns:
        dataset: TensorDataset with (inputs, targets)
        stats: Dictionary with normalization statistics (either computed or input)
    """
    if train_indices is None:
        train_indices = range(labeling_df.shape[1])
    
    # Convert train_indices to list if it's a range for consistent indexing
    train_indices = list(train_indices)
    
    # Prepare inputs
    labeling_data = labeling_df.values[:, train_indices].T  # (num_samples, labeling_dim)
    
    # Normalize if requested
    if normalize:
        # Log transform labeling data
        eps = 1e-9
        labeling_data = np.log(np.clip(labeling_data, 0, None) + eps)
        
        # Apply or compute normalization statistics
        if norm_stats_input is not None:
            # Use provided statistics
            mean = float(norm_stats_input['labeling_mean'])
            std = float(norm_stats_input['labeling_std'])
            if std < 1e-12:
                std = 1.0
            labeling_data = (labeling_data - mean) / std
            norm_stats = norm_stats_input.copy()
            norm_stats['labeling_std'] = std
        else:
            # Compute new statistics
            mean = labeling_data.mean()
            std = labeling_data.std()
            if std < 1e-12:
                std = 1.0
            labeling_data = (labeling_data - mean) / std
            norm_stats = {'labeling_mean': mean, 'labeling_std': std}
        labeling_data = np.nan_to_num(labeling_data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm_stats = {}
    
    # Extract tracer indicators for each experiment
    # Get unique tracers and their indicators
    tracers = labeling_df.index.get_level_values('Tracer')
    unique_tracers = tracers.unique().tolist()
    # For each unique tracer, get its multi-hot encoding (all rows for same tracer have same encoding)
    tracer_indicators = []
    for tracer in unique_tracers:
        idx = np.where(tracers == tracer)[0][0]  # Find first occurrence
        tracer_indicators.append(tracer_multi_hot[idx])
    tracer_indicators = np.array(tracer_indicators)  # (num_experiments, num_base_tracers)
    
    # Flatten tracer indicators and repeat for all samples
    tracer_data = np.tile(tracer_indicators.flatten(), (len(train_indices), 1))  # (num_samples, num_experiments * num_base_tracers)
    
    # Combine inputs: [labeling_data, tracer_data, measured_fluxes_data]
    inputs = np.hstack([labeling_data, tracer_data])

    # Extract and append measured flux features if requested
    if measured_fluxes:
        measured_fluxes_data, feature_names = extract_measured_fluxes(
            flux_df, train_indices, reaction_ids=measured_fluxes
        )

        if measured_fluxes_data is not None and len(feature_names) > 0:
            if normalize:
                if norm_stats_input is not None and 'measured_fluxes_mean' in norm_stats_input:
                    measured_fluxes_mean = np.array(norm_stats_input['measured_fluxes_mean'])
                    measured_fluxes_std = np.array(norm_stats_input['measured_fluxes_std'])

                    if len(measured_fluxes_mean) != measured_fluxes_data.shape[1]:
                        raise ValueError(
                            f"Measured-fluxes normalization stats dimension mismatch: "
                            f"expected {len(measured_fluxes_mean)} features but got {measured_fluxes_data.shape[1]}. "
                            f"Make sure measured_fluxes matches the training configuration."
                        )

                    measured_fluxes_data = (measured_fluxes_data - measured_fluxes_mean) / measured_fluxes_std
                    norm_stats['measured_fluxes_mean'] = norm_stats_input['measured_fluxes_mean']
                    norm_stats['measured_fluxes_std'] = norm_stats_input['measured_fluxes_std']
                    if 'measured_fluxes_feature_names' in norm_stats_input:
                        norm_stats['measured_fluxes_feature_names'] = norm_stats_input['measured_fluxes_feature_names']
                else:
                    measured_fluxes_mean = measured_fluxes_data.mean(axis=0)
                    measured_fluxes_std = measured_fluxes_data.std(axis=0)
                    measured_fluxes_std = np.where(measured_fluxes_std < 1e-6, 1.0, measured_fluxes_std)
                    measured_fluxes_data = (measured_fluxes_data - measured_fluxes_mean) / measured_fluxes_std
                    norm_stats['measured_fluxes_mean'] = measured_fluxes_mean.tolist()
                    norm_stats['measured_fluxes_std'] = measured_fluxes_std.tolist()
                    norm_stats['measured_fluxes_feature_names'] = feature_names

            inputs = np.hstack([inputs, measured_fluxes_data])
            print(f"Extracted {len(feature_names)} measured fluxes: {feature_names}")
    
    # Prepare targets (flux + reversibility)
    flux_data = flux_df.values[:, train_indices].T
    rev_data = reversibility_df.values[:, train_indices].T
    targets = np.hstack([flux_data, rev_data])
    
    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    
    return dataset, norm_stats


def apply_random_masking(inputs, mask_prob=0.3, min_experiments=1, num_base_tracers=21, measured_fluxes_dim=0, num_experiments=None):
    """
    Apply random masking augmentation to input data during training.

    Randomly masks out entire experiments (tracers) to teach the model
    to handle missing data. For each sample, randomly selects experiments
    to mask based on mask_prob.

    Args:
        inputs: (batch_size, total_input_dim) tensor with labeling + tracer data
        mask_prob: Probability of masking each experiment (0.0 to 1.0)
        min_experiments: Minimum number of experiments to keep (at least 1)
        num_base_tracers: Number of base tracers (to determine num_experiments)
        measured_fluxes_dim: Number of bypass features appended at end of input (e.g. 3 for VO2/VCO2/RER)
        num_experiments: Number of experiments in input. Must be provided; cannot be
                         reliably inferred from input dimensions alone.

    Returns:
        masked_inputs: (batch_size, total_input_dim) tensor with masked data
        mask: (batch_size, num_experiments) boolean tensor (True = valid, False = masked)
    """
    batch_size, total_dim = inputs.shape
    device = inputs.device

    # Input layout: [labeling_exp0 | ... | labeling_expN | tracer_exp0 | ... | tracer_expN | bypass]
    labeling_and_tracer_dim = total_dim - measured_fluxes_dim

    if num_experiments is None:
        raise ValueError(
            "num_experiments must be provided to apply_random_masking — it cannot be reliably "
            "inferred from input dimensions. Pass model.input_dim and model.num_base_tracers to "
            "compute it: num_experiments = (total_dim - measured_fluxes_dim) // (input_dim + num_base_tracers)"
        )

    labeling_total = labeling_and_tracer_dim - num_experiments * num_base_tracers
    input_dim_per_exp = labeling_total // num_experiments
    
    # Create random mask for each sample
    # Each sample gets different random experiments masked
    mask = torch.ones(batch_size, num_experiments, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        # Randomly decide which experiments to mask
        mask_indices = torch.rand(num_experiments, device=device) < mask_prob
        
        # Ensure at least min_experiments remain
        if mask_indices.sum() > num_experiments - min_experiments:
            # Too many masked, randomly keep some
            keep_count = max(min_experiments, num_experiments - int(num_experiments * mask_prob))
            all_indices = torch.randperm(num_experiments, device=device)
            keep_indices = all_indices[:keep_count]
            mask_indices = torch.ones(num_experiments, dtype=torch.bool, device=device)
            mask_indices[keep_indices] = False
        
        mask[b] = ~mask_indices
    
    # Apply masking to inputs
    masked_inputs = inputs.clone()
    
    for exp_idx in range(num_experiments):
        # Zero out labeling data for masked experiments
        labeling_start = exp_idx * input_dim_per_exp
        labeling_end = (exp_idx + 1) * input_dim_per_exp
        
        # Zero out tracer indicators for masked experiments
        tracer_start = labeling_total + exp_idx * num_base_tracers
        tracer_end = labeling_total + (exp_idx + 1) * num_base_tracers
        
        # Apply mask (broadcast across batch dimension)
        masked_inputs[:, labeling_start:labeling_end] *= mask[:, exp_idx:exp_idx+1]
        masked_inputs[:, tracer_start:tracer_end] *= mask[:, exp_idx:exp_idx+1]
    
    return masked_inputs, mask


def extract_mask_from_data(inputs, num_base_tracers=21, threshold=1e-6, measured_fluxes_dim=0, num_experiments=None):
    """
    Extract mask from data by detecting which experiments are all zeros/missing.

    Useful for handling real missing data during prediction.

    Args:
        inputs: (batch_size, total_input_dim) tensor
        num_base_tracers: Number of base tracers
        threshold: Threshold for considering data as missing
        measured_fluxes_dim: Number of bypass features appended at end of input (e.g. 3 for VO2/VCO2/RER)
        num_experiments: Number of experiments in input. Must be provided; cannot be
                         reliably inferred from input dimensions alone.

    Returns:
        mask: (batch_size, num_experiments) boolean tensor (True = valid, False = missing)
    """
    batch_size, total_dim = inputs.shape
    device = inputs.device

    labeling_and_tracer_dim = total_dim - measured_fluxes_dim

    if num_experiments is None:
        raise ValueError(
            "num_experiments must be provided to extract_mask_from_data — it cannot be reliably "
            "inferred from input dimensions. Pass model.input_dim and model.num_base_tracers to "
            "compute it: num_experiments = (total_dim - measured_fluxes_dim) // (input_dim + num_base_tracers)"
        )

    labeling_total = labeling_and_tracer_dim - num_experiments * num_base_tracers
    
    # Create mask by checking if tracer indicators are all zero
    mask = torch.zeros(batch_size, num_experiments, dtype=torch.bool, device=device)
    
    for exp_idx in range(num_experiments):
        tracer_start = labeling_total + exp_idx * num_base_tracers
        tracer_end = labeling_total + (exp_idx + 1) * num_base_tracers
        
        # Check if any tracer indicator is non-zero
        has_data = inputs[:, tracer_start:tracer_end].abs().sum(dim=1) > threshold
        mask[:, exp_idx] = has_data
    
    return mask
