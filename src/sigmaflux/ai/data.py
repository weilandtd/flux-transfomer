"""
Data loading and preprocessing utilities for flux prediction.

Provides functions for loading data from HDF5 files, building radial-embedding
vocabularies, tokenizing labeling experiments, and preparing datasets for training.
"""

import math
import os
import glob
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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
    Extract unique base tracer names from labeling DataFrame.

    Supports single and multi-tracer experiments (e.g., "tracer1|tracer2").

    Args:
        labeling_df: DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer)

    Returns:
        tracer_multi_hot: Array of multi-hot vectors (num_rows, num_base_tracers)
            (kept for backwards-compatibility; no longer used by prepare_dataset)
        base_tracer_names: Sorted list of unique base tracer names
    """
    print("\nExtracting tracer names...")

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

    # Create multi-hot encoding (kept for backwards-compatibility)
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


# ---------------------------------------------------------------------------
# Radial embedding vocabulary
# ---------------------------------------------------------------------------

def radial_embedding_table(vocab_size, num_harmonics):
    """
    Compute a fixed multi-frequency radial embedding matrix.

    Each index k in [0, vocab_size) is mapped to a 2*num_harmonics vector:
        e(k) = [sin θ(k,1), cos θ(k,1), ..., sin θ(k,H), cos θ(k,H)]
    where θ(k,h) = 2π * k/vocab_size * h.

    Args:
        vocab_size: Number of items in the alphabet (N).
        num_harmonics: Number of frequency harmonics (H). Vector length = 2H.

    Returns:
        table: (vocab_size, 2 * num_harmonics) float32 array
    """
    k = np.arange(vocab_size, dtype=np.float64)          # (N,)
    h = np.arange(1, num_harmonics + 1, dtype=np.float64)  # (H,)
    theta = 2.0 * np.pi * k[:, None] / vocab_size * h[None, :]  # (N, H)
    return np.concatenate([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)  # (N, 2H)


def build_radial_vocabs(cobra_model, labeling_df, base_tracer_names,
                        h_compound=None, h_label=None, h_tracer=None):
    """
    Build radial-embedding vocabularies for compound, isotopomer label, and tracer.

    Compound vocabulary is taken from the full COBRA model so embeddings
    generalise to any metabolite in the network (not just measured ones).
    Label and tracer vocabularies are built from what is present in labeling_df
    and base_tracer_names respectively.

    H values default to ceil(log2(N)) per vocabulary; pass explicit values to
    override (e.g. h_compound=16 as suggested in the design spec).

    Args:
        cobra_model: COBRA model object (from load_stoichiometric_matrix).
        labeling_df: DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer).
        base_tracer_names: Sorted list of unique base tracer names.
        h_compound: Number of harmonics for compound embeddings (default: auto).
        h_label: Number of harmonics for isotopomer label embeddings (default: auto).
        h_tracer: Number of harmonics for tracer embeddings (default: auto).

    Returns:
        vocabs: dict with keys:
            compound_vocab, label_vocab, tracer_vocab  — sorted string lists
            h_compound, h_label, h_tracer              — int H values used
            token_dim                                   — total token dimension
            compound_emb_table                         — (N_c, 2*H_c) float32 array
            label_emb_table                            — (N_l, 2*H_l) float32 array
            tracer_emb_table                           — (N_t, 2*H_t) float32 array
    """
    # Compound vocab: full metabolic network
    compound_vocab = sorted([met.id for met in cobra_model.metabolites])
    n_compound = len(compound_vocab)
    if h_compound is None:
        h_compound = max(1, math.ceil(math.log2(n_compound)))

    # Label vocab: unique isotopomers in data
    label_vocab = sorted(labeling_df.index.get_level_values('Isotopomer').unique().tolist())
    n_label = len(label_vocab)
    if h_label is None:
        h_label = max(1, math.ceil(math.log2(n_label)))

    # Tracer vocab: base tracers
    tracer_vocab = sorted(base_tracer_names)
    n_tracer = len(tracer_vocab)
    if h_tracer is None:
        h_tracer = max(1, math.ceil(math.log2(n_tracer)))

    token_dim = 2 * h_compound + 2 * h_label + 2 * h_tracer + 1  # +1 for fraction

    print(f"\nRadial embedding vocabularies:")
    print(f"  Compounds: {n_compound} → H={h_compound}, emb_dim={2*h_compound}")
    print(f"  Labels:    {n_label}   → H={h_label},  emb_dim={2*h_label}")
    print(f"  Tracers:   {n_tracer}  → H={h_tracer},  emb_dim={2*h_tracer}")
    print(f"  Token dim: {token_dim} (incl. fraction scalar)")

    return {
        'compound_vocab': compound_vocab,
        'label_vocab': label_vocab,
        'tracer_vocab': tracer_vocab,
        'h_compound': h_compound,
        'h_label': h_label,
        'h_tracer': h_tracer,
        'token_dim': token_dim,
        'compound_emb_table': radial_embedding_table(n_compound, h_compound),
        'label_emb_table': radial_embedding_table(n_label, h_label),
        'tracer_emb_table': radial_embedding_table(n_tracer, h_tracer),
    }


def build_token_static_features(labeling_df, vocabs):
    """
    Precompute the fixed (non-fraction) part of each token from labeling_df.

    Each row of labeling_df corresponds to one token: its compound, isotopomer
    label, and tracer identity are looked up in the vocabulary and their radial
    embeddings are concatenated.  Multi-tracer rows (e.g., "A|B") get the sum
    of the individual base-tracer embeddings.

    Args:
        labeling_df: DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer).
        vocabs: dict from build_radial_vocabs.

    Returns:
        static_features: (num_tokens, 2*H_c + 2*H_l + 2*H_t) float32 array
        token_experiment_ids: (num_tokens,) int64 array, 0-based experiment index
    """
    compound_vocab = vocabs['compound_vocab']
    label_vocab = vocabs['label_vocab']
    tracer_vocab = vocabs['tracer_vocab']
    compound_emb_table = vocabs['compound_emb_table']
    label_emb_table = vocabs['label_emb_table']
    tracer_emb_table = vocabs['tracer_emb_table']

    compound_to_idx = {c: i for i, c in enumerate(compound_vocab)}
    label_to_idx = {l: i for i, l in enumerate(label_vocab)}
    tracer_to_idx = {t: i for i, t in enumerate(tracer_vocab)}

    tracers = labeling_df.index.get_level_values('Tracer')
    metabolites = labeling_df.index.get_level_values('Metabolite')
    isotopomers = labeling_df.index.get_level_values('Isotopomer')
    unique_tracers = tracers.unique().tolist()

    num_tokens = len(labeling_df)
    compound_emb_dim = compound_emb_table.shape[1]
    label_emb_dim = label_emb_table.shape[1]
    tracer_emb_dim = tracer_emb_table.shape[1]
    static_dim = compound_emb_dim + label_emb_dim + tracer_emb_dim

    static_features = np.zeros((num_tokens, static_dim), dtype=np.float32)
    token_experiment_ids = np.zeros(num_tokens, dtype=np.int64)

    missing_compounds = set()
    missing_labels = set()
    missing_tracers = set()

    c_end = compound_emb_dim
    l_end = compound_emb_dim + label_emb_dim

    for i, (tracer_str, met_id, isotopomer) in enumerate(
            zip(tracers, metabolites, isotopomers)):

        # Compound embedding
        c_idx = compound_to_idx.get(met_id)
        if c_idx is None:
            missing_compounds.add(met_id)
            c_idx = 0
        static_features[i, :c_end] = compound_emb_table[c_idx]

        # Label (isotopomer) embedding
        l_idx = label_to_idx.get(isotopomer)
        if l_idx is None:
            missing_labels.add(isotopomer)
            l_idx = 0
        static_features[i, c_end:l_end] = label_emb_table[l_idx]

        # Tracer embedding: sum of base tracer embeddings (handles "A|B" combos)
        tracer_emb = np.zeros(tracer_emb_dim, dtype=np.float32)
        for bt in (t.strip() for t in tracer_str.split('|')):
            t_idx = tracer_to_idx.get(bt)
            if t_idx is None:
                missing_tracers.add(bt)
                t_idx = 0
            tracer_emb += tracer_emb_table[t_idx]
        static_features[i, l_end:] = tracer_emb

        # Experiment index (which unique tracer combination this token belongs to)
        token_experiment_ids[i] = unique_tracers.index(tracer_str)

    if missing_compounds:
        print(f"WARNING: {len(missing_compounds)} metabolite(s) not in COBRA model vocab "
              f"(using index 0): {sorted(missing_compounds)}")
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} isotopomer(s) not in label vocab "
              f"(using index 0): {sorted(missing_labels)}")
    if missing_tracers:
        print(f"WARNING: {len(missing_tracers)} base tracer(s) not in tracer vocab "
              f"(using index 0): {sorted(missing_tracers)}")

    return static_features, token_experiment_ids


# ---------------------------------------------------------------------------
# Tokenized dataset
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """
    Memory-efficient dataset for tokenized labeling experiments.

    Stores the fixed token ID embeddings (compound / label / tracer radial
    embeddings) once and assembles the full token tensor on-the-fly in
    __getitem__ by appending the per-sample fraction value.

    This avoids materialising the full (num_samples, num_tokens, token_dim)
    array, which can be large for deep learning datasets.
    """

    def __init__(self, static_features, fractions, targets):
        """
        Args:
            static_features: (num_tokens, token_dim-1) array of ID embeddings
            fractions: (num_samples, num_tokens) array of normalised fraction values
            targets: (num_samples, output_dim) array of target values
        """
        self.static = torch.tensor(static_features, dtype=torch.float32)   # (num_tokens, D-1)
        self.fractions = torch.tensor(fractions, dtype=torch.float32)       # (S, num_tokens)
        self.targets = torch.tensor(targets, dtype=torch.float32)            # (S, out_dim)

    def __len__(self):
        return len(self.fractions)

    def __getitem__(self, idx):
        fracs = self.fractions[idx].unsqueeze(1)             # (num_tokens, 1)
        tokens = torch.cat([self.static, fracs], dim=1)      # (num_tokens, token_dim)
        return tokens, self.targets[idx]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(labeling_df, flux_df, reversibility_df, vocabs,
                    train_indices=None, normalize=True, measured_fluxes=None,
                    norm_stats_input=None):
    """
    Prepare a TokenizedDataset from DataFrames using radial-embedding tokenization.

    Each row of labeling_df becomes one token: [compound_emb | label_emb |
    tracer_emb | fraction].  Optionally, measured bypass fluxes are prepended
    as extra tokens with zero ID embeddings and the normalised flux value in
    the fraction slot.

    Args:
        labeling_df: DataFrame with labeling data (MultiIndex: Tracer/Metabolite/Isotopomer).
        flux_df: DataFrame with flux data.
        reversibility_df: DataFrame with reversibility data.
        vocabs: dict from build_radial_vocabs.
        train_indices: Sample indices to use (None = all).
        normalize: Whether to log-transform and standardise labeling fractions.
        measured_fluxes: List of reaction IDs whose fluxes are prepended as
            bypass tokens, or None to disable.
        norm_stats_input: Pre-computed normalisation stats (for inference/val).

    Returns:
        dataset: TokenizedDataset
        norm_stats: dict with normalisation statistics
        token_experiment_ids: (num_tokens,) int64 tensor mapping tokens to
            experiment indices (-1 = bypass token, never masked)
    """
    if train_indices is None:
        train_indices = range(labeling_df.shape[1])
    train_indices = list(train_indices)
    num_samples = len(train_indices)

    # ------------------------------------------------------------------
    # 1. Labeling fractions: (num_samples, num_regular_tokens)
    # ------------------------------------------------------------------
    labeling_data = labeling_df.values[:, train_indices].T  # (num_samples, num_tokens)

    if normalize:
        eps = 1e-9
        labeling_data = np.log(np.clip(labeling_data, 0, None) + eps)

        if norm_stats_input is not None:
            mean = float(norm_stats_input['labeling_mean'])
            std = float(norm_stats_input['labeling_std'])
            if std < 1e-12:
                std = 1.0
            labeling_data = (labeling_data - mean) / std
            norm_stats = norm_stats_input.copy()
            norm_stats['labeling_std'] = std
        else:
            mean = labeling_data.mean()
            std = labeling_data.std()
            if std < 1e-12:
                std = 1.0
            labeling_data = (labeling_data - mean) / std
            norm_stats = {'labeling_mean': mean, 'labeling_std': std}

        labeling_data = np.nan_to_num(labeling_data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        norm_stats = {}

    # ------------------------------------------------------------------
    # 2. Token static features and experiment IDs
    # ------------------------------------------------------------------
    static_features, token_experiment_ids_np = build_token_static_features(labeling_df, vocabs)
    num_regular_tokens = len(labeling_df)
    token_dim = vocabs['token_dim']
    static_dim = token_dim - 1

    # ------------------------------------------------------------------
    # 3. Measured bypass flux tokens (optional)
    # ------------------------------------------------------------------
    bypass_fractions = None
    if measured_fluxes:
        measured_fluxes_data, feature_names = extract_measured_fluxes(
            flux_df, train_indices, reaction_ids=measured_fluxes
        )

        if measured_fluxes_data is not None and len(feature_names) > 0:
            if normalize:
                if norm_stats_input is not None and 'measured_fluxes_mean' in norm_stats_input:
                    bypass_mean = np.array(norm_stats_input['measured_fluxes_mean'])
                    bypass_std = np.array(norm_stats_input['measured_fluxes_std'])

                    if len(bypass_mean) != measured_fluxes_data.shape[1]:
                        raise ValueError(
                            f"Measured-fluxes normalisation stats dimension mismatch: "
                            f"expected {len(bypass_mean)} features but got "
                            f"{measured_fluxes_data.shape[1]}."
                        )
                    norm_stats['measured_fluxes_mean'] = norm_stats_input['measured_fluxes_mean']
                    norm_stats['measured_fluxes_std'] = norm_stats_input['measured_fluxes_std']
                    if 'measured_fluxes_feature_names' in norm_stats_input:
                        norm_stats['measured_fluxes_feature_names'] = \
                            norm_stats_input['measured_fluxes_feature_names']
                else:
                    bypass_mean = measured_fluxes_data.mean(axis=0)
                    bypass_std = measured_fluxes_data.std(axis=0)
                    bypass_std = np.where(bypass_std < 1e-6, 1.0, bypass_std)
                    norm_stats['measured_fluxes_mean'] = bypass_mean.tolist()
                    norm_stats['measured_fluxes_std'] = bypass_std.tolist()
                    norm_stats['measured_fluxes_feature_names'] = feature_names

                measured_fluxes_data = (measured_fluxes_data - bypass_mean) / bypass_std

            bypass_fractions = measured_fluxes_data  # (num_samples, num_bypass)
            print(f"Extracted {len(feature_names)} measured fluxes as bypass tokens: {feature_names}")

    # ------------------------------------------------------------------
    # 4. Assemble static features and experiment IDs across all tokens
    # ------------------------------------------------------------------
    if bypass_fractions is not None:
        num_bypass = bypass_fractions.shape[1]
        bypass_static = np.zeros((num_bypass, static_dim), dtype=np.float32)
        bypass_exp_ids = np.full(num_bypass, -1, dtype=np.int64)

        all_static = np.concatenate([bypass_static, static_features], axis=0)
        all_exp_ids = np.concatenate([bypass_exp_ids, token_experiment_ids_np])
        all_fractions = np.concatenate([bypass_fractions, labeling_data], axis=1)
    else:
        all_static = static_features
        all_exp_ids = token_experiment_ids_np
        all_fractions = labeling_data

    # ------------------------------------------------------------------
    # 5. Targets
    # ------------------------------------------------------------------
    flux_data = flux_df.values[:, train_indices].T
    rev_data = reversibility_df.values[:, train_indices].T
    targets = np.hstack([flux_data, rev_data])

    # ------------------------------------------------------------------
    # 6. Build dataset
    # ------------------------------------------------------------------
    dataset = TokenizedDataset(all_static, all_fractions, targets)
    token_experiment_ids = torch.tensor(all_exp_ids, dtype=torch.int64)

    return dataset, norm_stats, token_experiment_ids


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def apply_random_masking(tokens, token_experiment_ids, mask_prob=0.3, min_experiments=1):
    """
    Apply random experiment-group masking for missing-data augmentation.

    Randomly zeroes out all tokens belonging to a randomly selected subset of
    experiments.  Tokens with experiment id == -1 (bypass tokens) are never
    masked.

    Args:
        tokens: (batch_size, num_tokens, token_dim) tensor
        token_experiment_ids: (num_tokens,) int64 tensor (model buffer or plain tensor)
        mask_prob: Probability of masking each experiment
        min_experiments: Minimum number of experiments guaranteed to remain unmasked

    Returns:
        masked_tokens: (batch_size, num_tokens, token_dim) tensor
        mask: (batch_size, num_tokens) bool tensor (True = valid / unmasked)
    """
    batch_size, num_tokens, token_dim = tokens.shape
    device = tokens.device
    exp_ids = token_experiment_ids.to(device)

    # Identify valid experiment indices (exclude bypass tokens with id == -1)
    valid_mask = exp_ids >= 0
    unique_exp_ids = exp_ids[valid_mask].unique()
    num_experiments = unique_exp_ids.shape[0]

    # Initialise all tokens as valid
    token_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)

    if num_experiments == 0 or mask_prob <= 0.0:
        return tokens, token_mask

    # Per-sample experiment-level mask: True = keep
    # shape: (batch_size, num_experiments)
    exp_keep = torch.rand(batch_size, num_experiments, device=device) >= mask_prob

    # Ensure min_experiments are kept
    for b in range(batch_size):
        num_kept = exp_keep[b].sum().item()
        if num_kept < min_experiments:
            # Randomly enable some masked experiments
            masked_idxs = torch.where(~exp_keep[b])[0]
            to_enable = masked_idxs[
                torch.randperm(len(masked_idxs), device=device)[
                    :min_experiments - int(num_kept)
                ]
            ]
            exp_keep[b, to_enable] = True

    # Expand experiment mask to token level
    for exp_i in range(num_experiments):
        exp_id = unique_exp_ids[exp_i]
        tok_idxs = (exp_ids == exp_id)                          # (num_tokens,)
        # For each sample: if this experiment is masked, set token_mask to False
        token_mask[:, tok_idxs] &= exp_keep[:, exp_i].unsqueeze(1)

    masked_tokens = tokens * token_mask.unsqueeze(-1).float()
    return masked_tokens, token_mask


def extract_mask_from_data(tokens, token_experiment_ids, threshold=1e-6):
    """
    Infer an experiment-group mask from zero-valued fractions in the token tensor.

    For each experiment group, if all token fraction values (last column) are
    below threshold the entire group is considered masked.  Bypass tokens
    (experiment id == -1) are always considered valid.

    Args:
        tokens: (batch_size, num_tokens, token_dim) tensor
        token_experiment_ids: (num_tokens,) int64 tensor
        threshold: Absolute value threshold for detecting missing data

    Returns:
        mask: (batch_size, num_tokens) bool tensor (True = valid)
    """
    batch_size, num_tokens, _ = tokens.shape
    device = tokens.device
    exp_ids = token_experiment_ids.to(device)

    token_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)

    valid_mask = exp_ids >= 0
    unique_exp_ids = exp_ids[valid_mask].unique()

    fractions = tokens[:, :, -1]  # (batch, num_tokens)

    for exp_id in unique_exp_ids:
        tok_idxs = (exp_ids == exp_id)                              # (num_tokens,)
        exp_fracs = fractions[:, tok_idxs]                          # (batch, n)
        exp_has_data = exp_fracs.abs().max(dim=1).values > threshold  # (batch,)
        token_mask[:, tok_idxs] &= exp_has_data.unsqueeze(1)


# ---------------------------------------------------------------------------
# ID-based token features (for FluxDecoder)
# ---------------------------------------------------------------------------

def build_id_token_features(labeling_df, vocabs):
    """
    Build integer ID arrays for FluxDecoder's internal embedding lookups.

    Unlike build_token_static_features (which precomputes float embeddings),
    this returns integer indices so the model performs embedding lookups at
    runtime.  Multi-tracer combinations (e.g. "A|B") are handled by summing
    base tracer embeddings and treating each unique combination string as a
    single vocabulary entry.

    Args:
        labeling_df: DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer).
        vocabs: dict from build_radial_vocabs.

    Returns:
        compound_ids:         (num_tokens,) int64
        label_ids:            (num_tokens,) int64
        tracer_ids:           (num_tokens,) int64
        token_experiment_ids: (num_tokens,) int64  (same index as tracer_ids)
        tracer_combo_vocab:   sorted list of unique tracer strings
        tracer_combo_emb_table: (n_combos, d_tracer) float32
    """
    compound_vocab       = vocabs['compound_vocab']
    label_vocab          = vocabs['label_vocab']
    tracer_base_vocab    = vocabs['tracer_vocab']
    tracer_base_emb_table = vocabs['tracer_emb_table']

    compound_to_idx      = {c: i for i, c in enumerate(compound_vocab)}
    label_to_idx         = {l: i for i, l in enumerate(label_vocab)}
    tracer_base_to_idx   = {t: i for i, t in enumerate(tracer_base_vocab)}

    tracers     = labeling_df.index.get_level_values('Tracer')
    metabolites = labeling_df.index.get_level_values('Metabolite')
    isotopomers = labeling_df.index.get_level_values('Isotopomer')

    # Combined tracer vocab: one entry per unique tracer string (incl. "A|B")
    tracer_combo_vocab    = sorted(tracers.unique().tolist())
    tracer_combo_to_idx   = {t: i for i, t in enumerate(tracer_combo_vocab)}

    # Precompute combined embeddings by summing base tracer embeddings
    tracer_emb_dim = tracer_base_emb_table.shape[1]
    tracer_combo_emb_table = np.zeros(
        (len(tracer_combo_vocab), tracer_emb_dim), dtype=np.float32
    )
    for combo_str, combo_idx in tracer_combo_to_idx.items():
        for bt in (t.strip() for t in combo_str.split('|')):
            bt_idx = tracer_base_to_idx.get(bt, 0)
            tracer_combo_emb_table[combo_idx] += tracer_base_emb_table[bt_idx]

    num_tokens           = len(labeling_df)
    compound_ids         = np.zeros(num_tokens, dtype=np.int64)
    label_ids            = np.zeros(num_tokens, dtype=np.int64)
    tracer_ids           = np.zeros(num_tokens, dtype=np.int64)
    token_experiment_ids = np.zeros(num_tokens, dtype=np.int64)

    missing_compounds = set()
    missing_labels    = set()

    for i, (tracer_str, met_id, isotopomer) in enumerate(
            zip(tracers, metabolites, isotopomers)):
        c_idx = compound_to_idx.get(met_id)
        if c_idx is None:
            missing_compounds.add(met_id)
            c_idx = 0
        compound_ids[i] = c_idx

        l_idx = label_to_idx.get(isotopomer)
        if l_idx is None:
            missing_labels.add(isotopomer)
            l_idx = 0
        label_ids[i] = l_idx

        combo_idx = tracer_combo_to_idx.get(tracer_str, 0)
        tracer_ids[i]           = combo_idx
        token_experiment_ids[i] = combo_idx  # experiment ≡ tracer combination

    if missing_compounds:
        print(f"WARNING: {len(missing_compounds)} metabolite(s) not in COBRA model vocab "
              f"(using index 0): {sorted(missing_compounds)}")
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} isotopomer(s) not in label vocab "
              f"(using index 0): {sorted(missing_labels)}")

    return (
        compound_ids, label_ids, tracer_ids, token_experiment_ids,
        tracer_combo_vocab, tracer_combo_emb_table,
    )


class IDTokenizedDataset(Dataset):
    """
    Dataset for FluxDecoder: stores integer IDs + fractions rather than
    precomputed float embeddings.

    __getitem__ returns:
        ((compound_ids, label_ids, tracer_ids, fracs), targets)
        where compound/label/tracer_ids are (num_tokens,) int64 tensors shared
        across all samples, and fracs is (num_tokens, 1) per-sample float32.
    """

    def __init__(self, compound_ids, label_ids, tracer_ids, fractions, targets):
        """
        Args:
            compound_ids: (num_tokens,) int64
            label_ids:    (num_tokens,) int64
            tracer_ids:   (num_tokens,) int64
            fractions:    (num_samples, num_tokens) float32
            targets:      (num_samples, out_dim) float32
        """
        self.compound_ids = torch.tensor(compound_ids, dtype=torch.long)
        self.label_ids    = torch.tensor(label_ids,    dtype=torch.long)
        self.tracer_ids   = torch.tensor(tracer_ids,   dtype=torch.long)
        self.fractions    = torch.tensor(fractions,    dtype=torch.float32)
        self.targets      = torch.tensor(targets,      dtype=torch.float32)

    def __len__(self):
        return len(self.fractions)

    def __getitem__(self, idx):
        fracs = self.fractions[idx].unsqueeze(1)  # (num_tokens, 1)
        return (
            self.compound_ids,  # (num_tokens,)
            self.label_ids,     # (num_tokens,)
            self.tracer_ids,    # (num_tokens,)
            fracs,              # (num_tokens, 1)
        ), self.targets[idx]


def prepare_id_dataset(labeling_df, flux_df, reversibility_df, vocabs,
                       train_indices=None, normalize=True, measured_fluxes=None,
                       norm_stats_input=None):
    """
    Prepare an IDTokenizedDataset for use with FluxDecoder.

    Delegates all normalization and bypass-token handling to prepare_dataset,
    then extracts integer ID arrays so the model can perform embedding lookups
    at runtime rather than receiving precomputed float embeddings.

    Args:
        Same as prepare_dataset.

    Returns:
        dataset:               IDTokenizedDataset
        norm_stats:            dict (same schema as prepare_dataset)
        token_experiment_ids:  (num_tokens,) int64 tensor
        tracer_combo_vocab:    sorted list of unique tracer strings
        tracer_combo_emb_table:(n_combos, d_tracer) float32 array
    """
    float_dataset, norm_stats, token_experiment_ids = prepare_dataset(
        labeling_df, flux_df, reversibility_df, vocabs,
        train_indices=train_indices,
        normalize=normalize,
        measured_fluxes=measured_fluxes,
        norm_stats_input=norm_stats_input,
    )

    compound_ids, label_ids, tracer_ids, _, tracer_combo_vocab, tracer_combo_emb_table = \
        build_id_token_features(labeling_df, vocabs)

    # Prepend zero IDs for bypass tokens (prepended first in prepare_dataset)
    n_total  = float_dataset.static.shape[0]
    n_bypass = n_total - len(labeling_df)
    if n_bypass > 0:
        z            = np.zeros(n_bypass, dtype=np.int64)
        compound_ids = np.concatenate([z, compound_ids])
        label_ids    = np.concatenate([z, label_ids])
        tracer_ids   = np.concatenate([z, tracer_ids])

    fractions = float_dataset.fractions.numpy()  # (num_samples, n_total)
    targets   = float_dataset.targets.numpy()

    dataset = IDTokenizedDataset(compound_ids, label_ids, tracer_ids, fractions, targets)
    return dataset, norm_stats, token_experiment_ids, tracer_combo_vocab, tracer_combo_emb_table


def generate_experiment_mask(token_experiment_ids, batch_size, device,
                              mask_prob=0.3, min_experiments=1):
    """
    Generate a random experiment-group mask for missing-data augmentation.

    Equivalent to apply_random_masking but returns only the boolean mask
    without modifying any input tensor — suitable for FluxDecoder where
    inputs are integer IDs that cannot be zeroed out meaningfully.

    Args:
        token_experiment_ids: (num_tokens,) int64 tensor; -1 = bypass token
        batch_size:           int
        device:               torch device
        mask_prob:            probability of masking any given experiment
        min_experiments:      minimum experiments kept unmasked per sample

    Returns:
        mask: (batch_size, num_tokens) bool, True = valid / unmasked
    """
    exp_ids       = token_experiment_ids.to(device)
    num_tokens    = exp_ids.shape[0]
    valid_exp     = exp_ids >= 0
    unique_exp_ids = exp_ids[valid_exp].unique()
    num_experiments = unique_exp_ids.shape[0]

    token_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device)

    if num_experiments == 0 or mask_prob <= 0.0:
        return token_mask

    exp_keep = torch.rand(batch_size, num_experiments, device=device) >= mask_prob

    for b in range(batch_size):
        num_kept = exp_keep[b].sum().item()
        if num_kept < min_experiments:
            masked_idxs = torch.where(~exp_keep[b])[0]
            to_enable   = masked_idxs[
                torch.randperm(len(masked_idxs), device=device)[
                    :min_experiments - int(num_kept)
                ]
            ]
            exp_keep[b, to_enable] = True

    for exp_i in range(num_experiments):
        exp_id    = unique_exp_ids[exp_i]
        tok_idxs  = (exp_ids == exp_id)
        token_mask[:, tok_idxs] &= exp_keep[:, exp_i].unsqueeze(1)

    return token_mask

    return token_mask
