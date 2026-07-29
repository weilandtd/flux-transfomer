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
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from cobra.io.json import load_json_model
from cobra.util.array import create_stoichiometric_matrix


def _read_hdf(filepath, key):
    """Read a pandas HDF5 file using h5py, bypassing the broken tables package."""
    with h5py.File(filepath, 'r') as f:
        g = f[key]
        columns = g['axis0'][:]
        values  = g['block0_values'][:]  # (n_rows, n_cols)

        if 'axis1_level0' in g:
            # MultiIndex (labeling data has 3 levels: Tracer, Metabolite, Isotopomer)
            n_levels = sum(1 for k in g.keys() if k.startswith('axis1_level'))
            arrays, names = [], []
            for i in range(n_levels):
                ds   = g[f'axis1_level{i}']
                lvls = [x.decode() for x in ds[:]]
                lbs  = g[f'axis1_label{i}'][:]
                arrays.append([lvls[lb] for lb in lbs])
                names.append(ds.attrs.get('name', b'').decode())
            index = pd.MultiIndex.from_arrays(arrays, names=names)
        else:
            index = pd.Index([x.decode() for x in g['axis1'][:]])

        return pd.DataFrame(values, index=index, columns=columns)


# ── Measured ("bypass") fluxes ───────────────────────────────────────────────
# Fluxes measured by orthogonal methods and fed straight to the transformer,
# skipping the labeling pathway. Which reactions those are is a property of the
# organism and the experiment, so it is described by a JSON config supplied by
# the caller rather than hard-coded here.
#
# Config schema — every top-level object may declare:
#     "<group name>": {
#         "reaction_ids": ["RXN_A", "RXN_B"],           # direct flux features
#         "reactions":    {"RXN_C": "description"},      # ... or as a mapping
#         "derived": {                                   # optional derived features
#             "NAME": {"reaction": "RXN_C", "sign": -1},   # signed flux
#             "OTHER": {"ratio": ["NAME_A", "NAME_B"]}     # ratio of two derived
#         }
#     }
# A derived feature's source reactions are read but not emitted as features
# themselves unless they also appear in reaction_ids.


def load_measured_fluxes_config(config_path):
    """Load a measured-flux config from JSON.

    There is no default path: the reactions that can be measured orthogonally
    depend on the organism and the experimental setup.
    """
    if config_path is None:
        raise ValueError(
            "load_measured_fluxes_config() requires an explicit config_path; "
            "measured-flux definitions are organism-specific."
        )
    with open(config_path, 'r') as f:
        return json.load(f)


def _group_reaction_ids(group):
    """Reaction ids declared by one config group, from either supported key."""
    ids = list(group.get('reaction_ids', []))
    ids += [r for r in group.get('reactions', {}) if r not in ids]
    return ids


def get_measured_fluxes_reaction_ids(spec=None, config_path=None):
    """Resolve a measured-flux *spec* into reactions and derived features.

    Args:
        spec: one of
            - ``None`` / ``False``: no measured-flux features
            - ``list``: explicit reaction ids, no derived features
            - ``"all"``: every group in the config
            - ``dict``: ``{"groups": [...], "derived": bool}`` selecting groups
              by name (``derived`` defaults to True)
        config_path: JSON config, required unless *spec* is a list or empty.

    Returns:
        reaction_ids: reaction ids whose flux must be read
        derived_specs: ``{name: definition}`` for computed features
        feature_names: feature order in the emitted array
    """
    if spec is None or spec is False:
        return [], {}, []

    if isinstance(spec, (list, tuple)):
        return list(spec), {}, list(spec)

    config = load_measured_fluxes_config(config_path)

    if spec == "all":
        group_names, want_derived = None, True
    elif isinstance(spec, dict):
        group_names = spec.get("groups")
        want_derived = spec.get("derived", True)
    else:
        raise ValueError(
            f"Unsupported measured-flux spec {spec!r}. Expected None, a list of "
            'reaction ids, "all", or {"groups": [...], "derived": bool}.'
        )

    reaction_ids, derived_specs, feature_names = [], {}, []
    for name, group in config.items():
        if not isinstance(group, dict):
            continue                                  # description/notes strings
        if group_names is not None and name not in group_names:
            continue
        if 'reaction_ids' not in group and 'reactions' not in group and 'derived' not in group:
            continue                                  # e.g. a summary block

        derived = group.get('derived', {}) if want_derived else {}
        for rxn_id in _group_reaction_ids(group):
            if rxn_id not in reaction_ids:
                reaction_ids.append(rxn_id)
            # A reaction that only exists to feed a derived feature is not
            # itself emitted (that is what "reactions" vs "reaction_ids" means).
            if rxn_id in group.get('reaction_ids', []) and rxn_id not in feature_names:
                feature_names.append(rxn_id)

        for feat_name, definition in derived.items():
            derived_specs[feat_name] = definition
            if feat_name not in feature_names:
                feature_names.append(feat_name)

    return reaction_ids, derived_specs, feature_names


def _normalize_derived(definition, name):
    """Accept both the descriptive string form and the explicit dict form."""
    if isinstance(definition, dict):
        return definition
    # Legacy/descriptive configs carry a human-readable string; infer nothing
    # from it and let the caller know it cannot be computed.
    raise ValueError(
        f"Derived feature {name!r} is described by a string, not a definition. "
        'Use {"reaction": "<id>", "sign": 1} or {"ratio": ["<a>", "<b>"]}.'
    )


def extract_measured_fluxes(flux_df, train_indices, spec=None, config_path=None,
                            reaction_ids=None):
    """Extract measured-flux features from simulated flux data.

    Args:
        flux_df: DataFrame of fluxes (reactions x samples)
        train_indices: sample indices to extract
        spec: measured-flux spec, see :func:`get_measured_fluxes_reaction_ids`.
            Ignored when *reaction_ids* is given.
        config_path: JSON config path. Ignored when *reaction_ids* is given.
        reaction_ids: explicit reaction ids; skips spec resolution entirely.

    Returns:
        measured_fluxes_data: ``(n_samples, n_features)`` array, or None
        feature_names: names matching the array columns
    """
    if reaction_ids is not None:
        direct_rxn_ids = list(reaction_ids)
        derived_specs = {}
        feature_names = list(reaction_ids)
    else:
        direct_rxn_ids, derived_specs, feature_names = \
            get_measured_fluxes_reaction_ids(spec, config_path)
        if not direct_rxn_ids and not derived_specs:
            return None, []

    flux_data_subset = flux_df.values[:, train_indices].T   # (n_samples, n_rxns)
    reaction_list = flux_df.index.tolist()

    def _flux(rxn_id):
        if rxn_id not in reaction_list:
            return None
        return flux_data_subset[:, reaction_list.index(rxn_id)]

    # ── Derived features first: they may be referenced by later ratios ────────
    derived_values = {}
    for name, definition in derived_specs.items():
        definition = _normalize_derived(definition, name)
        if 'reaction' in definition:
            values = _flux(definition['reaction'])
            if values is None:
                print(f"WARNING: reaction {definition['reaction']} for derived "
                      f"feature {name} not found in flux data; skipping {name}")
                continue
            derived_values[name] = float(definition.get('sign', 1)) * values
        elif 'ratio' in definition:
            num_name, den_name = definition['ratio']
            num = derived_values.get(num_name, _flux(num_name))
            den = derived_values.get(den_name, _flux(den_name))
            if num is None or den is None:
                print(f"WARNING: operands for derived feature {name} unavailable; skipping")
                continue
            # np.where would evaluate num/den eagerly and warn on zeros.
            safe = np.abs(den) > 1e-6
            derived_values[name] = np.divide(
                num, den, out=np.zeros_like(np.asarray(num, dtype=float)), where=safe)
        else:
            raise ValueError(f"Derived feature {name!r} has no 'reaction' or 'ratio' key.")

    # ── Direct fluxes, minus any reaction consumed only by a derived feature ──
    derived_sources = {
        d['reaction'] for d in derived_specs.values()
        if isinstance(d, dict) and 'reaction' in d
    }
    missing_rxns = []
    direct_values = {}
    for rxn_id in direct_rxn_ids:
        if rxn_id in derived_sources and rxn_id not in feature_names:
            continue
        values = _flux(rxn_id)
        if values is None:
            missing_rxns.append(rxn_id)
            continue
        direct_values[rxn_id] = values

    if missing_rxns:
        print(f"WARNING: {len(missing_rxns)} measured-flux reactions not found in flux data:")
        for rxn in missing_rxns[:5]:
            print(f"         - {rxn}")
        if len(missing_rxns) > 5:
            print(f"         ... and {len(missing_rxns) - 5} more")

    # Emit in declared feature order, dropping whatever could not be computed.
    columns, kept_names = [], []
    for name in feature_names:
        if name in derived_values:
            columns.append(derived_values[name])
            kept_names.append(name)
        elif name in direct_values:
            columns.append(direct_values[name])
            kept_names.append(name)

    if not columns:
        return None, []

    return np.column_stack(columns), kept_names


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
            return _read_hdf(single_file, key=key)
        else:
            raise FileNotFoundError(f"No batch files found for {base_filename} in {base_dir}")

    dfs = []
    total_samples = 0

    for start_idx, end_idx, filepath in batch_files:
        df = _read_hdf(filepath, key=key)
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


def find_checkpoint(data_dir):
    """Return the path to the single trained .pt checkpoint in ``data_dir``."""
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".pt"):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No .pt checkpoint found in {data_dir}")


def split_sample_indices(split, checkpoint_path=None, n_total=None,
                         val_fraction=None, seed=None):
    """Sorted HDF5 column indices for a benchmark split: ``'val'``, ``'train'``, or ``'all'``.

    The validation set is the one withheld from AI training, so the trained checkpoint's
    stored ``val_indices`` / ``train_indices`` are the single source of truth. When a
    checkpoint is given and carries them, they are used directly; otherwise the split is
    recomputed from (n_total, val_fraction, seed) — identical to
    ``train_ai_model.split_indices`` (shuffle ``arange(n_total)`` with ``default_rng(seed)``,
    hold out the last ``val_fraction``). The two paths produce the same set for matching
    parameters. Indices are returned sorted so ``--start-ix`` tiles the split deterministically.
    """
    if split == "all":
        if n_total is None:
            raise ValueError("split_sample_indices('all', ...) requires n_total")
        return np.arange(n_total)
    if split not in ("val", "train"):
        raise ValueError(f"split must be 'val', 'train', or 'all', got {split!r}")

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        key = "val_indices" if split == "val" else "train_indices"
        if isinstance(ckpt, dict) and ckpt.get(key) is not None:
            return np.sort(np.asarray(ckpt[key]))

    if n_total is None or val_fraction is None or seed is None:
        raise ValueError("split_sample_indices: provide a checkpoint carrying the split, "
                         "or all of n_total/val_fraction/seed to recompute it")
    indices = np.arange(n_total)
    np.random.default_rng(seed).shuffle(indices)
    split_pt = int((1.0 - val_fraction) * n_total)
    return np.sort(indices[split_pt:] if split == "val" else indices[:split_pt])


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
    # Label vocab: unique isotopomers in data
    label_vocab = sorted(labeling_df.index.get_level_values('Isotopomer').unique().tolist())
    # Tracer vocab: base tracers
    tracer_vocab = sorted(base_tracer_names)

    return radial_vocabs_from_lists(
        compound_vocab, label_vocab, tracer_vocab,
        h_compound=h_compound, h_label=h_label, h_tracer=h_tracer,
    )


def radial_vocabs_from_lists(compound_vocab, label_vocab, tracer_vocab,
                             h_compound=None, h_label=None, h_tracer=None,
                             verbose=True):
    """
    Build the radial-embedding tables from three already-derived vocabularies.

    This is the half of :func:`build_radial_vocabs` that needs no COBRA model and
    no labeling DataFrame — only the three sorted string lists. An exported model
    stores those lists, so it can rebuild identical embedding tables without the
    network JSON or the training data.

    Args:
        compound_vocab: Sorted metabolite IDs (the full network, not just measured).
        label_vocab:    Sorted isotopomer labels.
        tracer_vocab:   Sorted base tracer names.
        h_compound/h_label/h_tracer: Harmonics per vocabulary (default: ceil(log2(N))).
        verbose:        Print the vocabulary summary.

    Returns:
        vocabs: dict with keys:
            compound_vocab, label_vocab, tracer_vocab  — sorted string lists
            h_compound, h_label, h_tracer              — int H values used
            token_dim                                   — total token dimension
            compound_emb_table                         — (N_c, 2*H_c) float32 array
            label_emb_table                            — (N_l, 2*H_l) float32 array
            tracer_emb_table                           — (N_t, 2*H_t) float32 array
    """
    n_compound, n_label, n_tracer = len(compound_vocab), len(label_vocab), len(tracer_vocab)
    if h_compound is None:
        h_compound = max(1, math.ceil(math.log2(n_compound)))
    if h_label is None:
        h_label = max(1, math.ceil(math.log2(n_label)))
    if h_tracer is None:
        h_tracer = max(1, math.ceil(math.log2(n_tracer)))

    token_dim = 2 * h_compound + 2 * h_label + 2 * h_tracer + 1  # +1 for fraction

    if verbose:
        print(f"\nRadial embedding vocabularies:")
        print(f"  Compounds: {n_compound} → H={h_compound}, emb_dim={2*h_compound}")
        print(f"  Labels:    {n_label}   → H={h_label},  emb_dim={2*h_label}")
        print(f"  Tracers:   {n_tracer}  → H={h_tracer},  emb_dim={2*h_tracer}")
        print(f"  Token dim: {token_dim} (incl. fraction scalar)")

    return {
        'compound_vocab': list(compound_vocab),
        'label_vocab': list(label_vocab),
        'tracer_vocab': list(tracer_vocab),
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
                    train_indices=None, measured_fluxes=None,
                    norm_stats_input=None, normalize=False, normalize_mids=False):
    """
    Prepare a TokenizedDataset from DataFrames using radial-embedding tokenization.

    Each row of labeling_df becomes one token: [compound_emb | label_emb |
    tracer_emb | fraction].  Measured flux tokens are built separately inside
    FluxDecoder.forward() using reaction embeddings — they are NOT prepended
    here.

    Args:
        labeling_df: DataFrame with labeling data (MultiIndex: Tracer/Metabolite/Isotopomer).
        flux_df: DataFrame with flux data.
        reversibility_df: DataFrame with reversibility data.
        vocabs: dict from build_radial_vocabs.
        train_indices: Sample indices to use (None = all).
        normalize: Whether to standardise flux/reversibility targets.
        measured_fluxes: Accepted for API compatibility; no longer used to build
            bypass tokens. Measured flux values are present in the targets tensor
            and passed to forward() at training time.
        norm_stats_input: Pre-computed normalisation stats (for inference/val).

    Returns:
        dataset: TokenizedDataset
        norm_stats: dict with normalisation statistics
        token_experiment_ids: (num_tokens,) int64 tensor mapping tokens to
            experiment indices
    """
    if train_indices is None:
        train_indices = range(labeling_df.shape[1])
    train_indices = list(train_indices)
    num_samples = len(train_indices)

    # ------------------------------------------------------------------
    # 1. Labeling fractions: (num_samples, num_regular_tokens)
    # ------------------------------------------------------------------
    labeling_data = labeling_df.values[:, train_indices].T  # (num_samples, num_tokens)

    norm_stats = {}

    # 1b. Optionally per-token z-score the labeling (MID) fractions so they match the
    # unit-variance scale of the z-scored measured-flux inputs (fractions are otherwise
    # fed raw ~[0,1], std ~0.04, ~24x weaker than flux inputs). Applied if explicitly
    # requested (training) OR if the checkpoint's stats carry mid_mean (val/inference),
    # so old checkpoints (no mid stats) keep the raw behavior.
    _apply_mid = normalize_mids or (norm_stats_input is not None and 'mid_mean' in norm_stats_input)
    if _apply_mid:
        if norm_stats_input is not None and 'mid_mean' in norm_stats_input:
            mid_mean = np.asarray(norm_stats_input['mid_mean'], dtype=np.float32)
            mid_std  = np.asarray(norm_stats_input['mid_std'],  dtype=np.float32)
        else:
            mid_mean = labeling_data.mean(axis=0).astype(np.float32)
            mid_std  = np.where(labeling_data.std(axis=0) < 1e-6, 1.0,
                                labeling_data.std(axis=0)).astype(np.float32)
        labeling_data = (labeling_data - mid_mean) / mid_std
        norm_stats['mid_mean'] = mid_mean.tolist()
        norm_stats['mid_std']  = mid_std.tolist()
        print(f"MID fractions per-token z-scored: input std now "
              f"~{labeling_data.std(axis=0).mean():.2f} (raw was ~0.04)")

    # ------------------------------------------------------------------
    # 2. Token static features and experiment IDs
    # ------------------------------------------------------------------
    static_features, token_experiment_ids_np = build_token_static_features(labeling_df, vocabs)
    num_regular_tokens = len(labeling_df)
    token_dim = vocabs['token_dim']
    static_dim = token_dim - 1

    # ------------------------------------------------------------------
    # 3. Targets — normalize flux, then slice bypass tokens from it
    # ------------------------------------------------------------------
    flux_data = flux_df.values[:, train_indices].T        # (num_samples, n_fluxes)
    rev_data  = reversibility_df.values[:, train_indices].T

    if normalize:
        if norm_stats_input is None:
            # Compute from current slice (used when caller doesn't pre-compute stats)
            norm_stats_input = {
                'flux_mean': flux_data.mean(axis=0).tolist(),
                'flux_std':  np.where(flux_data.std(axis=0) < 1e-6, 1.0,
                                      flux_data.std(axis=0)).tolist(),
                'rev_mean':  rev_data.mean(axis=0).tolist(),
                'rev_std':   np.where(rev_data.std(axis=0) < 1e-6, 1.0,
                                      rev_data.std(axis=0)).tolist(),
            }
        flux_mean = np.array(norm_stats_input['flux_mean'], dtype=np.float32)
        flux_std  = np.array(norm_stats_input['flux_std'],  dtype=np.float32)
        rev_mean  = np.array(norm_stats_input['rev_mean'],  dtype=np.float32)
        rev_std   = np.array(norm_stats_input['rev_std'],   dtype=np.float32)
        norm_stats['flux_mean'] = norm_stats_input['flux_mean']
        norm_stats['flux_std']  = norm_stats_input['flux_std']
        norm_stats['rev_mean']  = norm_stats_input['rev_mean']
        norm_stats['rev_std']   = norm_stats_input['rev_std']
        flux_data = (flux_data - flux_mean) / flux_std
        rev_data  = (rev_data  - rev_mean)  / rev_std
        print(f"Flux targets normalized: "
              f"mean range=[{flux_data.mean(axis=0).min():.4f}, {flux_data.mean(axis=0).max():.4f}], "
              f"std range=[{flux_data.std(axis=0).min():.4f}, {flux_data.std(axis=0).max():.4f}]")
        print(f"Rev targets normalized:  "
              f"mean range=[{rev_data.mean(axis=0).min():.4f}, {rev_data.mean(axis=0).max():.4f}], "
              f"std range=[{rev_data.std(axis=0).min():.4f}, {rev_data.std(axis=0).max():.4f}]")

    # ------------------------------------------------------------------
    # 4. Assemble static features and experiment IDs (labeling tokens only)
    # ------------------------------------------------------------------
    if measured_fluxes:
        print(f"Extracted {len(measured_fluxes)} measured fluxes as bypass tokens: {measured_fluxes}")

    targets = np.hstack([flux_data, rev_data])

    # ------------------------------------------------------------------
    # 5. Build dataset
    # ------------------------------------------------------------------
    dataset = TokenizedDataset(static_features, labeling_data, targets)
    token_experiment_ids = torch.tensor(token_experiment_ids_np, dtype=torch.int64)

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
                       train_indices=None, measured_fluxes=None,
                       norm_stats_input=None, normalize=False, normalize_mids=False):
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
        measured_fluxes=measured_fluxes,
        norm_stats_input=norm_stats_input,
        normalize=normalize,
        normalize_mids=normalize_mids,
    )

    compound_ids, label_ids, tracer_ids, _, tracer_combo_vocab, tracer_combo_emb_table = \
        build_id_token_features(labeling_df, vocabs)

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


def generate_metabolite_mask(compound_ids, batch_size, device, mask_prob=0.15):
    """
    Randomly mask entire metabolites (all isotopomers of a compound) per sample.

    Finer-grained complement to generate_experiment_mask: while that function
    drops whole tracer experiments, this one drops individual metabolites within
    experiments, simulating metabolites that were not detected in the MS run.

    Args:
        compound_ids: (N,) int64 tensor — compound index for each token (shared across batch)
        batch_size:   int
        device:       torch device
        mask_prob:    probability of masking each unique metabolite per sample

    Returns:
        mask: (batch_size, N) bool, True = valid / unmasked
    """
    compound_ids = compound_ids.to(device)
    unique_comps = compound_ids.unique()
    n_comps      = unique_comps.shape[0]
    n_tokens     = compound_ids.shape[0]

    token_mask = torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device)
    if mask_prob <= 0.0 or n_comps == 0:
        return token_mask

    # Per-sample, independently decide which metabolites to keep
    comp_keep = torch.rand(batch_size, n_comps, device=device) >= mask_prob

    # Ensure at least one metabolite is always unmasked per sample
    all_masked = ~comp_keep.any(dim=1)
    if all_masked.any():
        rescue = torch.randint(n_comps, (int(all_masked.sum().item()),), device=device)
        comp_keep[all_masked, rescue] = True

    for i, comp_id in enumerate(unique_comps):
        tok_idxs = (compound_ids == comp_id)
        token_mask[:, tok_idxs] &= comp_keep[:, i].unsqueeze(1)

    return token_mask
