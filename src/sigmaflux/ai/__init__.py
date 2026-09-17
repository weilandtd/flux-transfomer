"""
AI module for metabolic flux prediction from labeling data.
"""

# Model
from .model import CrossAttention, FluxDecoderConfig, FluxDecoder
from .flux_projection import FluxProjection, active_set_sizes

# Training
from .train import (
    train_model,
    evaluate_validation_correlations,
    var_norm_mse_loss,
    per_reaction_r2,
)

# Inference (experimental MID data -> fluxes)
from .inference import ModelSpec, SigmaFlux, EnsembleSigmaFlux, load_predictor

# Packaged models (.sfx)
from .export import (SFX_SUFFIX, export_model, inspect_model, load_model,
                     read_cobra_model, read_stoichiometry, validate_model)

# Checkpoint loading (training data -> model + the dataset it was scored on)
from .checkpoint import (
    ALL,
    VAL,
    EvalContext,
    build_flux_decoder,
    load_eval_context,
    resolve_device,
)

# Prediction
from .predict import (
    predict_fluxes,
    create_ensemble,
    ensemble_predict,
    train_ensemble
)

# Utilities (null-space, token scoring, file conversion)
from .utils import compute_sparse_nullspace, load_fisher_token_scores

# Data utilities
from .data import (
    load_data,
    load_data_batches,
    find_data_batches,
    find_checkpoint,
    split_sample_indices,
    extract_measured_fluxes,
    create_tracer_mapping,
    load_stoichiometric_matrix,
    radial_embedding_table,
    build_radial_vocabs,
    radial_vocabs_from_lists,
    build_token_static_features,
    build_id_token_features,
    TokenizedDataset,
    IDTokenizedDataset,
    prepare_dataset,
    prepare_id_dataset,
)

__all__ = [
    # Utilities
    'compute_sparse_nullspace',
    'load_fisher_token_scores',
    # Model
    'CrossAttention',
    'FluxDecoderConfig',
    'FluxDecoder',
    'FluxProjection',
    'active_set_sizes',
    # Training
    'train_model',
    'evaluate_validation_correlations',
    'var_norm_mse_loss',
    'per_reaction_r2',
    # Inference
    'ModelSpec',
    'SigmaFlux',
    'EnsembleSigmaFlux',
    'load_predictor',
    # Packaged models
    'SFX_SUFFIX',
    'export_model',
    'inspect_model',
    'load_model',
    'read_cobra_model',
    'read_stoichiometry',
    'validate_model',
    # Checkpoint loading
    'ALL',
    'VAL',
    'EvalContext',
    'build_flux_decoder',
    'load_eval_context',
    'resolve_device',
    # Prediction
    'predict_fluxes',
    'create_ensemble',
    'ensemble_predict',
    'train_ensemble',
    # Data
    'load_data',
    'load_data_batches',
    'find_data_batches',
    'find_checkpoint',
    'split_sample_indices',
    'extract_measured_fluxes',
    'create_tracer_mapping',
    'load_stoichiometric_matrix',
    'radial_embedding_table',
    'build_radial_vocabs',
    'radial_vocabs_from_lists',
    'build_token_static_features',
    'build_id_token_features',
    'TokenizedDataset',
    'IDTokenizedDataset',
    'prepare_dataset',
    'prepare_id_dataset',
]
