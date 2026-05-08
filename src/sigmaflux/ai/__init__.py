"""
AI module for metabolic flux prediction from labeling data.
"""

# Model
from .model import FBAProProjection, CrossAttention, FluxDecoderConfig, FluxDecoder

# Training
from .train import train_model, var_norm_mse_loss

# Prediction
from .predict import (
    predict_fluxes,
    create_ensemble,
    ensemble_predict,
    train_ensemble
)

# Data utilities
from .data import (
    load_data,
    load_data_batches,
    find_data_batches,
    create_tracer_mapping,
    load_stoichiometric_matrix,
    radial_embedding_table,
    build_radial_vocabs,
    build_token_static_features,
    build_id_token_features,
    TokenizedDataset,
    IDTokenizedDataset,
    prepare_dataset,
    prepare_id_dataset,
    apply_random_masking,
    extract_mask_from_data,
    generate_experiment_mask,
)

__all__ = [
    # Model
    'FBAProProjection',
    'CrossAttention',
    'FluxDecoderConfig',
    'FluxDecoder',
    # Training
    'train_model',
    'var_norm_mse_loss',
    # Prediction
    'predict_fluxes',
    'create_ensemble',
    'ensemble_predict',
    'train_ensemble',
    # Data
    'load_data',
    'load_data_batches',
    'find_data_batches',
    'create_tracer_mapping',
    'load_stoichiometric_matrix',
    'radial_embedding_table',
    'build_radial_vocabs',
    'build_token_static_features',
    'build_id_token_features',
    'TokenizedDataset',
    'IDTokenizedDataset',
    'prepare_dataset',
    'prepare_id_dataset',
    'apply_random_masking',
    'extract_mask_from_data',
    'generate_experiment_mask',
]
