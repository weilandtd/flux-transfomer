"""
AI module for metabolic flux prediction from labeling data.

This module provides deep learning models for predicting metabolic fluxes
and reversibilities from isotopic labeling experiments.
"""

# Model
from .model import LabelingToFluxTransformer, AttentionPooling

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
    prepare_dataset,
    apply_random_masking,
    extract_mask_from_data
)

__all__ = [
    # Model
    'LabelingToFluxTransformer',
    'AttentionPooling',
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
    'prepare_dataset',
    'apply_random_masking',
    'extract_mask_from_data',
]
