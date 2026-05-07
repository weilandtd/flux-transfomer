"""
Neural network models for flux prediction from labeling data.

This module provides transformer-based architectures for predicting metabolic fluxes
from isotopic labeling experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate contextualized experiment embeddings.
    Uses a learnable query to compute importance weights for each experiment.
    """
    
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.scale = embed_dim ** -0.5
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, num_exp, embed_dim) - contextualized experiment embeddings
            mask: (batch, num_exp) - True for valid, False for padding
        
        Returns:
            pooled: (batch, embed_dim) - aggregated representation
        """
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        scores = torch.bmm(query, x.transpose(1, 2)) * self.scale
        
        if mask is not None:
            # Check for empty masks (all False) - this would cause NaN in softmax
            all_masked = ~mask.any(dim=1)  # (batch,) - True if all experiments are masked
            
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
            # For samples with all-masked experiments, use uniform attention instead
            # This prevents NaN from softmax(-inf, -inf, ...)
            if all_masked.any():
                # Replace -inf with 0 for all-masked samples (uniform attention)
                all_masked_expanded = all_masked.unsqueeze(1).unsqueeze(2)
                scores = torch.where(all_masked_expanded, torch.zeros_like(scores), scores)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN that might still occur (safety check)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        pooled = torch.bmm(attn_weights, x).squeeze(1)
        
        # Handle any remaining NaN in output
        pooled = torch.nan_to_num(pooled, nan=0.0)
        
        return pooled


class LabelingToFluxTransformer(nn.Module):
    """
    Transformer-based architecture for predicting flux distributions from labeling experiments.
    
    Architecture:
    1. Project labeling data to embedding space
    2. Apply transformer encoder with multi-head self-attention
    3. Attention-based pooling over experiments
    4. Linear output layer for flux and reversibility prediction
    5. Apply stoichiometric constraint (Sv=0) to fluxes
    
    Args:
        stoichiometric_matrix: (m, n) stoichiometric matrix S
        input_dim: Dimension of labeling data per experiment
        flux_dim: Number of reactions (fluxes)
        reversibility_dim: Number of reactions (reversibilities)
        num_base_tracers: Number of unique tracers
        combined_dim: Total dimension for transformer (includes embedding + tracer info + measured fluxes)
        dropout: Dropout rate
        num_attn_heads: Number of attention heads
        num_transformer_layers: Number of transformer encoder layers
        measured_fluxes_dim: Number of directly measured flux features passed alongside labeling data
    """

    def __init__(self, stoichiometric_matrix, input_dim, flux_dim, reversibility_dim,
                 num_base_tracers, combined_dim=512, dropout=0.15,
                 num_attn_heads=8, num_transformer_layers=2, measured_fluxes_dim=0):
        super(LabelingToFluxTransformer, self).__init__()

        self.num_base_tracers = num_base_tracers
        self.input_dim = input_dim
        self.flux_dim = flux_dim
        self.reversibility_dim = reversibility_dim
        self.measured_fluxes_dim = measured_fluxes_dim

        # Compute FBApro matrix (I - S_pinv @ S) for Sv=0 constraint
        S = torch.tensor(stoichiometric_matrix, dtype=torch.float32)
        S_pinv = torch.linalg.pinv(S)
        identity = torch.eye(S.shape[1], dtype=torch.float32)
        fba_pro_matrix = identity - S_pinv @ S
        fba_pro_matrix_T = fba_pro_matrix.T  # Pre-compute transpose
        
        self.register_buffer('fba_pro_matrix', fba_pro_matrix)
        self.register_buffer('fba_pro_matrix_T', fba_pro_matrix_T)
        
        # Ensure combined_dim is divisible by num_attn_heads
        assert combined_dim % num_attn_heads == 0, \
            f"combined_dim ({combined_dim}) must be divisible by num_attn_heads ({num_attn_heads})"
        
        # Compute embedding dimension
        embed_dim = combined_dim - num_base_tracers - measured_fluxes_dim
        assert embed_dim > 0, \
            f"embed_dim must be positive (combined_dim={combined_dim}, num_base_tracers={num_base_tracers}, measured_fluxes_dim={measured_fluxes_dim})"
        
        self.embed_dim = embed_dim
        self.combined_dim = combined_dim
        self.num_transformer_layers = num_transformer_layers
        
        # Input projection for labeling data
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=num_attn_heads,
            dim_feedforward=combined_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(combined_dim)
        
        # Output layer
        self.output_layer = nn.Linear(combined_dim, flux_dim + reversibility_dim)
    
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer.
        
        Args:
            x: (batch_size, total_input_dim) - concatenated [labeling_data, tracer_data, measured_fluxes]
            mask: Optional mask for experiments
        
        Returns:
            fluxes: (batch_size, flux_dim) - predicted fluxes with Sv=0 constraint
            reversibilities: (batch_size, reversibility_dim) - predicted reversibilities
        """
        batch_size, total_dim = x.shape
        
        # Split inputs
        labeling_and_tracer_dim = total_dim - self.measured_fluxes_dim
        num_experiments = labeling_and_tracer_dim // (self.input_dim + self.num_base_tracers)

        labeling_data = x[:, :num_experiments * self.input_dim]
        tracer_data = x[:, num_experiments * self.input_dim:labeling_and_tracer_dim]
        measured_fluxes_data = x[:, labeling_and_tracer_dim:]
        
        # Reshape to 3D
        labeling_data = labeling_data.view(batch_size, num_experiments, self.input_dim)
        tracer_data = tracer_data.view(batch_size, num_experiments, self.num_base_tracers)
        
        # Project and normalize labeling data
        projected_labeling = self.input_projection(labeling_data)
        projected_labeling = F.layer_norm(projected_labeling, normalized_shape=(self.embed_dim,))
        
        # Combine with tracer info
        combined_input = torch.cat([projected_labeling, tracer_data], dim=-1)
        
        # Add measured fluxes if present
        if self.measured_fluxes_dim > 0:
            measured_fluxes_expanded = measured_fluxes_data.unsqueeze(1).expand(-1, num_experiments, -1)
            combined_input = torch.cat([combined_input, measured_fluxes_expanded], dim=-1)
        
        # Transformer encoding
        # src_key_padding_mask: True = ignore position (inverted from our convention where True=valid)
        src_key_padding_mask = ~mask if mask is not None else None
        contextualized_experiments = self.transformer_encoder(
            combined_input, src_key_padding_mask=src_key_padding_mask
        )
        
        # Handle any NaN values from transformer (e.g., from log of zero)
        contextualized_experiments = torch.nan_to_num(contextualized_experiments, nan=0.0)
        
        # Attention pooling (with optional masking)
        aggregated_representation = self.attention_pooling(contextualized_experiments, mask=mask)
        
        # Output layer
        combined_output = self.output_layer(aggregated_representation)
        fluxes = combined_output[:, :self.flux_dim]
        reversibilities = combined_output[:, self.flux_dim:]
        
        # Apply softplus to reversibilities (ensure non-negative)
        reversibilities = F.softplus(reversibilities)
        
        # Apply stoichiometric constraint to fluxes
        fluxes = fluxes @ self.fba_pro_matrix_T
        
        # Final NaN safety check
        fluxes = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)
        
        return fluxes, reversibilities
