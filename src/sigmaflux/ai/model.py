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
            x: (batch, num_tokens, embed_dim) - contextualized token embeddings
            mask: (batch, num_tokens) - True for valid, False for padding

        Returns:
            pooled: (batch, embed_dim) - aggregated representation
        """
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        scores = torch.bmm(query, x.transpose(1, 2)) * self.scale

        if mask is not None:
            # Check for empty masks (all False) — would cause NaN in softmax
            all_masked = ~mask.any(dim=1)  # (batch,)

            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

            # For samples with all-masked tokens, use uniform attention
            if all_masked.any():
                all_masked_expanded = all_masked.unsqueeze(1).unsqueeze(2)
                scores = torch.where(all_masked_expanded, torch.zeros_like(scores), scores)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = torch.bmm(attn_weights, x).squeeze(1)
        pooled = torch.nan_to_num(pooled, nan=0.0)

        return pooled


class LabelingToFluxTransformer(nn.Module):
    """
    Transformer-based architecture for predicting flux distributions from
    radially-tokenized labeling experiments.

    Each measurement (compound, isotopomer, tracer, fraction) is represented
    as one token.  Compound/label/tracer identities are encoded with fixed
    multi-frequency radial embeddings (precomputed in data.py); the fraction
    value occupies the last scalar dimension.

    Architecture:
    1. Linear projection of each token into embedding space (combined_dim)
    2. Layer norm per token
    3. Transformer encoder with multi-head self-attention across all tokens
    4. Attention-based pooling to a single vector
    5. Linear output layer for flux and reversibility prediction
    6. Stoichiometric constraint (Sv = 0) applied to flux output

    Args:
        stoichiometric_matrix: (m, n) stoichiometric matrix S
        token_dim: Dimension of each input token (2*H_c + 2*H_l + 2*H_t + 1)
        flux_dim: Number of reactions (fluxes)
        reversibility_dim: Number of reactions (reversibilities)
        token_experiment_ids: (num_tokens,) int64 tensor mapping each token to
            its experiment index; -1 for bypass tokens. Stored as a buffer and
            used by apply_random_masking / extract_mask_from_data in training.
        combined_dim: Transformer hidden dimension (must be divisible by num_attn_heads)
        dropout: Dropout rate
        num_attn_heads: Number of attention heads
        num_transformer_layers: Number of transformer encoder layers
    """

    def __init__(self, stoichiometric_matrix, token_dim, flux_dim, reversibility_dim,
                 token_experiment_ids, combined_dim=512, dropout=0.15,
                 num_attn_heads=8, num_transformer_layers=2):
        super(LabelingToFluxTransformer, self).__init__()

        self.token_dim = token_dim
        self.flux_dim = flux_dim
        self.reversibility_dim = reversibility_dim
        self.combined_dim = combined_dim
        self.num_transformer_layers = num_transformer_layers

        # Store token-to-experiment mapping as a non-trainable buffer
        if not isinstance(token_experiment_ids, torch.Tensor):
            token_experiment_ids = torch.tensor(token_experiment_ids, dtype=torch.int64)
        self.register_buffer('token_experiment_ids', token_experiment_ids)

        # Compute number of experiments (excludes bypass tokens with id == -1)
        valid = token_experiment_ids[token_experiment_ids >= 0]
        self.num_experiments = int(valid.max().item()) + 1 if len(valid) > 0 else 0

        # Stoichiometric constraint: Sv = 0  →  v = (I - S†S) v
        S = torch.tensor(stoichiometric_matrix, dtype=torch.float32)
        S_pinv = torch.linalg.pinv(S)
        identity = torch.eye(S.shape[1], dtype=torch.float32)
        fba_pro_matrix = identity - S_pinv @ S
        self.register_buffer('fba_pro_matrix', fba_pro_matrix)
        self.register_buffer('fba_pro_matrix_T', fba_pro_matrix.T)

        assert combined_dim % num_attn_heads == 0, (
            f"combined_dim ({combined_dim}) must be divisible by "
            f"num_attn_heads ({num_attn_heads})"
        )

        # Input projection: token_dim → combined_dim
        self.input_projection = nn.Linear(token_dim, combined_dim)

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
            num_layers=num_transformer_layers,
        )

        # Attention pooling over token sequence
        self.attention_pooling = AttentionPooling(combined_dim)

        # Output layer
        self.output_layer = nn.Linear(combined_dim, flux_dim + reversibility_dim)

    def forward(self, tokens, mask=None):
        """
        Forward pass.

        Args:
            tokens: (batch_size, num_tokens, token_dim)
            mask: (batch_size, num_tokens) bool tensor (True = valid token).
                  If None, all tokens are treated as valid.

        Returns:
            fluxes: (batch_size, flux_dim) with Sv = 0 constraint
            reversibilities: (batch_size, reversibility_dim) non-negative
        """
        # Project and normalise each token
        projected = self.input_projection(tokens)                   # (B, T, D)
        projected = F.layer_norm(projected, (self.combined_dim,))

        # Transformer: src_key_padding_mask convention is True = IGNORE
        src_key_padding_mask = ~mask if mask is not None else None
        contextualized = self.transformer_encoder(
            projected, src_key_padding_mask=src_key_padding_mask
        )
        contextualized = torch.nan_to_num(contextualized, nan=0.0)

        # Attention pooling → single vector per sample
        aggregated = self.attention_pooling(contextualized, mask=mask)  # (B, D)

        # Output
        combined_output = self.output_layer(aggregated)
        fluxes = combined_output[:, :self.flux_dim]
        reversibilities = combined_output[:, self.flux_dim:]

        reversibilities = F.softplus(reversibilities)
        fluxes = fluxes @ self.fba_pro_matrix_T
        fluxes = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)

        return fluxes, reversibilities
