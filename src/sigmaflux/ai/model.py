"""
Neural network models for flux prediction from labeling data.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class MeasuredFluxEncoder(nn.Module):
    """
    Projects measured fluxes into y-space for use by RotationFusion, with exact
    flux recovery guaranteed after null-space reconstruction.

    The naive projection b = v_measured @ Z_sub gives c_S = b_S in y-space, but
    reconstruction V = c @ Z_full.T returns:

        V[fixed_idx] = c_S @ Z_sub_S.T = v_measured @ (Z_sub_S @ Z_sub_S.T)
                     = v_measured @ G

    where G = Z_sub_S @ Z_sub_S.T is the Gram matrix of the S-columns of Z_sub.
    G ≠ I in general: diagonal entries < 1 (scale-down) and non-zero off-diagonal
    entries (cross-contamination between measured reactions), causing systematic
    under-prediction and sign errors at extreme flux ratios.

    Fix: replace Z_sub with its left pseudo-inverse restricted to S-columns,

        Z_pinv[:, S] = G⁻¹ @ Z_sub_S

    so that the RotationFusion LS scale sets c_S = b_S = v_measured @ G⁻¹ @ Z_sub_S,
    and reconstruction gives:

        V[fixed_idx] = c_S @ Z_sub_S.T = v_measured @ G⁻¹ @ G = v_measured  (exact)
    """

    def __init__(self, Z_full, fixed_idx,  eps=1e-4):
        super().__init__()
        self.fixed_idx = list(fixed_idx)
        if not isinstance(Z_full, torch.Tensor):
            Z_full = torch.tensor(Z_full, dtype=torch.float32)
        self.register_buffer('Z_full', Z_full)
        Z_np  = Z_full.numpy()
        Z_sub = Z_np[np.array(list(fixed_idx)), :]
        # Z_sub     = np.array(Z_sub)
        col_norms = np.linalg.norm(Z_sub, axis=0)
        S_mask    = col_norms > eps
        self.register_buffer('S_idx', torch.tensor(np.where(S_mask)[0], dtype=torch.long))

        Z_sub_S = Z_sub[:, S_mask]            # (n_measured, |S|)
        G       = Z_sub_S @ Z_sub_S.T         # (n_measured, n_measured) — small, PSD
        # Use pseudo-inverse: some measured reactions (e.g. biomass assembly) have near-zero
        # null-space participation, making G rank-deficient. pinv zeroes those directions
        # rather than inverting them, so x_m stays bounded for such reactions.
        G_inv   = np.linalg.pinv(G, rcond=eps)

        # Z_pinv: (n_measured, n_basis) with S-columns = G⁻¹ @ Z_sub_S, rest = 0
        Z_pinv             = np.zeros_like(Z_sub)
        Z_pinv[:, S_mask]  = G_inv @ Z_sub_S
        self.register_buffer('Z_pinv', torch.tensor(Z_pinv, dtype=torch.float32))

    def forward(self, v_measured):
        """
        Args:
            v_measured: (B, n_measured) flux values in physical units
        Returns:
            b: (B, n_basis)  b_S = v_measured @ G⁻¹ @ Z_sub_S; b_non-S = 0.
               After RotationFusion LS scale: c_S = b_S → V[fixed_idx] = v_measured.
        """
        return v_measured @ self.Z_pinv





@dataclass
class FluxDecoderConfig:
    d_model:     int   = 64
    n_heads:     int   = 4
    n_layers:    int   = 4
    dropout:     float = 0.3
    n_compounds: int   = 0
    n_labels:    int   = 0
    n_tracers:   int   = 0
    d_comp:      int   = 0
    d_label:     int   = 0
    d_tracer:    int   = 0
    n_fluxes:    int   = 0
    n_rev:       int   = 0
    n_measured:  int   = 0  # number of measured-flux bypass tokens
    n_free:      int   = 0  # null-space basis dimension (0 = use legacy FBAProProjection)


class CrossAttention(nn.Module):
    """
    Cross-attention with pre-norm applied to Q and K/V separately.

    forward(q, kv, key_padding_mask=None, attn_bias=None)
      q:               (batch, R, d) — flux query matrix
      kv:              (batch, N, d) — encoded token sequence (keys and values)
      key_padding_mask:(batch, N) bool — True = padded position (ignored)
      attn_bias:       (batch*n_heads, R, N) float — additive attention bias
    Returns:
      out: (batch, R, d)
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.q_norm  = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, q, kv, key_padding_mask=None, attn_bias=None, return_weights=False):
        q  = self.q_norm(q)
        kv = self.kv_norm(kv)
        out, weights = self.attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=return_weights,
            average_attn_weights=True,   # average over heads → (B, R, N)
        )
        if return_weights:
            return out, weights          # weights: (B, 38, N)
        return out


class FluxDecoder(nn.Module):
    """
    Perceiver-IO–style cross-attention decoder for metabolic flux prediction.

    Architecture:
      1. Embed compound/label/tracer IDs with frozen radial embeddings;
         concatenate with raw fraction scalar → token vectors (B, N, d_in)
      2. Linear(d_in → d_model) + LayerNorm
      3. TransformerEncoder — self-attention across the token sequence
      4. CrossAttention — R learned flux queries attend over encoded tokens
      5. Linear(d_model → 1) + squeeze → per-query scalars (B, R)
      6. Split into flux basis / reversibilities; reconstruct full flux vector

    If n_free > 0 (null-space mode):
      - R = n_free + n_rev  (n_free basis queries, n_rev reversibility queries)
      - flux output is (B, n_free) basis coefficients y
      - full flux vector reconstructed via NullSpaceDecoder: v = Z@y + correction
    Else (legacy mode):
      - R = n_fluxes + n_rev
      - flux output is (B, n_fluxes), projected via FBAProProjection
    """

    def __init__(
        self,
        config: FluxDecoderConfig,
        token_experiment_ids,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        null_space_data=None,   # dict with Z_free, correction_T, free_idx, fixed_idx
    ):
        super().__init__()
        cfg = config
        self.cfg = cfg
        self.flux_dim          = cfg.n_fluxes
        self.reversibility_dim = cfg.n_rev
        d_in = cfg.d_comp + cfg.d_label + cfg.d_tracer + 1

        # Token-to-experiment mapping buffer
        if not isinstance(token_experiment_ids, torch.Tensor):
            token_experiment_ids = torch.tensor(token_experiment_ids, dtype=torch.int64)
        self.register_buffer('token_experiment_ids', token_experiment_ids)
        valid = token_experiment_ids[token_experiment_ids >= 0]
        self.num_experiments = int(valid.max().item()) + 1 if len(valid) > 0 else 0

        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )

        # Frozen radial embedding lookups
        self.compound_emb = nn.Embedding(cfg.n_compounds, cfg.d_comp)
        self.label_emb    = nn.Embedding(cfg.n_labels,    cfg.d_label)
        self.tracer_emb   = nn.Embedding(cfg.n_tracers,   cfg.d_tracer)

        def _load_frozen(emb_layer, table):
            if table is not None:
                emb_layer.weight = nn.Parameter(
                    torch.tensor(table, dtype=torch.float32), requires_grad=False
                )

        _load_frozen(self.compound_emb, compound_emb_table)
        _load_frozen(self.label_emb,    label_emb_table)
        _load_frozen(self.tracer_emb,   tracer_emb_table)

        # Token encoder
        self.input_proj = nn.Linear(d_in, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # project measured fluxes into latent x space 
        self.known_flux_encoder = MeasuredFluxEncoder(
            Z_full=null_space_data['Z_free'],
            fixed_idx=null_space_data['fixed_idx'],
        )

        # Fusion gate for the known and learned flux components 
        self.alpha = nn.Parameter( torch.ones(cfg.n_free) )

        # Flux query matrix — one row per output query
        n_outputs = cfg.n_free + cfg.n_rev
        self.flux_queries = nn.Parameter(torch.empty(n_outputs, cfg.d_model))
        nn.init.xavier_uniform_(self.flux_queries)

        # Cross-attention decoder + output projection
        self.cross_attn  = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.output_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        compound_ids,         # (batch, N) long
        label_ids,            # (batch, N) long
        tracer_ids,           # (batch, N) long
        fractions,            # (batch, N, 1) float
        padding_mask,         # (batch, N) bool, True = padded
        fixed_fluxes=None,    # (batch, n_measured) float
        attn_bias=None,       # (batch*n_heads, n_queries, N) float, additive attention bias
        return_attn=False,    # if True, return attention weights (B, 38, N) as third output
    ):
        """
        Returns:
            fluxes:          (batch, n_fluxes) satisfying Sv=0
            reversibilities: (batch, n_rev) non-negative
            attn_weights:    (batch, 38, N) averaged over heads — only if return_attn=True
        """
        # Embed and concatenate
        x = torch.cat([
            self.compound_emb(compound_ids),
            self.label_emb(label_ids),
            self.tracer_emb(tracer_ids),
            fractions,
        ], dim=-1)

        x = self.input_norm(self.input_proj(x))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = torch.nan_to_num(x, nan=0.0)

        B = x.shape[0]
        q   = self.flux_queries.unsqueeze(0).expand(B, -1, -1)
        if return_attn:
            out, attn_weights = self.cross_attn(q, x, key_padding_mask=padding_mask,
                                                attn_bias=attn_bias, return_weights=True)
        else:
            out = self.cross_attn(q, x, key_padding_mask=padding_mask, attn_bias=attn_bias)

        logits = self.output_proj(out).squeeze(-1)   # (B, n_queries)

        n_free = self.cfg.n_free
        x_l    = logits[:, :n_free] 

        # Revs are always non-negative, so apply softplus activation to the corresponding output logits
        reversibilities = F.softplus(logits[:, n_free:])

        # Encode known fluxes into the same latent space as x_l
        x_m = self.known_flux_encoder(fixed_fluxes) if fixed_fluxes is not None else torch.zeros_like(x_l)

        # Project from latent x space back to flux space using the full null-space basis Z_full
        fluxes = x @ self.known_flux_encoder.Z_full.T
        
        fluxes          = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)
        if return_attn:
            return fluxes, reversibilities, attn_weights
        return fluxes, reversibilities
