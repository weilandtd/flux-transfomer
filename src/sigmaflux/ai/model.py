"""
Neural network models for flux prediction from labeling data.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FBAProProjection(nn.Module):
    """
    Projects fluxes onto ker(S) with optional per-sample fixed reactions.

    Without fixed_indices: plain null-space projection v → (I − S†S) v.

    With fixed_indices H and free indices nH = [r]\\H:
        x_nH = (I − S_nH† S_nH) v_nH − S_nH† S_H v_H
        x_H  = v_H
    guaranteeing Sv = 0 and x_H = v_H.
    """

    def __init__(self, stoichiometric_matrix, fixed_indices=None):
        super().__init__()
        S = torch.tensor(stoichiometric_matrix, dtype=torch.float64)
        _, r = S.shape

        S_pinv  = torch.linalg.pinv(S)
        fba_pro = torch.eye(r, dtype=torch.float64) - S_pinv @ S
        self.register_buffer('matrix_T', fba_pro.T.to(torch.float32))

        if fixed_indices is not None and len(fixed_indices) > 0:
            fixed = [int(i) for i in fixed_indices]
            free  = [i for i in range(r) if i not in set(fixed)]
            S_nH  = S[:, free]
            S_H   = S[:, fixed]
            S_nH_pinv = torch.linalg.pinv(S_nH)
            self.register_buffer(
                'proj_free_T',
                (torch.eye(len(free), dtype=torch.float64) - S_nH_pinv @ S_nH).T.to(torch.float32),
            )
            self.register_buffer('correction_T', (S_nH_pinv @ S_H).T.to(torch.float32))
            self.register_buffer('free_idx',  torch.tensor(free,  dtype=torch.long))
            self.register_buffer('fixed_idx', torch.tensor(fixed, dtype=torch.long))
        else:
            self.proj_free_T  = None
            self.correction_T = None
            self.free_idx     = None
            self.fixed_idx    = None

    def forward(self, fluxes, fixed_fluxes=None):
        if fixed_fluxes is not None and self.proj_free_T is not None:
            out = torch.empty_like(fluxes)
            out[:, self.free_idx]  = (
                fluxes[:, self.free_idx] @ self.proj_free_T - fixed_fluxes @ self.correction_T
            )
            out[:, self.fixed_idx] = fixed_fluxes
            return out
        return fluxes @ self.matrix_T


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


class CrossAttention(nn.Module):
    """
    Cross-attention with pre-norm applied to Q and K/V separately.

    forward(q, kv, key_padding_mask=None)
      q:               (batch, R, d) — flux query matrix
      kv:              (batch, N, d) — encoded token sequence (keys and values)
      key_padding_mask:(batch, N) bool — True = padded position (ignored)
    Returns:
      out: (batch, R, d)
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.q_norm  = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, q, kv, key_padding_mask=None):
        q  = self.q_norm(q)
        kv = self.kv_norm(kv)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
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
      5. Linear(d_model → 1) + squeeze → per-flux scalars (B, R)
      6. Split into fluxes / reversibilities; apply Sv=0 and softplus

    Args:
        config:                FluxDecoderConfig
        stoichiometric_matrix: (m, n) numpy array
        token_experiment_ids:  (num_tokens,) int64; -1 = bypass token
        compound_emb_table:    (n_compounds, d_comp) float32 — frozen weights
        label_emb_table:       (n_labels,    d_label) float32 — frozen weights
        tracer_emb_table:      (n_tracers,   d_tracer) float32 — frozen weights"""

    def __init__(
        self,
        config: FluxDecoderConfig,
        stoichiometric_matrix,
        token_experiment_ids,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        fixed_flux_indices=None,
    ):
        super().__init__()
        cfg = config
        self.cfg = cfg
        self.flux_dim          = cfg.n_fluxes
        self.reversibility_dim = cfg.n_rev
        d_in = cfg.d_comp + cfg.d_label + cfg.d_tracer + 1

        # Token-to-experiment mapping buffer (used by masking utilities)
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

        # Flux query matrix — one row per output (n_fluxes + n_rev)
        n_outputs = cfg.n_fluxes + cfg.n_rev
        self.flux_queries = nn.Parameter(torch.empty(n_outputs, cfg.d_model))
        nn.init.xavier_uniform_(self.flux_queries)

        # Cross-attention decoder
        self.cross_attn  = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.output_proj = nn.Linear(cfg.d_model, 1)

        # Stoichiometric constraint (Sv = 0), optionally with fixed measured fluxes
        self.fba_pro = FBAProProjection(stoichiometric_matrix, fixed_flux_indices)

    def forward(
        self,
        compound_ids,         # (batch, N) long
        label_ids,            # (batch, N) long
        tracer_ids,           # (batch, N) long
        fractions,            # (batch, N, 1) float, values in [0, 1]
        padding_mask,         # (batch, N) bool, True = padded position
        fixed_fluxes=None,    # (batch, n_measured) float — normalized measured flux values
    ):
        """
        Returns:
            fluxes:          (batch, n_fluxes) satisfying Sv = 0
            reversibilities: (batch, n_rev) non-negative
        """
        # Embed and concatenate
        x = torch.cat([
            self.compound_emb(compound_ids),  # (B, N, d_comp)
            self.label_emb(label_ids),        # (B, N, d_label)
            self.tracer_emb(tracer_ids),      # (B, N, d_tracer)
            fractions,                        # (B, N, 1)
        ], dim=-1)                            # (B, N, d_in)

        # Project + normalize
        x = self.input_norm(self.input_proj(x))  # (B, N, d_model)

        # Self-attention over all tokens
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = torch.nan_to_num(x, nan=0.0)

        # Cross-attention: flux queries attend over encoded tokens
        B = x.shape[0]
        q   = self.flux_queries.unsqueeze(0).expand(B, -1, -1)  # (B, R, d)
        out = self.cross_attn(q, x, key_padding_mask=padding_mask)  # (B, R, d)

        # Per-flux scalar
        logits = self.output_proj(out).squeeze(-1)  # (B, R)

        fluxes          = logits[:, :self.cfg.n_fluxes]
        reversibilities = logits[:, self.cfg.n_fluxes:]

        reversibilities = F.softplus(reversibilities)
        fluxes          = self.fba_pro(fluxes, fixed_fluxes)
        fluxes          = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)

        return fluxes, reversibilities
