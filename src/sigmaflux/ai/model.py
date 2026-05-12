"""
Neural network models for flux prediction from labeling data.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NullSpaceDecoder(nn.Module):
    """
    Reconstructs the full flux vector from full null-space basis coefficients.

    The null space of the full stoichiometric matrix S has dimension n_basis
    (38 for the I. orientalis model). The model predicts y ∈ R^n_basis and
    this module recovers the complete 386-dim flux vector via:

        V = y @ Z_full.T

    where Z_full is the orthonormal null-space basis of S (shape: n_fluxes × n_basis).
    S @ V = 0 is guaranteed by construction. Measured fluxes are a soft input
    feature fed through a fusion_layer in FluxDecoder — not hard-enforced here.

    free_idx / fixed_idx are retained as informational metadata (used by
    analysis scripts) but are not used in forward or encode_targets.
    """

    def __init__(self, Z_free, free_idx, fixed_idx, n_fluxes,
                 y_mean=None, y_std=None, obs_weights=None):
        """
        Args:
            Z_free:      (n_fluxes, n_basis) orthonormal null-space basis of S
                         (named Z_free for checkpoint compatibility; now holds Z_full)
            free_idx:    list[int] — indices of free (non-measured) reactions (metadata)
            fixed_idx:   list[int] — indices of measured reactions (metadata)
            n_fluxes:    total number of reactions (e.g. 386)
            y_mean:      (n_basis,) mean of y_true across training set (for normalization)
            y_std:       (n_basis,) std  of y_true across training set (for normalization)
            obs_weights: (n_basis,) observability weight per y-component = max|Pearson r|
                         against all labeling features.
        """
        super().__init__()
        self.n_fluxes = n_fluxes
        self.register_buffer('Z_free',    torch.tensor(np.array(Z_free), dtype=torch.float32))
        self.register_buffer('free_idx',  torch.tensor(list(free_idx),   dtype=torch.long))
        self.register_buffer('fixed_idx', torch.tensor(list(fixed_idx),  dtype=torch.long))

        n_basis = self.Z_free.shape[1]
        _mean = np.zeros(n_basis)  if y_mean is None else np.array(y_mean)
        _std  = np.ones(n_basis)   if y_std  is None else np.array(y_std)
        _obs  = np.ones(n_basis)   if obs_weights is None else np.array(obs_weights, dtype=np.float32)
        self.register_buffer('y_mean',      torch.tensor(_mean, dtype=torch.float32))
        self.register_buffer('y_std',       torch.tensor(_std,  dtype=torch.float32))
        self.register_buffer('obs_weights', torch.tensor(_obs,  dtype=torch.float32))

    @property
    def n_basis(self):
        return self.Z_free.shape[1]

    def forward(self, y_norm):
        """
        Args:
            y_norm: (B, n_basis) normalized basis coefficients (model output, ~N(0,1))
        Returns:
            V: (B, n_fluxes) full reconstructed flux vector satisfying S@V=0
        """
        y = y_norm * self.y_std + self.y_mean   # (B, n_basis) — unnormalize
        return y @ self.Z_free.T                 # (B, n_fluxes)

    def encode_targets(self, v_targets):
        """
        Project ground-truth flux vectors to normalized y-space for loss computation.

        Returns y_norm = (y_true - y_mean) / y_std  ≈ N(0,1).
        The model predicts y_norm; this function gives the matching target.

        Args:
            v_targets: (B, n_fluxes) ground-truth full flux vector
        Returns:
            y_norm: (B, n_basis) normalized y-space targets
        """
        y_full = v_targets @ self.Z_free          # (B, n_basis) — project onto null space
        return (y_full - self.y_mean) / self.y_std


class FBAProProjection(nn.Module):
    """
    Legacy projection: enforces Sv=0 via null-space projection.
    Kept for backward compatibility with checkpoints that don't use
    NullSpaceDecoder.

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

    def forward(self, q, kv, key_padding_mask=None, attn_bias=None):
        q  = self.q_norm(q)
        kv = self.kv_norm(kv)
        out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
        )
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
        stoichiometric_matrix,
        token_experiment_ids,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        fixed_flux_indices=None,
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

        # Null-space mode vs legacy mode
        if cfg.n_free > 0 and null_space_data is not None:
            self.null_decoder = NullSpaceDecoder(
                Z_free      = null_space_data['Z_free'],
                free_idx    = null_space_data['free_idx'],
                fixed_idx   = null_space_data['fixed_idx'],
                n_fluxes    = cfg.n_fluxes,
                y_mean      = null_space_data.get('y_mean'),
                y_std       = null_space_data.get('y_std'),
                obs_weights = null_space_data.get('obs_weights'),
            )
            self.fba_pro = None
            n_flux_queries = cfg.n_free   # 38 basis queries
            # Additive correction: transformer owns y_raw; measured fluxes add a residual
            # correction learned from v_known only. Residual design keeps the transformer
            # signal intact — fusion_layer cannot learn to bypass y_raw.
            if cfg.n_measured > 0:
                self.fusion_layer = nn.Linear(cfg.n_measured, cfg.n_free)
                # Zero-init: correction starts at 0 so transformer trains freely first.
                # Unobservable y-directions stay near zero — no spurious Adam drift.
                nn.init.zeros_(self.fusion_layer.weight)
                nn.init.zeros_(self.fusion_layer.bias)
            else:
                self.fusion_layer = None
            # Normalization statistics for fixed_fluxes (mmol/gDCW/h → ~N(0,1)).
            # Without this, v_known O(10) × Xavier-init weights O(0.15) = O(1.5) initial
            # correction in y-space — large enough to corrupt unobservable directions.
            fm = null_space_data.get('fixed_flux_mean')
            fs = null_space_data.get('fixed_flux_std')
            if fm is not None and fs is not None:
                self.register_buffer('fixed_flux_mean',
                                     torch.tensor(np.array(fm), dtype=torch.float32))
                self.register_buffer('fixed_flux_std',
                                     torch.tensor(np.array(fs), dtype=torch.float32))
            else:
                self.fixed_flux_mean = None
                self.fixed_flux_std  = None
        else:
            self.null_decoder = None
            self.fba_pro = FBAProProjection(stoichiometric_matrix, fixed_flux_indices)
            self.fusion_layer = None
            n_flux_queries = cfg.n_fluxes  # 386 reaction queries (legacy)

        # Flux query matrix — one row per output query
        n_outputs = n_flux_queries + cfg.n_rev
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
    ):
        """
        Returns:
            fluxes:          (batch, n_fluxes) satisfying Sv=0
            reversibilities: (batch, n_rev) non-negative
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
        out = self.cross_attn(q, x, key_padding_mask=padding_mask, attn_bias=attn_bias)

        logits = self.output_proj(out).squeeze(-1)   # (B, n_queries)

        if self.null_decoder is not None:
            n_free = self.cfg.n_free
            y_raw           = logits[:, :n_free]                         # (B, 38) raw transformer output
            reversibilities = F.softplus(logits[:, n_free:])
            # Additive correction: y_norm = transformer output + residual from v_known.
            # fixed_fluxes are normalized to ~N(0,1) before fusion_layer to prevent
            # scale mismatch (mmol/gDCW/h ≫ y-space units) from corrupting unobservable
            # y-directions via Adam drift.
            if self.fusion_layer is not None and fixed_fluxes is not None:
                ff = fixed_fluxes
                if self.fixed_flux_mean is not None:
                    ff = (ff - self.fixed_flux_mean) / self.fixed_flux_std
                # Unmasked: let BIOMASS inform stoichiometrically-coupled unobservable
                # pathways (sulfur, nucleotides). y_l2_reg in the training loss suppresses
                # y_raw noise for those directions, leaving the fusion correction to carry
                # the stoichiometric signal from the measured fluxes.
                y_norm = y_raw + self.fusion_layer(ff)
            else:
                y_norm = y_raw
            fluxes = self.null_decoder(y_norm)                           # (B, n_fluxes)
        else:
            fluxes          = logits[:, :self.cfg.n_fluxes]
            reversibilities = F.softplus(logits[:, self.cfg.n_fluxes:])
            fluxes          = self.fba_pro(fluxes, fixed_fluxes)

        fluxes          = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)
        return fluxes, reversibilities
