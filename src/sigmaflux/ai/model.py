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
    S @ V = 0 is guaranteed by construction. Measured fluxes enter via
    MeasuredFluxEncoder + RotationFusion in FluxDecoder — not hard-enforced here.

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
    n_free:      int   = 0  # null-space basis dimension


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

    def __init__(self, Z_sub, eps=1e-4):
        super().__init__()
        Z_sub     = np.array(Z_sub)
        col_norms = np.linalg.norm(Z_sub, axis=0)
        S_mask    = col_norms > eps
        self.register_buffer('S_idx', torch.tensor(np.where(S_mask)[0], dtype=torch.long))

        Z_sub_S = Z_sub[:, S_mask]            # (n_measured, |S|)
        G       = Z_sub_S @ Z_sub_S.T         # (n_measured, n_measured) — small, PSD
        G_inv   = np.linalg.inv(G)            # well-conditioned; n_measured is tiny (2–5)

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


class RotationFusion(nn.Module):
    """
    Fuses a (labeling-informed, physical y-units) and b (measurement-informed, physical
    y-units) by rotating a toward b on the S-subspace, then applying a least-squares scale.

    Both inputs are in the same physical y-space — no normalization required.
    b is produced by MeasuredFluxEncoder using the left pseudo-inverse of Z_sub_S,
    so the LS scale step (step 4) hard-enforces V[fixed_idx] = v_measured exactly.

    Steps:
      1. Extract a_S, b_S at S_idx — y-components constrained by measurements
      2. Embed unit vectors û (from â_S), v̂ (from b̂_S) into ℝ^n_basis
      3. Batched Rodrigues rotation of a from û toward v̂:
           Ra = a + (cθ-1)*((û·a)û + (ŵ·a)ŵ) + sθ*((û·a)ŵ - (ŵ·a)û)
         where ŵ = (v̂ - cθ û)/sθ  (Gram-Schmidt unit vector)
      4. Scale λ = (b_S · (Ra)_S) / ‖(Ra)_S‖²  → c_S = b_S exactly
      5. Return c = λ * Ra  (physical y-units; V = c @ Z_full.T gives V[fixed_idx] = v_measured)
    """

    def __init__(self, S_idx):
        super().__init__()
        self.register_buffer('S_idx', torch.tensor(np.array(S_idx), dtype=torch.long))

    def forward(self, a, b):
        S = self.S_idx
        a_S = a[:, S]
        b_S = b[:, S]

        norm_a = a_S.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_b = b_S.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        n_basis = a.shape[-1]
        u = a.new_zeros(a.shape[0], n_basis)
        v = a.new_zeros(a.shape[0], n_basis)
        u[:, S] = a_S / norm_a   # "from": labeling direction
        v[:, S] = b_S / norm_b   # "to":   measurement direction

        ct = (u * v).sum(-1, keepdim=True).clamp(-1.0, 1.0)
        st = (1.0 - ct**2).clamp(min=0.0).sqrt()

        # near-parallel guard — skip rotation, keep a
        parallel = (st < 1e-6).squeeze(-1)   # (B,) bool

        st_safe = st.clamp(min=1e-8)
        w = (v - ct * u) / st_safe           # (B, n_basis)

        ua = (u * a).sum(-1, keepdim=True)
        wa = (w * a).sum(-1, keepdim=True)

        Ra = a + ua * ((ct - 1.0) * u + st * w) + wa * ((ct - 1.0) * w - st * u)

        Ra[parallel] = a[parallel]

        # LS scale: λ·Ra_S = b_S (exactly enforces measurement on S-subspace)
        Ra_S    = Ra[:, S]
        lam_num = (b_S * Ra_S).sum(-1, keepdim=True)
        lam_den = (Ra_S * Ra_S).sum(-1, keepdim=True).clamp(min=1e-8)

        return (lam_num / lam_den) * Ra


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

    R = n_free + n_rev  (n_free basis queries, n_rev reversibility queries);
    full flux vector reconstructed via NullSpaceDecoder: V = y @ Z_full.T.
    """

    def __init__(
        self,
        config: FluxDecoderConfig,
        token_experiment_ids,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        null_space_data=None,   # dict with Z_free, free_idx, fixed_idx, y_mean, y_std, obs_weights
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

        self.null_decoder = NullSpaceDecoder(
            Z_free      = null_space_data['Z_free'],
            free_idx    = null_space_data['free_idx'],
            fixed_idx   = null_space_data['fixed_idx'],
            n_fluxes    = cfg.n_fluxes,
            y_mean      = null_space_data.get('y_mean'),
            y_std       = null_space_data.get('y_std'),
            obs_weights = null_space_data.get('obs_weights'),
        )
        n_flux_queries = cfg.n_free
        if cfg.n_measured > 0:
            nsd    = self.null_decoder
            Z_sub  = nsd.Z_free.numpy()[np.array(list(null_space_data['fixed_idx'])), :]
            self.flux_encoder    = MeasuredFluxEncoder(Z_sub=Z_sub)
            self.rotation_fusion = RotationFusion(S_idx=self.flux_encoder.S_idx.numpy())
        else:
            self.flux_encoder    = None
            self.rotation_fusion = None

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
        y_raw           = logits[:, :n_free]
        reversibilities = F.softplus(logits[:, n_free:])

        if self.rotation_fusion is not None and fixed_fluxes is not None:
            nsd    = self.null_decoder
            y_a    = y_raw * nsd.y_std + nsd.y_mean          # unnormalize to physical y-space
            b      = self.flux_encoder(fixed_fluxes)          # physical y-space
            c_phys = self.rotation_fusion(y_a, b)             # physical y-space
            fluxes = c_phys @ nsd.Z_free.T                    # direct reconstruction
        else:
            fluxes = self.null_decoder(y_raw)

        fluxes          = torch.nan_to_num(fluxes, nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)

        if return_attn:
            return fluxes, reversibilities, attn_weights
        return fluxes, reversibilities
