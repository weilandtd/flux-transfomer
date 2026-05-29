"""
Neural network models for flux prediction from labeling data.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarFeatureEncoder(nn.Module):
	"""
	Encodes a scalar value as a d_model-dimensional embedding via learned projection.

	Maps: z = W · φ(x)
	  where φ(x) = [x, log(|x|+ε), x², sign(x)] (4-dim featurization)
	        W ∈ ℝ^(d_model × 4) learned projection
    
	"""

	def __init__(self, d_model):
		super().__init__()
		self.d_model = d_model
		self.feature_dim = 4
		self.projection = nn.Linear(self.feature_dim, d_model)
		nn.init.xavier_uniform_(self.projection.weight)

	def _featurize(self, x):
		"""
		Featurize scalar values with non-linear transformations.

		Args:
		  x: (...) tensor of scalar values

		Returns:
		  features: (..., 4) with [x, log(|x|+1), x², sign(x)]
		"""
		log_feat = torch.log(torch.abs(x) + 1)
		quad_feat = x ** 2
		sign_feat = torch.sign(x)

		return torch.stack([x, log_feat, quad_feat, sign_feat], dim=-1)

	def forward(self, x):
		"""
		Args:
		  x: (B, N) or (B, N, 1) scalar values

		Returns:
		  (B, N, d_model) projected embedding
		"""
		if x.dim() == 3 and x.shape[-1] == 1:
			x = x.squeeze(-1)  # (B, N, 1) → (B, N)

		features = self._featurize(x)  # (B, N, 4)
		embedding = self.projection(features)  # (B, N, d_model)
		return embedding


class FractionEncoder(ScalarFeatureEncoder):
	"""Encoder for labeling fractions (0-1 bounded values)."""
	pass


class FluxEncoder(ScalarFeatureEncoder):
	"""Encoder for flux values (unbounded, often normalized)."""
	pass


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
    n_free:      int   = 0
    d_reaction:  int   = 0     # frozen reaction radial embedding dim; 0 = no flux tokens
    d_type:      int   = 0     # frozen token-type radial embedding dim (= 2 * h_type)


class CrossAttention(nn.Module):
    """
    Cross-attention with pre-norm applied to Q and K/V separately.

    forward(q, kv, key_padding_mask=None, attn_bias=None)
      q:               (B, R, d)
      kv:              (B, N, d)
      key_padding_mask:(B, N) bool — True = padded/hidden
      attn_bias:       (B*n_heads, R, N) float — additive bias
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
            average_attn_weights=True,
        )
        if return_weights:
            return out, weights
        return out


class FluxDecoder(nn.Module):
    """
    Perceiver-IO–style cross-attention decoder for metabolic flux prediction.

    Architecture:
      1. Build a unified token sequence from two token types, both sharing the same
         d_in feature vector and single input_proj + input_norm:

           Labeling token: [comp_emb | label_emb | tracer_emb | type_emb[0] | fraction]
           Flux token:     [reaction_emb | zeros(pad) | type_emb[1] | norm_flux]

         type_emb is a frozen 2-vocabulary radial embedding (vocab_size=2, H=h_type).
         Reaction embeddings are zero-padded to d_entity_lab = d_comp+d_label+d_tracer.

      2. TransformerEncoder — self-attention over the combined token sequence
      3. CrossAttention — R queries attend over encoded tokens
      4. Linear(d_model → 1) → basis coefficients x_l and reversibilities
      5. Flux reconstruction: fluxes = x_l @ Z_full.T  (satisfies Sv=0)

    d_in = d_comp + d_label + d_tracer + d_type + 1
    R    = n_free + n_rev
    """

    def __init__(
        self,
        config: FluxDecoderConfig,
        token_experiment_ids,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        null_space_data=None,
        reaction_emb_table=None,    # (n_fluxes, d_reaction) frozen table; required when d_reaction > 0
        type_emb_table=None,        # (2, d_type) frozen table; required when d_type > 0
        flux_norm_stats=None,       # dict with 'mean' and 'std' (n_fluxes,) arrays
    ):
        super().__init__()
        cfg = config
        self.cfg = cfg
        self.flux_dim          = cfg.n_fluxes
        self.reversibility_dim = cfg.n_rev

        # Unified input dimension (no +1 for scalar; projections add via modulation)
        d_entity_lab = cfg.d_comp + cfg.d_label + cfg.d_tracer
        self.d_entity_lab = d_entity_lab
        d_in = d_entity_lab + cfg.d_type

        # Token-to-experiment mapping
        if not isinstance(token_experiment_ids, torch.Tensor):
            token_experiment_ids = torch.tensor(token_experiment_ids, dtype=torch.int64)
        self.register_buffer('token_experiment_ids', token_experiment_ids)
        valid = token_experiment_ids[token_experiment_ids >= 0]
        self.num_experiments = int(valid.max().item()) + 1 if len(valid) > 0 else 0

        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )

        def _load_frozen(emb_layer, table):
            if table is not None:
                emb_layer.weight = nn.Parameter(
                    torch.tensor(table, dtype=torch.float32), requires_grad=False
                )

        # Frozen labeling-token embeddings
        self.compound_emb = nn.Embedding(cfg.n_compounds, cfg.d_comp)
        self.label_emb    = nn.Embedding(cfg.n_labels,    cfg.d_label)
        self.tracer_emb   = nn.Embedding(cfg.n_tracers,   cfg.d_tracer)
        _load_frozen(self.compound_emb, compound_emb_table)
        _load_frozen(self.label_emb,    label_emb_table)
        _load_frozen(self.tracer_emb,   tracer_emb_table)

        # Frozen reaction embeddings (flux tokens + optional structured queries)
        if cfg.d_reaction > 0:
            assert cfg.d_reaction <= d_entity_lab, (
                f"d_reaction ({cfg.d_reaction}) must be <= d_entity_lab ({d_entity_lab}); "
                "increase h_reaction or reduce compound/label/tracer harmonics"
            )
            self.reaction_emb = nn.Embedding(cfg.n_fluxes, cfg.d_reaction)
            _load_frozen(self.reaction_emb, reaction_emb_table)  # no-op if None (load_state_dict)

        # Frozen token-type embedding (0 = labeling, 1 = flux)
        if cfg.d_type > 0:
            self._token_type_emb = nn.Embedding(2, cfg.d_type)
            _load_frozen(self._token_type_emb, type_emb_table)  # no-op if None (load_state_dict)

        # Measured-flux reaction indices (used by training loop; forward() also accepts them explicitly)
        fixed_idx = list(null_space_data['fixed_idx'])
        self.fixed_idx = fixed_idx
        self.register_buffer('fixed_idx_t', torch.tensor(fixed_idx, dtype=torch.long))

        # Per-reaction flux normalization statistics (applied inside forward())
        if flux_norm_stats is not None:
            self.register_buffer('flux_norm_mean',
                torch.tensor(flux_norm_stats['mean'], dtype=torch.float32))
            self.register_buffer('flux_norm_std',
                torch.tensor(flux_norm_stats['std'],  dtype=torch.float32))
        else:
            self.register_buffer('flux_norm_mean', None)
            self.register_buffer('flux_norm_std',  None)

        # Scalar value encoders (featurize + project to d_model)
        self.fraction_encoder = FractionEncoder(cfg.d_model)
        self.flux_encoder = FluxEncoder(cfg.d_model)

        # Unified input projection for all token types
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

        # Null-space basis (orthonormal columns; Z.T @ Z = I)
        Z_full = null_space_data['Z_free']
        if not isinstance(Z_full, torch.Tensor):
            Z_full = torch.tensor(Z_full, dtype=torch.float32)
        self.register_buffer('Z_full', Z_full)

        # Basis-space normalization statistics (y_mean/y_std from training data)
        # Used to denormalize logits from N(0,1) to actual y-space in forward()
        _y_mean = null_space_data.get('y_mean')
        _y_std  = null_space_data.get('y_std')
        if _y_mean is not None:
            self.register_buffer('y_norm_mean',
                torch.tensor(_y_mean, dtype=torch.float32))
            self.register_buffer('y_norm_std',
                torch.tensor(_y_std,  dtype=torch.float32))
        else:
            self.register_buffer('y_norm_mean', None)
            self.register_buffer('y_norm_std',  None)

        # Flux query matrix
        n_outputs = cfg.n_free + cfg.n_rev

        self.flux_queries = nn.Parameter(torch.empty(n_outputs, cfg.d_model))
        nn.init.xavier_uniform_(self.flux_queries)

        # Cross-attention decoder + output projection
        self.cross_attn  = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.output_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        compound_ids,             # (B, N_lab) long
        label_ids,                # (B, N_lab) long
        tracer_ids,               # (B, N_lab) long
        fractions,                # (B, N_lab, 1) float
        labeling_mask=None,       # (B, N_lab) bool, True = hidden
        flux_mask=None,           # (B, n_fx)  bool, True = hidden; requires measured_fluxes
        measured_fluxes=None,     # (B, n_fx) float, raw flux values
        measured_flux_ids=None,   # (n_fx,) long; uses self.fixed_idx_t when None
        attn_bias=None,           # (B*n_heads, n_queries, N) float additive bias
        return_attn=False,
    ):
        """
        Returns:
            fluxes:          (B, n_fluxes) satisfying Sv=0
            reversibilities: (B, n_rev) non-negative
            attn_weights:    (B, n_queries, N_total) — only when return_attn=True
        """
        B      = compound_ids.shape[0]
        device = compound_ids.device

        # ── Labeling tokens ────────────────────────────────────────────────
        lab_entity = torch.cat([
            self.compound_emb(compound_ids),  # (B, N_lab, d_comp)
            self.label_emb(label_ids),        # (B, N_lab, d_label)
            self.tracer_emb(tracer_ids),      # (B, N_lab, d_tracer)
        ], dim=-1)                            # (B, N_lab, d_entity_lab)
        N_lab = lab_entity.shape[1]

        if self.cfg.d_type > 0:
            lab_type = self._token_type_emb(
                torch.zeros(B, N_lab, dtype=torch.long, device=device)
            )
            lab_base = torch.cat([lab_entity, lab_type], dim=-1)
        else:
            lab_base = lab_entity

        # Apply learned projection to fraction: z = y + W · φ(fraction)
        fraction_proj = self.fraction_encoder(fractions)  # (B, N_lab, d_model)
        # First project base to d_model, then add projection
        lab_base_proj = self.input_proj(lab_base)
        x_lab = self.input_norm(lab_base_proj + fraction_proj)  # (B, N_lab, d_model)

        # ── Flux tokens ────────────────────────────────────────────────────
        x_flux = None
        N_fx   = 0
        if measured_fluxes is not None and self.cfg.d_reaction > 0:
            fids = measured_flux_ids if measured_flux_ids is not None else self.fixed_idx_t
            N_fx = fids.shape[0]

            # Per-reaction standardization using stored statistics
            if self.flux_norm_mean is not None:
                norm_mean = self.flux_norm_mean[fids]                      # (n_fx,)
                norm_std  = self.flux_norm_std[fids]                       # (n_fx,)
                norm_flux = (measured_fluxes - norm_mean) / norm_std       # (B, n_fx)
            else:
                norm_flux = measured_fluxes

            r_emb = self.reaction_emb(
                fids.unsqueeze(0).expand(B, -1)
            )                                                               # (B, n_fx, d_reaction)

            # Zero-pad to d_entity_lab so both token types share the same d_in
            pad_size = self.d_entity_lab - self.cfg.d_reaction
            if pad_size > 0:
                r_emb = F.pad(r_emb, (0, pad_size))                        # (B, n_fx, d_entity_lab)

            if self.cfg.d_type > 0:
                flux_type = self._token_type_emb(
                    torch.ones(B, N_fx, dtype=torch.long, device=device)
                )
                flux_base = torch.cat([r_emb, flux_type], dim=-1)
            else:
                flux_base = r_emb

            # Apply learned projection to flux: z = y + W · φ(flux)
            flux_proj = self.flux_encoder(norm_flux)  # (B, n_fx, d_model)
            flux_base_proj = self.input_proj(flux_base)
            x_flux = self.input_norm(flux_base_proj + flux_proj)  # (B, n_fx, d_model)

        # ── Combine ─────────────────────────────────────────────────────────
        if x_flux is not None:
            x = torch.cat([x_lab, x_flux], dim=1)                          # (B, N_lab+n_fx, d_model)
        else:
            x = x_lab

        # Build encoder mask (None = no masking)
        if labeling_mask is None and flux_mask is None:
            full_mask = None
        else:
            lm = labeling_mask if labeling_mask is not None else \
                 torch.zeros(B, N_lab, dtype=torch.bool, device=device)
            if x_flux is not None:
                fm = flux_mask if flux_mask is not None else \
                     torch.zeros(B, N_fx, dtype=torch.bool, device=device)
                full_mask = torch.cat([lm, fm], dim=1)
            else:
                full_mask = lm

        x = self.encoder(x, src_key_padding_mask=full_mask)
        x = torch.nan_to_num(x, nan=0.0)

        # ── Queries ─────────────────────────────────────────────────────────
        q = self.flux_queries.unsqueeze(0).expand(B, -1, -1)

        if return_attn:
            out, attn_weights = self.cross_attn(
                q, x, key_padding_mask=full_mask, attn_bias=attn_bias, return_weights=True
            )
        else:
            out = self.cross_attn(q, x, key_padding_mask=full_mask, attn_bias=attn_bias)

        logits = self.output_proj(out).squeeze(-1)  # (B, n_queries)

        n_free = self.cfg.n_free
        x_l_raw = logits[:, :n_free]
        # Denormalize from standard normal to original y-space scale/offset
        if self.y_norm_mean is not None:
            x_l = x_l_raw * self.y_norm_std + self.y_norm_mean
        else:
            x_l = x_l_raw
        fluxes = x_l @ self.Z_full.T

        reversibilities = F.softplus(logits[:, n_free:])

        fluxes          = torch.nan_to_num(fluxes,          nan=0.0)
        reversibilities = torch.nan_to_num(reversibilities, nan=0.0)

        if return_attn:
            return fluxes, reversibilities, attn_weights
        return fluxes, reversibilities
