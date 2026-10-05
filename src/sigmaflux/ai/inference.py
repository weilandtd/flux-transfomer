"""
SigmaFlux inference engine.

Predicts fluxes from experimental MID measurements and (optionally) known
exchange fluxes.

A predictor can be built three ways, all ending at the same
:class:`ModelSpec` + weights pair:

- :class:`SigmaFlux` / :class:`EnsembleSigmaFlux` from a training checkpoint
  plus its config YAML (needs the run directory and the COBRA JSON);
- :meth:`EnsembleSigmaFlux.from_export` from a packaged ``.sfx`` model
  (self-contained — see :mod:`sigmaflux.ai.export`);
- :meth:`SigmaFlux.from_spec` / :meth:`EnsembleSigmaFlux.from_spec` directly.
"""

import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import flux_decoder_config, promote_norm_buffers, resolve_device
from .data import (
    build_radial_vocabs,
    build_tracer_combo_emb_table,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
    radial_vocabs_from_lists,
)
from .model import FluxDecoder, MISSING_LABEL_VALUE
from .uncertainty import _medoid_weights
from ..core.fbapro import fbapro


def bounds_from_cobra(model, flux_names, source="model"):
    """Reaction bounds read off a prepared COBRA model, aligned to *flux_names*.

    No FVA is run -- a prepared model already carries the bounds sampling and training
    saw, and re-deriving them would re-solve against whatever the solver does today.

    Args:
        model: a prepared :class:`cobra.Model`.
        flux_names: predicted reaction ids, defining the output order.
        source: what to name in the error when a reaction is missing.

    Returns ``{'lb', 'ub'}`` as float64 arrays, or None if the model does not cover
    every predicted reaction.
    """
    by_id = {r.id: (r.lower_bound, r.upper_bound) for r in model.reactions}
    missing = [n for n in flux_names if n not in by_id]
    if missing:
        print(f"WARNING: {source} is missing {len(missing)} predicted reaction(s), so it "
              f"cannot supply their bounds: {missing[:5]}")
        return None
    return {"lb": np.array([by_id[n][0] for n in flux_names], dtype=np.float64),
            "ub": np.array([by_id[n][1] for n in flux_names], dtype=np.float64)}


def bounds_from_model_json(model_json, flux_names):
    """:func:`bounds_from_cobra` for a ``model_prepared_for_mfa.json`` on disk."""
    from cobra.io.json import load_json_model

    return bounds_from_cobra(load_json_model(model_json), flux_names, source=model_json)


#: Moves the inference-time box repair is allowed. Sized by measurement, not taste: on
#: S. cerevisiae the exact projection of a trained affine model's validation predictions
#: needs at most 290 moves (p90 17, median 4), and 512 leaves headroom while reaching 0%
#: of states outside at 9.9e-16 residual. 0 restores the old clip-and-reproject repair.
#: This is affordable only because it runs once per prediction, not per training step.
_REPAIR_DEPTH = 512

#: Hold every repaired flux this far off its bounds. An exactly-feasible state may sit *on*
#: a bound, and roughly half these reactions have a lower bound of 0, so "on it" means zero
#: flux -- which degenerates the EMU forward pass below its 0.5e-6 row-sum threshold.
#: Measured per member on I. orientalis: 1000/1000 validation states carried at least one
#: sub-threshold flux without this, 0/1000 with it, at no cost in R2. The polytope admits a
#: uniform inset of only ~1.6e-5 before it empties, and 1e-5 already costs R2 (0.93 -> 0.70),
#: so this is deliberately near the bottom of the usable range.
_REPAIR_MIN_SLACK = 1e-6


@dataclass
class ModelSpec:
    """Everything a predictor needs except the trained weights — config-free,
    COBRA-free, HDF5-free. Treat instances as immutable; ``_cache`` is an
    internal memo for the embedding tables, excluded from equality."""

    model_config:         dict
    vocabs:               dict          # compound/label/tracer vocab + h_* + token_dim
    tracer_combo_vocab:   list
    token_index:          list          # (Tracer, Metabolite, Isotopomer), TRAINING order
    token_experiment_ids: list
    flux_names:           list
    reversibility_names:  list
    measured_fluxes:      list
    null_space_data:      dict          # y_*/rev_log_* already promoted
    flux_bounds:          dict = None   # {'lb', 'ub'}, aligned to flux_names
    flux_norm_stats:      dict = None   # {'mean', 'std'}
    norm_stats:           dict = field(default_factory=dict)
    h_reaction:           int = 0
    h_type:               int = 0
    tracer_display_names: dict = field(default_factory=dict)   # display name -> internal id
    batch_size:           int = 64
    _cache:               dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        n_tok, n_ids = len(self.token_index), len(self.token_experiment_ids)
        if n_ids and n_tok != n_ids:
            raise ValueError(
                f"token_index has {n_tok} entries but token_experiment_ids has {n_ids}; "
                "the two describe the same token sequence and must agree"
            )

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, ckpt, *, config=None, model_json=None, data_dir=None):
        """Build a spec from a training checkpoint, optionally with its config YAML.
        ``config`` supplies human-readable tracer names and the batch size;
        without it, tracer names fall back to the internal ids.

        Parameters
        ----------
        ckpt
            Loaded training checkpoint (``torch.load`` output).
        config
            Path to the run's config YAML.
        model_json
            Path to the COBRA JSON; used only for pre-``vocabs`` checkpoints.
        data_dir
            Training data directory, used to derive ``token_index`` when the
            checkpoint predates it.
        """
        cfg = {}
        if config is not None:
            with open(config) as fh:
                cfg = yaml.safe_load(fh) or {}

        state_dict = ckpt['model_state_dict']
        null_space_data = promote_norm_buffers(ckpt['null_space_data'], state_dict)

        token_index = ckpt.get('token_index')
        if token_index is None:
            # Older checkpoint: derive the token structure from one training sample.
            print("Checkpoint does not contain token_index; "
                  "loading one training sample to derive token structure ...")
            labeling_df, _, _ = load_data(data_dir or ckpt['data_dir'], max_samples=1)
            token_index = list(labeling_df.index)
        token_index = [tuple(t) for t in token_index]

        vocabs = ckpt.get('vocabs')
        if not vocabs or 'compound_vocab' not in vocabs:
            # Pre-`vocabs` checkpoint: the COBRA JSON is the only way back.
            resolved = model_json or ckpt.get('model_json')
            if not resolved or not os.path.exists(resolved):
                resolved = cfg.get('model_file')
            _, cobra_model = load_stoichiometric_matrix(resolved)
            labeling_df = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    token_index, names=['Tracer', 'Metabolite', 'Isotopomer'])
            )
            _, base_tracers = create_tracer_mapping(labeling_df)
            cfg_dict = ckpt['model_config']
            vocabs = build_radial_vocabs(
                cobra_model, labeling_df, base_tracers,
                h_compound=cfg_dict.get('d_comp', 32) // 2,
                h_label=cfg_dict.get('d_label', 10) // 2,
                h_tracer=cfg_dict.get('d_tracer', 2) // 2,
            )

        flux_names = list(ckpt.get('flux_names', []))

        # Bounds are mandatory at prediction time. Older checkpoints predate them, so
        # fall back to the prepared model the run was trained against -- the same
        # legacy-recovery pattern token_index and vocabs use above.
        flux_bounds = ckpt.get('flux_bounds')
        if flux_bounds is None and flux_names:
            from .export import _find_network_json
            network = _find_network_json(ckpt, model_json=model_json, data_dir=data_dir,
                                         config=config)
            if network is None:
                raise ValueError(
                    "Checkpoint stores no flux_bounds and its prepared model "
                    "(model_prepared_for_mfa.json) could not be found, so predictions "
                    "cannot be constrained to the reaction bounds. Pass data_dir=<run "
                    "directory>, or re-export the model with sigmaflux-export."
                )
            flux_bounds = bounds_from_model_json(network, flux_names)

        tracers = {
            t['name']: f"{t['tracer_id']}_{t['labeling']}_{t['enrichment']}"
            for t in cfg.get('labeling', {}).get('tracers', [])
        }

        token_ids = ckpt.get('token_experiment_ids')
        if token_ids is None:
            token_ids = state_dict['token_experiment_ids'].tolist()

        return cls(
            model_config         = dict(ckpt['model_config']),
            vocabs               = {k: v for k, v in vocabs.items() if not k.endswith('_emb_table')},
            tracer_combo_vocab   = list(ckpt.get('tracer_combo_vocab', []) or []),
            token_index          = token_index,
            token_experiment_ids = [int(i) for i in token_ids],
            flux_names           = flux_names,
            reversibility_names  = list(ckpt.get('reversibility_names', flux_names)),
            measured_fluxes      = list(ckpt.get('measured_fluxes')
                                        or cfg.get('training', {}).get('measured_fluxes')
                                        or []),
            null_space_data      = null_space_data,
            flux_bounds          = flux_bounds,
            flux_norm_stats      = ckpt.get('flux_norm_stats'),
            norm_stats           = ckpt.get('norm_stats') or {},
            h_reaction           = int(ckpt.get('h_reaction', 0)),
            h_type               = int(ckpt.get('h_type', 0)),
            tracer_display_names = tracers,
            batch_size           = int(cfg.get('training', {}).get('batch_size', 64)),
        )

    # ── Derived data ──────────────────────────────────────────────────────────

    def emb_tables(self):
        """Vocabularies plus their radial embedding tables (memoized).

        Rebuilt from the three stored vocabularies, so this needs neither the
        COBRA model nor the training data.
        """
        if 'vocabs' not in self._cache:
            self._cache['vocabs'] = radial_vocabs_from_lists(
                self.vocabs['compound_vocab'],
                self.vocabs['label_vocab'],
                self.vocabs['tracer_vocab'],
                h_compound=self.vocabs.get('h_compound'),
                h_label=self.vocabs.get('h_label'),
                h_tracer=self.vocabs.get('h_tracer'),
                verbose=False,
            )
        return self._cache['vocabs']

    def tracer_combo_emb_table(self):
        """Embedding table over ``tracer_combo_vocab`` (memoized); combined tracers
        are summed base embeddings, mirroring :func:`~sigmaflux.ai.data.prepare_dataset`
        at training time. Falls back to the base table when no combo vocab was recorded."""
        if 'tracer_combo' not in self._cache:
            base = self.emb_tables()['tracer_emb_table']
            combo = (build_tracer_combo_emb_table(
                        self.tracer_combo_vocab, self.vocabs['tracer_vocab'], base)
                     if self.tracer_combo_vocab else base)
            self._cache['tracer_combo'] = combo
        return self._cache['tracer_combo']

    def tracer_map(self):
        """Display name → internal tracer id, with internal ids mapping to themselves."""
        internal = {t: t for t in self.vocabs.get('tracer_vocab', [])}
        return {**internal, **self.tracer_display_names}


    def build_decoder(self, state_dict, *, device=None, strict=True):
        """Rebuild a :class:`FluxDecoder` and load *state_dict* into it. Frozen
        embedding tables are regenerated from the vocabularies and harmonics
        rather than read from the weights, which is why an exported model can
        omit them entirely.

        Parameters
        ----------
        state_dict
            Trained weights for one ensemble member.
        device
            Target device; resolved via :func:`resolve_device`.
        strict
            Passed through to ``load_state_dict``.
        """
        tables = self.emb_tables()
        n_reactions = len(self.flux_names) or self.model_config.get('n_fluxes')
        model = FluxDecoder(
            flux_decoder_config(self.model_config),
            token_experiment_ids=torch.as_tensor(self.token_experiment_ids, dtype=torch.int64),
            compound_emb_table=tables['compound_emb_table'],
            label_emb_table=tables['label_emb_table'],
            tracer_emb_table=self.tracer_combo_emb_table(),
            reaction_emb_table=radial_embedding_table(n_reactions, self.h_reaction),
            type_emb_table=radial_embedding_table(2, self.h_type),
            null_space_data=self.null_space_data,
            flux_norm_stats=self.flux_norm_stats,
        )
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        model.to(resolve_device(device))
        return model


#: A simulated mass fraction may sit this far outside [0, 1] and still be rounding rather
#: than a failed solve. The real failures overshoot by many orders of magnitude.
MID_TOL = 1e-6
def _knockout_indices(flux_names, knockout_rxn_ids):
    """Map knockout reaction ids to flux-vector indices, erroring on unknown ids."""
    missing = [r for r in knockout_rxn_ids if r not in flux_names]
    if missing:
        raise ValueError(f"unknown reaction id(s) for knockout: {missing}")
    return [flux_names.index(r) for r in knockout_rxn_ids]


def _inverse_member_variance(flux_blocks):
    """Per-reaction FBApro weights: inverse member variance, pooled over samples.

    ``flux_blocks`` is ``(n_members, n_samples, n_flux)``.
    """
    return 1.0 / (flux_blocks.var(axis=0).mean(axis=0) + 1e-8)


def _fbapro_decoder(model, V, vfixed, weights):
    """FBApro onto the decoder's flux space ``flux_offset + span(Z_full)``.

    ``Z_full`` spans only the free directions; pinned fluxes live in ``flux_offset``, so the
    projection runs on ``V - flux_offset`` and adds it back.
    """
    Z = model.Z_full.detach().cpu().numpy().astype(np.float64)
    v0 = model.flux_offset.detach().cpu().numpy().astype(np.float64)
    return fbapro(V - v0, Z=Z, vfixed=vfixed - v0, weights=weights) + v0


def _cleaned_knockout_df(base_df, full_values, n_flux, n_rev, model, ko_idx, weights):
    """Replace the flux block of a prediction with its knockout-cleaned projection.

    ``full_values`` is the long-format 'Value' column (flux then reversibility per sample);
    only the flux block is projected onto {S x = 0, x[ko] = 0} via
    :func:`~sigmaflux.core.fbapro.fbapro` (knockouts pinned to 0 through ``vfixed``).
    Reversibilities are left untouched. Returns a copy of ``base_df`` with a new 'Value'.
    """
    stride = n_flux + n_rev
    grid = np.asarray(full_values, dtype=np.float64).reshape(-1, stride).copy()
    vfixed = np.full(n_flux, np.nan)
    vfixed[list(ko_idx)] = 0.0
    grid[:, :n_flux] = _fbapro_decoder(model, grid[:, :n_flux], vfixed, weights)
    out = base_df.copy()
    out['Value'] = grid.reshape(-1)
    return out


class SigmaFlux:
    """Inference wrapper for a trained FluxDecoder model.

    Usage: ``SigmaFlux(config_yaml, checkpoint_pt).predict(df_L, df_Vm)``
    """

    def __init__(self, config, checkpoint, device=None, data_dir=None,
                 repair_depth=None):
        """Load a predictor from a config YAML and a training checkpoint.

        Parameters
        ----------
        config
            Path to the run's config YAML.
        checkpoint
            Path to a ``.pt`` training checkpoint.
        device
            Target device; resolved via :func:`resolve_device`.
        data_dir
            Training data directory, used only for legacy checkpoints without
            a stored ``token_index``.
        repair_depth
            Moves allowed for the box repair; ``None`` uses :data:`_REPAIR_DEPTH`,
            ``0`` restores the old clip-and-reproject operator.
        """
        with open(config) as f:
            self.cfg = yaml.safe_load(f)
        self.ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        spec = ModelSpec.from_checkpoint(self.ckpt, config=config, data_dir=data_dir)
        self._init_from_spec(spec, self.ckpt['model_state_dict'], device=device,
                             strict=False, repair_depth=repair_depth)
        # Only :meth:`mfa_context` needs these -- it has to locate the run's prepared
        # model, which the checkpoint itself does not carry.
        self._config_path, self._checkpoint, self._data_dir = config, checkpoint, data_dir

    @classmethod
    def from_spec(cls, spec, state_dict, *, device=None, strict=True):
        """Build a predictor from a :class:`ModelSpec` and one member's weights —
        no config YAML, run directory, or COBRA JSON needed; this is the path an
        exported model takes.

        Parameters
        ----------
        spec
            Model architecture and vocabularies.
        state_dict
            Trained weights for this member.
        device
            Target device; resolved via :func:`resolve_device`.
        strict
            Passed through to ``load_state_dict``.
        """
        self = cls.__new__(cls)
        self.cfg = None
        self.ckpt = None
        self._init_from_spec(spec, state_dict, device=device, strict=strict)
        return self

    def _init_from_spec(self, spec, state_dict, *, device=None, strict=True,
                        repair_depth=None):
        """Shared setup core for :meth:`__init__` and :meth:`from_spec`.

        Parameters
        ----------
        spec
            Model architecture and vocabularies.
        state_dict
            Trained weights to load into the rebuilt decoder.
        device
            Target device; resolved via :func:`resolve_device`.
        strict
            Passed through to ``load_state_dict``.
        """
        self.spec = spec
        self.device = resolve_device(device)
        # Set by __init__ when the model came from a run directory; a from_spec or .sfx
        # predictor has no run to find a prepared model in, and mfa_context() says so.
        self._config_path = self._checkpoint = self._data_dir = None
        self._mfa = None

        # Tracer name → internal tracer string mapping. The internal string is
        # Tracer.__repr__ = f"{tracer_id}_{labeling}_{enrichment}", which is what
        # appears in the training labeling_df index; user-facing names come from
        # the config (or, for an export, the frozen map stored with the model).
        self._tracer_name_to_id = spec.tracer_display_names or dict(
            zip(spec.vocabs['tracer_vocab'], spec.vocabs['tracer_vocab'])
        )
        self.tracer_names = list(self._tracer_name_to_id.keys())

        self.flux_names          = spec.flux_names
        self.reversibility_names = spec.reversibility_names
        self.measured_flux_ids   = spec.measured_fluxes
        self._batch_size         = spec.batch_size

        # Token skeleton: the exact (Tracer, Metabolite, Isotopomer) sequence the
        # model was trained on.
        self._token_index = spec.token_index
        self._known_metabolites = {tup[1] for tup in self._token_index}

        # Embedding tables come from the stored vocabularies, so no COBRA model
        # is needed; kept as an attribute only for backwards compatibility.
        self.cobra_model = None
        self._vocabs = spec.emb_tables()

        self._model = spec.build_decoder(state_dict, device=self.device, strict=strict)
        self._cfg_dict = spec.model_config
        self._nsd      = spec.null_space_data

        # Bound projection: the reaction bounds, plus the origin the rays are cast from.
        # The origin is the training flux mean rebuilt through the decoder's own map,
        # v = y @ Z_free.T + flux_offset. The offset holds the directions
        # split_constant_directions() pinned, whose Z_free rows are zero; without it
        # those reactions rebuild as 0 and the origin falls outside the box.
        if not spec.flux_bounds:
            raise ValueError(
                "This model carries no flux_bounds, so its predictions cannot be "
                "constrained to the reaction bounds. Re-export it with sigmaflux-export, "
                "or build the predictor with data_dir=<run directory> so the bounds can "
                "be read from model_prepared_for_mfa.json."
            )
        # The box repair runs here, on a trained model's output, and deliberately not
        # inside the forward pass -- see FluxProjection's docstring for the measurement.
        self._repair_depth = (_REPAIR_DEPTH if repair_depth is None else int(repair_depth))
        self._flux_projection = None
        self._flux_lb = np.asarray(spec.flux_bounds['lb'], dtype=np.float64)
        self._flux_ub = np.asarray(spec.flux_bounds['ub'], dtype=np.float64)
        _Z_free = np.asarray(self._nsd['Z_free'], dtype=np.float64)
        # Projector onto range(Z_free), factored through the Gram matrix rather than
        # formed as an (n_fluxes, n_fluxes) dense matrix. targeted_ica's basis is not
        # orthonormal -- Gram condition 5.5e3 on I. orientalis -- so Z @ Z.T is not it.
        self._null_basis = _Z_free
        self._null_dual = np.linalg.solve(_Z_free.T @ _Z_free, _Z_free.T).T
        # Absent on pre-offset checkpoints and regenerated exports; zeros matches them.
        _offset = self._nsd.get('flux_offset')
        _offset = (np.zeros(_Z_free.shape[0]) if _offset is None
                   else np.asarray(_offset, dtype=np.float64))
        self._null_offset = _offset
        self._flux_origin = (np.asarray(self._nsd['y_mean'], dtype=np.float64)
                             @ _Z_free.T) + _offset
        slack = np.minimum(self._flux_origin - self._flux_lb,
                           self._flux_ub - self._flux_origin)
        if slack.min() <= 0:
            worst = int(np.argmin(slack))
            raise ValueError(
                f"The projection origin (training flux mean) is not strictly inside the "
                f"reaction bounds: {self.flux_names[worst]} has slack {slack.min():.3g}. "
                f"Every ray from it would have zero length there, collapsing predictions "
                f"onto the origin. The bounds and the checkpoint likely disagree."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, df_L, df_Vm=None, *, verbose=True, with_mid=False):
        """Predict fluxes from labeling data and measured exchange fluxes.

        Tokens the experiment does not measure are filled with ``MISSING_LABEL_VALUE``
        rather than a fraction of zero: a zero fraction is a real measurement -- "this
        isotopomer is absent" -- so filling missing tokens with it told the model
        something the data never said, for every metabolite the platform cannot resolve
        and for every tracer the experiment did not run. The sentinel keeps the token
        attended (its identity still informs the rest of the sequence) while remaining
        unambiguous, since it lies outside a fraction's valid range.

        Parameters
        ----------
        df_L : pd.DataFrame
            Columns 'Tracer Name', 'Metabolite ID', 'Isotopomer', 'Fraction' (one row
            per isotopomer measurement); optional 'Sample' column for multiple samples
            (default: all rows treated as Sample=0). Metabolite IDs must match the
            COBRA model exactly, with compartment suffix — see expected_metabolites().
        df_Vm : pd.DataFrame or None
            Measured exchange fluxes, columns 'Flux Name', 'Value' (optional 'Sample'
            matching df_L); names must match expected_measured_fluxes().
        with_mid : bool
            Also simulate the labeling each predicted flux state implies, and return
            ``(predictions, mids)`` instead of just the predictions. One EMU forward
            pass per sample, so it is off by default; see :meth:`simulate_mid`.
        verbose : bool
            Print the input-coverage report. An ensemble sets this only for its first
            member, since every member sees identical inputs.

        Returns
        -------
        pd.DataFrame with columns 'Flux Name', 'Value' (plus 'Sample' if multiple
        samples were provided).
        """
        # Handle single-sample case
        if 'Sample' not in df_L.columns:
            df_L = df_L.copy()
            df_L['Sample'] = 0
            if df_Vm is not None and 'Sample' not in df_Vm.columns:
                df_Vm = df_Vm.copy()
                df_Vm['Sample'] = 0

        samples   = sorted(df_L['Sample'].unique().tolist())
        n_samples = len(samples)

        # Drop M+0 — training data never includes it (fractions sum to 1,
        # so M+0 is fully determined by M+1…M+N and was excluded at generation time).
        m0_rows = (df_L['Isotopomer'] == 'M+0').sum()
        if m0_rows > 0:
            if verbose:
                print(f"Dropping {m0_rows} M+0 row(s) — M+0 is not used by the model.")
            df_L = df_L[df_L['Isotopomer'] != 'M+0'].copy()

        # Convert human-readable tracer names → internal tracer IDs
        user_tracers = set(df_L['Tracer Name'].unique())
        unknown = user_tracers - set(self.tracer_names)
        if unknown and verbose:
            print(f"WARNING: unknown tracer name(s): {unknown}")
            print(f"  Expected: {self.tracer_names}")
        df_L = df_L.copy()
        df_L['Tracer Name'] = df_L['Tracer Name'].map(self._tracer_name_to_id)

        # Metabolites present in user data
        user_metabolites = set(df_L['Metabolite ID'].unique())

        # Drop metabolites not in the training set (e.g. _m compartment variants
        # absent from training data). Report once as a summary, not per-token.
        unknown_mets = user_metabolites - self._known_metabolites
        if unknown_mets:
            if verbose:
                print(f"Ignoring {len(unknown_mets)} metabolite(s) not in training set: "
                      f"{', '.join(sorted(unknown_mets))}")
            df_L = df_L[~df_L['Metabolite ID'].isin(unknown_mets)].copy()
            user_metabolites -= unknown_mets

        # ── Build the labeling frame on the training token skeleton ───────────
        # Rows stay in token_index order rather than being lexicographically sorted:
        # norm_stats' mid_mean/mid_std are *positional* arrays in training order, so a
        # re-sorted frame would z-score every token against another token's statistics.
        # One pivot instead of a row-by-row assignment loop -- bootstrapped inputs push
        # the sample count into the hundreds, and the loop was O(rows x samples) in Python.
        skeleton_index = pd.MultiIndex.from_tuples(
            self._token_index, names=['Tracer', 'Metabolite', 'Isotopomer']
        )
        col_of = {s: i for i, s in enumerate(samples)}
        if df_L.empty:
            # Nothing survived the tracer/metabolite filters. An all-NaN frame keeps the
            # shape contract so this fails as "no coverage" below rather than on a pivot
            # over an empty frame, which produces no index levels to reindex against.
            wide = pd.DataFrame(np.nan, index=skeleton_index, columns=range(n_samples))
        else:
            wide = (df_L.assign(_col=df_L['Sample'].map(col_of))
                        .pivot_table(index=['Tracer Name', 'Metabolite ID', 'Isotopomer'],
                                     columns='_col', values='Fraction', aggfunc='mean')
                        .reindex(index=skeleton_index, columns=range(n_samples)))

        # NaN is "no measurement for this token in this sample". Filled with
        # MISSING_LABEL_VALUE, not 0 -- a real 0 fraction is itself a measurement ("this
        # isotopomer is absent"), so zero-filling told the model something the data never
        # said. The sentinel is outside a fraction's valid [0, 1] range and finite through
        # the encoder's log(|x|+1)/|x|/sign(x) featurization (model.py), and (unlike the
        # attention-masking this replaced) keeps the token visible, so its identity still
        # informs the rest of the sequence.
        labeling_df = wide.fillna(MISSING_LABEL_VALUE).astype(np.float32)

        if verbose:
            self._report_coverage(wide)

        # ── Flux DataFrames ───────────────────────────────────────────────────
        flux_df, rev_df = self._build_flux_dfs(df_Vm, n_samples, samples)

        # ── Dataset ───────────────────────────────────────────────────────────
        # Vocabularies are fixed by the token skeleton, so they are built once in
        # __init__ rather than rebuilt on every predict() call.
        (dataset, _, _, _, _) = prepare_id_dataset(
            labeling_df, flux_df, rev_df,
            vocabs          = self._vocabs,
            train_indices   = list(range(n_samples)),
            normalize       = False,   # targets not used at inference time
            measured_fluxes = None,
            norm_stats_input= self.spec.norm_stats,  # applies MID z-score if model trained with it
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)

        # ── Pre-build fixed_fluxes tensor ─────────────────────────────────────
        # Normalize to match training if checkpoint used normalize_targets.
        # Shape: (n_samples, n_measured) — sliced per batch in the loop below.
        _fixed_flux_raw = None
        if len(self._model.fixed_idx) > 0:
            fixed_idx = list(self._nsd.get('fixed_idx', []))
            fvals = flux_df.values[fixed_idx, :].T.copy()   # (n_samples, n_measured)
            # Only apply norm_stats normalization when the model has no internal FluxNorm
            # (i.e., trained with normalize=True so targets were pre-normalized).
            if self._model.flux_norm_mean is None:
                _ns_ff = self.spec.norm_stats
                if 'flux_mean' in _ns_ff:
                    _fmean = np.array(_ns_ff['flux_mean'], dtype=np.float32)[fixed_idx]
                    _fstd  = np.array(_ns_ff['flux_std'],  dtype=np.float32)[fixed_idx]
                    fvals  = (fvals - _fmean) / _fstd
            _fixed_flux_raw = torch.tensor(fvals, dtype=torch.float32)

        # ── Inference ─────────────────────────────────────────────────────────
        all_flux, all_rev = [], []
        batch_offset = 0
        with torch.no_grad():
            for batch in loader:
                (cids, lids, tids, fracs), _ = batch
                cids  = cids.to(self.device)
                lids  = lids.to(self.device)
                tids  = tids.to(self.device)
                fracs = fracs.to(self.device)
                B     = cids.shape[0]

                fixed_fluxes = None
                if _fixed_flux_raw is not None:
                    fixed_fluxes = _fixed_flux_raw[batch_offset:batch_offset + B].to(self.device)

                fp, rp = self._model(cids, lids, tids, fracs,
                                     measured_fluxes=fixed_fluxes)
                all_flux.append(fp.cpu())
                all_rev.append(rp.cpu())
                batch_offset += B

        flux_arr = torch.cat(all_flux).numpy()
        rev_arr  = torch.cat(all_rev).numpy()

        # Flux: already denormalized inside model.forward() via y_norm_mean/std
        # (basis coefficients → physical mmol/gDW/h via Z_full.T). The reconstruction
        # guarantees S v = 0 but not the reaction bounds, so every state is projected
        # back inside them before it leaves the predictor -- a prediction that is not a
        # flux state is not a usable answer.
        flux_arr = self._project_into_bounds(flux_arr)

        # Reversibility: decode from z-scored log1p space → physical values.
        # model.forward() returns raw z-score logits when rev_norm_mean is set.
        _rev_log_mean = self._nsd.get('rev_log_mean')
        if _rev_log_mean is not None:
            _rev_log_mean = np.array(_rev_log_mean, dtype=np.float32)
            _rev_log_std  = np.array(self._nsd['rev_log_std'], dtype=np.float32)
            rev_arr = np.expm1(rev_arr * _rev_log_std + _rev_log_mean).clip(0)

        out = self._format_output(flux_arr, rev_arr, samples)
        if with_mid:
            return out, self.simulate_mid(flux_arr, rev_arr, samples)
        return out

    def _project_into_bounds(self, flux_arr):
        """Pull each predicted flux state back toward the reaction bounds.

        The decoder reconstructs ``v = y @ Z_full.T + flux_offset``, which satisfies
        ``S v = 0`` by construction but nothing keeps inside the box. Unprojected states
        are not merely inaccurate: measured on I. orientalis, every one of 1000 validation
        states sat outside, 35 reactions out apiece, and the EMU forward pass on them
        returned a largest "fraction" of 9.9e+39 -- not a MID at all.

        Clip into ``[lb, ub]``, then project back onto the affine null space. The second
        step is exact, so ``S v = 0`` is restored to 5.8e-10 while the clip has already
        removed the gross excursions the forward pass cannot survive. It does not reach
        exact box feasibility -- the two sets meet at too shallow an angle near these
        predictions for that to be cheap, and iterating toward it (Dykstra) makes the
        predictions monotonically *worse*, R2 0.267 at one sweep against 0.190 at fifty.

        This replaced a single ray from the training flux mean,
        ``x = x0 + min(1, alpha) * (v - x0)``, whose ``alpha`` was one ratio-test scalar
        for the whole vector -- so the reaction with the narrowest interval relative to the
        error on it set the step for every other flux. Measured on I. orientalis,
        ``alpha < 1`` on **100%** of predictions (median 0.0046), gated 99.6% of the time
        by ``AHSERL2_c``, ``METB1_c``, ``SADT_c`` and ``SLFAT_c`` -- sulfur assimilation,
        boxes ~0.027 wide against a glucose uptake of 8.6. Every state came back on the
        training mean: per-reaction R2 -0.005 against 0.592 unprojected, while Spearman rho
        barely moved (0.771 vs 0.781), a positive scalar preserving rank and erasing scale.

        Bound violation is not itself a symptom of a bad model -- the pre-pool run that
        scored 0.986 violated harder (worst 275 mmol/gDW/h, 27.7% of its largest flux) and
        was simply never projected. Only the operator matters.

        Args:
            flux_arr: predicted fluxes, shape (n_samples, n_fluxes), in flux_names order.
        """
        if self._repair_depth > 0:
            return self._project_exactly(flux_arr)
        clipped = np.clip(flux_arr.astype(np.float64), self._flux_lb, self._flux_ub)
        r = clipped - self._null_offset
        return self._null_offset + (r @ self._null_basis) @ self._null_dual.T

    def _project_exactly(self, flux_arr):
        """Euclidean projection of the latent onto the box, so both constraints hold.

        The clip-and-reproject operator this falls back to reaches neither constraint: it
        restores ``S v = 0`` but leaves the state outside the box, and iterating it toward
        feasibility makes predictions worse. Projecting the latent instead makes
        ``S v = 0`` automatic, since the reconstruction never leaves ``range(Z_free)``, and
        lets the box be satisfied exactly -- which is what the EMU forward pass needs, as
        it returns non-MIDs on states outside the bounds.

        Args:
            flux_arr: predicted fluxes, shape (n_samples, n_fluxes), in flux_names order.
        """
        if self._flux_projection is None:
            from .flux_projection import FluxProjection
            self._flux_projection = FluxProjection(
                self._null_basis,
                self._flux_lb - self._null_offset,
                self._flux_ub - self._null_offset,
                depth=self._repair_depth,
                min_slack=_REPAIR_MIN_SLACK,
            )
        v = flux_arr.astype(np.float64)
        y = (v - self._null_offset) @ self._null_dual
        with torch.no_grad():
            y = self._flux_projection(torch.from_numpy(y)).numpy()
        return y @ self._null_basis.T + self._null_offset

    def _report_coverage(self, wide):
        """Print how much of the trained token set the experiment actually measured.

        A tracer the model was trained on but the experiment never ran used to vanish
        silently -- only *unknown* tracer names warned -- while contributing a full block
        of zero-valued tokens. Reporting per tracer makes that visible at the point it
        happens.

        Args:
            wide: the token skeleton frame, NaN wherever there is no measurement.
        """
        measured = wide.notna().any(axis=1).groupby(level='Tracer').agg(['sum', 'count'])
        print(f"Input coverage: {int(measured['sum'].sum())}/{len(wide)} "
              f"trained tokens measured (the rest are filled with MISSING_LABEL_VALUE, "
              f"not a real 0)")
        for tracer, row in measured.iterrows():
            note = "  <-- tracer not measured at all" if row['sum'] == 0 else ""
            print(f"  {tracer}: {int(row['sum'])}/{int(row['count'])}{note}")

    def predict_knockout(self, df_L, knockout_rxn_ids, df_Vm=None, weights=None):
        """Predict fluxes for a knockout, forcing the knocked-out reactions to exactly zero.

        Runs :meth:`predict` and then projects the predicted flux onto
        ``{S x = 0} ∩ {x[ko] = 0}`` (minimal, inverse-variance-weighted adjustment), so the
        knocked-out reactions are exactly zero while the rest stay close to the model's
        estimate. A single model has no ensemble spread to weight by, so ``weights`` is
        required here; :class:`EnsembleSigmaFlux` derives them from the member spread
        automatically. See :func:`~sigmaflux.core.fbapro.fbapro`.

        Parameters
        ----------
        df_L, df_Vm
            Labeling data and measured exchange fluxes; see :meth:`predict`.
        knockout_rxn_ids : list of str
            Reaction ids to force to zero (must be in :attr:`flux_names`).
        weights : array-like, shape (n_fluxes,)
            Per-reaction non-negative weights (e.g. inverse variance); larger = held closer
            to the prediction. Required for a single model.

        Returns
        -------
        pd.DataFrame with the same columns as :meth:`predict`, knockouts at exactly 0.
        """
        if weights is None:
            raise ValueError(
                "predict_knockout on a single SigmaFlux needs per-reaction `weights` "
                "(inverse-variance); the weighting normally comes from the ensemble "
                "spread -- use EnsembleSigmaFlux, or pass weights explicitly.")
        pred = self.predict(df_L, df_Vm=df_Vm)
        ko_idx = _knockout_indices(self.flux_names, knockout_rxn_ids)
        return _cleaned_knockout_df(
            pred, pred['Value'].to_numpy(), len(self.flux_names),
            len(self.reversibility_names), self._model, ko_idx,
            np.asarray(weights, dtype=np.float64))

    # ── Introspection helpers ─────────────────────────────────────────────────

    def expected_metabolites(self):
        """Metabolite IDs the model knows about (returns sorted list)."""
        return sorted(self._known_metabolites)

    def expected_tracers(self):
        """Tracer experiment names from the config."""
        return list(self.tracer_names)

    def tracer_ids(self):
        """Display tracer name -> internal tracer id, as used in the token index.

        Anything that has to line measured data up against the token index (or against a
        simulated labeling vector, whose tracer level is the same internal id) needs this
        map; rebuilding it from the config at each call site is how the two drift apart.
        """
        return dict(self._tracer_name_to_id)

    def expected_measured_fluxes(self):
        """Reaction IDs to provide in df_Vm."""
        return list(self.measured_flux_ids)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_flux_dfs(self, df_Vm, n_samples, samples):
        """Build zero-filled flux/reversibility frames and fill in measured values.

        Parameters
        ----------
        df_Vm
            Measured exchange fluxes, or None.
        n_samples
            Number of sample columns to build.
        samples
            Sample ids, in column order.
        """
        flux_df = pd.DataFrame(
            np.zeros((len(self.flux_names), n_samples), dtype=np.float32),
            index   = self.flux_names,
            columns = list(range(n_samples)),
        )
        rev_df = pd.DataFrame(
            np.zeros((len(self.reversibility_names), n_samples), dtype=np.float32),
            index   = self.reversibility_names,
            columns = list(range(n_samples)),
        )
        if df_Vm is not None and not df_Vm.empty:
            for col_idx, s in enumerate(samples):
                sub = df_Vm[df_Vm['Sample'] == s][['Flux Name', 'Value']]
                for _, row in sub.iterrows():
                    rxn = row['Flux Name']
                    if rxn in flux_df.index:
                        flux_df.loc[rxn, col_idx] = np.float32(row['Value'])
        return flux_df, rev_df

    def mfa_context(self):
        """The prepared EMU network this model's fluxes are simulated through (memoized).

        Built from the run's own ``model_prepared_for_mfa.json`` rather than re-derived:
        rebuilding re-solves FVA and depends on the solver still behaving as it did, so
        the saved model is the one training and sampling actually saw. Expansion targets
        are restricted to the metabolites the measurement platform maps, which is what
        keeps the network small enough to simulate thousands of states through.

        Returns ``(mfa_model, reaction_ids, labeling_to, labeling_index)``.
        """
        if self._mfa is not None:
            return self._mfa
        # Imported here rather than at module scope: the EMU machinery pulls in cobra and
        # the expansion code, which a predictor that never simulates should not pay for.
        from ..pipeline._benchmark import build_mfa_context, load_prepared_model
        from ..pipeline._paths import load_config

        if self._checkpoint is None and self._data_dir is None:
            raise ValueError("this predictor was built without a run directory, so the "
                             "prepared model it would simulate through cannot be located")
        roots = [d for d in (self._data_dir,
                             os.path.dirname(os.path.abspath(self._checkpoint))
                             if self._checkpoint else None,
                             os.path.dirname(os.path.dirname(os.path.abspath(
                                 self._checkpoint))) if self._checkpoint else None)
                 if d]
        run_dir = next((d for d in roots
                        if os.path.isfile(os.path.join(d, "model_prepared_for_mfa.json"))),
                       None)
        if run_dir is None:
            raise FileNotFoundError(
                f"no model_prepared_for_mfa.json in any of {roots}; labeling cannot be "
                f"simulated without the prepared model")
        cfg, root = load_config(self._config_path) if self._config_path else (self.cfg, None)
        cobra_model, mapped = load_prepared_model(run_dir, cfg, root)
        mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
            cobra_model, cfg["labeling"]["tracers"],
            pd.Series(0.0, index=[r.id for r in cobra_model.reactions]),
            "predict_mid", with_backward=False, restrict_targets_to=mapped)
        self._mfa = (mfa_model, reaction_ids, labeling_to, labeling_index)
        return self._mfa

    def flux_bounds(self):
        """``{flux name: (lb, ub)}`` -- the box predictions are projected into.

        Exposed so a caller resampling measured fluxes can truncate its draws to the same
        box the prediction will be projected into, rather than to the error bar alone.
        """
        return {n: (float(lo), float(hi)) for n, lo, hi
                in zip(self.flux_names, self._flux_lb, self._flux_ub)}

    def simulate_mid(self, flux_arr, rev_arr, samples, context=None):
        """Labeling each predicted flux state implies, in this predictor's input schema.

        One EMU forward pass per row, so the cost is linear in the number of states --
        the reason :meth:`predict` only does this when asked. Reversibility is predicted
        for a subset of the reactions and fills with zero elsewhere, which is inert:
        MFAModel multiplies it by the reversibility mask before simulating.

        Args:
            flux_arr: predicted fluxes, shape (n_samples, n_fluxes), in flux_names order.
            rev_arr: matching reversibilities, in reversibility_names order.
            samples: sample ids, in row order.
            context: an ``mfa_context()`` to reuse, so an ensemble builds one network
                rather than one per member.

        Returns a long frame with the same columns the measured labeling arrives in:
        'Sample', 'Tracer Name', 'Metabolite ID', 'Isotopomer', 'Fraction'.
        """
        from ..pipeline._benchmark import forward_labeling

        mfa_model, reaction_ids, labeling_to, labeling_index = context or self.mfa_context()
        flux_pos = {n: i for i, n in enumerate(self.flux_names)}
        rev_pos = {n: i for i, n in enumerate(self.reversibility_names)}
        take_flux = [flux_pos.get(r) for r in reaction_ids]
        take_rev = [rev_pos.get(r) for r in reaction_ids]

        def _align(row, take):
            return np.array([0.0 if i is None else float(row[i]) for i in take])

        values = np.empty((len(samples), len(labeling_index)), dtype=float)
        failed = invalid = 0
        for i in range(len(samples)):
            try:
                row = forward_labeling(mfa_model, _align(flux_arr[i], take_flux),
                                       _align(rev_arr[i], take_rev), labeling_to)
            except Exception:
                # A predicted state can be one whose EMU system is singular -- a subnet
                # with near-zero row sums has no labeling to solve for. That state has no
                # labeling, which is a fact about it, not a reason to lose every other
                # state in the call.
                values[i] = np.nan
                failed += 1
                continue
            # A mass fraction is in [0, 1] for any steady-state flux state with
            # non-negative exchange, so a row outside it is a failed solve that did not
            # raise -- an ill-conditioned subnet returning a number rather than an error.
            # Measured on feasible, in-box states: fractions up to 4.5e10. Passing one on
            # is worse than losing the state, because nothing downstream can tell it from
            # a real fraction, and a single one sets a bootstrap percentile.
            if (not np.all(np.isfinite(row)) or row.min() < -MID_TOL
                    or row.max() > 1.0 + MID_TOL):
                values[i] = np.nan
                invalid += 1
                continue
            values[i] = row
        if failed:
            warnings.warn(f"{failed}/{len(samples)} predicted flux state(s) have no "
                          f"solvable labeling; their fractions are NaN")
        if invalid:
            warnings.warn(f"{invalid}/{len(samples)} predicted flux state(s) simulated a "
                          f"fraction outside [0, 1]; the solve failed without raising and "
                          f"their fractions are NaN")
        # Report the tracer under the name the caller supplied it as, not the internal
        # repr, so the result lines up with the df_L it was predicted from.
        display = {v: k for k, v in self._tracer_name_to_id.items()}
        levels = {lvl: labeling_index.get_level_values(lvl).to_numpy()
                  for lvl in ("Tracer", "Metabolite", "Isotopomer")}
        out = pd.DataFrame({
            "Sample": np.repeat(np.asarray(samples), len(labeling_index)),
            "Tracer Name": np.tile([display.get(t, t) for t in levels["Tracer"]], len(samples)),
            "Metabolite ID": np.tile(levels["Metabolite"], len(samples)),
            "Isotopomer": np.tile(levels["Isotopomer"], len(samples)),
            "Fraction": values.ravel(),
        })
        return out

    def _format_output(self, flux_arr, rev_arr, samples):
        """Assemble the long-format 'Sample'/'Flux Name'/'Value' output DataFrame.

        Parameters
        ----------
        flux_arr
            Predicted fluxes, shape (n_samples, n_fluxes).
        rev_arr
            Predicted reversibilities, shape (n_samples, n_rev).
        samples
            Sample ids, in row order.
        """
        flux_names = self.flux_names or [f'flux_{i}' for i in range(flux_arr.shape[1])]
        rev_names  = self.reversibility_names or [f'rev_{i}' for i in range(rev_arr.shape[1])]
        # zip() would silently truncate a mismatch and mislabel every row.
        if len(flux_names) != flux_arr.shape[1] or len(rev_names) != rev_arr.shape[1]:
            raise ValueError(
                f"output names do not match model outputs: {len(flux_names)} flux names "
                f"for {flux_arr.shape[1]} fluxes, {len(rev_names)} reversibility names "
                f"for {rev_arr.shape[1]} reversibilities"
            )

        records = []
        for s_idx, s in enumerate(samples):
            for name, val in zip(flux_names, flux_arr[s_idx]):
                records.append({'Sample': s, 'Flux Name': name, 'Value': float(val)})
            for name, val in zip(rev_names, rev_arr[s_idx]):
                records.append({'Sample': s, 'Flux Name': name, 'Value': float(val)})

        result = pd.DataFrame(records)
        if result['Sample'].nunique() == 1:
            result = result.drop(columns='Sample')
        return result


class EnsembleSigmaFlux:
    """Ensemble of trained SigmaFlux members sharing one architecture and split.

    Loads every ``model_*.pt`` in ``ensemble_dir`` and exposes the same ``predict``
    interface, returning the medoid_convex aggregate over members (see
    :func:`sigmaflux.ai.uncertainty._medoid_weights`) -- a convex combination, so it
    stays a flux state like the mean does, chosen over the mean for measuring closer to
    the real-data reference on SC (see the comment in ``predict()`` below).
    """

    def __init__(self, config, ensemble_dir, device=None, data_dir=None):
        """Load every ``model_*.pt`` member found in *ensemble_dir*.

        Parameters
        ----------
        config
            Path to the run's config YAML, shared by every member.
        ensemble_dir
            Directory containing ``model_*.pt`` checkpoint files.
        device
            Target device; resolved via :func:`resolve_device`.
        data_dir
            Training data directory, used only for legacy checkpoints without
            a stored ``token_index``.
        """
        self.member_paths = sorted(glob.glob(os.path.join(ensemble_dir, 'model_*.pt')))
        if not self.member_paths:
            raise FileNotFoundError(f"No model_*.pt in ensemble dir: {ensemble_dir}")
        members = [SigmaFlux(config, p, device=device, data_dir=data_dir)
                   for p in self.member_paths]
        # Any member's checkpoint carries the shared train/val split (val_indices).
        self._adopt(members, split_source=self.member_paths[0])

    @classmethod
    def from_spec(cls, spec, state_dicts, *, device=None, member_ids=None, strict=True):
        """Build an ensemble from one shared :class:`ModelSpec` and N weight sets.

        Parameters
        ----------
        spec
            Model architecture and vocabularies, shared by every member.
        state_dicts
            One trained weight set per member.
        device
            Target device; resolved via :func:`resolve_device`.
        member_ids
            Labels for ``self.member_paths``; defaults to ``range(len(state_dicts))``.
        strict
            Passed through to ``load_state_dict``.
        """
        self = cls.__new__(cls)
        self.member_paths = list(member_ids or range(len(state_dicts)))
        members = [SigmaFlux.from_spec(spec, sd, device=device, strict=strict)
                   for sd in state_dicts]
        self._adopt(members, split_source=None)
        return self

    @classmethod
    def from_export(cls, path, *, device=None):
        """Load a packaged ``.sfx`` model. A one-member export is still an ensemble."""
        from .export import load_model
        return load_model(path, device=device)

    def _adopt(self, members, *, split_source):
        """Re-expose the attributes the pipeline entry points read off a predictor.

        Parameters
        ----------
        members
            Loaded :class:`SigmaFlux` members, sharing one architecture.
        split_source
            Path the shared train/val split was read from, or None.
        """
        self.members = members
        m0 = members[0]
        self.spec                = m0.spec
        self.flux_names          = m0.flux_names
        self.reversibility_names = m0.reversibility_names
        self.tracer_names        = m0.tracer_names
        self.measured_flux_ids   = m0.measured_flux_ids
        self.device              = m0.device
        self._tracer_name_to_id  = m0._tracer_name_to_id
        self._token_index        = m0._token_index
        self._known_metabolites  = m0._known_metabolites
        self.split_source        = split_source
        self.manifest            = None
        self.name                = None
        self.model_version       = None
        self.source_path         = None
        self._projector          = None   # built lazily by _null_space_projector()

    def predict(self, df_L, df_Vm=None, *, aggregate=True, with_mid=False):
        """Ensemble prediction. Members share inputs, so per-member outputs have
        identical row order and are combined positionally.

        Parameters
        ----------
        df_L
            Labeling data; see :meth:`SigmaFlux.predict`.
        df_Vm
            Measured exchange fluxes, or None.
        aggregate
            ``True`` (default): medoid_convex aggregate over members, unchanged schema.
            ``False``: every member's own raw prediction, long-format with an added
            'Member' column (one row per Sample x Member x Flux Name) -- no aggregation,
            no other post-processing.
        with_mid
            Also simulate each member's labeling, returning ``(predictions, mids)``.
            The mids carry a 'Member' column whether or not the predictions are
            aggregated: the EMU pass is nonlinear, so the labeling of the aggregate flux
            state is not the aggregate labeling, and collapsing members here would decide
            that question for the caller. Every member simulates through member 0's
            network -- they share a model, and building one per member would prepare
            the same EMU expansion twenty times.
        """
        # Members see identical inputs, so only the first reports coverage.
        preds, mids = [], []
        context = None
        if with_mid:
            # Members share one model, so they share one network: seed every member's
            # memo with member 0's rather than preparing the same EMU expansion once
            # per member.
            context = self.members[0].mfa_context()
            for m in self.members:
                m._mfa = context
        # Per-member labeling is only simulated when per-member output is what is asked
        # for. The aggregate path simulates the aggregate state instead, once -- see the
        # comment on mid_out below.
        per_member_mid = with_mid and not aggregate
        for i, m in enumerate(self.members):
            if per_member_mid:
                pred, mid = m.predict(df_L, df_Vm=df_Vm, verbose=(i == 0), with_mid=True)
                mids.append(mid.assign(Member=i))
                print(f"     simulated labeling for member {i + 1}/{len(self.members)}",
                      flush=True)
            else:
                pred = m.predict(df_L, df_Vm=df_Vm, verbose=(i == 0))
            preds.append(pred)
        mid_out = pd.concat(mids, ignore_index=True) if per_member_mid else None
        if not aggregate:
            raw = []
            for midx, p in enumerate(preds):
                p = p.copy()
                if 'Sample' not in p.columns:      # single-sample call dropped it
                    p.insert(0, 'Sample', 0)
                p.insert(1, 'Member', midx)
                raw.append(p)
            combined = pd.concat(raw, ignore_index=True)
            return (combined, mid_out) if with_mid else combined
        # medoid_convex, not the plain mean or a per-reaction median: every member
        # predicts v = y @ Z_full.T and so lies in the stoichiometric null space, and
        # medoid_convex's simplex weights are a convex combination of those -- it stays
        # a flux state (S.v=0, inside the box whenever every member is), the same
        # guarantee the mean had. A per-reaction median is not a convex combination and
        # can leave the null space entirely (measured: max|S.v| = 5.74 on E. coli core
        # LC-MS, versus 2e-6 for the mean). Unlike the plain mean, medoid_convex weights
        # concentrate near the coordinatewise median instead of spreading uniformly
        # across every member, which measured better on real SC data (pooled flux R²
        # 0.242 vs the mean's 0.228, `[[medoid_convex_production_default]]` /
        # compare_aggregation_sc.py) with no feasibility cost, since it is still a
        # convex combination.
        values = np.stack([p['Value'].to_numpy() for p in preds], axis=0)
        weights = _medoid_weights(values, how="medoid_convex")
        out = preds[0].copy()
        out['Value'] = weights @ values
        if with_mid:
            # The labeling of the flux state we report, not the average of the labelings
            # each member implies. The EMU pass is nonlinear, so those differ, and only
            # the first is the labeling that corresponds to the published fluxes. It is
            # also better conditioned: the aggregate is a convex combination and so stays
            # inside the box, where individual members often sit on or outside it and
            # return fractions the forward pass cannot produce.
            mid_out = self.members[0].simulate_mid(
                *self._aggregate_arrays(out, df_L), context=context)
        return (out, mid_out) if with_mid else out

    def _aggregate_arrays(self, out, df_L):
        """``out``'s long Value column back as (flux_arr, rev_arr, samples).

        ``_format_output`` writes every flux name then every reversibility name for one
        sample before moving to the next, so the column reshapes by position. Names are
        not unique across the two blocks -- a reaction appears in both -- so position is
        what identifies a value here, not a pivot on the name.
        """
        samples = sorted(df_L['Sample'].unique().tolist()) if 'Sample' in df_L else [0]
        n_flux, n_rev = len(self.flux_names), len(self.reversibility_names)
        block = n_flux + n_rev
        expected = block * len(samples)
        if len(out) != expected:
            raise ValueError(f"cannot reshape {len(out)} rows into {len(samples)} samples "
                             f"x ({n_flux} fluxes + {n_rev} reversibilities)")
        wide = out['Value'].to_numpy().reshape(len(samples), block)
        return wide[:, :n_flux], wide[:, n_flux:], samples
    def predict_knockout(self, df_L, knockout_rxn_ids, df_Vm=None):
        """Ensemble prediction with the knocked-out reactions forced to exactly zero.

        Projects the ensemble-mean flux onto ``{S x = 0}∩{x[ko] = 0}`` weighted by the
        inverse variance across members: the rerouting needed to zero the knockout is
        absorbed by the reactions the ensemble is least certain about, keeping the confident
        ones close to their predicted values. Returns the same long DataFrame as
        :meth:`predict`. See :func:`~sigmaflux.core.fbapro.fbapro`.

        Parameters
        ----------
        df_L, df_Vm
            Labeling data and measured exchange fluxes; see :meth:`SigmaFlux.predict`.
        knockout_rxn_ids : list of str
            Reaction ids to force to zero (must be in :attr:`flux_names`).
        """
        preds  = [m.predict(df_L, df_Vm=df_Vm) for m in self.members]
        values = np.stack([p['Value'].to_numpy(dtype=np.float64) for p in preds], axis=0)
        n_flux = len(self.flux_names or [])
        n_rev  = len(self.reversibility_names or [])
        stride = n_flux + n_rev
        n_members, n_rows = values.shape
        n_samples = n_rows // stride

        flux_blocks = values.reshape(n_members, n_samples, stride)[:, :, :n_flux]
        weights = _inverse_member_variance(flux_blocks)

        ko_idx = _knockout_indices(self.flux_names, knockout_rxn_ids)
        return _cleaned_knockout_df(
            preds[0], values.mean(axis=0), n_flux, n_rev, self.members[0]._model, ko_idx,
            weights)

    # Introspection helpers delegate to member 0.
    def expected_metabolites(self):     return self.members[0].expected_metabolites()
    def expected_tracers(self):         return self.members[0].expected_tracers()
    def expected_measured_fluxes(self): return self.members[0].expected_measured_fluxes()
    def tracer_ids(self):               return self.members[0].tracer_ids()
    def flux_bounds(self):              return self.members[0].flux_bounds()


def load_predictor(config, checkpoint, data_dir=None, device=None):
    """Build a predictor, preferring an ensemble when one is available.

    Looks for an ensemble directory in priority order: ``checkpoint`` itself if
    it's a directory, ``<dir(checkpoint)>/ensemble``, or ``<data_dir>/ensemble``;
    if found and non-empty it wins even when ``checkpoint`` names a single ``.pt``
    file. A ``.sfx`` path is detected by suffix and loaded directly (``config`` is
    unused in that case).

    Parameters
    ----------
    config
        Path to the run's config YAML.
    checkpoint
        Path to a ``.pt``/``.sfx`` file, or an ensemble directory.
    data_dir
        Training data directory, used only for legacy checkpoints without a
        stored ``token_index``.
    device
        Target device; resolved via :func:`resolve_device`.
    """
    from ._sfx import SFX_SUFFIX
    if checkpoint and str(checkpoint).endswith(SFX_SUFFIX) and os.path.isfile(checkpoint):
        from .export import load_model
        print(f"Loading EXPORTED model: {checkpoint}")
        return load_model(checkpoint, device=device)

    candidates = []
    if checkpoint and os.path.isdir(checkpoint):
        candidates.append(checkpoint)
    if checkpoint:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(checkpoint)), 'ensemble'))
    if data_dir:
        candidates.append(os.path.join(data_dir, 'ensemble'))

    for cand in candidates:
        members = glob.glob(os.path.join(cand, 'model_*.pt')) if cand else []
        if members:
            print(f"Loading ENSEMBLE predictor: {len(members)} members from {cand}")
            return EnsembleSigmaFlux(config, cand, device=device, data_dir=data_dir)

    print(f"Loading single-model predictor: {checkpoint}")
    return SigmaFlux(config, checkpoint, device=device, data_dir=data_dir)
