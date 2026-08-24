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
from .model import FluxDecoder
from ..core.fbapro import fbapro


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


def _knockout_indices(flux_names, knockout_rxn_ids):
    """Map knockout reaction ids to flux-vector indices, erroring on unknown ids."""
    missing = [r for r in knockout_rxn_ids if r not in flux_names]
    if missing:
        raise ValueError(f"unknown reaction id(s) for knockout: {missing}")
    return [flux_names.index(r) for r in knockout_rxn_ids]


def _cleaned_knockout_df(base_df, full_values, n_flux, n_rev, Z, ko_idx, weights):
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
    grid[:, :n_flux] = fbapro(grid[:, :n_flux], Z=Z, vfixed=vfixed, weights=weights)
    out = base_df.copy()
    out['Value'] = grid.reshape(-1)
    return out


class SigmaFlux:
    """Inference wrapper for a trained FluxDecoder model.

    Usage: ``SigmaFlux(config_yaml, checkpoint_pt).predict(df_L, df_Vm)``
    """

    def __init__(self, config, checkpoint, device=None, data_dir=None):
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
        """
        with open(config) as f:
            self.cfg = yaml.safe_load(f)
        self.ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        spec = ModelSpec.from_checkpoint(self.ckpt, config=config, data_dir=data_dir)
        self._init_from_spec(spec, self.ckpt['model_state_dict'], device=device,
                             strict=False)

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

    def _init_from_spec(self, spec, state_dict, *, device=None, strict=True):
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

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, df_L, df_Vm=None):
        """Predict fluxes from labeling data and measured exchange fluxes.

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
            print(f"Dropping {m0_rows} M+0 row(s) — M+0 is not used by the model.")
            df_L = df_L[df_L['Isotopomer'] != 'M+0'].copy()

        # Convert human-readable tracer names → internal tracer IDs
        user_tracers = set(df_L['Tracer Name'].unique())
        unknown = user_tracers - set(self.tracer_names)
        if unknown:
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
            print(f"Ignoring {len(unknown_mets)} metabolite(s) not in training set: "
                  f"{', '.join(sorted(unknown_mets))}")
            df_L = df_L[~df_L['Metabolite ID'].isin(unknown_mets)].copy()
            user_metabolites -= unknown_mets

        # ── Build full-skeleton labeling_df ───────────────────────────────────
        # One column per sample; ALL training tokens present; zero where user
        # has no measurement.
        skeleton_index = pd.MultiIndex.from_tuples(
            self._token_index, names=['Tracer', 'Metabolite', 'Isotopomer']
        )
        labeling_df = pd.DataFrame(
            0.0,
            index   = skeleton_index,
            columns = list(range(n_samples)),
            dtype   = np.float32,
        ).sort_index()   # sort MultiIndex for efficient .loc access

        # Fill in user's values (Tracer Name already converted to internal ID)
        for col_idx, s in enumerate(samples):
            sub = df_L[df_L['Sample'] == s]
            for _, row in sub.iterrows():
                key = (row['Tracer Name'], row['Metabolite ID'], row['Isotopomer'])
                if key in labeling_df.index:
                    labeling_df.loc[key, col_idx] = np.float32(row['Fraction'])

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
        # (basis coefficients → physical mmol/gDW/h via Z_full.T).

        # Reversibility: decode from z-scored log1p space → physical values.
        # model.forward() returns raw z-score logits when rev_norm_mean is set.
        _rev_log_mean = self._nsd.get('rev_log_mean')
        if _rev_log_mean is not None:
            _rev_log_mean = np.array(_rev_log_mean, dtype=np.float32)
            _rev_log_std  = np.array(self._nsd['rev_log_std'], dtype=np.float32)
            rev_arr = np.expm1(rev_arr * _rev_log_std + _rev_log_mean).clip(0)

        return self._format_output(flux_arr, rev_arr, samples)

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
        Z = self._model.Z_full.detach().cpu().numpy().astype(np.float64)
        return _cleaned_knockout_df(
            pred, pred['Value'].to_numpy(), len(self.flux_names),
            len(self.reversibility_names), Z, ko_idx,
            np.asarray(weights, dtype=np.float64))

    # ── Introspection helpers ─────────────────────────────────────────────────

    def expected_metabolites(self):
        """Metabolite IDs the model knows about (returns sorted list)."""
        return sorted(self._known_metabolites)

    def expected_tracers(self):
        """Tracer experiment names from the config."""
        return list(self.tracer_names)

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
    interface, returning the per-reaction median over members, re-projected onto
    the null-space basis since the median (unlike any single member) does not
    satisfy ``S v = 0`` — see :meth:`_project_flux_blocks`.
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

    def predict(self, df_L, df_Vm=None):
        """Ensemble-median prediction. Members share inputs, so per-member outputs
        have identical row order and are combined positionally.

        Parameters
        ----------
        df_L
            Labeling data; see :meth:`SigmaFlux.predict`.
        df_Vm
            Measured exchange fluxes, or None.
        """
        preds  = [m.predict(df_L, df_Vm=df_Vm) for m in self.members]
        values = np.stack([p['Value'].to_numpy() for p in preds], axis=0)
        out = preds[0].copy()
        out['Value'] = self._project_flux_blocks(np.median(values, axis=0))
        return out

    def predict_knockout(self, df_L, knockout_rxn_ids, df_Vm=None):
        """Ensemble prediction with the knocked-out reactions forced to exactly zero.

        Projects the ensemble-median flux onto ``{S x = 0} ∩ {x[ko] = 0}`` weighted by the
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

        # inverse-variance weight per reaction: member disagreement, pooled over samples
        flux_blocks = values.reshape(n_members, n_samples, stride)[:, :, :n_flux]
        weights = 1.0 / (flux_blocks.var(axis=0).mean(axis=0) + 1e-8)

        ko_idx = _knockout_indices(self.flux_names, knockout_rxn_ids)
        Z = self.members[0]._model.Z_full.detach().cpu().numpy().astype(np.float64)
        return _cleaned_knockout_df(
            preds[0], np.median(values, axis=0), n_flux, n_rev, Z, ko_idx, weights)

    def _null_space_projector(self):
        """``Z (Z^T Z)^-1 Z^T`` for the members' shared basis, or None if absent.
        Varimax bases are orthonormal (Gram = I, so this collapses to ``Z Z^T``),
        but a column-normalised ``targeted_ica`` basis is not, hence the Gram solve."""
        if self._projector is not None:
            return self._projector
        inner = getattr(self.members[0], '_model', None)
        Z = getattr(inner, 'Z_full', None)
        if Z is None:
            return None
        Z = Z.detach().cpu().numpy().astype(np.float64)
        gram = Z.T @ Z
        if np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6):
            self._projector = Z @ Z.T
        else:
            self._projector = Z @ np.linalg.solve(gram, Z.T)
        return self._projector

    def _project_flux_blocks(self, values):
        """Return ``values`` with every sample's flux block re-projected onto the null
        space: each member's prediction satisfies ``S v = 0`` by construction, but a
        per-reaction median across members is a non-linear combination and does not.
        Only the flux block moves; reversibilities are not null-space coordinates."""
        P = self._null_space_projector()
        n_flux = len(self.flux_names or [])
        if P is None or not n_flux:
            return values
        stride = n_flux + len(self.reversibility_names or [])
        out = np.asarray(values, dtype=np.float64).copy()
        for start in range(0, len(out), stride):        # one block per sample
            out[start:start + n_flux] = P @ out[start:start + n_flux]
        return out

    # Introspection helpers delegate to member 0.
    def expected_metabolites(self):     return self.members[0].expected_metabolites()
    def expected_tracers(self):         return self.members[0].expected_tracers()
    def expected_measured_fluxes(self): return self.members[0].expected_measured_fluxes()


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
