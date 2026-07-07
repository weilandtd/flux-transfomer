"""
SigmaFlux inference engine.

Loads a trained checkpoint + config and predicts fluxes from experimental
MID measurements and (optionally) known exchange fluxes.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


class SigmaFlux:
    """
    Inference wrapper for a trained FluxDecoder model.

    Usage
    -----
    sf = SigmaFlux('paper/configs/i_orientalis.yaml',
                   'I_orientalis_20260508/model/imisor_transformer.pt')

    predictions = sf.predict(df_L, df_Vm)

    Parameters
    ----------
    config : str
        Path to the training config YAML (e.g. i_orientalis.yaml).
    checkpoint : str
        Path to the trained .pt model file.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, config, checkpoint, device=None, data_dir=None):
        with open(config) as f:
            self.cfg = yaml.safe_load(f)

        self.ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Tracer name → internal tracer string mapping.
        # The internal string is Tracer.__repr__ = f"{tracer_id}_{labeling}_{enrichment}"
        # which is what actually appears in the training labeling_df index.
        # User-facing names come from config labeling.tracers[*].name.
        self._tracer_name_to_id = {
            t['name']: f"{t['tracer_id']}_{t['labeling']}_{t['enrichment']}"
            for t in self.cfg['labeling']['tracers']
        }
        self.tracer_names = list(self._tracer_name_to_id.keys())  # human-readable

        # Flux / reversibility output names
        self.flux_names        = self.ckpt.get('flux_names', [])
        self.reversibility_names = self.ckpt.get('reversibility_names', self.flux_names)

        # Measured flux reaction IDs expected in df_Vm.
        # Fall back to config if not stored in checkpoint (older checkpoints).
        self.measured_flux_ids = (
            self.ckpt.get('measured_fluxes')
            or self.cfg.get('training', {}).get('measured_fluxes')
            or []
        )
        self._batch_size = self.cfg.get('training', {}).get('batch_size', 64)

        # ── Token skeleton ────────────────────────────────────────────────────
        # Full list of (Tracer, Metabolite, Isotopomer) tuples the model was
        # trained on.  Defines the exact token sequence for inference.
        token_index = self.ckpt.get('token_index')
        if token_index is not None:
            self._token_index = token_index          # list of 3-tuples
        else:
            # Older checkpoint without token_index — load one training sample
            print("Checkpoint does not contain token_index; "
                  "loading one training sample to derive token structure ...")
            _data_dir = data_dir or self.ckpt['data_dir']
            labeling_df, _, _ = load_data(_data_dir, max_samples=1)
            self._token_index = list(labeling_df.index)

        # Set of unique metabolite IDs the model knows about
        self._known_metabolites = {tup[1] for tup in self._token_index}

        # ── COBRA model (compound vocabulary) ─────────────────────────────────
        cfg_dict   = self.ckpt['model_config']
        model_json = self.ckpt.get('model_json')
        if not model_json or not os.path.exists(model_json):
            model_json = self.cfg.get('model_file')
        _, self.cobra_model = load_stoichiometric_matrix(model_json)

        # ── Reconstruct FluxDecoder from checkpoint ───────────────────────────
        cfg = FluxDecoderConfig(**{
            k: cfg_dict[k]
            for k in FluxDecoderConfig.__dataclass_fields__
            if k in cfg_dict
        })
        # Enrich null_space_data with normalization buffers from the state_dict.
        # Older checkpoints omit y_mean/y_std/rev_log_mean/rev_log_std from the
        # null_space_data dict even though the trained model weights have them.
        # PyTorch's load_state_dict(strict=False) cannot promote a None buffer to
        # a tensor, so we must seed the constructor with the correct values.
        nsd = dict(self.ckpt['null_space_data'])
        sd  = self.ckpt['model_state_dict']
        if nsd.get('y_mean') is None and sd.get('y_norm_mean') is not None:
            nsd['y_mean'] = sd['y_norm_mean'].numpy()
            nsd['y_std']  = sd['y_norm_std'].numpy()
        if nsd.get('rev_log_mean') is None and sd.get('rev_norm_mean') is not None:
            nsd['rev_log_mean'] = sd['rev_norm_mean'].numpy()
            nsd['rev_log_std']  = sd['rev_norm_std'].numpy()

        token_ids       = sd['token_experiment_ids']
        flux_norm_stats = self.ckpt.get('flux_norm_stats')
        self._model = FluxDecoder(
            cfg,
            token_experiment_ids=token_ids,
            null_space_data=nsd,
            flux_norm_stats=flux_norm_stats,
        )
        self._model.load_state_dict(self.ckpt['model_state_dict'], strict=False)
        self._model.to(self.device)
        self._model.eval()

        self._cfg_dict = cfg_dict
        self._nsd      = nsd

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, df_L, df_Vm=None):
        """
        Predict fluxes from labeling data and measured exchange fluxes.

        Parameters
        ----------
        df_L : pd.DataFrame
            Labeling measurements.  Required columns:
              'Tracer Name', 'Metabolite ID', 'Isotopomer', 'Fraction'

            One row = one isotopomer measurement.
            For multiple samples: add a 'Sample' column (integer 0, 1, ...).
            If absent, all rows are treated as a single sample (Sample=0).

            Metabolite IDs must match the COBRA model exactly (with compartment
            suffix, e.g. 'pyr_c', 'akg_c' — run sf.expected_metabolites() to
            see the full list).

        df_Vm : pd.DataFrame or None
            Measured exchange fluxes (optional, but use when available).
            Required columns: 'Flux Name', 'Value'
            For multiple samples: add a 'Sample' column matching df_L.

            Flux names must match measured_fluxes in the config YAML.
            Run sf.expected_measured_fluxes() to see the expected list.

        Returns
        -------
        pd.DataFrame
            Columns: 'Flux Name', 'Value'
            (plus 'Sample' if multiple samples were provided).
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

        # ── Vocabs ────────────────────────────────────────────────────────────
        _, base_tracer_names = create_tracer_mapping(labeling_df)
        vocabs = build_radial_vocabs(
            self.cobra_model, labeling_df, base_tracer_names,
            h_compound = self._cfg_dict.get('d_comp',   32) // 2,
            h_label    = self._cfg_dict.get('d_label',  10) // 2,
            h_tracer   = self._cfg_dict.get('d_tracer',  2) // 2,
        )

        (dataset, _, _, _, _) = prepare_id_dataset(
            labeling_df, flux_df, rev_df,
            vocabs          = vocabs,
            train_indices   = list(range(n_samples)),
            normalize       = False,   # targets not used at inference time
            measured_fluxes = None,
            norm_stats_input= self.ckpt.get('norm_stats'),  # applies MID z-score if model trained with it
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
                _ns_ff = self.ckpt.get('norm_stats', {})
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
        flux_names = self.flux_names or [f'flux_{i}' for i in range(flux_arr.shape[1])]
        rev_names  = self.reversibility_names or [f'rev_{n}' for n in flux_names]

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
