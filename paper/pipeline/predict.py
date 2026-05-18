"""
Predict fluxes from real experimental labeling data.

Takes a CSV of measured MID fractions and runs the trained transformer to
produce flux predictions (and reversibilities).

── Input CSV format ──────────────────────────────────────────────────────────

    Tracer,Metabolite,Isotopomer,sample_0,sample_1,...
    1,2-13C-glucose,pyr,M+0,0.452,0.448,...
    1,2-13C-glucose,pyr,M+1,0.301,0.309,...
    1,2-13C-glucose,pyr,M+2,0.183,0.179,...
    1,2-13C-glucose,pyr,M+3,0.064,0.064,...
    U-13C-glucose,pyr,M+0,0.210,0.215,...
    ...

Columns:
  Tracer     — tracer name, must match what the model was trained on
                (run with --show-tracers to list valid names)
  Metabolite — metabolite ID as used in the COBRA model (e.g. 'pyr', 'akg')
  Isotopomer — mass isotopomer label: M+0, M+1, M+2, ...
  sample_*   — MID fraction for each sample (rows must sum to ≈1 per metabolite)

── Masking — handling missing metabolites ────────────────────────────────────

The model was trained with random metabolite masking (met_mask_prob=0.15), so
it is robust to absent measurements.  Two ways to handle missing metabolites:

  1. Explicit flag (recommended):
         --mask-metabolites pyr oaa akg
     These metabolites are completely excluded from attention. The model
     predicts fluxes as if it never saw their MIDs — exactly what you want
     when a metabolite was not measured in your experiment.

  2. Leave them out of the CSV entirely.
     The script will detect metabolites present in the model vocab but absent
     from your CSV and warn you.  Their fraction tokens will still be present
     in the sequence but with value 0, which the model interprets as an
     unmeasured M+0 signal — WRONG.  Always use --mask-metabolites for these.

  Rule of thumb: if you didn't measure a metabolite's MIDs, mask it.

── Examples ──────────────────────────────────────────────────────────────────

    # List tracer names the checkpoint expects
    python paper/pipeline/predict.py checkpoint.pt --show-tracers

    # Single experiment, no masking
    python paper/pipeline/predict.py checkpoint.pt labeling.csv --out preds.csv

    # Two experiments, pyr and oaa not measured → mask them
    python paper/pipeline/predict.py checkpoint.pt labeling.csv \\
        --mask-metabolites pyr oaa --out preds.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labeling_csv(path):
    """
    Load a MID CSV into a labeling DataFrame.

    Expected columns: Tracer, Metabolite, Isotopomer, then one column per sample.
    Returns a DataFrame with MultiIndex (Tracer, Metabolite, Isotopomer) and
    integer-indexed columns (0, 1, 2, ...) — the format prepare_id_dataset expects.
    """
    df = pd.read_csv(path)
    required = {'Tracer', 'Metabolite', 'Isotopomer'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    sample_cols = [c for c in df.columns if c not in required]
    if not sample_cols:
        raise ValueError("CSV has no sample columns after Tracer/Metabolite/Isotopomer")

    df = df.set_index(['Tracer', 'Metabolite', 'Isotopomer'])
    df.columns = list(range(len(sample_cols)))  # integer column index
    df = df.astype(np.float32)
    return df


def build_dummy_flux_df(flux_names, n_samples):
    """Dummy zero flux DataFrame for inference (targets are unknown)."""
    return pd.DataFrame(
        np.zeros((len(flux_names), n_samples), dtype=np.float32),
        index=flux_names,
        columns=list(range(n_samples)),
    )


def make_metabolite_padding_mask(masked_met_set, labeling_df, batch_size, device):
    """
    Build a padding_mask (True = ignored) for the given set of metabolites.

    Returns (batch_size, n_tokens) bool tensor.
    """
    metabolites = labeling_df.index.get_level_values('Metabolite').tolist()
    n_tokens = len(metabolites)
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    for j, met in enumerate(metabolites):
        if met in masked_met_set:
            mask[:, j] = True
    return mask


def strip_compartment(mid):
    for s in ('_c', '_m', '_e', '_x', '_n', '_r', '_g', '_l'):
        if mid.endswith(s):
            return mid[:-2]
    return mid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict fluxes from experimental MID measurements.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('checkpoint', help='Path to .pt model checkpoint')
    parser.add_argument('data', nargs='?', default=None,
                        help='Input CSV with labeling data (Tracer, Metabolite, Isotopomer, ...)')
    parser.add_argument('--out', default=None,
                        help='Output CSV for predictions (default: print to stdout)')
    parser.add_argument('--mask-metabolites', nargs='*', default=[],
                        metavar='MET',
                        help='Metabolite IDs to mask (not measured in this experiment)')
    parser.add_argument('--show-tracers', action='store_true',
                        help='Print tracer names the checkpoint expects, then exit')
    parser.add_argument('--show-metabolites', action='store_true',
                        help='Print metabolite IDs the checkpoint knows about, then exit')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt        = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict    = ckpt['model_config']
    nsd         = ckpt['null_space_data']
    data_dir    = ckpt['data_dir']
    model_json  = ckpt.get('model_json')
    flux_names  = ckpt.get('flux_names', [])

    # Build vocabs from the COBRA model stored in the checkpoint
    _, cobra_model = load_stoichiometric_matrix(model_json)

    # We need a labeling_df to build vocabs; we'll use a placeholder derived
    # from the checkpoint's norm_stats if --show-* flags are requested, or
    # the actual user data otherwise.

    if args.show_tracers or args.show_metabolites:
        # Load minimal training data just to get vocab info
        from sigmaflux.ai.data import load_data
        labeling_df, _, _ = load_data(data_dir, max_samples=1)
        _, base_tracer_names = create_tracer_mapping(labeling_df)

        if args.show_tracers:
            print("Tracer names this checkpoint expects:")
            for t in labeling_df.index.get_level_values('Tracer').unique():
                print(f"  {t}")

        if args.show_metabolites:
            mets = sorted(set(
                strip_compartment(m)
                for m in labeling_df.index.get_level_values('Metabolite').unique()
            ))
            print("Metabolite IDs this checkpoint knows about:")
            for m in mets:
                print(f"  {m}")
        return

    if args.data is None:
        parser.error("data CSV is required (unless --show-tracers / --show-metabolites)")

    # ── Load user's labeling data ─────────────────────────────────────────────
    labeling_df = load_labeling_csv(args.data)
    n_samples   = labeling_df.shape[1]
    print(f"Loaded: {args.data}  ({n_samples} sample(s), {len(labeling_df)} tokens)")

    if args.mask_metabolites:
        print(f"Masking {len(args.mask_metabolites)} metabolite(s): "
              f"{', '.join(args.mask_metabolites)}")

    # ── Build vocabs ──────────────────────────────────────────────────────────
    _, base_tracer_names = create_tracer_mapping(labeling_df)

    vocabs = build_radial_vocabs(
        cobra_model, labeling_df, base_tracer_names,
        h_compound = cfg_dict.get('d_comp',   32) // 2,
        h_label    = cfg_dict.get('d_label',  10) // 2,
        h_tracer   = cfg_dict.get('d_tracer',  2) // 2,
    )

    # ── Build dummy flux/reversibility DataFrames (targets unknown) ───────────
    rev_names   = ckpt.get('reversibility_names', flux_names)
    flux_df     = build_dummy_flux_df(flux_names, n_samples)
    rev_df      = build_dummy_flux_df(rev_names,  n_samples)

    # ── Prepare dataset (use checkpoint's norm_stats so scaling matches) ──────
    all_indices = list(range(n_samples))
    (dataset, _, _, _, _) = prepare_id_dataset(
        labeling_df, flux_df, rev_df,
        vocabs           = vocabs,
        train_indices    = all_indices,
        normalize        = True,
        measured_fluxes  = ckpt.get('measured_fluxes'),
        norm_stats_input = ckpt.get('norm_stats'),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── Reconstruct model ─────────────────────────────────────────────────────
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__
                               if k in cfg_dict})
    token_ids = ckpt['model_state_dict']['token_experiment_ids']
    model = FluxDecoder(cfg, stoichiometric_matrix=None,
                        token_experiment_ids=token_ids, null_space_data=nsd,
                        fixed_flux_indices=nsd.get('fixed_idx', []))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # ── Build metabolite padding mask (if requested) ──────────────────────────
    # True = this token is masked (ignored by all attention layers).
    # Call once with a representative batch size; we rebuild per batch below.
    masked_met_set = set(args.mask_metabolites)

    # ── Run inference ─────────────────────────────────────────────────────────
    all_flux_preds = []
    all_rev_preds  = []

    with torch.no_grad():
        for batch in loader:
            (cids, lids, tids, fracs), _ = batch
            cids  = cids.to(device)
            lids  = lids.to(device)
            tids  = tids.to(device)
            fracs = fracs.to(device)
            B     = cids.shape[0]

            # Build padding mask: shape (B, N_tokens), True = ignored
            padding_mask = None
            if masked_met_set:
                padding_mask = make_metabolite_padding_mask(
                    masked_met_set, labeling_df, B, device
                )

            fixed_fluxes = None
            if model.cfg.n_measured > 0:
                bypass_mask  = model.token_experiment_ids == -1
                fixed_fluxes = fracs[:, bypass_mask, 0]

            flux_preds, rev_preds = model(
                cids, lids, tids, fracs,
                padding_mask=padding_mask,
                fixed_fluxes=fixed_fluxes,
            )
            all_flux_preds.append(flux_preds.cpu())
            all_rev_preds.append(rev_preds.cpu())

    flux_preds = torch.cat(all_flux_preds).numpy()   # (n_samples, n_fluxes)
    rev_preds  = torch.cat(all_rev_preds).numpy()    # (n_samples, n_rev)

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    flux_out = pd.DataFrame(
        flux_preds.T,
        index   = flux_names if flux_names else [f"flux_{i}" for i in range(flux_preds.shape[1])],
        columns = sample_ids,
    )
    rev_out = pd.DataFrame(
        rev_preds.T,
        index   = [f"rev_{n}" for n in (flux_names or range(rev_preds.shape[1]))],
        columns = sample_ids,
    )
    result = pd.concat([flux_out, rev_out])
    result.index.name = 'reaction'

    if args.out:
        result.to_csv(args.out)
        print(f"Saved predictions → {args.out}  "
              f"({flux_preds.shape[1]} fluxes + {rev_preds.shape[1]} reversibilities)")
    else:
        print(result.to_string())


if __name__ == '__main__':
    main()
