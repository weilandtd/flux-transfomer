"""
tSNE visualization of pre-transformer token representations.

For each data sample the model constructs two kinds of tokens:
  - labeling tokens: compound/label/tracer embeddings + fraction scalar
  - flux tokens:     reaction embedding + measured flux scalar

This script collects those pre-transformer tokens across many samples,
subsamples for tractability, and visualizes them with tSNE.

Usage:
    python paper/analysis/analyze_learned_embeddings.py \
        /path/to/checkpoint.pt \
        --output paper/analysis/output

Requires environment setup:
    module load anaconda3/2025.12 cplex/22.1.0
    eval "$(conda shell.bash hook)"
    conda activate sigmaflux
    export PYTHONPATH=<repo>/src:${PYTHONPATH:+:$PYTHONPATH}
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


def collect_tokens(model, dataset, max_samples, device):
    """
    Run the model on up to max_samples data points and capture pre-transformer
    token vectors via a forward hook on model.encoder.

    Returns:
        tokens: (n_tokens, d_model) float32 ndarray
        types:  (n_tokens,) list of 'labeling' or 'flux'
    """
    n_lab = dataset.compound_ids.shape[0]
    n_fx = len(model.fixed_idx)

    captured = {}

    def _pre_hook(module, args):
        captured['x'] = args[0].detach().cpu()

    hook = model.encoder.register_forward_pre_hook(_pre_hook)

    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    lab_tokens, fx_tokens = [], []
    n_collected = 0

    model.eval()
    with torch.no_grad():
        for (compound_ids, label_ids, tracer_ids, fractions), targets in loader:
            if n_collected >= max_samples:
                break

            compound_ids = compound_ids.to(device)
            label_ids    = label_ids.to(device)
            tracer_ids   = tracer_ids.to(device)
            fractions    = fractions.to(device)
            targets      = targets.to(device)

            measured_fluxes = targets[:, model.fixed_idx] if n_fx > 0 else None

            model(compound_ids, label_ids, tracer_ids, fractions,
                  measured_fluxes=measured_fluxes)

            x = captured['x']  # (B, N_lab [+ n_fx], d_model)
            B = x.shape[0]

            lab_tokens.append(x[:, :n_lab, :].reshape(-1, x.shape[-1]).numpy())
            if n_fx > 0:
                fx_tokens.append(x[:, n_lab:n_lab + n_fx, :].reshape(-1, x.shape[-1]).numpy())

            n_collected += B

    hook.remove()

    lab_arr = np.vstack(lab_tokens)
    tokens  = [lab_arr]
    types   = ['labeling'] * len(lab_arr)

    if fx_tokens:
        fx_arr  = np.vstack(fx_tokens)
        tokens.append(fx_arr)
        types  += ['flux'] * len(fx_arr)

    return np.vstack(tokens), types


def subsample(tokens, types, max_tokens):
    """Stratified subsample to max_tokens total, preserving class balance."""
    unique_types = list(dict.fromkeys(types))
    per_class = max_tokens // len(unique_types)
    types_arr = np.array(types)

    idx_keep = []
    for t in unique_types:
        idx = np.where(types_arr == t)[0]
        n = min(per_class, len(idx))
        idx_keep.append(np.random.choice(idx, size=n, replace=False))

    idx_keep = np.concatenate(idx_keep)
    np.random.shuffle(idx_keep)
    return tokens[idx_keep], types_arr[idx_keep].tolist()


def main():
    parser = argparse.ArgumentParser(
        description='tSNE visualization of pre-transformer token representations'
    )
    parser.add_argument('checkpoint', help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output', default='paper/analysis/output',
                        help='Output directory for plots (default: paper/analysis/output)')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Max data samples to collect tokens from (default: 5000)')
    parser.add_argument('--max-tokens', type=int, default=20000,
                        help='Max tokens for tSNE after subsampling (default: 20000)')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='tSNE perplexity (default: 30.0)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (headless mode)')
    args = parser.parse_args()

    device = 'cpu'
    np.random.seed(args.random_state)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict  = ckpt['model_config']
    nsd       = ckpt['null_space_data']
    data_dir  = ckpt['data_dir']
    model_json = ckpt.get('model_json')
    flux_names = ckpt.get('flux_names', [])
    fixed_idx  = nsd.get('fixed_idx', [])
    measured_flux_names = [flux_names[i] for i in fixed_idx] if flux_names else None

    # ── Rebuild vocabs ────────────────────────────────────────────────────────
    print("Building vocabs...")
    labeling_df, flux_df, reversibility_df = load_data(data_dir, max_samples=None)
    _, base_tracer_names = create_tracer_mapping(labeling_df)
    _, cobra_model = load_stoichiometric_matrix(model_json)

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound=cfg_dict.get('d_comp', 32) // 2,
        h_label=cfg_dict.get('d_label', 10) // 2,
        h_tracer=cfg_dict.get('d_tracer', 2) // 2,
    )

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("Building dataset...")
    dataset, _, token_experiment_ids, _, _ = prepare_id_dataset(
        labeling_df,
        flux_df,
        reversibility_df,
        vocabs=vocabs,
        measured_fluxes=measured_flux_names,
    )
    print(f"  Dataset size: {len(dataset)} samples")

    # ── Reconstruct model ─────────────────────────────────────────────────────
    print("Reconstructing model...")
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__
                               if k in cfg_dict})
    flux_norm_stats = None
    if 'flux_norm_mean' in ckpt['model_state_dict'] and 'flux_norm_std' in ckpt['model_state_dict']:
        flux_norm_stats = {
            'mean': ckpt['model_state_dict']['flux_norm_mean'].numpy(),
            'std':  ckpt['model_state_dict']['flux_norm_std'].numpy(),
        }

    model = FluxDecoder(
        cfg,
        token_experiment_ids=token_experiment_ids,
        null_space_data=nsd,
        compound_emb_table=None,
        label_emb_table=None,
        tracer_emb_table=None,
        flux_norm_stats=flux_norm_stats,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    n_lab = dataset.compound_ids.shape[0]
    n_fx  = len(model.fixed_idx)
    print(f"  Labeling tokens/sample: {n_lab}   Flux tokens/sample: {n_fx}")

    # ── Collect pre-transformer tokens ────────────────────────────────────────
    print(f"Collecting tokens from up to {args.max_samples} samples...")
    tokens, types = collect_tokens(model, dataset, args.max_samples, device)
    print(f"  Collected {len(tokens)} tokens  "
          f"(labeling: {types.count('labeling')}  flux: {types.count('flux')})")

    if len(tokens) > args.max_tokens:
        print(f"  Subsampling to {args.max_tokens} tokens for tSNE...")
        tokens, types = subsample(tokens, types, args.max_tokens)
        print(f"  After subsample: labeling={types.count('labeling')}  flux={types.count('flux')}")

    # ── Apply tSNE ────────────────────────────────────────────────────────────
    print(f"Applying tSNE (perplexity={args.perplexity}, n={len(tokens)})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=1000,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=1,
    )
    tokens_2d = tsne.fit_transform(tokens)

    # ── Plot ──────────────────────────────────────────────────────────────────
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})

    COLORS = {
        'labeling': '#1b9e77',
        'flux':     '#d95f02',
    }

    df = pd.DataFrame({
        'tsne_1': tokens_2d[:, 0],
        'tsne_2': tokens_2d[:, 1],
        'type':   types,
    })

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for ttype in ['labeling', 'flux']:
        mask = df['type'] == ttype
        ax.scatter(
            df.loc[mask, 'tsne_1'],
            df.loc[mask, 'tsne_2'],
            label=ttype,
            alpha=0.5,
            s=5,
            color=COLORS[ttype],
            rasterized=True,
        )

    ax.set_xlabel('tSNE 1')
    ax.set_ylabel('tSNE 2')
    ax.legend(frameon=True, fontsize=10, markerscale=3)
    sns.despine(ax=ax)
    fig.tight_layout()

    output_file = output_dir / 'embedding_tsne.svg'
    fig.savefig(output_file, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

    if not args.no_show:
        plt.show()

    df.to_csv(output_dir / 'embeddings_2d.csv', index=False)
    print(f"Saved embedding data to {output_dir / 'embeddings_2d.csv'}")


if __name__ == '__main__':
    main()
