"""
Plot the distribution of token_information scores and tSNE of labeling token
structural embeddings.

token_information[j] = max_i |Pearson r(L_j, y_i)| across all null-space basis
components y_i — measures how informative each labeling token is for flux prediction.

Usage:
    python paper/analysis/plot_token_information.py path/to/transformer.pt
    python paper/analysis/plot_token_information.py path/to/transformer.pt \\
        --pathway-csv models/I_orientalis/metabolite_pathways.csv
"""
import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

# --- style ------------------------------------------------------------------
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})

COLORS = {
    'main':            '#269d89',
    'tsne_info_cmap':  'viridis',
    'tsne_path_pal':   'tab20',
}

OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _build_token_vecs(ckpt):
    """Return (n_tokens, d) static embedding matrix from checkpoint weights."""
    state     = ckpt['model_state_dict']
    comp_emb  = state['compound_emb.weight'].numpy()   # (n_comp, d_comp)
    lab_emb   = state['label_emb.weight'].numpy()      # (n_lab,  d_label)
    trac_emb  = state['tracer_emb.weight'].numpy()     # (n_trac, d_tracer)

    comp_idx  = {m: i for i, m in enumerate(ckpt['vocabs']['compound_vocab'])}
    label_idx = {l: i for i, l in enumerate(ckpt['vocabs']['label_vocab'])}
    trac_idx  = {t: i for i, t in enumerate(ckpt['tracer_combo_vocab'])}

    token_index = ckpt['token_index']
    vecs = np.stack([
        np.concatenate([
            comp_emb[comp_idx.get(met, 0)],
            lab_emb[label_idx.get(iso, 0)],
            trac_emb[trac_idx.get(trac, 0)],
        ])
        for trac, met, iso in token_index
    ])
    return vecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to .pt checkpoint file')
    parser.add_argument('--bins', type=int, default=40)
    parser.add_argument('--pathway-csv', default=None,
                        help='Path to metabolite_pathways.csv (skips pathway plot if absent)')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-seed', type=int, default=42)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    ti = ckpt.get('token_information')
    if ti is None:
        raise ValueError(
            "Checkpoint does not contain 'token_information'. "
            "Re-train with the updated pipeline."
        )
    ti = np.asarray(ti, dtype=float)

    token_index = ckpt.get('token_index', [])

    # Build DataFrame ----------------------------------------------------------
    if token_index and len(token_index) == len(ti):
        idx = pd.MultiIndex.from_tuples(token_index, names=['Tracer', 'Metabolite', 'Isotopomer'])
        df = pd.DataFrame({'token_information': ti}, index=idx).reset_index()
    else:
        df = pd.DataFrame({'token_information': ti})

    # Print summary stats ------------------------------------------------------
    print(f"Tokens: {len(ti)}")
    print(f"  min    = {ti.min():.4f}")
    print(f"  median = {np.median(ti):.4f}")
    print(f"  mean   = {ti.mean():.4f}")
    print(f"  max    = {ti.max():.4f}")
    print(f"  >0.10  = {(ti > 0.10).sum()}  ({100*(ti > 0.10).mean():.1f}%)")
    print(f"  >0.30  = {(ti > 0.30).sum()}  ({100*(ti > 0.30).mean():.1f}%)")

    if 'Metabolite' in df.columns:
        top = df.nlargest(15, 'token_information')[['Tracer', 'Metabolite', 'Isotopomer', 'token_information']]
        print(f"\nTop-15 tokens by token_information:")
        print(top.to_string(index=False))

    # Plot histogram -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(
        data=df, x='token_information',
        bins=args.bins, ax=ax,
        color=COLORS['main'], edgecolor='white', linewidth=0.4,
        stat='count',
    )
    ax.set_xlabel('Token information  (max |r|)')
    ax.set_ylabel('Count')
    sns.despine(ax=ax)

    fig.savefig(OUTPUT_DIR / 'token_information.svg', bbox_inches='tight')
    print(f"\nSaved {OUTPUT_DIR / 'token_information.svg'}")
    plt.show()

    # tSNE plots ---------------------------------------------------------------
    if not token_index:
        print("No token_index in checkpoint — skipping tSNE plots.")
        return

    print(f"\nComputing token embeddings ({len(token_index)} tokens)...")
    vecs = _build_token_vecs(ckpt)

    print(f"Running tSNE (perplexity={args.tsne_perplexity})...")
    xy = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=1000,
        random_state=args.tsne_seed,
    ).fit_transform(vecs)

    tsne_df = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1], 'token_information': ti})

    # Plot A: colored by token_information -------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(
        data=tsne_df, x='x', y='y', hue='token_information',
        palette=COLORS['tsne_info_cmap'],
        s=3, alpha=0.6, rasterized=True, ax=ax,
    )
    ax.legend(
        title='Token information\n(max |r|)', fontsize=6, title_fontsize=6,
        markerscale=3, frameon=False,
        bbox_to_anchor=(1.01, 1), loc='upper left',
    )
    ax.set_xlabel('tSNE 1')
    ax.set_ylabel('tSNE 2')
    sns.despine(ax=ax)
    fig.savefig(OUTPUT_DIR / 'token_information_tsne_info.svg', bbox_inches='tight')
    print(f"Saved {OUTPUT_DIR / 'token_information_tsne_info.svg'}")
    plt.show()

    # Plot B: colored by pathway -----------------------------------------------
    if args.pathway_csv is None:
        print("--pathway-csv not provided — skipping pathway tSNE plot.")
        return

    pathway_csv = pathlib.Path(args.pathway_csv)
    if not pathway_csv.exists():
        print(f"Warning: {pathway_csv} not found — skipping pathway tSNE plot.")
        return

    pw_lookup = pd.read_csv(pathway_csv).set_index('metabolite_id')['subsystem'].to_dict()
    tsne_df['pathway'] = [pw_lookup.get(met, 'Other') for _, met, _ in token_index]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(
        data=tsne_df, x='x', y='y', hue='pathway',
        palette=COLORS['tsne_path_pal'],
        s=3, alpha=0.6, rasterized=True, ax=ax,
        hue_order=sorted(tsne_df['pathway'].unique()),
    )
    ax.legend(
        fontsize=6, markerscale=3, frameon=False,
        bbox_to_anchor=(1.01, 1), loc='upper left',
    )
    ax.set_xlabel('tSNE 1')
    ax.set_ylabel('tSNE 2')
    sns.despine(ax=ax)
    fig.savefig(OUTPUT_DIR / 'token_information_tsne_pathway.svg', bbox_inches='tight')
    print(f"Saved {OUTPUT_DIR / 'token_information_tsne_pathway.svg'}")
    plt.show()


if __name__ == '__main__':
    main()
