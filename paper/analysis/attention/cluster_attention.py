"""
Biclustered heatmap: what labeling metabolites inform what fluxes.

Loads flux_token_attn.npz, aggregates the 538 tokens → 48 metabolites
(picks the most influential isotopomer per metabolite), then produces a
dendrogram-clustered heatmap with Ward linkage on both axes.

Usage:
    python paper/cluster_attention.py I_orientalis_20260508/flux_token_attn.npz
    python paper/cluster_attention.py ...npz --out figure.pdf --top-fluxes 80
    python paper/cluster_attention.py ...npz --normalize col   # per-metabolite z-score
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns


def parse_met(token_label):
    """Extract metabolite name from 'met M+N [[tracer]]'."""
    return token_label.split(' ')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz',
                        help='flux_token_attn.npz saved by show_attention.py')
    parser.add_argument('--out', default=None,
                        help='Output file (default: same dir as npz, cluster_heatmap.pdf)')
    parser.add_argument('--top-fluxes', type=int, default=0,
                        help='Keep top N fluxes by total influence (0 = keep all, default)')
    parser.add_argument('--normalize', choices=['none', 'row', 'col'], default='row',
                        help='"row" = z-score per flux (default), "col" = per metabolite, "none" = raw abs')
    parser.add_argument('--method', default='ward',
                        help='Linkage method for clustering (default: ward)')
    args = parser.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.npz), 'cluster_heatmap.pdf')

    # ── Load matrix ───────────────────────────────────────────────────────────
    d           = np.load(args.npz, allow_pickle=True)
    flux_token  = d['flux_token']       # (386, 538)
    flux_names  = list(d['flux_names'])
    token_labels = list(d['token_labels'])

    # ── Aggregate tokens → metabolites ────────────────────────────────────────
    # For each unique metabolite, pick the isotopomer with the highest total
    # absolute influence across all fluxes (L2 norm of its column).
    met_names  = [parse_met(lbl) for lbl in token_labels]
    unique_mets = []
    seen        = set()
    for m in met_names:
        if m not in seen:
            unique_mets.append(m)
            seen.add(m)

    met_matrix = np.zeros((flux_token.shape[0], len(unique_mets)))  # (386, 48)
    chosen_isotopomers = []   # track which isotopomer was chosen per metabolite

    for col_i, met in enumerate(unique_mets):
        token_indices = [j for j, m in enumerate(met_names) if m == met]
        # Influence = L2 norm of each token's column (how much it moves any flux)
        norms = [np.linalg.norm(flux_token[:, j]) for j in token_indices]
        best  = token_indices[int(np.argmax(norms))]
        met_matrix[:, col_i] = flux_token[:, best]
        chosen_isotopomers.append(token_labels[best])

    # ── Take absolute values — we care about association magnitude, not sign ──
    mat = np.abs(met_matrix)   # (386, 48)

    # ── Filter to top-N most influential fluxes ───────────────────────────────
    flux_influence = mat.sum(axis=1)   # total attention received per flux
    if args.top_fluxes > 0 and args.top_fluxes < len(flux_names):
        top_idx    = np.argsort(-flux_influence)[:args.top_fluxes]
        top_idx    = np.sort(top_idx)   # keep original order before clustering
        mat        = mat[top_idx]
        row_labels = [flux_names[i] for i in top_idx]
    else:
        row_labels = flux_names

    col_labels = unique_mets

    # ── Normalize ─────────────────────────────────────────────────────────────
    if args.normalize == 'row':
        # Per-flux z-score: highlights which metabolites each flux relies on
        mu  = mat.mean(axis=1, keepdims=True)
        std = mat.std(axis=1,  keepdims=True).clip(min=1e-9)
        mat = (mat - mu) / std
    elif args.normalize == 'col':
        # Per-metabolite z-score: highlights which fluxes each metabolite informs
        mu  = mat.mean(axis=0, keepdims=True)
        std = mat.std(axis=0,  keepdims=True).clip(min=1e-9)
        mat = (mat - mu) / std

    cmap   = 'RdBu_r' if args.normalize != 'none' else 'YlOrRd'
    center = 0.0 if args.normalize != 'none' else None
    vmax   = np.percentile(np.abs(mat), 95)
    vmin   = -vmax if args.normalize != 'none' else 0

    # ── Cluster ───────────────────────────────────────────────────────────────
    # Rows (fluxes): cluster by their attention profile across metabolites
    row_dist = ssd.pdist(mat,   metric='euclidean')
    col_dist = ssd.pdist(mat.T, metric='euclidean')
    row_link = sch.linkage(row_dist, method=args.method)
    col_link = sch.linkage(col_dist, method=args.method)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows, n_cols = mat.shape
    fig_h = max(10, n_rows * 0.14 + 3)
    fig_w = max(10, n_cols * 0.40 + 4)

    g = sns.clustermap(
        mat,
        row_linkage  = row_link,
        col_linkage  = col_link,
        figsize      = (fig_w, fig_h),
        cmap         = cmap,
        center       = center,
        vmin         = vmin,
        vmax         = vmax,
        xticklabels  = col_labels,
        yticklabels  = row_labels,
        dendrogram_ratio = (0.12, 0.08),
        cbar_pos     = (0.02, 0.8, 0.03, 0.15),
        linewidths   = 0,
    )

    # Rotate x-axis labels for readability
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=45, ha='right', fontsize=9
    )
    fontsize_y = max(4, min(8, 200 // n_rows))
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0, fontsize=fontsize_y
    )

    norm_label = {'row': 'row z-score (per flux)', 'col': 'col z-score (per metabolite)',
                  'none': 'absolute attention'}[args.normalize]
    g.figure.suptitle(
        f'Flux ← Labeling attention  ({n_rows} fluxes × {n_cols} metabolites)\n'
        f'Normalization: {norm_label}  |  Linkage: {args.method}',
        y=1.01, fontsize=11
    )

    g.figure.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f'Saved → {out_path}')

    # ── Print chosen isotopomers ──────────────────────────────────────────────
    print(f'\nChosen representative isotopomers ({len(unique_mets)} metabolites):')
    for met, iso in zip(unique_mets, chosen_isotopomers):
        print(f'  {met:<20s}  {iso}')


if __name__ == '__main__':
    main()
