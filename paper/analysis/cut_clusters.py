"""
Cut the reaction dendrogram at a chosen threshold and report top metabolites per cluster.

Step 1 (no --cut): print how many reaction clusters form at each dendrogram level.
Step 2 (with --cut N): for each cluster at that level, list top metabolites.

Usage:
    # Step 1 — survey thresholds
    python paper/cut_clusters.py I_orientalis_20260508/flux_token_attn.npz

    # Step 2 — analyse clusters at a chosen cut
    python paper/cut_clusters.py ...npz --cut 8.5 --top-mets 5
"""

import argparse
import os

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


def parse_met(token_label):
    return token_label.split(' ')[0]


def build_matrix(flux_token, token_labels):
    """Aggregate tokens → metabolites (max-norm isotopomer), abs + row z-score."""
    met_tags    = [parse_met(lbl) for lbl in token_labels]
    unique_mets = []
    seen        = set()
    for m in met_tags:
        if m not in seen:
            unique_mets.append(m)
            seen.add(m)

    met_matrix         = np.zeros((flux_token.shape[0], len(unique_mets)))
    chosen_isotopomers = []
    for col_i, met in enumerate(unique_mets):
        indices = [j for j, m in enumerate(met_tags) if m == met]
        norms   = [np.linalg.norm(flux_token[:, j]) for j in indices]
        best    = indices[int(np.argmax(norms))]
        met_matrix[:, col_i] = flux_token[:, best]
        chosen_isotopomers.append(token_labels[best])

    mat = np.abs(met_matrix)
    mu  = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1,  keepdims=True).clip(min=1e-9)
    mat = (mat - mu) / std
    return mat, unique_mets, chosen_isotopomers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz')
    parser.add_argument('--cut',      type=float, default=None,
                        help='Dendrogram distance threshold to cut (see table first)')
    parser.add_argument('--top-mets', type=int,   default=5,
                        help='Top metabolites to show per cluster (default 5)')
    parser.add_argument('--method',   default='ward')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    d            = np.load(args.npz, allow_pickle=True)
    flux_token   = d['flux_token']
    flux_names   = list(d['flux_names'])
    token_labels = list(d['token_labels'])

    mat, unique_mets, chosen_isos = build_matrix(flux_token, token_labels)

    row_link = sch.linkage(ssd.pdist(mat, metric='euclidean'), method=args.method)

    # ── Step 1: threshold survey ───────────────────────────────────────────────
    distances = row_link[:, 2]          # merge heights
    d_min, d_max = distances[0], distances[-1]

    # Sample 35 evenly-spaced thresholds + every unique merge height
    thresholds = np.unique(np.concatenate([
        np.linspace(d_min, d_max, 35),
        distances,
    ]))
    thresholds = thresholds[thresholds >= d_min]

    print(f"Reaction dendrogram  ({len(flux_names)} reactions, {args.method} linkage)")
    print(f"Distance range: {d_min:.3f} – {d_max:.3f}\n")
    print(f"{'Threshold':>10}  {'N clusters':>10}  bar")
    print("─" * 55)
    prev_n = None
    for t in thresholds:
        labels = sch.fcluster(row_link, t, criterion='distance')
        n      = len(set(labels))
        if n == prev_n:
            continue          # skip duplicates — only show when cluster count changes
        prev_n = n
        bar = '█' * min(n, 60)
        marker = '  ◄' if n in (2, 3, 4, 5, 6, 8, 10) else ''
        print(f"{t:10.3f}  {n:10d}  {bar}{marker}")

    print()
    print("Run again with  --cut <threshold>  to analyse clusters at that level.")

    if args.cut is None:
        return

    # ── Step 2: cluster analysis ───────────────────────────────────────────────
    labels   = sch.fcluster(row_link, args.cut, criterion='distance')
    n_clust  = len(set(labels))
    print(f"\n{'═'*60}")
    print(f"Cut at {args.cut:.3f}  →  {n_clust} clusters")
    print(f"{'═'*60}\n")

    # Sort clusters by size (largest first)
    cluster_ids   = sorted(set(labels), key=lambda c: -(labels == c).sum())

    for rank, cid in enumerate(cluster_ids, 1):
        member_mask    = labels == cid
        member_indices = np.where(member_mask)[0]
        n_members      = len(member_indices)

        sub        = mat[member_indices]          # (n_members, n_mets)
        mean_score = sub.mean(axis=0)             # (n_mets,)  — mean z-score per metabolite
        top_idx    = np.argsort(-mean_score)[:args.top_mets]

        print(f"Cluster {rank}  ({n_members} reactions)")
        print(f"  Top metabolites (by mean attention z-score in cluster):")
        for m_i in top_idx:
            iso_label = chosen_isos[m_i].split(' ')[1]   # just M+N
            print(f"    {unique_mets[m_i]:<20s}  {mean_score[m_i]:+.3f}  [{iso_label}]")

        # Print reaction list (first 12, then ...)
        rxn_list = [flux_names[i] for i in member_indices]
        if len(rxn_list) <= 12:
            rxn_str = '  '.join(rxn_list)
        else:
            rxn_str = '  '.join(rxn_list[:12]) + f'  … (+{len(rxn_list)-12} more)'
        print(f"  Reactions: {rxn_str}")
        print()


if __name__ == '__main__':
    main()
