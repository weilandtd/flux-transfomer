"""
Fisher information vs learned attention.

Fisher/D-optimal selection (features.csv) says which MID measurements are
*theoretically* most informative about the fluxes; the trained model's
cross-attention says which the model *actually* relies on. This compares them.

Two granularities:
  - per metabolite : total attention vs the metabolite's peak Fisher info
                     → does the model focus on the informative metabolites?
  - per token (met, isotopomer), colored by isotopomer
                     → finer view; exposes any isotopomer (M+k) bias.

Usage:
    python paper/analysis/fisher_attention.py \
        paper/.results/i_orientalis_20260620/attention/flux_token_attn.npz
    # --features defaults to <npz dir>/../features.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def strip_compartment(mid):
    return mid[:-2] if len(mid) > 2 and mid[-2] == '_' else mid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', help='flux_token_attn.npz from show_attention.py')
    ap.add_argument('--features', default=None,
                    help='features.csv (default: <npz dir>/../features.csv)')
    ap.add_argument('--out', default=None,
                    help='Output SVG (default: <npz dir>/fisher_attention.svg)')
    args = ap.parse_args()

    npz_dir = os.path.dirname(args.npz)
    feat_path = args.features or os.path.join(npz_dir, os.pardir, 'features.csv')

    # ── Attention received per token (sum over fluxes) ─────────────────────────
    d = np.load(args.npz, allow_pickle=True)
    recv = np.abs(d['flux_token']).sum(axis=0)              # (n_tokens,)
    labels = list(d['token_labels'])

    tok = {}                                                # (met, iso) -> attention
    for j, l in enumerate(labels):
        if l.endswith('[flux]'):                            # skip measured-flux tokens
            continue
        p = l.split(' ')
        tok[(p[0], p[1])] = tok.get((p[0], p[1]), 0.0) + recv[j]

    # ── Fisher info per (met, iso) ─────────────────────────────────────────────
    f = pd.read_csv(feat_path)
    fish = {(strip_compartment(r.metabolite_id), r.isotopomer): float(r.standalone_nats)
            for r in f.itertuples() if not np.isnan(r.standalone_nats)}

    keys = [k for k in tok if k in fish]
    t = pd.DataFrame({'met': [k[0] for k in keys], 'iso': [k[1] for k in keys],
                      'attn': [tok[k] for k in keys], 'fisher': [fish[k] for k in keys]})

    # ── Per-metabolite aggregation ─────────────────────────────────────────────
    m = t.groupby('met').agg(attn=('attn', 'sum'), fisher=('fisher', 'max')).reset_index()

    r_tok, p_tok = spearmanr(t['attn'], t['fisher'])
    r_met, p_met = spearmanr(m['attn'], m['fisher'])
    print(f"matched: {len(t)} tokens, {len(m)} metabolites")
    print(f"per-token      attn vs Fisher: Spearman r={r_tok:+.3f} p={p_tok:.1e}")
    print(f"per-metabolite attn vs Fisher: Spearman r={r_met:+.3f} p={p_met:.1e}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    out = args.out or os.path.join(npz_dir, 'fisher_attention.svg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # (A) per metabolite — the headline agreement
    ax1.scatter(m['fisher'], m['attn'], s=22, color='#2c7fb8', edgecolor='none')
    for _, r in m.nlargest(8, 'attn').iterrows():
        ax1.annotate(r['met'], (r['fisher'], r['attn']), fontsize=6,
                     xytext=(2, 2), textcoords='offset points')
    ax1.set_xlabel('Fisher information [nats]  (metabolite peak)')
    ax1.set_ylabel('total attention received')
    ax1.set_title(f'Per metabolite: r={r_met:.2f} (p={p_met:.0e})', fontsize=10)

    # (B) per token, colored by isotopomer — exposes the M+k bias
    isos = sorted(t['iso'].unique(), key=lambda s: int(s.split('+')[1]))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(isos)))
    for io, c in zip(isos, cmap):
        s = t[t['iso'] == io]
        ax2.scatter(s['fisher'], s['attn'], s=16, color=c, edgecolor='none', label=io)
    ax2.set_xlabel('Fisher information [nats]  (per isotopomer)')
    ax2.set_ylabel('attention received')
    ax2.set_title(f'Per token: r={r_tok:.2f} (p={p_tok:.0e})', fontsize=10)
    ax2.legend(frameon=False, fontsize=7, title='isotopomer', title_fontsize=7)

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
