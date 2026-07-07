"""
Pathway x metabolite attention map.

1. Build the cross-attention reaction x metabolite-token matrix from flux_token_attn.npz
   (|flux_token| = |Z @ attn_mean|, the reaction-level cross-attention), max-aggregated
   over each metabolite's (isotopomer, tracer) tokens.
2. Collapse reactions into pathways (GEM 'subsystem'): one row per pathway.
3. For each pathway, the metabolites SHOWN are the union of the top-k metabolites of each
   reaction in that pathway; their colour = the pathway's peak attention to that metabolite.
   Every other metabolite is 0 (white).

Excluded subsystems: Exchange, Transport, Pseudoreaction, Unassigned, Glycerol secretion (-> 32).

Usage:
    python paper/analysis/pathway_attention.py \
        paper/.results/i_orientalis_20260620/attention/flux_token_attn.npz \
        --gem models/I_orientalis/imIsor_GEM.xlsx --top-k 3
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import json
import re
import seaborn as sns
from matplotlib.patches import Rectangle

EXCLUDE_EXACT = {'Pseudoreaction', 'Unassigned', 'Glycerol secretion'}
_ATOM = re.compile(r'([A-Za-z0-9_\-\[\]]+)\s*\(([a-zA-Z]+)\)')


def parse_met(label):
    return label.split(' ')[0]


def strip_compartment(mid):
    return mid[:-2] if len(mid) > 2 and mid[-2] == '_' else mid


def cluster_order(mat, optimal=False):
    if mat.shape[0] < 3:
        return list(range(mat.shape[0]))
    d = ssd.pdist(mat)
    if not np.isfinite(d).all() or d.sum() == 0:
        return list(range(mat.shape[0]))
    return list(sch.leaves_list(sch.linkage(d, method='ward', optimal_ordering=optimal)))


def seriate(M, method):
    """Return (row_order, col_order). 'ca' = correspondence-analysis seriation
    (couples rows & cols on one latent axis -> diagonal); others cluster axes independently."""
    if method == 'ca':
        X = M.astype(float) + 1e-9
        P = X / X.sum()
        r = P.sum(1, keepdims=True)
        c = P.sum(0, keepdims=True)
        Smat = (P - r @ c) / np.sqrt(r @ c)             # chi-square standardized residuals
        U, s, Vt = np.linalg.svd(Smat, full_matrices=False)
        rs = U[:, 0] / np.sqrt(r[:, 0])                  # first CA axis (row scores)
        cs = Vt[0] / np.sqrt(c[0])                       # first CA axis (col scores)
        return list(np.argsort(rs)), list(np.argsort(cs))
    if method == 'optimal':
        return cluster_order(M, optimal=True), cluster_order(M.T, optimal=True)
    return cluster_order(M), cluster_order(M.T)          # 'cluster'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', help='flux_token_attn.npz from show_attention.py')
    ap.add_argument('--gem', required=True, help='GEM xlsx with Rxn ID + subsystem columns')
    ap.add_argument('--model-json', required=True,
                    help='annotated model JSON (for atom_mapping -> in-pathway carbon metabolites)')
    ap.add_argument('--top-k', type=int, default=3,
                    help='top metabolites per reaction to keep (union over the pathway)')
    ap.add_argument('--checkpoint', default=None,
                    help='trained .pt; if given, exclude reactions with validation R2 < --min-r2')
    ap.add_argument('--min-r2', type=float, default=0.2,
                    help='minimum per-reaction validation R2 to include (default 0.2)')
    ap.add_argument('--order', choices=['ca', 'optimal', 'cluster'], default='ca',
                    help="seriation method (default ca)")
    ap.add_argument('--order-by', choices=['membership', 'attention'], default='membership',
                    help="order rows/cols by the in-pathway binary map (default) or by attention")
    ap.add_argument('--cmap', default='mako',
                    help="seaborn/matplotlib colormap, dark->colour (e.g. mako, rocket, 'dark:steelblue')")
    ap.add_argument('--raw', action='store_true',
                    help='colour by absolute attention (default: row-normalized per pathway)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # ── reaction x metabolite attention (max over isotopomer/tracer tokens) ─────
    d = np.load(args.npz, allow_pickle=True)
    FT = np.abs(d['flux_token'])                              # (n_rxn, n_tok)
    flux_names = list(d['flux_names'])
    labels = list(d['token_labels'])
    keep = [j for j, l in enumerate(labels) if not str(l).endswith('[flux]')]
    tok_met = [parse_met(labels[j]) for j in keep]
    mets = sorted(set(tok_met))
    mcol = {m: i for i, m in enumerate(mets)}
    A = np.zeros((len(flux_names), len(mets)))               # reaction x metabolite
    for k, j in enumerate(keep):
        c = mcol[tok_met[k]]
        A[:, c] = np.maximum(A[:, c], FT[:, j])              # max over tokens of this metabolite

    # ── reaction -> pathway (subsystem) ────────────────────────────────────────
    g = pd.read_excel(args.gem, sheet_name='Reactions')
    sub = dict(zip(g['Rxn ID'].astype(str), g['subsystem'].astype(str).str.strip()))

    def keep_path(p):
        pl = p.lower()
        return not ('exchange' in pl or 'transport' in pl or p in EXCLUDE_EXACT or p in ('', 'nan'))

    # optional per-reaction validation-R2 filter
    r2map = {}
    if args.checkpoint:
        import torch
        vc = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('val_correlations', {})
        r2map = dict(zip(list(vc.get('flux_reaction_names', [])),
                         np.asarray(vc.get('flux_r2_per_reaction', []), float)))

    paths, n_dropped = {}, 0
    for ri, r in enumerate(flux_names):
        if args.checkpoint and r2map.get(r, -np.inf) < args.min_r2:
            n_dropped += 1
            continue
        p = sub.get(r)
        if p and keep_path(p):
            paths.setdefault(p, []).append(ri)
    pnames = sorted(paths)
    if args.checkpoint:
        print(f"R2 filter (>= {args.min_r2}): dropped {n_dropped} reactions")
    print(f"reactions mapped to {len(pnames)} pathways; {sum(len(v) for v in paths.values())} reactions")

    # ── pathway x metabolite: union of per-reaction top-k, colour = pathway peak ─
    P = np.zeros((len(pnames), len(mets)))
    pathway_top = {}
    for pi, p in enumerate(pnames):
        rxns = paths[p]
        union = set()
        for ri in rxns:
            row = A[ri]
            topm = [mm for mm in np.argsort(-row)[:args.top_k] if row[mm] > 0]
            union.update(topm)
        for m in union:
            P[pi, m] = max(A[ri, m] for ri in rxns)          # pathway peak attention to m
        pathway_top[p] = sorted(union, key=lambda m: -P[pi, m])

    # ── ground-truth: in-pathway carbon metabolites (from atom_mapping) ────────
    _mdl = json.load(open(args.model_json))
    amap = {r['id']: (r.get('annotation') or {}).get('atom_mapping', '') for r in _mdl['reactions']}
    name_map = {strip_compartment(m['id']): m.get('name', m['id']) for m in _mdl['metabolites']}
    inpath = {}                                              # pathway -> set of metabolite indices (into mets)
    for rid, p in sub.items():
        if p not in paths:                                   # only kept pathways
            continue
        for mid, _ in _ATOM.findall(amap.get(rid, '') or ''):
            b = strip_compartment(mid)
            if b in mcol:                                    # carbon metabolite that is also measured
                inpath.setdefault(p, set()).add(mcol[b])

    # columns = metabolites that are attended by some pathway OR in-pathway-measured
    attended_cols = set(np.where(P.sum(axis=0) > 0)[0])
    inpath_cols = set().union(*inpath.values()) if inpath else set()
    col_idx = sorted(attended_cols | inpath_cols)
    P = P[:, col_idx]
    shown_mets = [mets[i] for i in col_idx]
    newpos = {old: i for i, old in enumerate(col_idx)}
    B = np.zeros((len(pnames), len(col_idx)), bool)          # in-pathway membership over shown cols
    for pi, p in enumerate(pnames):
        for old in inpath.get(p, set()):
            B[pi, newpos[old]] = True
    print(f"metabolites shown: {len(shown_mets)}  (attended {len(attended_cols)} + in-pathway {len(inpath_cols)})")

    # ── recovery accuracy: of the in-pathway measured metabolites, how many did
    #    attention's top-k recover? (recall); of attended, how many in-pathway? (precision) ─
    print("\nPathway-metabolite recovery (attention top-k vs carbon-mapping membership):")
    print(f"  {'pathway':40s}  {'recovered/in-path':>17}  {'recall':>6}  {'precision':>9}")
    tot_rec, tot_in, tot_att, recalls = 0, 0, 0, []
    for pi, p in enumerate(pnames):
        att = set(np.where(P[pi] > 0)[0])
        inp = set(np.where(B[pi])[0])
        rec = att & inp
        tot_rec += len(rec); tot_in += len(inp); tot_att += len(att)
        recall = len(rec) / len(inp) if inp else float('nan')
        prec = len(rec) / len(att) if att else float('nan')
        if inp:
            recalls.append(recall)
        print(f"  {p:40s}  {len(rec):7d}/{len(inp):<9d}  "
              f"{('  n/a' if not inp else f'{recall:5.2f}'):>6}  "
              f"{('   n/a' if not att else f'{prec:6.2f}'):>9}")
    micro_recall = tot_rec / tot_in if tot_in else float('nan')
    macro_recall = float(np.mean(recalls)) if recalls else float('nan')
    micro_prec = tot_rec / tot_att if tot_att else float('nan')
    print(f"\n  OVERALL recovery of in-pathway metabolites: "
          f"micro-recall={micro_recall:.2f} ({tot_rec}/{tot_in})  "
          f"macro-recall={macro_recall:.2f}  micro-precision={micro_prec:.2f}")

    # ── ENRICHMENT (threshold-free; the principled metric — robust to top-k,
    #    tiny vocabulary, and hub metabolites). Uses continuous attention, not top-k. ─
    try:
        from sklearn.metrics import roc_auc_score
        Praw = np.stack([A[paths[p]].max(0) for p in pnames])   # pathway x ALL metabolites (continuous)
        Yall, Sall, macro = [], [], []
        for pi, p in enumerate(pnames):
            y = np.zeros(len(mets))
            for old in inpath.get(p, ()):
                y[old] = 1
            if 0 < y.sum() < len(y):
                macro.append(roc_auc_score(y, Praw[pi]))
            Yall += list(y); Sall += list(Praw[pi])
        Yall, Sall = np.array(Yall), np.array(Sall)
        ratio = Sall[Yall == 1].mean() / Sall[Yall == 0].mean()
        print(f"  ENRICHMENT (continuous attention, in- vs out-of-pathway): "
              f"pooled AUROC={roc_auc_score(Yall, Sall):.3f}  "
              f"macro AUROC={np.mean(macro):.3f}  in/out attention ratio={ratio:.2f}x")
        print("  (AUROC ~0.5 = no preference, >0.5 = attends in-pathway more; "
              "this is the metric to quote, not precision)")
    except Exception as e:
        print(f"  (AUROC enrichment skipped: {e})")

    # ── colour scaling ─────────────────────────────────────────────────────────
    if args.raw:
        Pdisp = P.copy()
        vmax = np.percentile(P[P > 0], 95) if (P > 0).any() else 1.0
        cbar_label = 'attention (|flux·token|, max)'
    else:
        rowmax = P.max(axis=1, keepdims=True)
        rowmax[rowmax == 0] = 1.0
        Pdisp = P / rowmax                                   # per-pathway relative
        vmax = 1.0
        cbar_label = 'attention (row-normalized per pathway)'

    # ── order rows & cols (default: CA seriation -> diagonal) ──────────────────
    # order by the BINARY in-pathway map (so the black squares diagonalize), or by attention
    order_mat = B.astype(float) if args.order_by == 'membership' else Pdisp
    ro, co = seriate(order_mat, args.order)
    Pord = Pdisp[np.ix_(ro, co)]
    Bord = B[np.ix_(ro, co)]
    row_lab = [pnames[i] for i in ro]                                    # pathways
    col_lab = [name_map.get(shown_mets[i], shown_mets[i]) for i in co]    # metabolite NAMES

    # Horizontal layout: pathways on x-axis, metabolites on y-axis; square cells.
    Pmask = np.ma.masked_equal(Pord.T, 0)                                 # metabolites x pathways
    try:
        cmap = sns.color_palette(args.cmap, as_cmap=True)
    except Exception:
        cmap = plt.get_cmap(args.cmap)
    cmap = cmap.copy()
    cmap.set_bad('white')

    nrows, ncols = len(col_lab), len(row_lab)                            # mets (y), pathways (x)
    fig, ax = plt.subplots(figsize=(0.30 * ncols + 5, 0.30 * nrows + 2))
    im = ax.imshow(Pmask, aspect='equal', cmap=cmap, vmin=0, vmax=vmax)
    for j in range(Bord.shape[0]):                                        # pathway (x)
        for i in range(Bord.shape[1]):                                    # metabolite (y)
            if Bord[j, i]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='black', lw=1.0))
    ax.set_xticks(range(ncols)); ax.set_xticklabels(row_lab, rotation=90, fontsize=7)
    ax.set_yticks(range(nrows)); ax.set_yticklabels(col_lab, fontsize=7)
    ax.set_title(f'Pathway attention (top-{args.top_k}); black outline = metabolite in pathway',
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015, shrink=0.35, aspect=12)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.npz), 'pathway_attention.svg')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print(f"saved -> {out} (+ .png)")

    # ── text summary: each pathway's top attended metabolites ──────────────────
    print("\nTop attended metabolites per pathway:")
    for p in pnames:
        tops = [mets[m] for m in pathway_top[p][:6]]
        print(f"  {p:42s}: {', '.join(tops)}")


if __name__ == '__main__':
    main()
