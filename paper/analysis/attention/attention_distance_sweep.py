"""
Sweep: which attention-vs-distance setup gives the strongest correlation?

Tries the full grid and prints a ranked table:
  attention object : self-attention (layer/head-averaged AND every single head) | cross-attention
  distance graph   : S bipartite metabolite-reaction graph | carbon atom-transition graph
  token->metabolite: mean | max over (isotopomer, tracer) tokens
  correlation      : raw Spearman | partial Spearman controlling for metabolite hub-degree

Self-attention is metabolite<->metabolite (even distances on the S graph; direct hops on the
atom graph). Cross-attention is reaction->metabolite (S graph only — the atom graph has no
reaction nodes). The carbon atom-transition graph connects two metabolites iff a reaction shares
mapped carbon atoms between them (from each reaction's annotation['atom_mapping']).

Usage:
    python paper/analysis/attention_distance_sweep.py \
        --self  paper/.results/i_orientalis_20260620/attention/self_attn.npz \
        --cross paper/.results/i_orientalis_20260620/attention/flux_token_attn.npz \
        --model-json models/I_orientalis/imIsor_annotated_hierarchical.json
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr, rankdata, pearsonr
from cobra.io import load_json_model

CURRENCY = {'h', 'h2o', 'atp', 'adp', 'amp', 'pi', 'ppi', 'nad', 'nadh',
            'nadp', 'nadph', 'co2', 'o2', 'coa', 'nh4', 'so4', 'fad', 'fadh2'}
_ATOM = re.compile(r'([A-Za-z0-9_\-\[\]]+)\s*\(([a-zA-Z]+)\)')


def strip_compartment(mid):
    return mid[:-2] if len(mid) > 2 and mid[-2] == '_' else mid


def parse_met(label):
    return label.split(' ')[0]


def build_s_graph(model, keep_currency=False):
    """Bipartite reaction<->metabolite graph (metabolite ids compartment-stripped)."""
    G = nx.Graph()
    for rxn in model.reactions:
        for met in rxn.metabolites:
            b = strip_compartment(met.id)
            if not keep_currency and b in CURRENCY:
                continue
            G.add_edge(('R', rxn.id), ('M', b))
    return G


def build_atom_graph(model):
    """Metabolite<->metabolite graph: edge iff a reaction shares mapped carbon atoms."""
    G = nx.Graph()
    for rxn in model.reactions:
        ann = rxn.annotation if isinstance(rxn.annotation, dict) else {}
        amap = ann.get('atom_mapping', '') or ''
        if not amap.strip():
            continue
        mets = {}
        for mid, letters in _ATOM.findall(amap):
            mets.setdefault(strip_compartment(mid), set()).update(letters)
        names = list(mets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if mets[names[i]] & mets[names[j]]:        # shared atom letter = carbon transferred
                    G.add_edge(names[i], names[j])
    return G


def partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z (rank-residualize both on z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    def resid(a):
        c, *_ = np.linalg.lstsq(Z, a, rcond=None)
        return a - Z @ c
    r, p = pearsonr(resid(rx), resid(ry))
    return r, p


def met_pair_value(M, idx_a, idx_b, agg):
    block = M[np.ix_(idx_a, idx_b)]
    return block.max() if agg == 'max' else block.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self', dest='self_npz', required=True)
    ap.add_argument('--cross', dest='cross_npz', required=True)
    ap.add_argument('--model-json', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    model = load_json_model(args.model_json)
    G_S = build_s_graph(model)
    G_A = build_atom_graph(model)
    print(f"S graph: {G_S.number_of_nodes()} nodes, {G_S.number_of_edges()} edges | "
          f"atom graph: {G_A.number_of_nodes()} mets, {G_A.number_of_edges()} edges")

    spS = {n: nx.single_source_shortest_path_length(G_S, n) for n in G_S if n[0] == 'M'}
    spA = {n: nx.single_source_shortest_path_length(G_A, n) for n in G_A}

    # ── distance + degree helpers ──────────────────────────────────────────────
    def dS_mm(a, b):   return spS.get(('M', a), {}).get(('M', b))
    def dA_mm(a, b):   return spA.get(a, {}).get(b)
    def dS_rm(r, m):   return spS.get(('M', m), {}).get(('R', r))
    def degS(m):       return G_S.degree(('M', m)) if ('M', m) in G_S else 0
    def degA(m):       return G_A.degree(m) if m in G_A else 0
    def degSr(r):      return G_S.degree(('R', r)) if ('R', r) in G_S else 0

    results = []   # (label, raw_r, raw_p, part_r, part_p, n)

    # ════════════════ SELF-ATTENTION (metabolite <-> metabolite) ════════════════
    sd = np.load(args.self_npz, allow_pickle=True)
    s_tok_met = [parse_met(l) for l in sd['token_labels']]
    s_mets = sorted(set(s_tok_met))
    s_idx = {m: [j for j, mm in enumerate(s_tok_met) if mm == m] for m in s_mets}
    lh = sd['self_attn_lh']                         # (L, H, n, n)
    L, H = lh.shape[0], lh.shape[1]

    def eval_self(M, graph, agg, label, collect=None):
        dist = dS_mm if graph == 'S' else dA_mm
        deg = degS if graph == 'S' else degA
        ds, at, hub = [], [], []
        for i in range(len(s_mets)):
            for j in range(i + 1, len(s_mets)):
                a, b = s_mets[i], s_mets[j]
                d = dist(a, b)
                if d is None or d == 0:
                    continue
                ds.append(d); at.append(met_pair_value(M, s_idx[a], s_idx[b], agg))
                hub.append(deg(a) + deg(b))
        if len(ds) < 10:
            return
        ds, at, hub = np.array(ds), np.array(at), np.array(hub)
        rr, rp = spearmanr(at, ds)
        pr, pp = partial_spearman(at, ds, hub)
        results.append((label, rr, rp, pr, pp, len(ds)))
        if collect is not None:
            collect['ds'], collect['at'] = ds, at

    # layer/head-averaged
    M_avg = sd['self_attn']
    for graph in ('S', 'atom'):
        for agg in ('mean', 'max'):
            eval_self(M_avg, graph, agg, f"self-AVG      | {graph:4s} | {agg}")

    # per-head scan: find best single head for each (graph, agg)
    for graph in ('S', 'atom'):
        for agg in ('mean', 'max'):
            best = None
            for l in range(L):
                for h in range(H):
                    M = lh[l, h]
                    dist = dS_mm if graph == 'S' else dA_mm
                    ds, at = [], []
                    for i in range(len(s_mets)):
                        for j in range(i + 1, len(s_mets)):
                            a, b = s_mets[i], s_mets[j]
                            d = dist(a, b)
                            if d is None or d == 0:
                                continue
                            ds.append(d); at.append(met_pair_value(M, s_idx[a], s_idx[b], agg))
                    if len(ds) < 10:
                        continue
                    rr, _ = spearmanr(at, ds)
                    if best is None or rr < best[0]:        # most negative
                        best = (rr, l, h)
            if best is not None:
                _, l, h = best
                eval_self(lh[l, h], graph, agg, f"self-L{l}H{h:<2d}  | {graph:4s} | {agg}")

    # ════════════════ CROSS-ATTENTION (reaction -> metabolite, S graph) ══════════
    cd = np.load(args.cross_npz, allow_pickle=True)
    FT = np.abs(cd['flux_token'])                    # (n_flux, n_tok)
    flux_names = list(cd['flux_names'])
    c_labels = list(cd['token_labels'])
    keep = [j for j, l in enumerate(c_labels) if not str(l).endswith('[flux]')]
    c_tok_met = [parse_met(c_labels[j]) for j in keep]
    c_mets = sorted(set(c_tok_met))
    c_idx = {m: [keep[k] for k, mm in enumerate(c_tok_met) if mm == m] for m in c_mets}

    def eval_cross(agg, label):
        ds, at, hub = [], [], []
        for ri, r in enumerate(flux_names):
            if ('R', r) not in G_S:
                continue
            for m in c_mets:
                d = dS_rm(r, m)
                if d is None:
                    continue
                block = FT[ri][c_idx[m]]
                at.append(block.max() if agg == 'max' else block.mean())
                ds.append(d); hub.append(degSr(r) + degS(m))
        if len(ds) < 10:
            return
        ds, at, hub = np.array(ds), np.array(at), np.array(hub)
        rr, rp = spearmanr(at, ds)
        pr, pp = partial_spearman(at, ds, hub)
        results.append((label, rr, rp, pr, pp, len(ds)))

    for agg in ('mean', 'max'):
        eval_cross(agg, f"cross-rxn>met | S    | {agg}")

    # ════════════════ Report ════════════════
    results.sort(key=lambda r: r[1])               # most-negative raw Spearman first
    print("\n" + "=" * 92)
    print(f"{'configuration':30s} | {'raw r':>7} {'raw p':>9} | {'partial r':>9} {'partial p':>9} | {'n':>6}")
    print("-" * 92)
    for label, rr, rp, pr, pp, n in results:
        print(f"{label:30s} | {rr:+7.3f} {rp:9.1e} | {pr:+9.3f} {pp:9.1e} | {n:6d}")
    print("=" * 92)
    best = results[0]
    print(f"\nBEST (raw Spearman): {best[0].strip()}  r={best[1]:+.3f}  p={best[2]:.1e}")

    # plot the best config's distance curve
    out = args.out or os.path.join(os.path.dirname(args.self_npz), 'attention_distance_sweep.svg')
    # re-collect the best config's points
    label = best[0]
    col = {}
    if label.startswith('cross'):
        agg = label.split('|')[-1].strip()
        ds, at = [], []
        for ri, r in enumerate(flux_names):
            if ('R', r) not in G_S:
                continue
            for m in c_mets:
                d = dS_rm(r, m)
                if d is None:
                    continue
                b = FT[ri][c_idx[m]]
                at.append(b.max() if agg == 'max' else b.mean()); ds.append(d)
        col['ds'], col['at'] = np.array(ds), np.array(at)
        xlabel = 'reaction→metabolite distance (S graph)'
    else:
        parts = [p.strip() for p in label.split('|')]
        graph = 'S' if parts[1] == 'S' else 'atom'
        agg = parts[2]
        if parts[0] == 'self-AVG':
            M = M_avg
        else:
            lh_str = parts[0].replace('self-L', '')
            l = int(lh_str[0]); h = int(lh_str.split('H')[1])
            M = lh[l, h]
        eval_self(M, graph, agg, '_plot', collect=col)
        xlabel = ('metabolite–metabolite distance ' +
                  ('(atom-transition graph)' if graph == 'atom' else '(S graph)'))

    ds, at = col['ds'], col['at']
    dvals = sorted(set(ds))
    means = [at[ds == dd].mean() for dd in dvals]
    sems = [at[ds == dd].std() / max(1, np.sqrt((ds == dd).sum())) for dd in dvals]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.errorbar(dvals, means, yerr=sems, marker='o', color='#2c7fb8', lw=2, capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('attention')
    ax.set_xticks(dvals)
    ax.set_title(f"BEST: {label.strip()}  (r={best[1]:.2f}, p={best[2]:.0e})", fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f"saved best-config figure -> {out}")


if __name__ == '__main__':
    main()
