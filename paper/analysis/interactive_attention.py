"""
Interactive HTML heatmap: what labeling metabolites inform what fluxes.

Loads flux_token_attn.npz and a metabolic model JSON, builds a Ward-clustered
heatmap with rich hover tooltips (full reaction equation, metabolite names,
flux type).

Usage:
    python paper/interactive_attention.py \\
        I_orientalis_20260508/flux_token_attn.npz \\
        --model-json models/I_orientalis/imIsor_annotated.json

    python paper/interactive_attention.py ...npz --model-json ...json --out fig.html
"""

import argparse
import json
import os
import re

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import plotly.graph_objects as go


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_compartment(rxn_or_met_id):
    """'ACGK_m' → 'ACGK',  'pyr_c' → 'pyr'."""
    return re.sub(r'_[a-z]\d?$', '', rxn_or_met_id)


def format_equation(rxn, met_name_map):
    """Format reaction stoichiometry as 'A + B → C + D'."""
    reactants, products = [], []
    for met_id, coeff in rxn.get('metabolites', {}).items():
        name = met_name_map.get(met_id, strip_compartment(met_id))
        coeff_str = '' if abs(coeff) == 1 else f'{abs(coeff):.4g} '
        if coeff < 0:
            reactants.append(coeff_str + name)
        else:
            products.append(coeff_str + name)
    if not reactants and not products:
        return '—'
    return ' + '.join(reactants) + ' → ' + ' + '.join(products)


def parse_met(token_label):
    return token_label.split(' ')[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='flux_token_attn.npz from show_attention.py')
    parser.add_argument('--model-json', default=None,
                        help='Annotated model JSON for full reaction/metabolite names')
    parser.add_argument('--out', default=None,
                        help='Output HTML file (default: same dir as npz)')
    parser.add_argument('--method', default='ward',
                        help='Linkage method (default: ward)')
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.npz)), 'cluster_heatmap_interactive.html'
    )

    # ── Load matrix ───────────────────────────────────────────────────────────
    d            = np.load(args.npz, allow_pickle=True)
    flux_token   = d['flux_token']          # (386, N_tokens)
    flux_names   = list(d['flux_names'])
    token_labels = list(d['token_labels'])
    fixed_idx    = set(d['fixed_idx'].tolist()) if 'fixed_idx' in d else set()
    Z            = d['Z']                   # (386, 38)

    # ── Load model for names/reactions ───────────────────────────────────────
    rxn_dict     = {}   # reaction_id -> {name, metabolites, ...}
    met_name_map = {}   # metabolite_id -> full name
    # map: base metabolite id (no compartment) -> full name (first match)
    base_met_names = {}

    if args.model_json:
        model = json.load(open(args.model_json))
        met_name_map = {m['id']: m.get('name', '') or m['id']
                        for m in model['metabolites']}
        for m in model['metabolites']:
            base = strip_compartment(m['id'])
            if base not in base_met_names and m.get('name'):
                base_met_names[base] = m['name']
        rxn_dict = {r['id']: r for r in model['reactions']}

    # ── Aggregate tokens → metabolites ────────────────────────────────────────
    met_tags     = [parse_met(lbl) for lbl in token_labels]
    unique_mets  = []
    seen         = set()
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

    # ── Row-normalise absolute values ─────────────────────────────────────────
    mat = np.abs(met_matrix)
    mu  = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1,  keepdims=True).clip(min=1e-9)
    mat = (mat - mu) / std

    # ── Ward hierarchical clustering ──────────────────────────────────────────
    row_link   = sch.linkage(ssd.pdist(mat,   metric='euclidean'), method=args.method)
    col_link   = sch.linkage(ssd.pdist(mat.T, metric='euclidean'), method=args.method)
    row_leaves = sch.leaves_list(row_link)
    col_leaves = sch.leaves_list(col_link)

    mat_ord        = mat[row_leaves][:, col_leaves]
    flux_ord       = [flux_names[i]   for i in row_leaves]
    met_ord        = [unique_mets[i]  for i in col_leaves]
    iso_ord        = [chosen_isotopomers[i] for i in col_leaves]

    # ── Build hover text (one string per cell) ────────────────────────────────
    hover = []
    for ri, r_idx in enumerate(row_leaves):
        rxn_id   = flux_names[r_idx]
        rxn      = rxn_dict.get(rxn_id, {})
        rxn_name = rxn.get('name', '') or rxn_id
        base_rxn = strip_compartment(rxn_id)
        equation = format_equation(rxn, met_name_map) if rxn else '—'
        flux_type = 'measured (fixed)' if r_idx in fixed_idx else 'free (predicted)'

        # Top y-dim contribution
        top_ydim = int(np.argmax(np.abs(Z[r_idx])))
        ydim_loading = float(Z[r_idx, top_ydim])

        row_hover = []
        for ci, c_idx in enumerate(col_leaves):
            met      = unique_mets[c_idx]
            iso_lbl  = chosen_isotopomers[c_idx]
            met_full = base_met_names.get(met, met)
            val      = mat_ord[ri, ci]

            txt = (
                f'<b>Reaction:</b> {rxn_id}<br>'
                f'<b>Base ID:</b> {base_rxn}<br>'
                f'<b>Name:</b> {rxn_name}<br>'
                f'<b>Equation:</b> {equation}<br>'
                f'<b>Type:</b> {flux_type}<br>'
                f'<b>Top y-dim:</b> y{top_ydim:02d}  (loading {ydim_loading:+.3f})<br>'
                f'<br>'
                f'<b>Metabolite:</b> {met}<br>'
                f'<b>Full name:</b> {met_full}<br>'
                f'<b>Representative isotopomer:</b> {iso_lbl}<br>'
                f'<b>Attention z-score:</b> {val:.3f}'
            )
            row_hover.append(txt)
        hover.append(row_hover)

    # ── Column hover (shown when hovering axis labels) — embed in customdata ──
    col_full_names = [base_met_names.get(m, m) for m in met_ord]

    # ── Plotly heatmap ────────────────────────────────────────────────────────
    vmax = float(np.percentile(np.abs(mat_ord), 95))

    fig = go.Figure(go.Heatmap(
        z           = mat_ord,
        x           = [f'{m}<br><i>{base_met_names.get(m, "")}</i>' for m in met_ord],
        y           = flux_ord,
        colorscale  = 'RdBu_r',
        zmid        = 0,
        zmin        = -vmax,
        zmax        =  vmax,
        hovertext   = hover,
        hoverinfo   = 'text',
        colorbar    = dict(
            title      = dict(text='z-score', side='right'),
            thickness  = 14,
            len        = 0.5,
        ),
    ))

    n_rows, n_cols = mat_ord.shape
    fig.update_layout(
        title=dict(
            text=(
                f'Flux ← Labeling attention  '
                f'({n_rows} reactions × {n_cols} metabolites)<br>'
                f'<sup>Row z-score | Ward hierarchical clustering | '
                f'hover for full reaction details</sup>'
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            tickangle=-50,
            tickfont=dict(size=10),
            title='Labeling metabolite',
            side='bottom',
        ),
        yaxis=dict(
            tickfont=dict(size=6),
            title='Reaction',
            autorange='reversed',
        ),
        width  = max(900,  n_cols * 18 + 300),
        height = max(1000, n_rows * 10 + 200),
        margin = dict(l=220, r=60, t=100, b=160),
        plot_bgcolor='white',
    )

    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Saved → {out_path}')
    print(f'  {n_rows} reactions × {n_cols} metabolites')


if __name__ == '__main__':
    main()
