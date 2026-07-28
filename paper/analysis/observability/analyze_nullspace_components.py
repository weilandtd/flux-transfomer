"""Analyze which biological modes each null-space basis column captures.

For each DOF column j of the saved Z_free matrix the script reports:
  - obs_weight[j]   (max |Pearson r| with any labeling feature)
  - all reactions with |Z[i,j]| >= rel_min * col_max  (relative threshold)
  - subsystem breakdown split into metabolic vs. transport
  - column type: measured DOF | free-observable | free-semi | free-unobservable

Run from repo root:
    python paper/analysis/analyze_nullspace_components.py \\
        --checkpoint paper/.results/s_cerevisiae_20260614/sc_transformer.pt \\
        --model-json models/S_cerevisiae/sc_annotated_hierarchical.json

    python paper/analysis/analyze_nullspace_components.py \\
        --checkpoint paper/.results/i_orientalis_20260614/imisor_transformer.pt \\
        --model-json models/I_orientalis/imIsor_annotated_hierarchical.json
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch



OBS_THRESH_HIGH = 0.30
OBS_THRESH_MID  = 0.10
REL_MIN_DEFAULT = 0.20   # reaction must be >= 20 % of the column's max |Z[i,j]|
ABS_MIN_FLOOR   = 0.01   # hard floor regardless of rel_min

# Subsystems that are pure transport — separated from the metabolic breakdown
TRANSPORT_SUBS = {
    'Transport',
    'Exchange',
    'Transport, extracellular',
    'Transport, mitochondrial',
    'Transport, peroxisomal',
    'Transport, vacuolar',
    'Transport, endoplasmic reticular',
}


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_subsystem_map(model_json_path):
    """Return {reaction_id: subsystem} from the *GEM.xlsx beside the model JSON."""
    import glob
    import pandas as pd
    model_dir = os.path.dirname(os.path.abspath(model_json_path))
    candidates = glob.glob(os.path.join(model_dir, '*GEM.xlsx'))
    if not candidates:
        from cobra.io import load_json_model
        model = load_json_model(model_json_path)
        return {r.id: (r.subsystem or 'Unknown') for r in model.reactions}
    df = pd.read_excel(candidates[0], sheet_name='Reactions', usecols=['Rxn ID', 'subsystem'])
    df['subsystem'] = df['subsystem'].fillna('Unknown')
    return dict(zip(df['Rxn ID'], df['subsystem']))


def classify_column(j, n_targets, obs_weight):
    if obs_weight is None:
        return 'unknown'
    if j < n_targets:
        return 'measured'
    if obs_weight >= OBS_THRESH_HIGH:
        return 'free-observable'
    if obs_weight >= OBS_THRESH_MID:
        return 'free-semi'
    return 'free-unobservable'


def significant_reactions(col, flux_names, subsys_map, rel_min=REL_MIN_DEFAULT, abs_floor=ABS_MIN_FLOOR):
    """Return (rxn_name, weight, subsystem) for reactions with |w| >= rel_min * col_max.

    rel_min is relative to the column maximum, so each DOF is scaled independently.
    abs_floor is a hard lower bound below which reactions are always excluded.
    """
    col_max = np.abs(col).max()
    if col_max < 1e-15:
        return []
    threshold = max(rel_min * col_max, abs_floor)
    order = np.argsort(np.abs(col))[::-1]
    results = []
    for i in order:
        w = col[i]
        if abs(w) < threshold:
            break
        results.append((flux_names[i], float(w), subsys_map.get(flux_names[i], 'Unknown')))
    return results


def split_metabolic_transport(sig_rxns):
    """Split significant reactions into (metabolic_list, transport_list).
    Returns two lists of (rxn, weight, subsystem) tuples.
    """
    metabolic = [(r, w, s) for r, w, s in sig_rxns if s not in TRANSPORT_SUBS]
    transport = [(r, w, s) for r, w, s in sig_rxns if s in TRANSPORT_SUBS]
    return metabolic, transport


def subsystem_breakdown(rxns):
    """Return list of (subsystem, frac_sq_weight, [(rxn, w)]) sorted by frac desc."""
    sub_sq = defaultdict(float)
    sub_rxns = defaultdict(list)
    for rxn, w, sub in rxns:
        sub_sq[sub] += w ** 2
        sub_rxns[sub].append((rxn, w))
    total = sum(sub_sq.values()) or 1.0
    result = []
    for sub in sorted(sub_sq, key=sub_sq.get, reverse=True):
        result.append((sub, sub_sq[sub] / total, sub_rxns[sub]))
    return result


def dominant_subsystem(metabolic, transport):
    """Dominant subsystem from metabolic reactions; fall back to transport if none."""
    if metabolic:
        bd = subsystem_breakdown(metabolic)
        return bd[0][0]
    if transport:
        bd = subsystem_breakdown(transport)
        return 'Transport:' + bd[0][2][0][0]   # transport + top reaction
    return 'Unknown'


def analyze_checkpoint(checkpoint_path, model_json_path,
                       rel_min=REL_MIN_DEFAULT, abs_floor=ABS_MIN_FLOOR,
                       out_tsv=None):
    ck = load_checkpoint(checkpoint_path)
    nsd = ck['null_space_data']

    Z = nsd['Z_free']
    obs_weights = nsd.get('obs_weights')
    flux_names = ck['flux_names']
    measured_fluxes = ck.get('measured_fluxes') or []
    n_rxn, k = Z.shape
    n_targets = len(measured_fluxes)

    print(f"\nCheckpoint : {checkpoint_path}")
    print(f"Flux names : {n_rxn} reactions")
    print(f"Null-space : k = {k} columns  (first {n_targets} are measured DOFs)")
    if obs_weights is not None:
        print(f"obs_weights: min={obs_weights.min():.3f}  max={obs_weights.max():.3f}")
    print(f"Significance: |Z[i,j]| >= {rel_min:.0%} × col_max  (floor {abs_floor})")
    print()

    subsys_map = {}
    if model_json_path:
        try:
            subsys_map = load_subsystem_map(model_json_path)
        except Exception as e:
            print(f"Warning: could not load subsystem annotations: {e}")

    measured_rxn_for_col = {j: rxn for j, rxn in enumerate(measured_fluxes[:n_targets])}

    rows = []
    for j in range(k):
        col = Z[:, j]
        sig = significant_reactions(col, flux_names, subsys_map, rel_min, abs_floor)
        met, tra = split_metabolic_transport(sig)
        dom_sub = dominant_subsystem(met, tra)

        obs_w = float(obs_weights[j]) if obs_weights is not None else None
        col_type = classify_column(j, n_targets, obs_w)

        rows.append({
            'dof_idx':      j,
            'type':         col_type,
            'obs_weight':   obs_w,
            'dominant_sub': dom_sub,
            'sig':          sig,
            'metabolic':    met,
            'transport':    tra,
            'measured_rxn': measured_rxn_for_col.get(j),
        })

    # Compute purity score per row (fraction of metabolic L2² in dominant subsystem)
    for r in rows:
        met_bd = subsystem_breakdown(r['metabolic'])
        r['met_bd'] = met_bd
        r['purity'] = met_bd[0][1] if met_bd else 0.0
        tra_sq = sum(w ** 2 for _, w, _ in r['transport'])
        sig_sq = sum(w ** 2 for _, w, _ in r['sig']) or 1e-30
        r['tra_frac'] = tra_sq / sig_sq

    rows.sort(key=lambda r: (0 if r['type'] == 'measured' else 1, -(r['obs_weight'] or 0)))

    # ── Summary table ─────────────────────────────────────────────────────────
    header = (f"{'DOF':>4}  {'Type':<20}  {'obs_w':>6}  "
              f"{'pur':>4}  {'Dominant metabolic subsystem':<42}  Top metabolic reactions")
    print(header)
    print('─' * 140)
    for r in rows:
        obs_str = f"{r['obs_weight']:.3f}" if r['obs_weight'] is not None else '  n/a'
        pur_str = f"{r['purity']:.0%}"
        if r['measured_rxn']:
            top_str = (f"[{r['measured_rxn']}]"
                       + (f"  dep: {', '.join(x[0] for x in r['metabolic'][:2])}" if r['metabolic'] else ''))
        else:
            top_str = ', '.join(x[0] for x in r['metabolic'][:4]) or '(only transport)'
        print(f"{r['dof_idx']:>4}  {r['type']:<20}  {obs_str:>6}  "
              f"{pur_str:>4}  {r['dominant_sub']:<42}  {top_str}")

    # ── Tier summary ──────────────────────────────────────────────────────────
    print('\n=== Tier summary ===')
    tier_keys = ['measured', 'free-observable', 'free-semi', 'free-unobservable']
    tiers = {t: [r for r in rows if r['type'] == t] for t in tier_keys}
    for tier, tier_rows in tiers.items():
        if not tier_rows:
            continue
        obs_vals = [r['obs_weight'] for r in tier_rows if r['obs_weight'] is not None]
        obs_str = f"obs=[{min(obs_vals):.3f},{max(obs_vals):.3f}]" if obs_vals else ''
        sub_sq = defaultdict(float)
        for r in tier_rows:
            for sub, frac, _ in r['met_bd']:
                sub_sq[sub] += frac
        top_subs = sorted(sub_sq, key=sub_sq.get, reverse=True)[:4]
        avg_pur = np.mean([r['purity'] for r in tier_rows]) if tier_rows else 0
        print(f"  {tier:<20}  n={len(tier_rows):>3}  {obs_str:<24}  avg_purity={avg_pur:.0%}  "
              f"metabolic: {', '.join(top_subs)}")

    # ── Per-DOF detail (compact: top subsystems + signed reaction groups) ──────
    PURITY_PURE  = 0.60
    PURITY_MIXED = 0.35

    print('\n\n=== Per-DOF detail ===')
    print('Format: subsystem (frac%): +increases  /  -decreases')
    print('        transport (frac%): +into_cell_or_mito  /  -out')
    for r in rows:
        obs_str = f"{r['obs_weight']:.3f}" if r['obs_weight'] is not None else ' n/a'
        meas_tag = f"  [{r['measured_rxn']}]" if r['measured_rxn'] else ''
        if r['purity'] >= PURITY_PURE:
            mode_tag = 'PURE'
        elif r['purity'] >= PURITY_MIXED:
            mode_tag = 'PART'
        else:
            mode_tag = 'MIXD'
        print(f"\nDOF {r['dof_idx']:>3}  {r['type']}  obs={obs_str}  {mode_tag}"
              f"  purity={r['purity']:.0%}  transport={r['tra_frac']:.0%}{meas_tag}")

        # Metabolic breakdown: top 4 subsystems, reactions split by sign
        for sub, frac, rxns in r['met_bd'][:4]:
            pos = '  '.join(f"{rn}(+{w:.2f})" for rn, w in sorted(rxns, key=lambda x: -x[1]) if w > 0)
            neg = '  '.join(f"{rn}({w:.2f})" for rn, w in sorted(rxns, key=lambda x: x[1]) if w < 0)
            parts = []
            if pos:
                parts.append(f"↑ {pos}")
            if neg:
                parts.append(f"↓ {neg}")
            print(f"  {frac:5.1%}  {sub:<44}  {'  /  '.join(parts)}")

        # Transport: top 5 by |weight|, split by sign
        if r['transport']:
            top_tra = sorted(r['transport'], key=lambda x: -abs(x[1]))[:5]
            pos_t = '  '.join(f"{rn}(+{w:.2f})" for rn, w, _ in top_tra if w > 0)
            neg_t = '  '.join(f"{rn}({w:.2f})" for rn, w, _ in top_tra if w < 0)
            tra_parts = []
            if pos_t:
                tra_parts.append(f"↑ {pos_t}")
            if neg_t:
                tra_parts.append(f"↓ {neg_t}")
            print(f"  {r['tra_frac']:5.1%}  {'Transport':<44}  {'  /  '.join(tra_parts)}")

    # ── TSV export ────────────────────────────────────────────────────────────
    if out_tsv:
        import csv
        with open(out_tsv, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'dof_idx', 'type', 'obs_weight', 'purity', 'transport_frac',
                'n_metabolic', 'n_transport',
                'dominant_metabolic_sub', 'metabolic_sub_breakdown',
                'metabolic_rxns', 'metabolic_weights',
                'transport_rxns', 'measured_rxn',
            ])
            for r in rows:
                sub_bd_str = ';'.join(f"{s}:{f:.3f}" for s, f, _ in r['met_bd'])
                writer.writerow([
                    r['dof_idx'], r['type'],
                    f"{r['obs_weight']:.4f}" if r['obs_weight'] is not None else '',
                    f"{r['purity']:.4f}",
                    f"{r['tra_frac']:.4f}",
                    len(r['metabolic']), len(r['transport']),
                    r['dominant_sub'],
                    sub_bd_str,
                    ';'.join(x[0] for x in r['metabolic']),
                    ';'.join(f"{x[1]:.4f}" for x in r['metabolic']),
                    ';'.join(x[0] for x in r['transport']),
                    r['measured_rxn'] or '',
                ])
        print(f"\nTSV written to {out_tsv}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-json', default=None)
    parser.add_argument('--rel-min', type=float, default=REL_MIN_DEFAULT,
                        help='Min |Z[i,j]| as fraction of column max (default 0.20)')
    parser.add_argument('--abs-floor', type=float, default=ABS_MIN_FLOOR,
                        help='Hard lower bound on |Z[i,j]| (default 0.01)')
    parser.add_argument('--out-tsv', default=None)
    args = parser.parse_args()

    analyze_checkpoint(
        checkpoint_path=args.checkpoint,
        model_json_path=args.model_json,
        rel_min=args.rel_min,
        abs_floor=args.abs_floor,
        out_tsv=args.out_tsv,
    )


if __name__ == '__main__':
    main()
