"""Diagnose the targeted null-space basis on the actual stoichiometric matrices.

Checks:
1. Reaction ordering alignment: cobra model vs flux_df
2. Target block values in Z (before and after column normalization)
3. Gram matrix condition number
4. Z_proj sanity: v @ Z_proj @ Z.T ≈ v for sampled fluxes
5. Coordinate recovery: y_true = v @ Z_proj; v_rec = y_true @ Z.T ≈ v

Run from the repo root:
    python paper/analysis/diagnose_targeted_basis.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from cobra.io import load_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.ai.data import load_data
from sigmaflux.ai.utils import compute_targeted_nullspace

# ── Config ───────────────────────────────────────────────────────────────────

CONFIGS = {
    'IO': {
        'model_json':  'models/I_orientalis/imIsor_annotated_hierarchical.json',
        'data_dir':    'paper/.results/i_orientalis_20260614',
        'targets': [
            'EX_glc__D_e', 'PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c',
            'CARB_c', 'EX_etoh_e', 'EX_ac_e', 'EX_glyc_e', 'EX_succ_e',
        ],
    },
}


def sep(title=''):
    print('\n' + '─' * 70)
    if title:
        print(f'  {title}')
        print('─' * 70)


def run(name, cfg):
    sep(f'Organism: {name}')

    # ── Load cobra model ──────────────────────────────────────────────────────
    cobra_model = load_json_model(cfg['model_json'])
    S_cobra  = create_stoichiometric_matrix(cobra_model, array_type='dense')
    cobra_rxns = [r.id for r in cobra_model.reactions]
    print(f'Cobra S shape: {S_cobra.shape}  (met × rxn)')
    print(f'Cobra reactions: {len(cobra_rxns)}')

    # ── Load flux data ────────────────────────────────────────────────────────
    _, flux_df, _ = load_data(cfg['data_dir'], max_samples=500)
    flux_rxns = flux_df.index.tolist()
    print(f'Flux data reactions: {len(flux_rxns)}')

    # ── Ordering alignment ────────────────────────────────────────────────────
    sep('1. Reaction ordering alignment')
    n_common = len(set(cobra_rxns) & set(flux_rxns))
    print(f'  Reactions in both:  {n_common}')
    print(f'  Only in cobra:      {len(set(cobra_rxns) - set(flux_rxns))}')
    print(f'  Only in flux_df:    {len(set(flux_rxns) - set(cobra_rxns))}')

    # Check if the orderings match for the common reactions
    common = [r for r in cobra_rxns if r in set(flux_rxns)]
    position_mismatch = sum(
        1 for r in common if cobra_rxns.index(r) != flux_rxns.index(r)
    )
    print(f'  Position mismatches (cobra vs flux_df): {position_mismatch}')
    if position_mismatch > 0:
        print('  *** ORDERING MISMATCH — stoichiometric matrix columns do not align with flux_df ***')
        # Show first few mismatches
        for r in common[:5]:
            ci = cobra_rxns.index(r)
            fi = flux_rxns.index(r)
            if ci != fi:
                print(f'    {r}: cobra col {ci}, flux_df row {fi}')

    # ── Reorder S to match flux_df ─────────────────────────────────────────
    sep('2. Reorder S to match flux_df ordering')
    # Only keep reactions present in both; put them in flux_df order
    shared = [r for r in flux_rxns if r in set(cobra_rxns)]
    print(f'  Using {len(shared)} reactions present in both cobra and flux_df')
    cobra_col_idx = [cobra_rxns.index(r) for r in shared]
    S = S_cobra[:, cobra_col_idx]

    # Restrict flux_df to shared reactions
    v_mat = flux_df.loc[shared].values.T.astype(float)   # (n_samples, n_rxn)
    rxn_names = shared
    print(f'  S (reordered): {S.shape}')
    print(f'  v_mat: {v_mat.shape}')

    # Check mass balance on sampled fluxes
    residual = np.abs(S @ v_mat.T).max()
    print(f'  Max |S @ v| (sampled): {residual:.2e}  (should be ~0)')

    # ── Targeted null-space ───────────────────────────────────────────────────
    sep('3. compute_targeted_nullspace')
    targets = [t for t in cfg['targets'] if t in set(rxn_names)]
    print(f'  Targets ({len(targets)}): {targets}')

    Z, status = compute_targeted_nullspace(S, rxn_names, targets)
    if not status['separable']:
        print(f'  FAILED: {status["fallback_reason"]}')
        return
    print(f'  Separable: YES   cond(S_D)={status["cond_SD"]:.2e}')

    # ── Column norms (already unit by construction) ───────────────────────────
    col_norms = np.linalg.norm(Z, axis=0)
    print(f'  Z col norms: min={col_norms.min():.4f}  max={col_norms.max():.4f}  (all should be 1.0)')

    # ── Target block ─────────────────────────────────────────────────────────
    sep('4. Target block Z[target_idx, :n_targets]')
    t_idx = status['target_indices']
    block = Z[np.ix_(t_idx, list(range(len(targets))))]
    print(f'  Diagonal values: {np.diag(block).round(4)}')
    off_diag = block - np.diag(np.diag(block))
    print(f'  Max |off-diagonal|: {np.abs(off_diag).max():.2e}  (should be ~0)')

    # ── Gram matrix ───────────────────────────────────────────────────────────
    sep('5. Gram matrix G = Z.T @ Z')
    G = Z.T @ Z
    print(f'  G diagonal: min={np.diag(G).min():.4f}  max={np.diag(G).max():.4f}  (should be 1.0)')
    off_diag_G = G - np.eye(G.shape[0])
    print(f'  Max |off-diagonal|: {np.abs(off_diag_G).max():.4f}')
    cond_G = np.linalg.cond(G)
    print(f'  cond(G): {cond_G:.2e}')

    # ── Z_proj ────────────────────────────────────────────────────────────────
    sep('6. Z_proj = Z @ G^{-1}')
    Z_proj = Z @ np.linalg.solve(G, np.eye(G.shape[0]))

    # Sanity: Z_proj @ Z.T should be the null-space projector P_null
    P = Z_proj @ Z.T    # (n_rxn, n_rxn)
    # P @ v should ≈ v for v in null space
    residuals = np.abs(v_mat - (v_mat @ P)).max(axis=1)
    print(f'  Max |v - v @ P_null| over samples: {residuals.max():.2e}  (should be ~0)')

    # ── Coordinate recovery ───────────────────────────────────────────────────
    sep('7. Coordinate recovery: y = v @ Z_proj, v_rec = y @ Z.T')
    y_true = v_mat @ Z_proj                  # (n_samples, k)
    v_rec  = y_true @ Z.T                    # (n_samples, n_rxn)
    rec_err = np.abs(v_mat - v_rec).max(axis=1)
    print(f'  Max reconstruction error: {rec_err.max():.2e}  (should be ~0)')
    print(f'  Mean reconstruction error: {rec_err.mean():.2e}')

    # ── y_true statistics ─────────────────────────────────────────────────────
    sep('8. y_true statistics (v @ Z_proj)')
    y_std  = y_true.std(axis=0)
    y_mean = y_true.mean(axis=0)
    print(f'  y_std: min={y_std.min():.3f}  max={y_std.max():.3f}  mean={y_std.mean():.3f}')
    print(f'  y_mean: min={y_mean.min():.3f}  max={y_mean.max():.3f}')
    print()
    print('  Target coordinates (first n_targets columns):')
    for i, t in enumerate(targets):
        v_t  = v_mat[:, rxn_names.index(t)]
        y_ti = y_true[:, i]
        r    = np.corrcoef(v_t, y_ti)[0, 1]
        print(f'    col {i} ({t}): std={y_ti.std():.4f}  corr(y_i, v_target)={r:.4f}')

    # ── Original training code (wrong) vs corrected ───────────────────────────
    sep('9. Original vs corrected y_train scale comparison')
    y_wrong = v_mat @ Z          # what the code was doing before Z_proj fix
    y_right = v_mat @ Z_proj

    print(f'  y_wrong (v @ Z):      std range [{y_wrong.std(axis=0).min():.2f}, {y_wrong.std(axis=0).max():.2f}]')
    print(f'  y_right (v @ Z_proj): std range [{y_right.std(axis=0).min():.2f}, {y_right.std(axis=0).max():.2f}]')


if __name__ == '__main__':
    for name, cfg in CONFIGS.items():
        run(name, cfg)
