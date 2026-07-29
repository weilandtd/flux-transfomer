"""
Reverse Fisher analysis — how many FLUXES are observable from the MIDs?

Forward Fisher ranks MIDs by their sensitivity to flux. The reverse question:
given the selected MID set, how many independent flux directions can it actually
resolve, and how many of the 387 reactions become predictable?

  1. Eigenspectrum of the total Fisher information relative to the flux prior
     (observable_spectrum): each eigenvalue = how much a flux direction's
     uncertainty shrinks. n_resolved / n_resolved_90/99 = # resolvable directions.
  2. Per-reaction: project the posterior vs prior covariance through the
     null-space basis Z -> variance reduction per reaction -> how many of the
     387 reactions are predictable from the MIDs.

Usage: python paper/analysis/fisher_observability.py --config paper/configs/i_orientalis.yaml --data-dir <run>
"""
import argparse, os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from sigmaflux.ai.data import load_data, load_stoichiometric_matrix
from sigmaflux.ai import features as feat_lib
from sigmaflux.ai.fisher_selection import empirical_sensitivity_grams, observable_spectrum, _prior_precision
from sigmaflux.ai.utils import compute_sparse_nullspace
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'main': '#4C72B0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-dir', required=True)
    add_root_argument(ap)
    args = ap.parse_args()
    cfg, root = load_config(args.config, args.root)
    fc = cfg.get('features', {})
    noise_std = fc.get('noise_std', 0.02)
    n_samples = fc.get('n_samples', 10000)

    model_json = os.path.join(args.data_dir, 'model_prepared_for_mfa.json')
    if not os.path.isfile(model_json): model_json = resolve_path(cfg['model_file'], root)

    lab, flux, rev = load_data(args.data_dir, max_samples=n_samples)
    # FORWARD flux directions only — no reversibility augmentation
    theta, token_states, d = feat_lib.build_theta_and_token_states(
        lab, flux, model_json, n_samples, reversibility_df=None, seed=cfg.get('seed', 42))
    print(f"forward free-flux directions d = {d} | candidate tokens = {len(token_states)} | samples = {theta.shape[0]}")
    measured_fluxes = cfg.get('training', {}).get('measured_fluxes') or []
    flux_noise = cfg.get('training', {}).get('flux_noise_std', 0.05)

    # restrict to the tokens the model actually uses (features.csv 'included')
    feats = os.path.join(args.data_dir, 'features.csv')
    if os.path.isfile(feats):
        f = pd.read_csv(feats)
        inc = set(f.loc[f['included'], 'metabolite_id'].astype(str) + '|' + f.loc[f['included'], 'isotopomer'].astype(str))
        token_states = {k: v for k, v in token_states.items() if k in inc}
        print(f"restricted to {len(token_states)} model-used tokens")

    grams = empirical_sensitivity_grams(theta, token_states)
    spec = observable_spectrum(grams, theta, noise_std=noise_std, resolve_ratio=2.0)

    print("\n=== Direction-space observability (of %d free-flux directions) ===" % d)
    print("top eigenvalues (posterior/prior precision ratio):", np.round(spec['eigenvalues'][:12], 2))
    print(f"  n_resolved (precision >= 2x)         : {spec['n_resolved']}")
    print(f"  n_resolved (>=90% variance removed)  : {spec['n_resolved_90']}")
    print(f"  n_resolved (>=99% variance removed)  : {spec['n_resolved_99']}")
    print(f"  effective rank (participation ratio) : {spec['effective_rank']:.2f}")

    # ── per-reaction observability via Z projection ──────────────────────────
    S, cobra = load_stoichiometric_matrix(model_json)
    Z = np.asarray(compute_sparse_nullspace(np.asarray(S, float)), float)     # (n_rxn, d)
    dd = theta.shape[1]                                                        # d (forward flux only)
    F0 = _prior_precision(theta, dd)
    Cov_prior = np.linalg.inv(F0)
    mid_F = sum((G for G in grams.values()), np.zeros((dd, dd))) / noise_std ** 2

    # known-flux measurements: each measured reaction k observes v[k] = Z[k]·y
    # with ~flux_noise relative to its prior std -> rank-1 Fisher (1/σ_k²) Z[k]ᵀZ[k]
    rxn_ids = [r.id for r in cobra.reactions]
    id2i = {r: i for i, r in enumerate(rxn_ids)}
    vp_prior = np.einsum('rd,de,re->r', Z, Cov_prior, Z)                       # prior var of each reaction
    flux_F = np.zeros((dd, dd))
    for m in measured_fluxes:
        if m in id2i:
            k = id2i[m]; sig2 = (flux_noise ** 2) * max(vp_prior[k], 1e-12)
            flux_F += np.outer(Z[k], Z[k]) / sig2
    print(f"known-flux inputs used: {sum(m in id2i for m in measured_fluxes)}/{len(measured_fluxes)}")

    n_rxn = Z.shape[0]
    def reduction(F_extra):
        Cq = np.linalg.inv(F0 + F_extra)
        vq = np.einsum('rd,de,re->r', Z, Cq, Z)
        return np.clip(1.0 - vq / np.clip(vp_prior, 1e-30, None), 0, 1)
    red_mid  = reduction(mid_F)
    red_flux = reduction(flux_F)
    red_all  = reduction(mid_F + flux_F)

    print(f"\n=== Per-reaction observability (of {n_rxn} FORWARD reactions) ===")
    print(f"{'threshold':>12} | {'MIDs only':>10} | {'known flux':>10} | {'MID+flux':>10}")
    for thr in [0.10, 0.25, 0.50, 0.90]:
        lab_thr = 'some (>=10%)' if thr == 0.10 else f'>= {int(thr*100)}%'
        print(f"{lab_thr:>12} | {int((red_mid>=thr).sum()):>10} | {int((red_flux>=thr).sum()):>10} | {int((red_all>=thr).sum()):>10}")
    print(f"{'median':>12} | {np.median(red_mid):>10.2f} | {np.median(red_flux):>10.2f} | {np.median(red_all):>10.2f}")
    red = red_mid  # for the histogram below

    # ── plots: direction scree + per-reaction histogram ──────────────────────
    dfd = pd.DataFrame({'direction': np.arange(1, len(spec['variance_reduction'])+1), 'var_removed': spec['variance_reduction']})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.lineplot(data=dfd, x='direction', y='var_removed', marker='o', color=COLORS['main'], ax=ax)
    ax.axhline(0.9, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Flux direction (sorted)'); ax.set_ylabel('Variance removed by MIDs')
    ax.set_ylim(0, 1.02); sns.despine(ax=ax)
    out = os.path.join(args.data_dir, 'fisher_observability_directions.svg')
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(red, bins=20, color=COLORS['main'], ax=ax)
    ax.set_xlabel('Per-reaction variance removed by MIDs'); ax.set_ylabel('# reactions')
    sns.despine(ax=ax)
    out2 = os.path.join(args.data_dir, 'fisher_observability_reactions.svg')
    fig.savefig(out2, bbox_inches='tight'); fig.savefig(out2.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nsaved -> {out}\nsaved -> {out2}")


if __name__ == '__main__':
    main()
