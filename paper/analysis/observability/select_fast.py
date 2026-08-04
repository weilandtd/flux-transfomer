"""
Fast significant-token selection — det-lemma standalone + permutation-max null, NO greedy, NO d×d grams.

Matches the semantics of the current Fisher selection's *significance* step, but computes each token's
standalone information via the matrix-determinant lemma instead of a full d×d log-determinant:

    F0 = inv(cov(theta) + ridge)          =>  F0^{-1} = Sigma = cov(theta)+ridge   (no inversion!)
    standalone(m) = logdet(F0 + G_m) - logdet(F0)
                  = logdet( I_k + (1/sigma^2) * B_m Sigma B_m^T )    # k×k = 2×2 determinant

with B_m = (pinv(theta) @ L_m)^T  (k×d).  Cost: O(C·k·d^2), memory O(C·d) — no dense d×d grams,
no per-token d^3 logdet, no greedy.  The null floor is the family-wise max standalone under a
permutation of the theta<->L pairing (pinv and Sigma are reused, so permutations are ~free).

Writes a features CSV in the same schema the trainer reads (included == significant; no redundancy trim).

Usage:
  python paper/analysis/observability/select_fast.py --config <cfg> --data-dir <run> --output features_fast.csv
"""
import argparse, os, sys, time, numpy as np, pandas as pd, yaml
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
from sigmaflux.ai.data import load_data
from sigmaflux.ai import features as feat_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output', default='features_fast.csv')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fc = cfg.get('features', {})
    noise_std = fc.get('noise_std', 0.01)
    n_perm = fc.get('n_permutations', 10)
    inc_rev = fc.get('include_reversibilities', True)
    restrict = fc.get('restrict_to_mapped', True)
    ridge = 1e-6
    seed = cfg.get('seed', 42)
    n_samples = fc.get('n_samples', 10000)

    def _abs(p): return p if os.path.isabs(p) else os.path.join(_REPO, p)
    model_json = os.path.join(args.data_dir, 'model_prepared_for_mfa.json')
    if not os.path.isfile(model_json): model_json = _abs(cfg['model_file'])
    mapping_csv = _abs(cfg['mapping_csv']) if cfg.get('mapping_csv') else None

    lab, flux, rev = load_data(args.data_dir, max_samples=n_samples)
    theta, token_states, d = feat_lib.build_theta_and_token_states(
        lab, flux, model_json, n_samples, reversibility_df=rev if inc_rev else None)
    if restrict and mapping_csv and os.path.isfile(mapping_csv):
        token_states = feat_lib.apply_mapping_filter(token_states, mapping_csv)
    keys = list(token_states)
    print(f"candidates={len(keys)} | theta dim D={theta.shape[1]} (free d={d}) | samples={theta.shape[0]}")

    t0 = time.time()
    tc = theta - theta.mean(0, keepdims=True)
    D = tc.shape[1]
    pinv = np.linalg.pinv(tc)                                   # (D, n), once
    cov = np.atleast_2d(np.cov(tc.T))
    Sigma = cov + ridge * np.trace(cov) / D * np.eye(D)         # = F0^{-1}, once
    inv_sig2 = 1.0 / (noise_std ** 2)
    Ik = np.eye(token_states[keys[0]].shape[1])                 # n_tracers x n_tracers

    def standalone(L):
        # det-lemma: logdet(I_k + (1/sigma^2) B Sigma B^T),  B = (pinv @ L)^T  (k x D)
        B = (pinv @ (L - L.mean(0, keepdims=True))).T
        s, v = np.linalg.slogdet(Ik + inv_sig2 * (B @ Sigma @ B.T))
        return float(v) if s > 0 else 0.0

    real = {tok: standalone(token_states[tok]) for tok in keys}

    # permutation-max null (family-wise): break the theta<->L pairing, reuse pinv & Sigma
    rng = np.random.default_rng(seed)
    n = tc.shape[0]; null_max = 0.0
    for _ in range(max(1, n_perm)):
        perm = rng.permutation(n)
        for tok in keys:
            null_max = max(null_max, standalone(token_states[tok][perm]))
    dt = time.time() - t0

    rows = []
    for tok in keys:
        met, iso = tok.split('|', 1)
        rows.append({'metabolite_id': met, 'isotopomer': iso,
                     'standalone_nats': real[tok], 'marginal_nats': real[tok],
                     'significant': real[tok] > null_max})
    df = pd.DataFrame(rows).sort_values('standalone_nats', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(len(df))
    df['included'] = df['significant']                          # no greedy: keep all significant
    out = os.path.join(args.data_dir, args.output)
    df.to_csv(out, index=False)
    n_sig = int(df.significant.sum())
    print(f"null floor(fam-wise max) = {null_max:.3f} | significant/included = {n_sig}/{len(df)} "
          f"| selection time = {dt:.2f}s -> {out}")


if __name__ == '__main__':
    main()
