"""
KO-detection evaluation. For every KO validation sample, does the model drive the
knocked-out reaction's predicted flux to ~0?

A "hit" = |v_pred[R]| < tau_rel * WT_std[R] (predicted flux is small relative to that
reaction's normal WT variation). Reported split by seen vs held-out KO reactions and
by free vs dependent reaction (null-space structure).

Usage:
  python paper/analysis/ko_eval.py --checkpoint <ko.pt> --run-dir <ko run> --tau-rel 0.1
"""
import argparse, json, os, numpy as np, pandas as pd, torch
from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tau-rel', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = resolve_device()
    ctx = load_eval_context(args.checkpoint)
    model = build_flux_decoder(ctx, device=device)

    nsd = ctx.null_space_data
    flux_names = ctx.flux_names
    name2idx = {n: i for i, n in enumerate(flux_names)}
    val_idx = ctx.sample_indices

    labels = pd.read_csv(os.path.join(args.run_dir, 'ko_labels.csv'), index_col=0)
    fs = pd.read_csv(os.path.join(args.run_dir, 'flux_samples.csv'), index_col=0)
    wt_std = fs[labels['ko_reaction'].values == 'WT'].std(axis=0)        # per-reaction WT scale

    # free (identity) vs dependent reactions from the null-space basis
    Z = np.asarray(nsd['Z_free'], float)
    is_free = np.array([(np.sum(np.abs(Z[i]) > 1e-6) == 1) for i in range(Z.shape[0])])

    (cids, lids, tids, fracs), targets = ctx.full_batch(device=device)
    fixed_idx = model.fixed_idx_t.to(device)
    with torch.no_grad():
        out = model(cids, lids, tids, fracs, measured_fluxes=targets.index_select(1, fixed_idx),
                    measured_flux_ids=fixed_idx, return_attn=False)
    v_pred = out[0].cpu().numpy()                                         # (n_val, 387)

    val_labels = labels.iloc[val_idx].reset_index(drop=True)
    rows = []
    for i in range(len(val_labels)):
        r = val_labels['ko_reaction'].iloc[i]
        if r == 'WT' or r not in name2idx:
            continue
        ridx = name2idx[r]
        pred = abs(float(v_pred[i, ridx]))
        scale = float(wt_std[r]) if wt_std[r] > 1e-9 else 1.0
        rows.append({'reaction': r, 'split': val_labels['split'].iloc[i],
                     'pred_abs': pred, 'wt_std': scale, 'rel': pred / scale,
                     'hit': pred < args.tau_rel * scale, 'is_free': bool(is_free[ridx])})
    df = pd.DataFrame(rows)

    def rate(sub, lab):
        if len(sub) == 0: print(f"  {lab}: (none)"); return
        print(f"  {lab}: n={len(sub)}  hit-rate={sub['hit'].mean():.2f}  "
              f"median rel={sub['rel'].median():.3f}  (reactions={sub['reaction'].nunique()})")
    print(f"KO-detection (hit = |v_pred[R]| < {args.tau_rel}·WT_std[R]) on {len(df)} KO val samples")
    rate(df, "ALL KO")
    rate(df[df.split == 'mixed'], "seen KO reactions")
    rate(df[df.split == 'valonly'], "held-out KO reactions")
    rate(df[df.is_free], "free (identity) reactions")
    rate(df[~df.is_free], "dependent reactions")
    print("\nper-reaction hit-rate:")
    per = df.groupby(['reaction', 'split', 'is_free']).agg(n=('hit', 'size'), hit=('hit', 'mean'),
                                                           med_rel=('rel', 'median')).reset_index()
    print(per.to_string(index=False))
    out = args.out or os.path.join(args.run_dir, 'ko_eval.csv')
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
