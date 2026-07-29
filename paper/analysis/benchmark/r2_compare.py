"""
Per-reaction validation-R2 scatter between two trained checkpoints (e.g. MID-raw
vs MID-normalized). Points above the diagonal improved in B; below worsened.
Keeps reactions with R2 > --min-r2 in at least one model.

Usage:
    python paper/analysis/r2_compare.py \
        --a paper/.results/i_orientalis_20260620/imisor_transformer.pt \
        --b paper/.results/i_orientalis_20260620/imisor_transformer_midnorm.pt \
        --la "MID raw" --lb "MID normalized" --min-r2 0.1
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_r2(path):
    vc = torch.load(path, map_location='cpu', weights_only=False)['val_correlations']
    return dict(zip(list(vc['flux_reaction_names']),
                    np.asarray(vc['flux_r2_per_reaction'], float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--la', default='model A'); ap.add_argument('--lb', default='model B')
    ap.add_argument('--min-r2', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # consistency check: same val split?
    for p, tag in [(args.a, 'A'), (args.b, 'B')]:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        globals()[f'_vi_{tag}'] = list(ck.get('val_indices', []))
    same_split = _vi_A == _vi_B
    print(f"same validation split: {same_split} (A:{len(_vi_A)} B:{len(_vi_B)} samples)")

    ra, rb = get_r2(args.a), get_r2(args.b)
    rxns = [r for r in ra if r in rb]
    a = np.array([ra[r] for r in rxns]); b = np.array([rb[r] for r in rxns])
    keep = (np.maximum(a, b) > args.min_r2)
    a, b, kr = a[keep], b[keep], [rxns[i] for i in np.where(keep)[0]]
    d = b - a
    print(f"\nreactions with R2>{args.min_r2} in at least one model: {len(a)}")
    print(f"  improved (B>A by >0.05): {int((d > 0.05).sum())}")
    print(f"  worsened (A>B by >0.05): {int((d < -0.05).sum())}")
    print(f"  ~unchanged (|Δ|<=0.05):  {int((np.abs(d) <= 0.05).sum())}")
    print(f"  median Δ (B-A) = {np.median(d):+.3f}   mean Δ = {d.mean():+.3f}")
    print(f"  median R2: A={np.median(a):.3f}  B={np.median(b):.3f}")
    # biggest movers
    order = np.argsort(d)
    print("  most worsened:", [(kr[i], round(a[i],2), round(b[i],2)) for i in order[:4]])
    print("  most improved:", [(kr[i], round(a[i],2), round(b[i],2)) for i in order[-4:]])

    out = args.out or 'r2_compare.svg'
    lo = max(-1.0, min(a.min(), b.min())); hi = 1.02
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    col = np.where(d > 0.05, '#2c7fb8', np.where(d < -0.05, '#d95f02', '#aaaaaa'))
    ax.scatter(a.clip(lo, hi), b.clip(lo, hi), s=14, c=col, alpha=0.6, edgecolor='none')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel(f'{args.la} — per-reaction R²')
    ax.set_ylabel(f'{args.lb} — per-reaction R²')
    ax.set_title(f'{args.lb} vs {args.la}  (n={len(a)}, R²>{args.min_r2} in either)\n'
                 f'blue=improved  orange=worsened  median Δ={np.median(d):+.2f}', fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nsaved -> {out} (+ .png)")


if __name__ == '__main__':
    main()
