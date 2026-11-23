"""
Build a train/val split for the KO experiment.

Plain RANDOM split over all samples (WT + every KO), so every KO reaction is seen
during training (a real KO in the field differs by condition/growth, so holding out
whole reactions is not the right generalization test). Reports the minimum per-reaction
training count to confirm every KO reaction is represented in the training data.

Indices are positional into flux_samples row order == load_data column order
(requires training.max_samples = null so no resubsampling reorders them).
Saves split.json {train_indices, val_indices} in the run dir.
"""
import argparse, json, os, numpy as np, pandas as pd


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--val-fraction', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    return ap


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)

    labels = pd.read_csv(os.path.join(args.run_dir, 'ko_labels.csv'), index_col=0)
    n = len(labels)

    # no held-out reactions any more — every KO reaction is 'mixed' (seen in training)
    labels['split'] = 'mixed'
    labels.to_csv(os.path.join(args.run_dir, 'ko_labels.csv'))

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int((1 - args.val_fraction) * n)
    train_idx = np.sort(idx[:cut])
    val_idx = np.sort(idx[cut:])

    json.dump({'train_indices': train_idx.tolist(), 'val_indices': val_idx.tolist()},
              open(os.path.join(args.run_dir, 'split.json'), 'w'))

    # coverage report: every KO reaction must have training samples
    tr = labels.iloc[train_idx]['ko_reaction'].value_counts()
    va = labels.iloc[val_idx]['ko_reaction'].value_counts()
    ko_rxns = [r for r in labels['ko_reaction'].unique() if r != 'WT']
    min_tr = min(int(tr.get(r, 0)) for r in ko_rxns)
    print(f"total={n}  train={len(train_idx)}  val={len(val_idx)}  (random split, seed={args.seed})")
    print(f"KO reactions: {len(ko_rxns)}  | every one in training? {all(tr.get(r,0)>0 for r in ko_rxns)}")
    print(f"per-KO-reaction train count: min={min_tr}  median={int(np.median([tr.get(r,0) for r in ko_rxns]))}")
    print(f"per-KO-reaction val count:   min={min(int(va.get(r,0)) for r in ko_rxns)}")
    print(f"saved -> {os.path.join(args.run_dir, 'split.json')}")


if __name__ == "__main__":
    cli()
