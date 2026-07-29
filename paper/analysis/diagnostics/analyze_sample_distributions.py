"""
Analyze flux sample distributions in a SigmaFlux dataset directory.

Prints three sections to stdout:
  1. Global flux statistics (all samples × all reactions)
  2. Per-reaction 95% interval width (most vs. least variable reactions)
  3. Per-reaction fraction near-zero (constrained vs. free reactions)

Usage:
    python paper/analysis/analyze_sample_distributions.py paper/.results/i_orientalis_20260611
"""
import argparse
import pathlib

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to dataset directory containing flux_samples.csv')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    csv_path = data_dir / 'flux_samples.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"flux_samples.csv not found in {data_dir}")

    print(f"Loading {csv_path} ...")
    flux_df = pd.read_csv(csv_path, index_col=0)
    # flux_df: rows=samples (30000), columns=reactions (388)
    n_samples, n_rxns = flux_df.shape
    print(f"  {n_samples:,} samples  ×  {n_rxns} reactions\n")

    stats = pd.DataFrame({
        'mean':  flux_df.mean(),
        'min':   flux_df.min(),
        'p5':    flux_df.quantile(0.05),
        'p95':   flux_df.quantile(0.95),
        'max':   flux_df.max(),
    })
    stats.index.name = 'reaction'

    col_w = max(len(r) for r in stats.index)
    header = f"{'reaction':<{col_w}}  {'mean':>10}  {'min':>10}  {'p5':>10}  {'p95':>10}  {'max':>10}"
    print(header)
    print("-" * len(header))
    for rxn, row in stats.iterrows():
        print(f"{rxn:<{col_w}}  {row['mean']:>10.4f}  {row['min']:>10.4f}  {row['p5']:>10.4f}  {row['p95']:>10.4f}  {row['max']:>10.4f}")


if __name__ == '__main__':
    main()
