"""
Extract informative labeling tokens for flux estimation — pipeline step 3.

Sits between generate_training_data.py (step 2) and train_ai_model.py (step 4).
Loads the generated HDF5 data, optionally filters to experimentally measurable
metabolites via the mapping CSV, runs Fisher/D-optimal token selection, and
writes features.csv to the run directory.

The training script consumes this CSV automatically when
  token_selection_method: fisher
is set in the config's features: section.

Usage:
    python paper/pipeline/extract_informative_features.py \\
        --config paper/configs/i_orientalis.yaml --data-dir <run_dir>
"""
import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.ai.data import load_data
from sigmaflux.ai import features as feat_lib


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fc               = cfg.get("features", {})
    n_samples        = fc.get("n_samples", 10000)
    subsample_n      = fc.get("subsample_n", None)
    noise_std        = fc.get("noise_std", 0.02)
    gain_threshold   = fc.get("gain_threshold", 0.01)
    budget           = fc.get("budget", None)
    run_mi_audit     = fc.get("run_mi_audit", False)
    restrict         = fc.get("restrict_to_mapped", False)
    out_name         = fc.get("output", "features.csv")
    seed             = cfg.get("seed", 42)

    output_path = os.path.join(args.data_dir, out_name)

    if os.path.isfile(output_path):
        print(f"[extract_informative_features] {output_path} already exists — skipping.")
        return

    print(f"Config   : {args.config}")
    print(f"Data dir : {args.data_dir}")
    print(f"Output   : {output_path}")

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

    model_json = _abs(cfg["model_file"])
    prepared = os.path.join(args.data_dir, "model_prepared_for_mfa.json")
    if os.path.isfile(prepared):
        model_json = prepared

    mapping_csv = _abs(cfg["mapping_csv"]) if cfg.get("mapping_csv") else None

    print(f"\nLoading generated data (up to {n_samples} samples) ...")
    labeling_df, flux_df, reversibility_df = load_data(args.data_dir, max_samples=n_samples)

    theta, token_states, d = feat_lib.build_theta_and_token_states(
        labeling_df, flux_df, model_json, n_samples,
        reversibility_df=reversibility_df,
        subsample_n=subsample_n, seed=seed,
    )
    print(f"  free-flux directions d = {d}; candidate tokens = {len(token_states)}; "
          f"samples = {theta.shape[0]}")

    if restrict:
        if mapping_csv and os.path.isfile(mapping_csv):
            token_states = feat_lib.apply_mapping_filter(token_states, mapping_csv)
        else:
            print("WARNING: restrict_to_mapped=true but mapping_csv not found — skipping filter")

    df, summary = feat_lib.select_informative_tokens(
        theta, token_states,
        noise_std=noise_std,
        gain_threshold=gain_threshold,
        budget=budget,
        run_mi_audit=run_mi_audit,
    )

    df.to_csv(output_path, index=False)

    n_sig = int(df["significant"].sum())
    n_sel = int(df["selected"].sum())

    print("\n=======================  FEATURE SELECTION SUMMARY  =======================")
    print(f"Free-flux directions                     : {summary['n_free_flux_directions']}")
    eff_rank = summary.get("effective_rank")
    if eff_rank is not None:
        print(f"Effective rank                           : {eff_rank:.2f}")
    print(f"Noise std                                : {summary['noise_std']}")
    print(f"Null significance floor (nats)           : {summary.get('null_floor_nats', 0.0):.4f}")
    print("-" * 74)
    print(f"TOTAL candidate tokens                   : {len(df)}")
    print(f"  ... significant (info > null floor)    : {n_sig}")
    print(f"  ... selected (greedy D-optimal subset) : {n_sel}")
    if n_sel > 0:
        print(f"  unique metabolites among selected      : "
              f"{df.loc[df['selected'], 'metabolite_id'].nunique()}")
    print("-" * 74)

    ev = summary.get("eigenvalues", [])
    if len(ev):
        print("Top eigenvalues: " + ", ".join(f"{e:.1f}" for e in ev[:10]))

    print(f"\nFeatures written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--data-dir", required=True,
                        help="Run directory containing generated *.h5 files")
    main(parser.parse_args())
