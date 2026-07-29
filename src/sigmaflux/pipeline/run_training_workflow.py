"""
Run the full SigmaFlux training workflow driven by a YAML config file.

Steps executed in order:
  1. sample_flux                 — sample flux distributions from the COBRA model
  2. generate_training_data      — simulate isotope labeling for each flux sample
  3. extract_informative_features— Fisher/D-optimal token selection → features.csv
  4. train_ai_model              — train the labeling-to-flux transformer

Usage (works from any directory):
    sigmaflux-workflow --config paper/configs/i_orientalis.yaml

Individual steps can be skipped if their outputs already exist:
    sigmaflux-workflow --config ... --skip-sampling --skip-generate --skip-features

Relative paths inside the config resolve against the config's project root
(see :mod:`sigmaflux.pipeline._paths`), so a downstream project can keep its
own models and configs and still drive this workflow.
"""

import argparse
import os

from sigmaflux.pipeline import (
    extract_informative_features,
    generate_training_data,
    sample_flux,
    train_ai_model,
)
from sigmaflux.pipeline._paths import (
    add_root_argument,
    default_output_dir,
    load_config,
)


def announce(step, title):
    print(f"\n{'=' * 60}")
    print(f">>> [Step {step}] {title}")
    print(f"{'=' * 60}\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the full SigmaFlux training workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Skip flux sampling (reuse existing flux_samples.csv)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip labeling simulation (reuse existing HDF5 files)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction (reuse existing features.csv)")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory containing HDF5 files (default: <name>_YYYYMMDD)")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, root = load_config(args.config, args.root)

    t = cfg.get("training", {})
    output_dir = default_output_dir(cfg, args.data_dir)
    output_model = os.path.join(output_dir, t.get("output_model", "transformer.pt"))
    config_path = cfg["_config_path"]

    print(f"Config       : {config_path}")
    print(f"Project root : {root}")
    print(f"Model file   : {cfg['model_file']}")
    print(f"Output dir   : {output_dir}")
    print(f"Output model : {output_model}")

    # The steps are invoked in-process through their cli() entry points rather
    # than by shelling out, so a failing step raises here instead of being
    # reduced to an exit code.
    common = ["--config", config_path, "--root", root]

    # ── Step 1: Sample flux distributions ────────────────────────────────────
    if args.skip_sampling:
        print("\n[Step 1] Skipping flux sampling.")
    else:
        announce(1, "Sampling flux distributions")
        sample_flux.cli(common + ["--output-dir", output_dir])

    # ── Step 2: Simulate isotope labeling ────────────────────────────────────
    if args.skip_generate:
        print("\n[Step 2] Skipping labeling data generation.")
    else:
        announce(2, "Simulating isotope labeling")
        generate_training_data.cli(common + ["--output-dir", output_dir])

    # ── Step 3: Extract informative features ─────────────────────────────────
    features_csv = os.path.join(output_dir,
                                cfg.get("features", {}).get("output", "features.csv"))
    if args.skip_features or os.path.isfile(features_csv):
        print(f"\n[Step 3] Skipping feature extraction "
              f"({'--skip-features' if args.skip_features else 'features.csv exists'}).")
    else:
        announce(3, "Extracting informative features")
        extract_informative_features.cli(common + ["--data-dir", output_dir])

    # ── Step 4: Train the transformer ────────────────────────────────────────
    if args.skip_train:
        print("\n[Step 4] Skipping model training.")
    else:
        announce(4, "Training the transformer")
        train_ai_model.cli(common + ["--data-dir", output_dir])

    print("\nWorkflow complete.")
    print(f"Features CSV : {features_csv}")
    print(f"Trained model: {output_model}")
    return {"output_dir": output_dir,
            "features_csv": features_csv,
            "output_model": output_model}


if __name__ == "__main__":
    cli()
