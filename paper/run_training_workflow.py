"""
Run the full SigmaFlux training workflow driven by a YAML config file.

Steps executed in order:
  1. sample_flux.py            — sample flux distributions from the COBRA model
  2. generate_training_data.py — simulate isotope labeling for each flux sample
  3. train_ai_model.py         — train the labeling-to-flux transformer

Usage (works from any directory):
    python paper/run_training_workflow.py --config paper/io_config.yaml

Individual steps can be skipped if their outputs already exist:
    python paper/run_training_workflow.py --config paper/io_config.yaml \\
        --skip-sampling --skip-generate
"""

import argparse
import datetime
import os
import subprocess
import sys

import yaml

# Absolute path to the paper/ directory — independent of cwd.
_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(cmd):
    print(f"\n{'='*60}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStep failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full SigmaFlux training workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip flux sampling (reuse existing flux_samples.csv)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip labeling simulation (reuse existing HDF5 files)")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--data-dir", default=None, help="Data directory containing HDF5 files (default: <name>_YYYYMMDD)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    t = cfg.get("training", {})
    output_dir = args.data_dir or f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"
    output_model = os.path.join(output_dir, t.get("output_model", "transformer.pt"))
    python = sys.executable

    print(f"Config       : {args.config}")
    print(f"Model file   : {cfg['model_file']}")
    print(f"Output dir   : {output_dir}")
    print(f"Output model : {output_model}")

    # ── Step 1: Sample flux distributions ────────────────────────────────────
    if args.skip_sampling:
        print("\n[Step 1] Skipping flux sampling.")
    else:
        run_step([python, os.path.join(_PAPER_DIR, "sample_flux.py"), "--config", config_path])

    # ── Step 2: Simulate isotope labeling ────────────────────────────────────
    if args.skip_generate:
        print("\n[Step 2] Skipping labeling data generation.")
    else:
        run_step([python, os.path.join(_PAPER_DIR, "generate_training_data.py"), "--config", config_path])

    # ── Step 3: Train the transformer ────────────────────────────────────────
    if args.skip_train:
        print("\n[Step 3] Skipping model training.")
    else:
        run_step([python, os.path.join(_PAPER_DIR, "train_ai_model.py"),
                  "--config", config_path, "--data-dir", output_dir])

    print("\nWorkflow complete.")
    print(f"Trained model: {output_model}")


if __name__ == "__main__":
    main()
