"""
Run the full SigmaFlux training workflow for a model folder.

Steps executed in order:
  1. sample_flux.py            — sample flux distributions from the COBRA model
  2. generate_training_data.py — simulate isotope labeling for each flux sample
  3. train_ai_model.py         — train the labeling-to-flux transformer

Usage (works from any directory):
    python paper/run_training_workflow.py I_orientalis
    python /abs/path/to/paper/run_training_workflow.py I_orientalis
    python run_training_workflow.py I_orientalis   # if cwd is paper/

Individual steps can be skipped if their outputs already exist:
    python paper/run_training_workflow.py I_orientalis --skip-sampling --skip-generate
"""

import argparse
import datetime
import glob
import os
import subprocess
import sys

# Absolute paths derived from the script's own location — independent of cwd.
_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PAPER_DIR)


def find_annotated_model(model_folder, models_root=None):
    if models_root is None:
        models_root = os.path.join(_REPO_ROOT, "models")
    pattern = os.path.join(models_root, model_folder, "*_annotated.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No *_annotated.json found in {os.path.join(models_root, model_folder)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple annotated models found: {matches}")
    return matches[0]


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
        description="Run the full SigmaFlux training workflow for a model folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_folder",
        help="Folder name under models/, e.g. I_orientalis",
    )

    # Path overrides
    parser.add_argument(
        "--models-root",
        default=None,
        help="Root directory for models (default: <repo>/models)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model_folder>_<YYYYMMDD>)",
    )
    parser.add_argument(
        "--output-model",
        default=None,
        help="Output .pt file (default: <output_dir>/<model_folder>_transformer.pt)",
    )

    # Shared
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    # Step 1: sampling
    sampling = parser.add_argument_group("Step 1 — flux sampling")
    sampling.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip flux sampling (reuse existing flux_samples.csv)",
    )
    sampling.add_argument("--n-samples", type=int, default=10000)
    sampling.add_argument("--processes", type=int, default=1)
    sampling.add_argument("--thinning", type=int, default=1000)

    # Step 2: labeling generation
    generate = parser.add_argument_group("Step 2 — labeling data generation")
    generate.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip labeling simulation (reuse existing HDF5 files)",
    )

    # Step 3: training
    train = parser.add_argument_group("Step 3 — transformer training")
    train.add_argument("--skip-train", action="store_true", help="Skip model training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)

    args = parser.parse_args()

    # Resolve paths
    datestamp = datetime.date.today().strftime("%Y%m%d")
    run_name = f"{args.model_folder}_{datestamp}"
    model_json = find_annotated_model(args.model_folder, args.models_root)
    output_dir = args.output_dir or run_name
    output_model = args.output_model or os.path.join(
        output_dir, f"{args.model_folder.lower()}_transformer.pt"
    )
    python = sys.executable

    print(f"Model folder : {args.model_folder}")
    print(f"Annotated JSON : {model_json}")
    print(f"Training data  : {output_dir}")
    print(f"Output model   : {output_model}")

    # ── Step 1: Sample flux distributions ────────────────────────────────────
    if args.skip_sampling:
        print("\n[Step 1] Skipping flux sampling.")
    else:
        run_step([
            python, os.path.join(_PAPER_DIR, "sample_flux.py"),
            "--model-file", model_json,
            "--output-dir", output_dir,
            "--n-samples", str(args.n_samples),
            "--processes", str(args.processes),
            "--thinning", str(args.thinning),
            "--seed", str(args.seed),
        ])

    # ── Step 2: Simulate isotope labeling ────────────────────────────────────
    if args.skip_generate:
        print("\n[Step 2] Skipping labeling data generation.")
    else:
        prepared_model = os.path.join(output_dir, "model_prepared_for_mfa.json")
        flux_samples = os.path.join(output_dir, "flux_samples.csv")
        run_step([
            python, os.path.join(_PAPER_DIR, "generate_training_data.py"),
            "--model_file", prepared_model,
            "--flux_samples", flux_samples,
            "--output_dir", output_dir,
            "--device", args.device,
            "--seed", str(args.seed),
        ])

    # ── Step 3: Train the transformer ────────────────────────────────────────
    if args.skip_train:
        print("\n[Step 3] Skipping model training.")
    else:
        run_step([
            python, os.path.join(_PAPER_DIR, "train_ai_model.py"),
            "--data-dir", output_dir,
            "--model-json", model_json,
            "--output", output_model,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
        ])

    print("\nWorkflow complete.")
    print(f"Trained model: {output_model}")


if __name__ == "__main__":
    main()
