"""
Generate SigmaFlux AI training data from pre-sampled flux distributions.

This follows the mammalian atlas workflow:
  1. Load an annotated COBRA model.
  2. Load a CSV of sampled fluxes.
  3. Process a slice of samples.
  4. Simulate isotope labeling with MFAModel.
  5. Save labeling, flux, and reversibility HDF5 files.

Example:
    PYTHONPATH=src python paper/generate_training_data.py --config paper/io_config.yaml

    # Process a specific slice (for parallel jobs):
    PYTHONPATH=src python paper/generate_training_data.py --config paper/io_config.yaml \\
        --start-idx 0 --end-idx 500
"""

import argparse
import datetime
import os

# Pin thread counts to the SLURM allocation before any BLAS/OpenMP library
# is imported.  Without this, PyTorch/OpenBLAS spawn as many threads as there
# are physical cores on the node, causing heavy contention when multiple jobs
# share the node and producing the large per-iteration time variance seen in
# practice.
_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_n_threads))

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
torch.set_num_threads(_n_threads)
torch.set_num_interop_threads(1)
from cobra.io.json import load_json_model
from tqdm import tqdm

from sigmaflux.core.data import MassIsotopomerData, create_data_index
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU



def validate_tracers(tracer_cfg, media_cfg):
    """Check that every tracer metabolite has an open exchange reaction in the media.

    The exchange reaction for metabolite ``met_id`` is expected to be named
    ``EX_<met_id>``.  A reaction is considered open when at least one of its
    bounds is non-zero (i.e. flux can flow through it).

    Raises ValueError listing all incompatible tracers.
    """
    errors = []
    for t in tracer_cfg:
        met_id = t["tracer_id"]
        ex_id = f"EX_{met_id}"
        bounds = media_cfg.get(ex_id)
        if bounds is None:
            errors.append(
                f"  tracer '{met_id}': exchange reaction '{ex_id}' is not in the media"
            )
        elif bounds["lb"] == 0 and bounds["ub"] == 0:
            errors.append(
                f"  tracer '{met_id}': exchange reaction '{ex_id}' is closed (lb=0, ub=0)"
            )
    if errors:
        raise ValueError(
            "Tracer/media incompatibility — the following tracers cannot enter the network:\n"
            + "\n".join(errors)
        )


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda:0" if device_arg == "cuda" else "cpu")


def build_experiments(mfa_model, tracer_cfg):
    tracers = []
    for tracer_info in tracer_cfg:
        metabolite = mfa_model.metabolites.get_by_id(tracer_info["tracer_id"])
        tracer = Tracer(
            metabolite,
            tracer_info["enrichment"],
            tracer_info["labeling"],
        )
        tracers.append(tracer)
        print(
            f"Loaded tracer: {tracer_info['tracer_id']}, "
            f"labeling: {tracer_info['labeling']}, "
            f"enrichment: {tracer_info['enrichment']}"
        )

    return [
        Experiment(mid_data=MassIsotopomerData(), tracers=[tracer], fluxes=None, enrichment=None)
        for tracer in tracers
    ]


def build_targets(mfa_model):
    targets = set()
    measured_metabolites = []
    metabolite_atom_patterns = {}

    for metabolite in mfa_model.metabolites:
        atom_pattern = metabolite.annotation.get("atom_pattern", "")
        if not atom_pattern:
            continue
        metabolite_atom_patterns[metabolite.id] = atom_pattern
        targets.add(EMU(metabolite, list(range(1, len(atom_pattern) + 1))))
        measured_metabolites.append(metabolite)

    if not targets:
        raise ValueError("No EMU targets found. Check model atom_pattern annotations.")

    print(f"Targets: {len(targets)} EMUs from {len(measured_metabolites)} metabolites")
    return targets, measured_metabolites, metabolite_atom_patterns


def build_labeling_index(mfa_model, experiments, measured_metabolites, metabolite_atom_patterns):
    labeling_to = []
    index = []

    for exp_ix, experiment in enumerate(experiments):
        tracer = "|".join(str(tracer) for tracer in experiment.tracers)
        for metabolite in measured_metabolites:
            metabolite_ix = mfa_model._metabolite_hash[metabolite.id]
            atom_pattern = metabolite_atom_patterns[metabolite.id]
            for isotopomer_ix in range(1, len(atom_pattern) + 1):
                ix = create_data_index(mfa_model, exp_ix, isotopomer_ix, metabolite_ix)
                labeling_to.append(ix)
                index.append((tracer, metabolite.id, f"M+{isotopomer_ix}"))

    return (
        torch.tensor(labeling_to, device=mfa_model.device),
        pd.MultiIndex.from_tuples(index, names=["Tracer", "Metabolite", "Isotopomer"]),
    )


def write_training_batch(output_dir, suffix, labeling_data, flux_data, reversibility_data, labeling_index):
    pd.DataFrame(labeling_data, columns=labeling_index).T.to_hdf(
        os.path.join(output_dir, f"wb_model_labeling_data{suffix}.h5"),
        key="labeling",
        mode="w",
    )
    pd.DataFrame(flux_data).T.to_hdf(
        os.path.join(output_dir, f"wb_model_flux_data{suffix}.h5"),
        key="fluxes",
        mode="w",
    )
    pd.DataFrame(reversibility_data).T.to_hdf(
        os.path.join(output_dir, f"wb_model_reversibility_data{suffix}.h5"),
        key="reversibility",
        mode="w",
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    device = resolve_device(args.device)
    dtype = torch.float64

    model_file = args.model_file
    flux_samples_file = args.flux_samples or os.path.join(args.output_dir, "flux_samples.csv")

    validate_tracers(args.tracer_cfg, args.media_cfg)

    print(f"Using device: {device}")
    print(f"Loading model: {model_file}")
    cobra_model = load_json_model(model_file)
    print(f"Cobra model has {len(cobra_model.reactions)} reactions and {len(cobra_model.metabolites)} metabolites")

    print(f"Loading flux samples: {flux_samples_file}")
    flux_samples = pd.read_csv(flux_samples_file, index_col=0)

    end_idx = args.end_idx if args.end_idx is not None else len(flux_samples)
    if args.n_samples is not None:
        end_idx = min(args.start_idx + args.n_samples, len(flux_samples))
    if args.start_idx < 0 or end_idx > len(flux_samples) or args.start_idx >= end_idx:
        raise ValueError(f"Invalid sample range {args.start_idx}:{end_idx}; available samples: 0:{len(flux_samples)}")

    flux_samples = flux_samples.iloc[args.start_idx:end_idx]
    print(f"Processing {len(flux_samples)} samples ({args.start_idx}:{end_idx})")

    mfa_model = MFAModel(
        cobra_model,
        "mfa_model",
        device=device,
        dtype=dtype,
        initial_flux_distribution=flux_samples.iloc[0],
    )

    experiments = build_experiments(mfa_model, args.tracer_cfg)
    targets, measured_metabolites, metabolite_atom_patterns = build_targets(mfa_model)

    print("Preparing MFAModel...")
    mfa_model.prepare(experiments, targets=targets, with_backward=False)

    # After pruning, restrict output to metabolites that actually exist in the EMU network.
    # mfa_model.emu_met_filter is keyed by subnet size and contains only post-pruning EMUs.
    emu_met_ids = {emu.id for emus in mfa_model.emu_met_filter.values() for emu in emus}
    before = len(measured_metabolites)
    measured_metabolites = [m for m in measured_metabolites if m.id in emu_met_ids]
    metabolite_atom_patterns = {k: v for k, v in metabolite_atom_patterns.items() if k in emu_met_ids}
    print(f"Metabolites after EMU network pruning: {len(measured_metabolites)} (removed {before - len(measured_metabolites)})")

    labeling_to, labeling_index = build_labeling_index(
        mfa_model,
        experiments,
        measured_metabolites,
        metabolite_atom_patterns,
    )

    labeling_data = []
    flux_data = []
    reversibility_data = []
    output_suffix = f"_{args.start_idx}_{end_idx}"
    skipped = 0

    for sample_ix in tqdm(range(len(flux_samples)), desc="Processing samples"):
        fluxes = flux_samples.iloc[sample_ix]
        mfa_model.reset_initial_fluxes(initial_flux_distribution=fluxes)

        try:
            with torch.no_grad():
                labeling = mfa_model()
        except RuntimeError as exc:
            print(f"Warning: skipping sample {sample_ix}: degenerate flux configuration ({exc})")
            skipped += 1
            continue

        labeling_vector = torch.take(labeling, labeling_to).cpu().detach().numpy()
        if labeling_vector.sum() == 0:
            raise ValueError("Predicted labeling is all zeros.")

        reversibilities = pd.Series(
            mfa_model.reversibility.cpu().detach().numpy(),
            index=[reaction.id for reaction in mfa_model.reactions],
        )
        reversibility_data.append(reversibilities)
        flux_data.append(fluxes)
        labeling_data.append(labeling_vector)

        accepted = sample_ix + 1 - skipped
        if accepted % args.batch_size == 0 or (sample_ix + 1) == len(flux_samples):
            write_training_batch(
                args.output_dir,
                output_suffix,
                labeling_data,
                flux_data,
                reversibility_data,
                labeling_index,
            )

    print(f"\nCompleted processing samples {args.start_idx}:{end_idx}")
    if skipped:
        print(f"Skipped {skipped} degenerate samples (inversion failure); {len(labeling_data)} samples written.")
    print(f"Output files saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data from pre-sampled fluxes.")
    parser.add_argument("--config", required=True, help="Path to YAML config file (e.g. paper/io_config.yaml)")
    parser.add_argument("--output-dir", default=None, help="Directory containing sampled inputs and where HDF5 outputs are written")
    # Optional slice overrides for parallel jobs
    parser.add_argument("--start-idx", type=int, default=None, help="Override start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="Override end sample index")
    parser.add_argument("--n-samples", type=int, default=None, help="Override number of samples to process")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    lab = cfg.get("labeling", {})
    output_dir = cli.output_dir or f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"

    args = argparse.Namespace(
        output_dir=output_dir,
        model_file=os.path.join(output_dir, "model_prepared_for_mfa.json"),
        flux_samples=os.path.join(output_dir, "flux_samples.csv"),
        device=cfg.get("device", "auto"),
        seed=cfg.get("seed", 42),
        batch_size=lab.get("batch_size", 500),
        start_idx=cli.start_idx if cli.start_idx is not None else 0,
        end_idx=cli.end_idx,
        n_samples=cli.n_samples,
        tracer_cfg=lab["tracers"],
        media_cfg=cfg["sampling"]["media"],
    )
    main(args)
