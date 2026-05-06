"""
Generate SigmaFlux AI training data from pre-sampled flux distributions.

This follows the mammalian atlas workflow:
  1. Load an annotated COBRA model.
  2. Load a CSV of sampled fluxes.
  3. Process a slice of samples.
  4. Simulate isotope labeling with MFAModel.
  5. Save labeling, flux, and reversibility HDF5 files.

Example:
    PYTHONPATH=src python paper/generate_training_data.py \
        --flux_samples training_data/flux_samples.csv \
        --start_idx 0 \
        --end_idx 500 \
        --output_dir training_data \
        --device cuda
"""

import argparse
import os
import re

import pandas as pd
import torch
from cobra.io.json import load_json_model
from tqdm import tqdm

from sigmaflux.core.data import MassIsotopomerData, create_data_index
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.model import MFAModel
from sigmaflux.expansion.emu import EMU


TRACERS = [
    {"tracer_id": "glc__D_out", "labeling": [1, 1, 1, 1, 1, 1], "enrichment": 0.5},
    {"tracer_id": "glc__D_out", "labeling": [1, 1, 0, 0, 0, 0], "enrichment": 0.5},
]


MEASURED_METABOLITE_IDS = [
    "glc__D_c", "g6p_c", "f6p_c", "fdp_c", "dhap_c", "g3p_c", "3pg_c", "pep_c", "pyr_c",
    "cit_c", "akg_c", "succ_c", "fum_c", "mal__L_c",
    "6pgc_c", "r5p_c", "s7p_c", "e4p_c",
    "glyc3p_c", "3mob_c", "2ippm_c", "orot_c",
    "ala__L_c", "arg__L_c", "asp__L_c", "gln__L_c", "glu__L_c", "gly_c",
    "his__L_c", "ile__L_c", "leu__L_c", "lys__L_c", "met__L_c",
    "phe__L_c", "pro__L_c", "ser__L_c", "thr__L_c", "trp__L_c", "tyr__L_c", "val__L_c",
    "citr__L_c", "orn_c",
    "atp_c", "amp_c", "imp_c", "ump_c", "utp_c", "dgmp_c",
    "gam6p_c",
]


DEFAULT_MODEL_FILE = "models/I_orientalis/imIsor_annotated.json"


def infer_atom_pattern(model, met_id):
    patterns = []
    for reaction in model.metabolites.get_by_id(met_id).reactions:
        atom_mapping = reaction.annotation.get("atom_mapping", "")
        pattern = r"\b{}\s*\(([^)]+)\)".format(re.escape(met_id))
        patterns.extend(re.findall(pattern, atom_mapping))

    if not patterns:
        return ""

    return atom_pattern_for_size(max(len(pattern) for pattern in patterns))


def atom_pattern_for_size(n_carbons):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if n_carbons <= len(alphabet):
        return alphabet[:n_carbons]
    return "".join(f"a{i}" for i in range(1, n_carbons + 1))


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda:0" if device_arg == "cuda" else "cpu")


def build_experiments(mfa_model):
    tracers = []
    for tracer_info in TRACERS:
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

    for met_id in MEASURED_METABOLITE_IDS:
        try:
            metabolite = mfa_model.metabolites.get_by_id(met_id)
            atom_pattern = metabolite.annotation.get("atom_pattern", "")
            if len(atom_pattern) == 0:
                atom_pattern = infer_atom_pattern(mfa_model, met_id)
            if len(atom_pattern) == 0:
                print(f"WARNING: {met_id} has no atom mapping, skipping")
                continue

            metabolite.annotation["atom_pattern"] = atom_pattern
            metabolite_atom_patterns[metabolite.id] = atom_pattern
            targets.add(EMU(metabolite, list(range(1, len(atom_pattern) + 1))))
            measured_metabolites.append(metabolite)
        except KeyError:
            print(f"WARNING: {met_id} not found in model, skipping")

    if not targets:
        raise ValueError("No EMU targets were built. Check measured metabolites and atom mappings.")

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

    model_file = args.model_file or DEFAULT_MODEL_FILE
    flux_samples_file = args.flux_samples or os.path.join(args.output_dir, "flux_samples.csv")

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

    experiments = build_experiments(mfa_model)
    targets, measured_metabolites, metabolite_atom_patterns = build_targets(mfa_model)

    print("Preparing MFAModel...")
    mfa_model.prepare(experiments, targets=targets, with_backward=False)

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

    for sample_ix in tqdm(range(len(flux_samples)), desc="Processing samples"):
        fluxes = flux_samples.iloc[sample_ix]
        mfa_model.reset_initial_fluxes(initial_flux_distribution=fluxes)

        reversibilities = pd.Series(
            mfa_model.reversibility.cpu().detach().numpy(),
            index=[reaction.id for reaction in mfa_model.reactions],
        )
        reversibility_data.append(reversibilities)
        flux_data.append(fluxes)

        with torch.no_grad():
            labeling = mfa_model()

        labeling_vector = torch.take(labeling, labeling_to).cpu().detach().numpy()
        if labeling_vector.sum() == 0:
            raise ValueError("Predicted labeling is all zeros.")
        labeling_data.append(labeling_vector)

        if (sample_ix + 1) % args.batch_size == 0 or (sample_ix + 1) == len(flux_samples):
            write_training_batch(
                args.output_dir,
                output_suffix,
                labeling_data,
                flux_data,
                reversibility_data,
                labeling_index,
            )

    print(f"\nCompleted processing samples {args.start_idx}:{end_idx}")
    print(f"Output files saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data from pre-sampled fluxes.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="training_data")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--flux_samples", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
