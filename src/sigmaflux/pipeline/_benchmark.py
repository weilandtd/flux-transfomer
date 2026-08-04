"""Shared plumbing for the accuracy benchmarks.

:mod:`sigmaflux.pipeline.run_ai_benchmark` and
:mod:`sigmaflux.pipeline.run_mfa_benchmark` score two different predictors on the
same held-out samples and write the same HDF5 schema, so everything that defines
"the same benchmark" — which samples, which model, which labeling vector, which
output layout — lives here. The two drivers differ only in how they turn ground
truth into a prediction.

These are small composable pieces rather than one setup call because the drivers
need a different order: the AI benchmark has to load the predictor first, since an
ensemble's shared validation split lives in its member checkpoint rather than in
the run directory's top-level ``.pt``.
"""

import os

import numpy as np
import pandas as pd
import torch
from cobra.io.json import load_json_model

from ..ai.data import find_checkpoint, load_data, split_sample_indices
from ..core.model import MFAModel
from .generate_training_data import build_experiments, build_labeling_index, build_targets


def pin_threads():
    """Pin the BLAS/torch thread pool to the cores SLURM actually granted.

    Without this a benchmark task oversubscribes its allocation: torch defaults to
    every core on the node, not the ``--cpus-per-task`` it may use.
    """
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)
    return n_threads


def load_benchmark_inputs(data_dir):
    """Load the run directory's model, observable metabolites and training data.

    Returns ``(cobra_model, mapped_met_ids, labeling_df, flux_df, rev_df)``.
    ``mapped_met_ids`` is every metabolite in ``features.csv`` — all of them, not
    only those flagged ``included``, so the simulated MID covers what the mapping
    CSV makes measurable.
    """
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)
    print(f"  {len(cobra_model.reactions)} reactions, "
          f"{len(cobra_model.metabolites)} metabolites")

    mapped_met_ids = set(pd.read_csv(os.path.join(data_dir, "features.csv"))
                           ["metabolite_id"].dropna())
    print(f"  {len(mapped_met_ids)} metabolite IDs from features.csv")

    print(f"Loading training data from: {data_dir}")
    return (cobra_model, mapped_met_ids, *load_data(data_dir))


def select_samples(split, split_source, n_total, cfg, start_ix, n_benchmarks):
    """Sample columns for this task, as an offset window into *split*.

    Restricting the benchmark to the AI model's split (default: the withheld
    validation set) is what makes MFA and AI comparable — both are scored on
    samples the AI never trained on. ``start_ix``/``n_benchmarks`` index the sorted
    split, not raw HDF5 columns, so a SLURM array tiles it deterministically.

    Returns ``(sel, tag)``; ``sel`` is empty when the offset window falls past the
    end of the split, and ``tag`` names the output files.
    """
    split_cols = split_sample_indices(
        split, split_source, n_total,
        cfg.get("training", {}).get("val_fraction"), cfg.get("seed"),
    )
    sel = split_cols[start_ix:start_ix + n_benchmarks]
    n_runs = len(sel)
    print(f"  split={split}: {len(split_cols)} samples; this task evaluates "
          f"offsets {start_ix}–{start_ix + n_runs - 1} → columns "
          f"{sel[0] if n_runs else '-'}..{sel[-1] if n_runs else '-'} of {n_total} total")
    return sel, f"{start_ix:05d}_{start_ix + max(n_runs - 1, 0):05d}"


def resolve_split_source(checkpoint, data_dir, split):
    """Checkpoint whose stored indices define the split, or ``None`` to recompute it."""
    if checkpoint is not None or split == "all":
        return checkpoint
    try:
        return find_checkpoint(data_dir)
    except FileNotFoundError:
        return None


def build_mfa_context(cobra_model, tracer_cfg, initial_flux, name, with_backward):
    """Build the prepared MFAModel and the index that flattens its output.

    ``with_backward`` is the expensive half of the EMU network: the MFA benchmark
    fits and needs gradients, the AI benchmark only runs forward passes.

    Returns ``(mfa_model, reaction_ids, experiments, labeling_to, labeling_index)``.
    """
    mfa_model = MFAModel(cobra_model, name, initial_flux_distribution=initial_flux)
    targets, measured_metabolites, atom_patterns = build_targets(mfa_model)
    experiments = build_experiments(mfa_model, tracer_cfg)
    print("Preparing EMU network (per-tracer experiments; may take several minutes) ...")
    mfa_model.prepare(experiments, targets=targets, with_backward=with_backward)
    print("  EMU network ready.")

    labeling_to, labeling_index = build_labeling_index(
        mfa_model, experiments, measured_metabolites, atom_patterns
    )
    return (mfa_model, [r.id for r in mfa_model.reactions], experiments,
            labeling_to, labeling_index)


def forward_labeling(mfa_model, fluxes, reversibility, labeling_to):
    """Simulated labeling vector for a flux state, in training-HDF5 row order.

    *fluxes* and *reversibility* are arrays already aligned to
    ``mfa_model.reactions``. The model is left holding that state.
    """
    dtype, device = mfa_model.dtype, mfa_model.device
    with torch.no_grad():
        mfa_model._fluxes.copy_(torch.tensor(np.asarray(fluxes), dtype=dtype, device=device))
        mfa_model._reversibility.copy_(
            torch.tensor(np.asarray(reversibility), dtype=dtype, device=device)
        )
        return torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()


def write_predictions(base, labeling_data, labeling_index, flux_data, reaction_ids,
                      sample_indices):
    """Write the predicted labeling and flux HDF5 pair.

    Columns are absolute HDF5 sample indices so every task's output pairs with
    ground truth independently; rewriting after each sample makes a killed job
    keep everything it had finished.
    """
    (pd.DataFrame(labeling_data, index=sample_indices, columns=labeling_index)
       .T.to_hdf(f"{base}_labeling.h5", key="labeling", mode="w"))
    (pd.DataFrame(flux_data, index=sample_indices, columns=reaction_ids)
       .T.to_hdf(f"{base}_flux.h5", key="fluxes", mode="w"))
