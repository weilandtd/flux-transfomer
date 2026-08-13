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
import warnings

import numpy as np
import pandas as pd
import torch
from cobra.io.json import load_json_model

from ..ai.data import find_checkpoint, load_data, split_sample_indices
from ..core.model import MFAModel
from .generate_training_data import build_experiments, build_labeling_index, build_targets


def pin_threads():
    """Pin the torch thread pool to the cores SLURM actually granted.

    Without this a benchmark task oversubscribes its allocation: torch defaults to
    every core on the node, not the ``--cpus-per-task`` it may use.

    The BLAS pools cannot be pinned from here. OpenMP and MKL read their environment
    when the shared library is first loaded, which happens on ``import torch`` — long
    before this function runs — so assigning to ``os.environ`` now has no effect on
    them. They must be exported by the submitting shell. Left unset, MKL takes one
    thread per core (192 on della) inside a 6-core allocation; on a contended node the
    resulting thrash cost ~10x, with one 30-iteration MFA fit measured at 25 s on one
    node and 258 s on another of the same type. That is silent and easily mistaken for
    a slow algorithm, so warn rather than let it pass.
    """
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    unpinned = [v for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                if v not in os.environ]
    if unpinned and n_threads > 1:
        warnings.warn(
            f"{', '.join(unpinned)} not set — BLAS may use every core on the node "
            f"rather than the {n_threads} allocated, which can cost an order of "
            f"magnitude in wall clock. Export them in the submitting shell; setting "
            f"them here would be too late to take effect.",
            RuntimeWarning, stacklevel=2,
        )
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


# ── Synthetic measurement noise ───────────────────────────────────────────────
#
# One implementation, used by both benchmarks, so "the AI and MFA saw the same data"
# is true by construction rather than by two code paths happening to agree. It
# reproduces the noise the model is *trained* under
# (:func:`sigmaflux.ai.train.apply_labeling_noise`); anything else evaluates the AI
# off-distribution and — worse — hands MFA data no flux state can reproduce.


def benchmark_rng(seed, sample_ix):
    """Per-sample generator, so a sample's noise never depends on its neighbours.

    Seeding per task instead would make the stream position depend on how many
    multi-start restarts earlier samples consumed, which is optimiser- and
    BLAS-thread-dependent — that is, not reproducible even by re-running.
    """
    return np.random.default_rng([int(seed), int(sample_ix)])


def add_labeling_noise(gt_labeling, noise_std, rng):
    """Noisy MIDs that still sum to one, per (tracer, metabolite) group.

    Mirrors :func:`sigmaflux.ai.train.apply_labeling_noise`: reconstruct the implicit
    M+0, perturb the stored fractions *and* M+0 with unclipped Gaussian noise, then
    renormalise by the group total.

    The renormalisation is the point. An EMU simulation always returns MIDs summing
    to one, so a measurement vector that does not is unreachable by every feasible
    flux — an irreducible residual that inflates chi-square and reads as an optimiser
    failing to converge when the data was simply not fittable. *gt_labeling* is
    indexed ``(Tracer, Metabolite, Isotopomer)`` and holds M+1..M+n only; M+0 stays
    implicit, exactly as the stored training data does.

    One deliberate departure from the training implementation: negatives are clamped
    *before* the division, not after. Training clamps the normalised fractions and
    never clamps M+0, so wherever the true M+0 sits near zero its noise can drive it
    negative and the stored fractions then sum above one. Measured on real T-cell
    data that leaves ~77 of 456 groups unfittable and ~400 of chi-square permanently
    unreachable; clamping first drives both to zero. Training arguably wants the same
    fix, but changing it would retrain every model, so it is not touched here.
    """
    if noise_std <= 0:
        return gt_labeling.copy()

    stored = gt_labeling.to_numpy(dtype=np.float64)
    group = pd.MultiIndex.from_arrays([gt_labeling.index.get_level_values("Tracer"),
                                       gt_labeling.index.get_level_values("Metabolite")])
    codes, _ = pd.factorize(group, sort=False)
    n_groups = int(codes.max()) + 1 if codes.size else 0

    m0 = 1.0 - np.bincount(codes, weights=stored, minlength=n_groups)
    # Draw order matches the training implementation: stored fractions, then M+0.
    noisy_stored = np.clip(stored + rng.normal(0.0, noise_std, size=stored.size), 0.0, None)
    noisy_m0 = np.clip(m0 + rng.normal(0.0, noise_std, size=n_groups), 0.0, None)

    total = noisy_m0 + np.bincount(codes, weights=noisy_stored, minlength=n_groups)
    total = np.clip(total, 1e-10, None)
    return pd.Series(noisy_stored / total[codes], index=gt_labeling.index)


def labeling_chi2_scale(measured, noise_std, rng, n_draws=16):
    """E[χ²]/n for the labeling block, in units of the *declared* σ.

    The χ² goodness-of-fit test assumes the residual sd equals the σ the objective
    divides by. Here it does not: :func:`add_labeling_noise` renormalises each group
    to sum to one, which correlates the perturbations and shrinks the residual well
    below ``noise_std``. Taking σ at face value therefore makes the test accept
    essentially every fit, which is why an earlier reduced-χ² bound was abandoned as
    uninformative.

    This measures the deflation instead of assuming it away, by re-noising the
    *measured* MIDs and reading off the mean squared standardised residual. It needs
    no ground truth: the shrinkage is a property of the MID values and the group
    structure, not of whether those values are the true ones — the same thing a lab
    does when it characterises its instrument rather than trusting the spec sheet.

    Only the stored fractions are scored, matching the objective. M+0 is deliberately
    excluded: its residual is a sum of the whole group's perturbations, so its variance
    grows with group size and averaging it in here would report ~1.05 for a block whose
    stored elements sit near 0.75.

    Returns the variance scale ``s``, so ``SSR/s`` is the statistic that follows
    χ²(dof) and ``s·χ²_{1-α}(dof)`` is the bound to compare a raw SSR against.

    Each draw already averages over every MID element, so a handful of draws pins ``s``
    to well under a percent and the estimate costs no simulation at all. Re-noising an
    already-noisy vector biases ``s`` up a few percent — the base is clipped at zero and
    most MIDs sit near it — which loosens the bound slightly rather than tightening it.
    """
    if noise_std <= 0:
        return 1.0
    stored = measured.to_numpy(dtype=np.float64)
    total = 0.0
    for _ in range(n_draws):
        draw = add_labeling_noise(measured, noise_std, rng).to_numpy(dtype=np.float64)
        total += ((draw - stored) ** 2).sum()
    return float(total / (n_draws * stored.size * noise_std ** 2))


def add_flux_noise(true_values, flux_noise_std, rng):
    """Relative flux noise, ``v·(1 + N(0, σ))`` — the training form.

    Unclipped on purpose: the AI is trained against this, so bounding the error here
    would test it on a gentler distribution than it learned.
    """
    values = np.asarray(true_values, dtype=np.float64)
    if flux_noise_std <= 0:
        return values.copy()
    return values * (1.0 + rng.normal(0.0, flux_noise_std, size=values.size))


def noisy_measurements(gt_labeling, gt_fluxes, noise_std, flux_noise_std, seed, sample_ix):
    """Both measurement blocks for one sample, drawn in a fixed order.

    Callers must not draw these separately: the order fixes the stream, and it is
    what makes the AI and MFA benchmarks see the identical realisation for a given
    ``(seed, sample_ix)``.
    """
    rng = benchmark_rng(seed, sample_ix)
    return (add_labeling_noise(gt_labeling, noise_std, rng),
            add_flux_noise(gt_fluxes, flux_noise_std, rng))


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
