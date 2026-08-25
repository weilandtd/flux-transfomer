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

    BLAS pools (OpenMP, MKL) cannot be pinned from here — they read their thread
    env when first loaded on ``import torch``, before this function runs — so
    those vars must be exported by the submitting shell; warn if they aren't.
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


def load_benchmark_inputs(data_dir, cfg=None, root=None):
    """Load the run directory's model, observable metabolites and training data.

    Args:
        data_dir: run directory holding the prepared model and training HDF5 files.
        cfg: organism config; supplies ``mapping_csv`` when given.
        root: project root for resolving config-relative paths.

    Returns ``(cobra_model, mapped_met_ids, labeling_df, flux_df, rev_df)``.
    ``mapped_met_ids`` comes from the config's ``mapping_csv`` when given, else
    every metabolite id in the run's ``features.csv``.
    """
    model_path = os.path.join(data_dir, "model_prepared_for_mfa.json")
    print(f"Loading model: {model_path}")
    cobra_model = load_json_model(model_path)
    print(f"  {len(cobra_model.reactions)} reactions, "
          f"{len(cobra_model.metabolites)} metabolites")

    mapped_met_ids = None
    if cfg is not None:
        from .check_mfa_model import measurable_metabolites
        mapped_met_ids = measurable_metabolites(cfg, root)
    if mapped_met_ids is not None:
        print(f"  {len(mapped_met_ids)} metabolite IDs from mapping_csv")
    else:
        mapped_met_ids = set(pd.read_csv(os.path.join(data_dir, "features.csv"))
                               ["metabolite_id"].dropna())
        print(f"  {len(mapped_met_ids)} metabolite IDs from features.csv")

    print(f"Loading training data from: {data_dir}")
    return (cobra_model, mapped_met_ids, *load_data(data_dir))


def select_samples(split, split_source, n_total, cfg, start_ix, n_benchmarks,
                   subsample=None):
    """Sample columns for this task, as an offset window into *split*.

    Args:
        split: which split to draw from ("val", "train", or "all").
        split_source: checkpoint (or ``None``) whose stored indices define the split.
        n_total: total number of samples in the training data.
        cfg: organism config, for the val_fraction/seed fallback when recomputing the split.
        start_ix: offset of the first sample within the split.
        n_benchmarks: number of samples to select from that offset.
        subsample: when given, first draw this many indices out of the full split
            (seeded at 0, independent of any run's ``--seed``, so every condition of
            a sensitivity study draws the identical subset regardless of what noise
            seed it runs with), sorted back into split order, then windowed by
            ``start_ix``/``n_benchmarks`` as usual. ``None`` uses the whole split.

    ``start_ix``/``n_benchmarks`` index the sorted split (post-subsampling, when
    given), not raw HDF5 columns, so a SLURM array tiles it deterministically.
    Returns ``(sel, tag)``; ``sel`` is empty when the offset window falls past the
    end of the split.
    """
    split_cols = split_sample_indices(
        split, split_source, n_total,
        cfg.get("training", {}).get("val_fraction"), cfg.get("seed"),
    )
    if subsample is not None and subsample < len(split_cols):
        split_cols = np.sort(
            np.random.default_rng(0).choice(split_cols, size=subsample, replace=False)
        )
    sel = split_cols[start_ix:start_ix + n_benchmarks]
    n_runs = len(sel)
    print(f"  split={split}: {len(split_cols)} samples"
          f"{' (subsampled)' if subsample is not None else ''}; this task evaluates "
          f"offsets {start_ix}–{start_ix + n_runs - 1} → columns "
          f"{sel[0] if n_runs else '-'}..{sel[-1] if n_runs else '-'} of {n_total} total")
    return sel, f"{start_ix:05d}_{start_ix + max(n_runs - 1, 0):05d}"


def metabolite_subset(mapped_met_ids, frac, seed=0):
    """Nested random subset of *mapped_met_ids*, keeping a fixed fraction of it.

    One permutation of the sorted ids is drawn from *seed*; level ``frac`` keeps its
    first ``max(1, round(frac * n))`` entries (matching the floor
    ``paper/analysis/diagnostics/test_missing_metabolites.py`` already uses for its
    masking rate), so subsets at different fractions nest (frac=0.25 ⊂ frac=0.5 ⊂
    frac=0.75 ⊂ frac=1.0) rather than being independent draws — the reading is "this
    platform resolves these metabolites", and degradation is monotone in how much of
    that set is dropped. The floor keeps a positive ``frac`` from ever rounding to
    an empty set on a small metabolite count (e.g. a toy/test model): a restriction
    that quietly reverted to "no restriction" downstream would be a worse bug than a
    slightly generous smallest subset.

    Args:
        mapped_met_ids: full set of metabolite ids the measurement platform can see.
        frac: fraction of them to keep, in [0, 1]. ``frac <= 0`` keeps nothing.
        seed: RNG seed for the permutation. Fixed by default (0) so independent
            callers — the AI and MFA arms of a sensitivity study — drop the
            identical metabolites without having to share any other state.
    """
    ids = np.array(sorted(mapped_met_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_keep = max(1, round(frac * len(ids))) if frac > 0 else 0
    return set(ids[order[:n_keep]].tolist())


def resolve_split_source(checkpoint, data_dir, split):
    """Checkpoint whose stored indices define the split, or ``None`` to recompute it.

    Args:
        checkpoint: explicit checkpoint path, used as-is when given.
        data_dir: run directory to search for an auto-detected checkpoint.
        split: split name; "all" always recomputes rather than using a checkpoint.
    """
    if checkpoint is not None or split == "all":
        return checkpoint
    try:
        return find_checkpoint(data_dir)
    except FileNotFoundError:
        return None


def build_mfa_context(cobra_model, tracer_cfg, initial_flux, name, with_backward,
                      restrict_targets_to=None):
    """Build the prepared MFAModel and the index that flattens its output.

    Args:
        cobra_model: the COBRA model to wrap.
        tracer_cfg: tracer definitions, as in the organism config's ``labeling.tracers``.
        initial_flux: flux vector the model is initialized at before preparation.
        name: name given to the constructed MFAModel.
        with_backward: enables gradients, needed for fitting but not forward-only inference.
        restrict_targets_to: optional metabolite ids the expansion targets are limited to.

    Returns ``(mfa_model, reaction_ids, experiments, labeling_to, labeling_index)``.
    """
    mfa_model = MFAModel(cobra_model, name, initial_flux_distribution=initial_flux)
    targets, measured_metabolites, atom_patterns = build_targets(
        mfa_model, restrict_to=restrict_targets_to)
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

    Args:
        mfa_model: model the flux state is written into and simulated with.
        fluxes: flux values, aligned to ``mfa_model.reactions``.
        reversibility: reversibility values, aligned to ``mfa_model.reactions``.
        labeling_to: index tensor gathering the flattened output into HDF5 row order.

    The model is left holding that state.
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
    """Per-sample RNG, so a sample's noise stream doesn't depend on run order or neighbours.

    Args:
        seed: base seed for the benchmark run.
        sample_ix: absolute HDF5 sample index the RNG stream belongs to.
    """
    return np.random.default_rng([int(seed), int(sample_ix)])


def add_labeling_noise(gt_labeling, noise_std, rng):
    """Noisy MIDs that still sum to one, per (tracer, metabolite) group.

    Reconstructs the implicit M+0, perturbs it and the stored fractions with Gaussian
    noise, then renormalises by the group total so the result stays a valid MID.
    Negatives are clamped before dividing, not after (unlike
    :func:`sigmaflux.ai.train.apply_labeling_noise`), so a true M+0 near zero can't
    push the group sum above one.

    Args:
        gt_labeling: ground-truth MIDs, indexed ``(Tracer, Metabolite, Isotopomer)``,
            holding M+1..M+n only.
        noise_std: Gaussian noise std applied to the fractions.
        rng: random generator to draw the noise from.
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
    """E[χ²]/n for the labeling block, in units of the declared σ.

    :func:`add_labeling_noise`'s renormalisation correlates the perturbations and
    shrinks the residual below ``noise_std``, so a χ² test using σ at face value
    would accept nearly every fit; this measures the deflation empirically by
    re-noising the *measured* MIDs (M+0 excluded, since its variance scales with
    group size).

    Args:
        measured: noisy MIDs the deflation is measured against.
        noise_std: Gaussian noise std used to re-noise the measured MIDs.
        rng: random generator to draw the re-noising from.
        n_draws: number of re-noising draws averaged over.

    Returns the variance scale ``s`` such that ``SSR/s`` follows χ²(dof).
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

    Args:
        true_values: ground-truth flux values to perturb.
        flux_noise_std: relative Gaussian noise std.
        rng: random generator to draw the noise from.
    """
    values = np.asarray(true_values, dtype=np.float64)
    if flux_noise_std <= 0:
        return values.copy()
    return values * (1.0 + rng.normal(0.0, flux_noise_std, size=values.size))


def noisy_measurements(gt_labeling, gt_fluxes, noise_std, flux_noise_std, seed, sample_ix):
    """Both measurement blocks for one sample, drawn in a fixed order.

    Args:
        gt_labeling: ground-truth MIDs to add labeling noise to.
        gt_fluxes: ground-truth measured flux values to add flux noise to.
        noise_std: Gaussian noise std for the labeling block.
        flux_noise_std: relative Gaussian noise std for the flux block.
        seed: base seed for the RNG stream.
        sample_ix: absolute HDF5 sample index the RNG stream belongs to.

    Order matters: it fixes the RNG stream so the AI and MFA benchmarks see the
    identical noise realisation for a given ``(seed, sample_ix)``.
    """
    rng = benchmark_rng(seed, sample_ix)
    return (add_labeling_noise(gt_labeling, noise_std, rng),
            add_flux_noise(gt_fluxes, flux_noise_std, rng))


def write_predictions(base, labeling_data, labeling_index, flux_data, reaction_ids,
                      sample_indices):
    """Write the predicted labeling and flux HDF5 pair.

    Args:
        base: output path prefix; writes ``{base}_labeling.h5`` and ``{base}_flux.h5``.
        labeling_data: rows of predicted MID vectors, one per (sample[, member]).
        labeling_index: column MultiIndex for the labeling HDF5.
        flux_data: rows of predicted flux vectors, one per (sample[, member]).
        reaction_ids: column labels for the flux HDF5.
        sample_indices: row labels, one per row of *_data. Either plain absolute
            HDF5 sample indices (one prediction per sample — MFA, or a single
            checkpoint), or ``(sample_ix, member_ix)`` pairs (every ensemble
            member's own raw prediction for that sample, kept separate rather than
            combined here — see :mod:`run_ai_benchmark`), in which case the written
            columns are a ``(sample, member)`` MultiIndex.

    Rewritten after each checkpoint so a killed job keeps whatever it finished.
    """
    index = (pd.MultiIndex.from_tuples(sample_indices, names=["sample", "member"])
             if sample_indices and isinstance(sample_indices[0], tuple)
             else pd.Index(sample_indices))
    (pd.DataFrame(labeling_data, index=index, columns=labeling_index)
       .T.to_hdf(f"{base}_labeling.h5", key="labeling", mode="w"))
    (pd.DataFrame(flux_data, index=index, columns=reaction_ids)
       .T.to_hdf(f"{base}_flux.h5", key="fluxes", mode="w"))
