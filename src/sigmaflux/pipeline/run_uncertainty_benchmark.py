"""
Uncertainty benchmark: missing-metabolite and measurement-noise sensitivity.

Runs the AI and MFA accuracy benchmarks (``sigmaflux-benchmark`` /
``sigmaflux-benchmark-mfa``) over a small, fixed grid of measurement scenarios --
metabolite coverage and noise level -- instead of the single default scenario each
of those drivers scores. This module never re-derives a prediction: a task
dispatches straight into the sibling driver's own ``cli()``, so "the same benchmark"
stays defined in exactly one place (:mod:`sigmaflux.pipeline._benchmark`).

Two things are fixed once and shared by every cell, so the grid answers "how does
accuracy degrade" rather than "did we get a different random draw":

  the sample pool  -- NOT an arbitrary subsample of the validation split. Drawn only
                       from samples MFA demonstrably solved at baseline
                       (:func:`resolve_solved_sample_pool`, default criterion
                       ``ssr_ratio <= 1`` against the existing production
                       ``<data-dir>/mfa_benchmark/`` run), then persisted to
                       ``<output-dir>/samples.json`` on first use so every condition,
                       every method, and every rerun of this module shares the exact
                       same 100 ids -- the pairing a paired test across conditions or
                       methods needs.
  metabolite subsets -- nested and seed-fixed per replicate (see
                       :func:`sigmaflux.pipeline._benchmark.metabolite_subset`): the
                       75% cut is a subset of 100%, 50% a subset of 75%, matching
                       "this measurement platform resolves these metabolites" rather
                       than an independent draw per condition. Each of the 20
                       replicates (``--n-met-reps``) is its OWN independent
                       permutation (seed = replicate index), so the missing-data
                       curve is a mean over draws, not the luck of one.

Conditions (built by :func:`_build_conditions`) are a deliberately small, hardcoded
grid -- this is a sensitivity study, not a benchmark parameter a user configuration
should extend -- so it is not read from the organism YAML:

  missing data : keep_met_frac in {1.00, 0.75, 0.50, 0.25} at the default noise;
                 1.00 needs no replicate (keeping everything is deterministic), the
                 other three each get ``--n-met-reps`` independent draws
  noise        : one-at-a-time cross around the default noise --
                 labeling_std in {0.001, 0.01, 0.02, 0.05} at flux_std=0.05,
                 flux_std in {0.001, 0.05, 0.10, 0.25} at labeling_std=0.01

Neither axis has a literal-0 cell. Both noise models weight the fit residual by
1/sigma, so sigma=0 is undefined, not merely awkward -- and the obvious stand-in
(a tiny positive sigma) does not rescue the condition, it just moves the failure from
a division-by-zero to a precision stall the optimiser cannot clear (verified: at
sigma=1e-6 MFA's SSR-at-the-fit was 10^6-10^7 times the SSR at the true flux, on the
two largest networks here -- an optimiser failure, not a property of MFA, but a real
one). 0.1% is comfortably above where that stall bites on either axis; see
:mod:`sigmaflux.pipeline.run_mfa_benchmark`'s ``FLUX_SIGMA_FLOOR`` for the matching
fix to the flux axis's own near-zero-measurement edge case.

The two studies share the ``met100_rep00_lab1_flux5`` baseline cell, run once —
67 unique cells in total.

Each (condition, method) cell writes into its own subdirectory of ``--output-dir``
(default ``<data-dir>/uncertainty_benchmark``):

    <output-dir>/samples.json
    <output-dir>/<condition>/ai/pred_<start>_<end>_{flux,labeling}.h5
    <output-dir>/<condition>/mfa/pred_<start>_<end>_{flux,labeling}.h5
                                  pred_<start>_<end>_convergence.csv

-- matching the two arms' own default subdirectory names (``ai_benchmark/``,
``mfa_benchmark/``) so evaluation code can glob each the same way
``plot_test_performance.py`` globs those.

Parallelism: one task = one (condition, method, sample-window) cell, tiled with
``--per-task`` exactly like the underlying drivers' own ``--start-ix``/
``--n-benchmarks`` (indeed that is literally how it is implemented -- see
:func:`tasks`). ``--task-id`` runs exactly one task, for a SLURM array; ``--list``
prints the grid without running anything or requiring a baseline MFA run to exist.

Usage:
    # print the task grid, run nothing
    sigmaflux-benchmark-uncertainty --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260822 --list

    # run one task (SLURM array: --task-id $SLURM_ARRAY_TASK_ID) -- needs
    # <data-dir>/mfa_benchmark/pred_*_convergence.csv from a prior full MFA run
    sigmaflux-benchmark-uncertainty --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260822 \\
        --backend x3cflux --per-task 10 --task-id 0
"""

import argparse
import dataclasses
import glob
import json
import os

import numpy as np
import pandas as pd

from ._paths import add_root_argument, load_config, resolve_path


@dataclasses.dataclass(frozen=True)
class Condition:
    """One cell of the sensitivity grid: a name plus the four knobs that vary."""
    name: str
    keep_met_frac: float
    keep_met_seed: int   # which replicate's metabolite draw; irrelevant at frac=1.0
    labeling_std: float
    flux_std: float
    studies: tuple   # which sensitivity stud(y/ies) this cell belongs to


DEFAULT_LAB_STD = 0.01
DEFAULT_FLUX_STD = 0.05
LAB_LEVELS = (0.001, 0.01, 0.02, 0.05)     # 0.1% / 1% / 2% / 5%, absolute
FLUX_LEVELS = (0.001, 0.05, 0.10, 0.25)    # 0.1% / 5% / 10% / 25%, relative
MISSING_FRACS = (0.75, 0.50, 0.25)
N_MET_REPS = 20
METHODS = ("ai", "mfa")


def _pct(x):
    """0.001 -> '0p1', 0.01 -> '1', 0.25 -> '25' -- percentage, 'p' stands for '.'.

    v1's zero-padded scheme (lab001 = 1%) cannot express 0.1% without colliding
    with 1%, hence this replacement.
    """
    return f"{x * 100:g}".replace(".", "p")


def _build_conditions(n_met_reps=N_MET_REPS):
    """The full sensitivity grid: 1 + 3*n_met_reps missing-data cells, 6 noise cells,
    plus the one baseline cell shared by both studies. Not config-driven -- this is a
    sensitivity study's own fixed design, not a benchmark parameter a user
    organism YAML should extend."""
    baseline_name = f"met100_rep00_lab{_pct(DEFAULT_LAB_STD)}_flux{_pct(DEFAULT_FLUX_STD)}"
    conditions = [Condition(baseline_name, 1.00, 0, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
                            ("missing", "noise"))]

    for frac in MISSING_FRACS:
        met_tag = f"met{round(frac * 100)}"
        for rep in range(n_met_reps):
            name = f"{met_tag}_rep{rep:02d}_lab{_pct(DEFAULT_LAB_STD)}_flux{_pct(DEFAULT_FLUX_STD)}"
            conditions.append(Condition(name, frac, rep, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
                                        ("missing",)))

    for lab in LAB_LEVELS:
        if lab == DEFAULT_LAB_STD:
            continue   # the shared baseline cell above already covers this level
        name = f"met100_rep00_lab{_pct(lab)}_flux{_pct(DEFAULT_FLUX_STD)}"
        conditions.append(Condition(name, 1.00, 0, lab, DEFAULT_FLUX_STD, ("noise",)))

    for flux in FLUX_LEVELS:
        if flux == DEFAULT_FLUX_STD:
            continue
        name = f"met100_rep00_lab{_pct(DEFAULT_LAB_STD)}_flux{_pct(flux)}"
        conditions.append(Condition(name, 1.00, 0, DEFAULT_LAB_STD, flux, ("noise",)))

    return tuple(conditions)


#: The default-replicate-count grid, built once at import time -- what --list shows
#: without needing --n-met-reps, and what the test suite imports directly. Runtime
#: dispatch goes through _build_conditions(args.n_met_reps) instead, so the flag
#: actually changes the grid when passed.
CONDITIONS = _build_conditions()


def tasks(conditions, methods, n_samples, per_task):
    """Flat, deterministic ``(condition, method, start_ix, n_this_task)`` task list.

    ``start_ix``/``n_this_task`` tile ``[0, n_samples)`` in windows of at most
    ``per_task`` -- exactly the ``--start-ix``/``--n-benchmarks`` contract
    :func:`sigmaflux.pipeline._benchmark.select_samples` already implements, applied
    on top of the fixed ``n_samples``-long solved-sample pool. There is no new tiling
    logic here: a task literally becomes ``--samples <path> --start-ix start_ix
    --n-benchmarks n_this_task`` on the sibling driver, so array resizing (a
    different ``--per-task``) can never change what a sample sees, the same
    guarantee the underlying drivers already give ``--start-ix``.

    Args:
        conditions: grid cells to include (a subset of ``_build_conditions()``).
        methods: predictors to include (a subset of ``METHODS``).
        n_samples: size of the fixed solved-sample pool every condition scores.
        per_task: maximum samples per task; smaller values buy more, cheaper-per-task
            parallelism (e.g. for x3cflux's slowest organisms).
    """
    per_task = per_task or n_samples
    out = []
    for c in conditions:
        for m in methods:
            for start_ix in range(0, n_samples, per_task):
                out.append((c, m, start_ix, min(per_task, n_samples - start_ix)))
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="sigmaflux-benchmark-uncertainty",
        description="Missing-data / noise sensitivity benchmark, both methods",
    )
    p.add_argument("--config", required=True, help="Organism YAML config")
    p.add_argument("--data-dir", required=True,
                   help="Training run directory (as for sigmaflux-benchmark[-mfa])")
    p.add_argument("--output-dir", default=None,
                   help="Results directory (default: <data-dir>/uncertainty_benchmark)")
    p.add_argument("--n-samples", type=int, default=100,
                   help="Size of the solved-sample pool every condition scores "
                        "(default: 100) -- see --solved-criterion/--baseline-mfa-dir")
    p.add_argument("--n-met-reps", type=int, default=N_MET_REPS,
                   help=f"Independent metabolite-subset draws per missing-data "
                        f"fraction (default: {N_MET_REPS})")
    p.add_argument("--solved-criterion", choices=["ssr_ratio", "reached_optimum"],
                   default="ssr_ratio",
                   help="Which column of the baseline convergence CSVs decides "
                        "'MFA solved this sample'. 'ssr_ratio' (default) keeps SSR "
                        "at the fit <= SSR at the true flux -- the fit did not stop "
                        "short of an optimum the truth itself achieves; looser than "
                        "'reached_optimum' (passed the chi2 goodness-of-fit test), "
                        "which for some organisms leaves too few samples.")
    p.add_argument("--baseline-mfa-dir", default=None,
                   help="Directory holding pred_*_convergence.csv from a prior full "
                        "MFA benchmark run, the source of the solved-sample pool "
                        "(default: <data-dir>/mfa_benchmark)")
    p.add_argument("--per-task", type=int, default=None,
                   help="Max samples per task, tiling --n-samples for a SLURM array "
                        "(default: all of --n-samples in one task). Set lower for "
                        "the mfa method on slow-fitting organisms.")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--methods", default="ai,mfa",
                   help="Comma-separated subset of {ai, mfa} to run (default: both)")
    p.add_argument("--studies", default="missing,noise",
                   help="Comma-separated subset of {missing, noise} conditions to "
                        "run (default: both; the shared baseline cell runs once "
                        "even when both studies are selected)")
    p.add_argument("--task-id", type=int, default=None,
                   help="Run exactly this task (0-indexed into the grid restricted "
                        "by --methods/--studies/--per-task) instead of every task "
                        "-- the SLURM array unit. Omit to run the whole grid "
                        "sequentially (fine for a toy/smoke run, not for production).")
    p.add_argument("--list", action="store_true",
                   help="Print the task grid (index, condition, method, sample "
                        "window) and exit; runs nothing.")
    p.add_argument("--seed", type=int, default=42,
                   help="Noise RNG seed, forwarded to the underlying benchmark. The "
                        "solved-sample pool and the metabolite subsets are seeded "
                        "independently (0, and the replicate index respectively), so "
                        "neither ever moves with this.")
    # Passthroughs to sigmaflux-benchmark-mfa: a 100-sample multi-start fit needs
    # the same tuning knobs the plain MFA benchmark exposes. Ignored for --methods ai.
    p.add_argument("--backend", choices=["native", "native-ipopt", "x3cflux"],
                   default="x3cflux", help="MFA backend (see sigmaflux-benchmark-mfa)")
    p.add_argument("--max-starts", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-3)
    add_root_argument(p)
    return p


def _selected_grid(args):
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    studies = [s.strip() for s in args.studies.split(",") if s.strip()]
    unknown_m = sorted(set(methods) - set(METHODS))
    unknown_s = sorted(set(studies) - {"missing", "noise"})
    if unknown_m:
        raise SystemExit(f"Unknown --methods value(s): {unknown_m}. Known: {METHODS}")
    if unknown_s:
        raise SystemExit(f"Unknown --studies value(s): {unknown_s}. Known: missing, noise")
    grid_conditions = _build_conditions(args.n_met_reps)
    conditions = [c for c in grid_conditions if set(c.studies) & set(studies)]
    return tasks(conditions, methods, args.n_samples, args.per_task)


def resolve_solved_sample_pool(baseline_mfa_dir, criterion, n_samples, seed=0):
    """Absolute HDF5 sample ids MFA solved at baseline, downsampled to *n_samples*.

    Every condition in this study restricts to samples MFA could actually fit at the
    default noise level -- otherwise the fits it never solved (a large majority for
    some organisms; see the module docstring) dominate a cell's spread with blowups
    rather than genuine sensitivity to the condition being varied.

    Args:
        baseline_mfa_dir: directory holding ``pred_*_convergence.csv`` from a prior,
            unrestricted MFA benchmark run over the full validation split.
        criterion: ``"ssr_ratio"`` keeps SSR at the fit <= SSR at the true flux (the
            true flux is a feasible point of the same optimisation, so a fit scoring
            worse than it demonstrably did not find the minimum -- see
            ``run_mfa_benchmark.py``'s own comment on this). ``"reached_optimum"``
            keeps only fits that passed the stricter chi2 goodness-of-fit test.
        n_samples: pool size to draw.
        seed: RNG seed for the draw, independent of any run's noise ``--seed`` (same
            convention as the old ``--subsample``) so the pool never moves with it.
    """
    files = sorted(glob.glob(os.path.join(baseline_mfa_dir, "pred_*_convergence.csv")))
    if not files:
        raise SystemExit(
            f"No pred_*_convergence.csv in {baseline_mfa_dir} -- run "
            f"sigmaflux-benchmark-mfa over the full split first; the solved-sample "
            f"pool is drawn from its output, never recomputed here.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("sample")
    if criterion == "ssr_ratio":
        eligible = df.loc[df["ssr_ratio"] <= 1.0, "sample"]
    else:
        eligible = df.loc[df["reached_optimum"].astype(bool), "sample"]
    eligible = np.sort(eligible.astype(int).unique())
    if len(eligible) < n_samples:
        raise SystemExit(
            f"Only {len(eligible)} samples pass --solved-criterion={criterion} in "
            f"{baseline_mfa_dir}, need --n-samples={n_samples}. Lower --n-samples or "
            f"pick a looser --solved-criterion.")
    chosen = np.random.default_rng(seed).choice(eligible, size=n_samples, replace=False)
    return sorted(int(i) for i in chosen)


def _resolve_sample_pool_path(args, output_dir, root):
    """Path to samples.json: resolved and written on first use, reused after that.

    Persisting rather than re-resolving means the pool cannot drift between
    conditions, methods, or a later rerun of this module -- it is read off disk once
    and every dispatched task is handed the same file.
    """
    path = os.path.join(output_dir, "samples.json")
    if os.path.exists(path):
        return path
    baseline_dir = (resolve_path(args.baseline_mfa_dir, root) if args.baseline_mfa_dir
                    else os.path.join(args.data_dir, "mfa_benchmark"))
    ids = resolve_solved_sample_pool(baseline_dir, args.solved_criterion, args.n_samples)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(ids, fh)
    print(f"Resolved {len(ids)}-sample solved pool from {baseline_dir} "
          f"(criterion={args.solved_criterion}) -> {path}")
    return path


def _resolve_ai_checkpoint(data_dir):
    """Best top-level ``.pt``, else the ``ensemble/`` directory, else ``None``.

    Mirrors ``submit_ai_benchmark.sh``'s own ``FIND_CHECKPOINT`` glob order.
    ``sigmaflux.ai.data.find_checkpoint`` (``run_ai_benchmark.py``'s own default
    when ``--checkpoint`` is omitted) only looks at the top level and raises when a
    run has ensemble members but no top-level ``.pt`` -- true for every current
    ``paper/.results/*_20260822`` run -- so this module resolves it explicitly
    rather than relying on that fallback. ``load_predictor`` (what the AI arm
    actually calls) already knows how to load a directory of ``model_*.pt``.
    """
    top = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if top:
        return top[0]
    ensemble = os.path.join(data_dir, "ensemble")
    if glob.glob(os.path.join(ensemble, "model_*.pt")):
        return ensemble
    return None   # nothing found; let run_ai_benchmark's own error explain why


def _run_task(condition, method, start_ix, n_this_task, args, output_dir, samples_path):
    """Build argv for the sibling driver and call its ``cli()`` -- never reimplemented."""
    sub_out = os.path.join(output_dir, condition.name, method)
    common = [
        "--config", args.config,
        "--data-dir", args.data_dir,
        "--output-dir", sub_out,
        "--split", args.split,
        "--samples", samples_path,
        "--start-ix", str(start_ix),
        "--n-benchmarks", str(n_this_task),
        "--seed", str(args.seed),
        "--noise-std", str(condition.labeling_std),
        "--flux-noise-std", str(condition.flux_std),
    ]
    if condition.keep_met_frac < 1.0:
        common += ["--keep-met-frac", str(condition.keep_met_frac),
                  "--keep-met-seed", str(condition.keep_met_seed)]
    if args.root:
        common += ["--root", args.root]

    if method == "ai":
        from .run_ai_benchmark import cli as ai_cli
        argv = list(common)
        checkpoint = _resolve_ai_checkpoint(args.data_dir)
        if checkpoint is not None:
            argv += ["--checkpoint", checkpoint]
        return ai_cli(argv)

    from .run_mfa_benchmark import cli as mfa_cli
    argv = common + [
        "--backend", args.backend,
        "--max-starts", str(args.max_starts),
        "--max-iter", str(args.max_iter),
        "--tol", str(args.tol),
    ]
    return mfa_cli(argv)


def main(args):
    cfg, root = load_config(args.config, args.root)
    data_dir = resolve_path(args.data_dir, root)
    output_dir = resolve_path(args.output_dir, root) if args.output_dir else \
        os.path.join(data_dir, "uncertainty_benchmark")
    args.data_dir = data_dir   # so every dispatched task sees the same resolved path

    grid = _selected_grid(args)
    if args.list:
        for i, (c, m, start_ix, n) in enumerate(grid):
            print(f"{i:3d}  {c.name:30s}  {m:3s}  samples[{start_ix}:{start_ix + n}]  "
                  f"met={c.keep_met_frac:.0%} rep={c.keep_met_seed} "
                  f"lab={c.labeling_std} flux={c.flux_std}")
        print(f"{len(grid)} tasks, {output_dir}")
        return 0

    if args.task_id is not None:
        if not 0 <= args.task_id < len(grid):
            raise SystemExit(f"--task-id {args.task_id} out of range [0, {len(grid)})")
        todo = [grid[args.task_id]]
    else:
        todo = grid

    # Resolved lazily -- --list must work with no baseline MFA run on disk yet, and
    # a toy/test invocation with an empty todo should never require one either.
    samples_path = _resolve_sample_pool_path(args, output_dir, root) if todo else None

    for condition, method, start_ix, n_this_task in todo:
        print(f"\n=== {condition.name} / {method}  samples[{start_ix}:{start_ix + n_this_task}]"
              f"  (met={condition.keep_met_frac:.0%} rep={condition.keep_met_seed} "
              f"lab={condition.labeling_std} flux={condition.flux_std}) ===")
        rc = _run_task(condition, method, start_ix, n_this_task, args, output_dir, samples_path)
        if rc:
            return rc
    return 0


def cli(argv=None):
    args = build_parser().parse_args(argv)
    # See mfa_entry.py's module docstring for why: cyipopt/x3cflux must be
    # pre-imported before torch when the mfa method is selected with a backend that
    # needs them (CXXABI_1.3.15 from the conda libstdc++ vs the system one torch
    # pulls in). This module dispatches into run_mfa_benchmark.cli() itself, so it
    # inherits that constraint whenever "mfa" is among --methods.
    methods = [m.strip() for m in args.methods.split(",")]
    if "mfa" in methods and args.backend == "x3cflux":
        from ..x3cflux import preimport, require
        if not preimport():
            require()          # not installed: raise with the install instructions
    elif "mfa" in methods and args.backend == "native-ipopt":
        try:
            import cyipopt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "the native-ipopt backend needs the optional cyipopt dependency, "
                "which pip cannot build (it needs Ipopt headers at build time): "
                "conda env create -f environment.yml"
            ) from exc
    return main(args)


if __name__ == "__main__":
    raise SystemExit(cli())
