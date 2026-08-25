"""
Uncertainty benchmark: missing-metabolite and measurement-noise sensitivity.

Runs the AI and MFA accuracy benchmarks (``sigmaflux-benchmark`` /
``sigmaflux-benchmark-mfa``) over a small, fixed grid of measurement scenarios --
metabolite coverage and noise level -- instead of the single default scenario each
of those drivers scores. This module never re-derives a prediction: a task
dispatches straight into the sibling driver's own ``cli()``, so "the same benchmark"
stays defined in exactly one place (:mod:`sigmaflux.pipeline._benchmark`).

Every condition scores the identical 100-sample (default) subset of the validation
split -- drawn once, seeded independently of ``--seed`` (see
:func:`sigmaflux.pipeline._benchmark.select_samples`'s ``subsample`` argument) -- so
the grid answers "how does accuracy degrade" rather than "did we get a different
random sample". Metabolite subsets are likewise nested and seed-fixed (see
:func:`sigmaflux.pipeline._benchmark.metabolite_subset`): the 75% cut is a subset of
100%, 50% a subset of 75%, and so on, matching "this measurement platform resolves
these metabolites" rather than an independent draw per condition.

Conditions (``CONDITIONS`` below) are a deliberately small, hardcoded grid -- this is
a sensitivity study, not a benchmark parameter a user configuration should extend --
so it is not read from the organism YAML:

  missing data : keep_met_frac in {1.00, 0.75, 0.50, 0.25} at the default noise
  noise        : one-at-a-time cross around the default noise --
                 labeling_std in {1e-6, 0.01, 0.05} at flux_std=0.05,
                 flux_std in {1e-6, 0.05, 0.25} at labeling_std=0.01
                 (1e-6 stands in for "0" -- both noise models weight residuals by
                 1/sigma, so a literal 0 divides by zero; 1e-6 is a MID fraction of
                 1e-4% / a flux of 1e-4% relative -- numerically noiseless, well-posed)

The two studies share the ``met100_lab001_flux005`` baseline cell, run once.

Each (condition, method) cell writes into its own subdirectory of ``--output-dir``
(default ``<data-dir>/uncertainty_benchmark``):

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
prints the grid without running anything.

Usage:
    # print the task grid, run nothing
    sigmaflux-benchmark-uncertainty --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260822 --list

    # run one task (SLURM array: --task-id $SLURM_ARRAY_TASK_ID)
    sigmaflux-benchmark-uncertainty --config paper/configs/i_orientalis.yaml \\
        --data-dir paper/.results/i_orientalis_20260822 \\
        --backend x3cflux --per-task 10 --task-id 0
"""

import argparse
import dataclasses
import os

from ._paths import add_root_argument, load_config, resolve_path


@dataclasses.dataclass(frozen=True)
class Condition:
    """One cell of the sensitivity grid: a name plus the three knobs that vary."""
    name: str
    keep_met_frac: float
    labeling_std: float
    flux_std: float
    studies: tuple   # which sensitivity stud(y/ies) this cell belongs to


ZERO = 1e-6                  # stand-in for sigma=0; see the module docstring
DEFAULT_LAB_STD = 0.01
DEFAULT_FLUX_STD = 0.05

CONDITIONS = (
    Condition("met100_lab001_flux005", 1.00, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
              ("missing", "noise")),
    Condition("met075_lab001_flux005", 0.75, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
              ("missing",)),
    Condition("met050_lab001_flux005", 0.50, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
              ("missing",)),
    Condition("met025_lab001_flux005", 0.25, DEFAULT_LAB_STD, DEFAULT_FLUX_STD,
              ("missing",)),
    Condition("met100_lab000_flux005", 1.00, ZERO,            DEFAULT_FLUX_STD,
              ("noise",)),
    Condition("met100_lab005_flux005", 1.00, 0.05,            DEFAULT_FLUX_STD,
              ("noise",)),
    Condition("met100_lab001_flux000", 1.00, DEFAULT_LAB_STD, ZERO,
              ("noise",)),
    Condition("met100_lab001_flux025", 1.00, DEFAULT_LAB_STD, 0.25,
              ("noise",)),
)
METHODS = ("ai", "mfa")


def tasks(conditions, methods, n_samples, per_task):
    """Flat, deterministic ``(condition, method, start_ix, n_this_task)`` task list.

    ``start_ix``/``n_this_task`` tile ``[0, n_samples)`` in windows of at most
    ``per_task`` -- exactly the ``--start-ix``/``--n-benchmarks`` contract
    :func:`sigmaflux.pipeline._benchmark.select_samples` already implements, applied
    on top of the fixed ``n_samples`` subsample. There is no new tiling logic here:
    a task literally becomes ``--subsample n_samples --start-ix start_ix
    --n-benchmarks n_this_task`` on the sibling driver, so array resizing (a
    different ``--per-task``) can never change what a sample sees, the same
    guarantee the underlying drivers already give ``--start-ix``.

    Args:
        conditions: grid cells to include (a subset of ``CONDITIONS``).
        methods: predictors to include (a subset of ``METHODS``).
        n_samples: size of the fixed validation subsample every condition scores.
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
                   help="Size of the fixed validation subsample every condition "
                        "scores (default: 100)")
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
                        "sample subsample and the metabolite subsets are seeded "
                        "independently at 0, so they never move with this.")
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
    conditions = [c for c in CONDITIONS if set(c.studies) & set(studies)]
    return tasks(conditions, methods, args.n_samples, args.per_task)


def _run_task(condition, method, start_ix, n_this_task, args, output_dir):
    """Build argv for the sibling driver and call its ``cli()`` -- never reimplemented."""
    sub_out = os.path.join(output_dir, condition.name, method)
    common = [
        "--config", args.config,
        "--data-dir", args.data_dir,
        "--output-dir", sub_out,
        "--split", args.split,
        "--subsample", str(args.n_samples),
        "--start-ix", str(start_ix),
        "--n-benchmarks", str(n_this_task),
        "--seed", str(args.seed),
        "--noise-std", str(condition.labeling_std),
        "--flux-noise-std", str(condition.flux_std),
    ]
    if condition.keep_met_frac < 1.0:
        common += ["--keep-met-frac", str(condition.keep_met_frac)]
    if args.root:
        common += ["--root", args.root]

    if method == "ai":
        from .run_ai_benchmark import cli as ai_cli
        return ai_cli(common)

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
            print(f"{i:3d}  {c.name:24s}  {m:3s}  samples[{start_ix}:{start_ix + n}]  "
                  f"met={c.keep_met_frac:.0%} lab={c.labeling_std} flux={c.flux_std}")
        print(f"{len(grid)} tasks, {output_dir}")
        return 0

    if args.task_id is not None:
        if not 0 <= args.task_id < len(grid):
            raise SystemExit(f"--task-id {args.task_id} out of range [0, {len(grid)})")
        todo = [grid[args.task_id]]
    else:
        todo = grid

    for condition, method, start_ix, n_this_task in todo:
        print(f"\n=== {condition.name} / {method}  samples[{start_ix}:{start_ix + n_this_task}]"
              f"  (met={condition.keep_met_frac:.0%} lab={condition.labeling_std} "
              f"flux={condition.flux_std}) ===")
        rc = _run_task(condition, method, start_ix, n_this_task, args, output_dir)
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
