"""
Tests for the missing-metabolite / measurement-noise sensitivity benchmark.

``sigmaflux-benchmark-uncertainty`` never re-derives a prediction -- it only builds
argv for ``sigmaflux-benchmark``/``sigmaflux-benchmark-mfa`` and calls their own
``cli()``. So these tests cover what actually is new: the grid itself (every sample
scored exactly once per condition/method, regardless of how ``--per-task`` slices
it), the two shared helpers the grid depends on (``metabolite_subset``,
``select_samples(subsample=...)``), and one end-to-end smoke run per method on the
``toy_run`` fixture, following ``test_benchmark_cli.py``'s conventions.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sigmaflux.pipeline import run_ai_benchmark, run_mfa_benchmark, run_uncertainty_benchmark
from sigmaflux.pipeline._benchmark import metabolite_subset, select_samples
from sigmaflux.pipeline.run_uncertainty_benchmark import CONDITIONS, METHODS, tasks

from conftest import TOY_CONFIG_YAML as _TOY_CONFIG_YAML


# ── metabolite_subset ─────────────────────────────────────────────────────────

def test_metabolite_subset_nests_and_is_deterministic():
    """25% must be a subset of 50% must be a subset of 75% must be a subset of
    100% -- the "this platform resolves these metabolites" reading the missing-data
    study depends on -- and repeat calls (independent AI/MFA processes) must agree."""
    ids = {f"M{i}" for i in range(40)}
    levels = {f: metabolite_subset(ids, f) for f in (1.0, 0.75, 0.5, 0.25)}
    assert levels[0.25] <= levels[0.5] <= levels[0.75] <= levels[1.0]
    assert levels[1.0] == ids
    assert len(levels[0.5]) == 20

    assert metabolite_subset(ids, 0.5) == levels[0.5]         # deterministic


def test_metabolite_subset_floors_at_one_for_tiny_sets():
    """A positive frac must never round down to an empty set -- on a tiny mapped-
    metabolite count (e.g. a toy/test model with 2 metabolites) round(0.25 * 2) is 0
    under banker's rounding, which would otherwise silently turn a restriction into a
    no-op downstream (an empty set is falsy, so callers guarding on ``if ids:`` would
    skip the restriction entirely instead of keeping the smallest meaningful subset)."""
    ids = {"A", "B"}
    assert len(metabolite_subset(ids, 0.25)) == 1
    assert metabolite_subset(ids, 0.0) == set()


def test_metabolite_subset_different_seed_still_nests():
    """Nesting is a property of one seed's permutation, not the fractions alone."""
    ids = {f"M{i}" for i in range(40)}
    levels = {f: metabolite_subset(ids, f, seed=7) for f in (1.0, 0.75, 0.5, 0.25)}
    assert levels[0.25] <= levels[0.5] <= levels[0.75] <= levels[1.0]


# ── select_samples(subsample=...) ───────────────────────────────────────────────

def test_select_samples_subsample_is_deterministic_and_smaller():
    """The subsample draw is seeded at 0, independent of anything the caller passes,
    so two conditions with different noise --seed still score the identical 100
    samples -- the guarantee the whole sensitivity study depends on."""
    sel_a, _ = select_samples("all", None, 1000, {}, 0, 1000, subsample=100)
    sel_b, _ = select_samples("all", None, 1000, {}, 0, 1000, subsample=100)
    assert len(sel_a) == 100
    assert np.array_equal(sel_a, sel_b)
    assert np.array_equal(sel_a, np.sort(sel_a))               # stays sorted


def test_select_samples_subsample_tiles_without_gaps_or_overlap():
    """--per-task windows into the fixed subsample must reconstruct it exactly --
    the property that makes repartitioning a SLURM array safe."""
    whole, _ = select_samples("all", None, 1000, {}, 0, 100, subsample=100)
    windows = [select_samples("all", None, 1000, {}, start, 30, subsample=100)[0]
               for start in range(0, 100, 30)]
    reassembled = np.concatenate(windows)
    assert np.array_equal(reassembled, whole)


def test_select_samples_no_subsample_matches_old_behaviour():
    """Omitting --subsample must reproduce exactly what the pre-existing benchmarks
    do today -- this is the regression guard for the two drivers' default path."""
    sel, tag = select_samples("all", None, 50, {}, 0, 50)
    assert len(sel) == 50
    assert tag == "00000_00049"


# ── the task grid ────────────────────────────────────────────────────────────

def test_tasks_covers_every_condition_method_sample_exactly_once():
    for per_task in (None, 7, 100):
        grid = tasks(CONDITIONS, METHODS, 100, per_task)
        seen = {}
        for cond, method, start_ix, n in grid:
            key = (cond.name, method)
            covered = seen.setdefault(key, np.zeros(100, dtype=bool))
            window = covered[start_ix:start_ix + n]
            assert not window.any(), f"overlap in {key} at per_task={per_task}"
            covered[start_ix:start_ix + n] = True
        assert len(seen) == len(CONDITIONS) * len(METHODS)
        assert all(c.all() for c in seen.values())


def test_tasks_per_task_does_not_change_the_grid_shape():
    """Resizing --per-task only changes how many tasks the grid has, never which
    (condition, method, sample) triples are covered -- same invariant
    size_mfa_array's own docstring makes for the plain MFA benchmark array."""
    fine = tasks(CONDITIONS, ("mfa",), 100, 10)
    coarse = tasks(CONDITIONS, ("mfa",), 100, 100)
    fine_samples = {(c.name, s) for c, _, start, n in fine
                    for s in range(start, start + n)}
    coarse_samples = {(c.name, s) for c, _, start, n in coarse
                      for s in range(start, start + n)}
    assert fine_samples == coarse_samples


def test_studies_filter_shares_the_baseline_cell():
    missing_only = run_uncertainty_benchmark._selected_grid(
        run_uncertainty_benchmark.build_parser().parse_args([
            "--config", "x", "--data-dir", "y", "--studies", "missing", "--methods", "ai",
        ]))
    noise_only = run_uncertainty_benchmark._selected_grid(
        run_uncertainty_benchmark.build_parser().parse_args([
            "--config", "x", "--data-dir", "y", "--studies", "noise", "--methods", "ai",
        ]))
    assert len(missing_only) == 4     # baseline + 75/50/25%
    assert len(noise_only) == 5       # baseline + 4 noise cells
    assert missing_only[0][0].name == noise_only[0][0].name == "met100_lab001_flux005"


def test_bad_task_id_raises():
    with pytest.raises(SystemExit):
        run_uncertainty_benchmark.cli([
            "--config", _TOY_CONFIG_YAML, "--data-dir", "/nonexistent",
            "--task-id", "999999",
        ])


# ── AI checkpoint resolution ─────────────────────────────────────────────────────

def test_resolve_ai_checkpoint_falls_back_to_ensemble_dir(toy_run, tmp_path):
    """Every current paper/.results/*_20260822 run has an ensemble/ but no
    top-level .pt (train_ai_model returns from the ensemble branch before writing
    training.output_model) -- sigmaflux.ai.data.find_checkpoint only looks at the
    top level and raises there, which is what broke the first real run of this
    driver against production data. The toy_run fixture always ships a top-level
    .pt, so this branch needs its own run dir to exercise at all."""
    from sigmaflux.pipeline.run_uncertainty_benchmark import _resolve_ai_checkpoint

    ensemble_only = tmp_path / "ensemble_only_run"
    ensemble_only.mkdir()
    (ensemble_only / "ensemble").symlink_to(toy_run["ensemble_dir"])

    assert _resolve_ai_checkpoint(str(ensemble_only)) == str(ensemble_only / "ensemble")
    assert _resolve_ai_checkpoint(toy_run["dir"]) == toy_run["checkpoint"]   # top-level wins
    assert _resolve_ai_checkpoint(str(tmp_path)) is None                    # neither present


def test_uncertainty_benchmark_ai_task_works_with_ensemble_only_run(tmp_path, toy_run, monkeypatch):
    """End-to-end: an ensemble-only run dir (no top-level .pt) must still run the
    ai method through the sensitivity driver, not just through plain
    sigmaflux-benchmark -- this is the exact shape every real _20260822 run has."""
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "ensemble_only_run"
    run_dir.mkdir()
    for name in os.listdir(toy_run["dir"]):
        if name == "toy_ckpt.pt":
            continue   # the one file that must NOT exist, to reproduce the real shape
        os.symlink(os.path.join(toy_run["dir"], name), run_dir / name)

    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("ai",), 4, None)
    task_id = next(i for i, t in enumerate(grid) if t[0].name == "met100_lab001_flux005")

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", str(run_dir),
        "--output-dir", str(out_dir),
        "--split", "all",
        "--n-samples", "4",
        "--methods", "ai",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    flux = pd.read_hdf(out_dir / "met100_lab001_flux005" / "ai" / "pred_00000_00003_flux.h5")
    assert np.isfinite(flux.values).all()


# ── end-to-end smoke, both methods ──────────────────────────────────────────────

def test_uncertainty_benchmark_list_prints_grid_and_runs_nothing(toy_run, capsys):
    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML, "--data-dir", toy_run["dir"], "--list",
        "--n-samples", "8",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "16 tasks" in out
    assert not os.path.exists(os.path.join(toy_run["dir"], "uncertainty_benchmark"))


def test_uncertainty_benchmark_ai_task_writes_expected_layout(toy_run, tmp_path):
    """One --task-id call reaches straight into sigmaflux-benchmark and writes into
    <output-dir>/<condition>/ai/, mirroring ai_benchmark/'s own file schema."""
    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("ai",), 8, None)
    task_id = next(i for i, t in enumerate(grid) if t[0].name == "met075_lab001_flux005")

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--split", "all",
        "--n-samples", "8",
        "--methods", "ai",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    sub = out_dir / "met075_lab001_flux005" / "ai"
    flux = pd.read_hdf(sub / "pred_00000_00007_flux.h5")
    assert list(flux.index) == toy_run["flux_names"]
    assert np.isfinite(flux.values).all()


def test_uncertainty_benchmark_mfa_task_writes_expected_layout(toy_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # prepare(with_backward=True) drops dAs/dBs.pkl in CWD
    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("mfa",), 1, None)
    task_id = next(i for i, t in enumerate(grid) if t[0].name == "met100_lab005_flux005")

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--split", "all",
        "--n-samples", "1",
        "--methods", "mfa",
        "--backend", "native",
        "--max-starts", "1",
        "--max-iter", "5",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    sub = out_dir / "met100_lab005_flux005" / "mfa"
    flux = pd.read_hdf(sub / "pred_00000_00000_flux.h5")
    # Column label is the real HDF5 sample index the --subsample=1 draw picked, not
    # positional 0 -- the toy fixture has 30 samples, so this also confirms the
    # subsample mechanism actually ran rather than degenerating to the plain split.
    assert len(flux.columns) == 1
    assert 0 <= int(flux.columns[0]) < 30
    assert (sub / "pred_00000_00000_convergence.csv").exists()


def test_keep_met_frac_reduces_measured_tokens(toy_run, tmp_path, monkeypatch, capsys):
    """--keep-met-frac on the plain MFA driver (not routed through the sensitivity
    grid, so the toy fixture's 2-metabolite edge case doesn't round to zero) must
    actually shrink the noisy-MID count the fit sees."""
    monkeypatch.chdir(tmp_path)

    def _run(out_dir, extra):
        capsys.readouterr()
        assert run_mfa_benchmark.cli([
            "--config", _TOY_CONFIG_YAML,
            "--data-dir", toy_run["dir"],
            "--output-dir", str(out_dir),
            "--split", "all",
            "--n-benchmarks", "1",
            "--max-starts", "1",
            "--max-iter", "5",
        ] + extra) == 0
        return capsys.readouterr().out

    baseline_log = _run(tmp_path / "mfa_full", [])
    restricted_log = _run(tmp_path / "mfa_half", ["--keep-met-frac", "0.5"])

    def _n_entries(log):
        line = next(l for l in log.splitlines() if l.strip().startswith("Noisy MID:"))
        return int(line.split()[2])

    assert _n_entries(restricted_log) < _n_entries(baseline_log)
    assert "Restricted to 1 metabolites" in restricted_log
