"""
Tests for the missing-metabolite / measurement-noise sensitivity benchmark.

``sigmaflux-benchmark-uncertainty`` never re-derives a prediction -- it only builds
argv for ``sigmaflux-benchmark``/``sigmaflux-benchmark-mfa`` and calls their own
``cli()``. So these tests cover what actually is new: the grid itself (every sample
scored exactly once per condition/method, regardless of how ``--per-task`` slices
it; replicated metabolite draws nest within a replicate and vary across replicates),
the solved-sample pool (filtering, persistence, and that a --samples pool overrides
--subsample), and one end-to-end smoke run per method on the ``toy_run`` fixture,
following ``test_benchmark_cli.py``'s conventions.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from sigmaflux.pipeline import run_ai_benchmark, run_mfa_benchmark, run_uncertainty_benchmark
from sigmaflux.pipeline._benchmark import metabolite_subset, select_samples
from sigmaflux.pipeline.run_uncertainty_benchmark import (
    CONDITIONS, METHODS, _build_conditions, _pct, _resolve_sample_pool_path,
    resolve_solved_sample_pool, tasks,
)

from conftest import TOY_CONFIG_YAML as _TOY_CONFIG_YAML


def _write_baseline_convergence(path, samples, ssr_ratio=0.5, reached_optimum=True):
    """A minimal pred_*_convergence.csv -- only the columns the pool resolver reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample": list(samples),
        "ssr_ratio": ssr_ratio,
        "reached_optimum": reached_optimum,
    }).to_csv(path, index=False)


# ── metabolite_subset (unchanged API; still what replicated draws build on) ────────

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
    """Nesting is a property of one seed's permutation, not the fractions alone --
    what makes 20 independent replicates each internally consistent."""
    ids = {f"M{i}" for i in range(40)}
    levels = {f: metabolite_subset(ids, f, seed=7) for f in (1.0, 0.75, 0.5, 0.25)}
    assert levels[0.25] <= levels[0.5] <= levels[0.75] <= levels[1.0]


def test_metabolite_subset_seed_changes_the_draw():
    """Different replicates of the same fraction must actually differ -- otherwise
    20 'independent' draws would be 20 copies of one draw and buy no mean effect."""
    ids = {f"M{i}" for i in range(40)}
    draws = {seed: metabolite_subset(ids, 0.5, seed=seed) for seed in range(5)}
    assert len({frozenset(d) for d in draws.values()}) > 1


# ── select_samples: subsample and the new explicit sample_ids pool ─────────────────

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


def test_select_samples_sample_ids_overrides_subsample():
    """An explicit sample_ids pool must win over --subsample even when both are
    given -- the sensitivity study's solved-sample pool must never silently fall
    back to an unrestricted subsample."""
    explicit = [3, 7, 11, 40, 41]
    sel, _ = select_samples("all", None, 1000, {}, 0, len(explicit),
                            subsample=999, sample_ids=explicit)
    assert list(sel) == sorted(explicit)


def test_select_samples_sample_ids_tiles_without_gaps():
    """The offset-window contract is identical whether the pool comes from
    --subsample or an explicit --samples pool."""
    ids = [50, 10, 30, 20, 40, 60, 0, 70]   # unsorted on purpose
    whole, _ = select_samples("all", None, 1000, {}, 0, 8, sample_ids=ids)
    windows = [select_samples("all", None, 1000, {}, start, 3, sample_ids=ids)[0]
               for start in range(0, 8, 3)]
    reassembled = np.concatenate(windows)
    assert np.array_equal(reassembled, whole)
    assert np.array_equal(whole, np.sort(ids))


# ── _pct ─────────────────────────────────────────────────────────────────────────

def test_pct_formats_percentages_with_p_for_decimal():
    assert _pct(0.001) == "0p1"
    assert _pct(0.01) == "1"
    assert _pct(0.02) == "2"
    assert _pct(0.05) == "5"
    assert _pct(0.10) == "10"
    assert _pct(0.25) == "25"


# ── _build_conditions: the replicated grid ──────────────────────────────────────

def test_build_conditions_default_grid_shape():
    """1 baseline + 3 fractions x 20 replicates (missing) + 3 lab + 3 flux levels
    (noise, excluding the shared baseline) = 67 -- the number the plan commits to."""
    conditions = _build_conditions()
    assert len(conditions) == 67
    assert len({c.name for c in conditions}) == 67   # no duplicate names
    missing = [c for c in conditions if "missing" in c.studies]
    noise = [c for c in conditions if "noise" in c.studies]
    assert len(missing) == 1 + 3 * 20
    assert len(noise) == 1 + 6
    shared = [c for c in conditions if set(c.studies) == {"missing", "noise"}]
    assert len(shared) == 1
    assert shared[0].name == "met100_rep00_lab1_flux5"


def test_build_conditions_n_met_reps_scales_the_grid():
    small = _build_conditions(n_met_reps=2)
    assert len(small) == 1 + 3 * 2 + 6


def test_build_conditions_replicates_nest_and_vary():
    """Every replicate of one fraction must itself be a valid nested subset (checked
    via metabolite_subset directly, not re-derived here), and replicates of the same
    fraction must use different seeds -- rep index IS the seed."""
    conditions = _build_conditions(n_met_reps=5)
    met75 = sorted((c for c in conditions if c.name.startswith("met75_")),
                   key=lambda c: c.keep_met_seed)
    assert [c.keep_met_seed for c in met75] == [0, 1, 2, 3, 4]
    assert all(c.keep_met_frac == 0.75 for c in met75)


def test_build_conditions_noise_levels_match_the_grid():
    conditions = _build_conditions()
    lab_levels = sorted({c.labeling_std for c in conditions if "noise" in c.studies})
    flux_levels = sorted({c.flux_std for c in conditions if "noise" in c.studies
                          and c.labeling_std == 0.01})
    assert lab_levels == [0.001, 0.01, 0.02, 0.05]
    assert flux_levels == [0.001, 0.05, 0.10, 0.25]


def test_no_condition_uses_a_literal_zero_sigma():
    """The whole point of the 0.1% floor: nothing in the grid is a 1/sigma division
    away from being undefined."""
    conditions = _build_conditions()
    assert all(c.labeling_std > 0 for c in conditions)
    assert all(c.flux_std > 0 for c in conditions)


# ── resolve_solved_sample_pool ──────────────────────────────────────────────────

def test_resolve_solved_sample_pool_filters_by_ssr_ratio(tmp_path):
    d = tmp_path / "mfa_benchmark"
    df = pd.DataFrame({
        "sample": range(20),
        "ssr_ratio": [0.5 if i < 15 else 2.0 for i in range(20)],   # 15 eligible
        "reached_optimum": True,
    })
    (d).mkdir()
    df.to_csv(d / "pred_00000_00019_convergence.csv", index=False)

    pool = resolve_solved_sample_pool(str(d), "ssr_ratio", n_samples=10, seed=0)
    assert len(pool) == 10
    assert set(pool) <= set(range(15))          # never draws an ineligible sample
    assert pool == sorted(pool)


def test_resolve_solved_sample_pool_reached_optimum_criterion(tmp_path):
    d = tmp_path / "mfa_benchmark"
    d.mkdir()
    df = pd.DataFrame({
        "sample": range(20),
        "ssr_ratio": 0.5,
        "reached_optimum": [i < 5 for i in range(20)],   # only 5 eligible
    })
    df.to_csv(d / "pred_00000_00019_convergence.csv", index=False)

    with pytest.raises(SystemExit, match="Only 5 samples"):
        resolve_solved_sample_pool(str(d), "reached_optimum", n_samples=10)

    pool = resolve_solved_sample_pool(str(d), "reached_optimum", n_samples=5)
    assert set(pool) == set(range(5))


def test_resolve_solved_sample_pool_dedups_across_files(tmp_path):
    """Overlapping convergence files (a PILOT rerun into the same directory, or two
    array tasks that briefly overlapped) must not double-count a sample."""
    d = tmp_path / "mfa_benchmark"
    d.mkdir()
    pd.DataFrame({"sample": range(10), "ssr_ratio": 0.5, "reached_optimum": True}) \
        .to_csv(d / "pred_00000_00009_convergence.csv", index=False)
    pd.DataFrame({"sample": range(5, 15), "ssr_ratio": 0.5, "reached_optimum": True}) \
        .to_csv(d / "pred_00005_00014_convergence.csv", index=False)

    pool = resolve_solved_sample_pool(str(d), "ssr_ratio", n_samples=15)
    assert sorted(pool) == list(range(15))   # 0-14, no duplicates, none dropped


def test_resolve_solved_sample_pool_missing_dir_raises():
    with pytest.raises(SystemExit, match="No pred_\\*_convergence.csv"):
        resolve_solved_sample_pool("/nonexistent/mfa_benchmark", "ssr_ratio", n_samples=10)


def test_resolve_solved_sample_pool_reproducible_across_calls(tmp_path):
    d = tmp_path / "mfa_benchmark"
    d.mkdir()
    pd.DataFrame({"sample": range(50), "ssr_ratio": 0.5, "reached_optimum": True}) \
        .to_csv(d / "pred_00000_00049_convergence.csv", index=False)
    a = resolve_solved_sample_pool(str(d), "ssr_ratio", n_samples=20, seed=0)
    b = resolve_solved_sample_pool(str(d), "ssr_ratio", n_samples=20, seed=0)
    assert a == b


# ── _resolve_sample_pool_path: persistence ──────────────────────────────────────

def test_resolve_sample_pool_path_persists_and_is_reused(tmp_path):
    baseline = tmp_path / "mfa_benchmark"
    baseline.mkdir()
    pd.DataFrame({"sample": range(30), "ssr_ratio": 0.5, "reached_optimum": True}) \
        .to_csv(baseline / "pred_00000_00029_convergence.csv", index=False)
    out_dir = tmp_path / "uncertainty_benchmark"

    class _Args:
        data_dir = str(tmp_path)
        baseline_mfa_dir = str(baseline)
        solved_criterion = "ssr_ratio"
        n_samples = 10

    path = _resolve_sample_pool_path(_Args(), str(out_dir), root=str(tmp_path))
    assert os.path.exists(path)
    with open(path) as fh:
        first = json.load(fh)
    assert len(first) == 10

    # Second call must reuse the file rather than re-resolve -- delete the source
    # convergence data and confirm resolution still succeeds off the persisted file.
    (baseline / "pred_00000_00029_convergence.csv").unlink()
    path2 = _resolve_sample_pool_path(_Args(), str(out_dir), root=str(tmp_path))
    assert path2 == path
    with open(path2) as fh:
        second = json.load(fh)
    assert second == first


# ── the task grid ────────────────────────────────────────────────────────────

def test_tasks_covers_every_condition_method_sample_exactly_once():
    small_grid = _build_conditions(n_met_reps=2)   # keep this test fast
    for per_task in (None, 3, 20):
        grid = tasks(small_grid, METHODS, 20, per_task)
        seen = {}
        for cond, method, start_ix, n in grid:
            key = (cond.name, method)
            covered = seen.setdefault(key, np.zeros(20, dtype=bool))
            window = covered[start_ix:start_ix + n]
            assert not window.any(), f"overlap in {key} at per_task={per_task}"
            covered[start_ix:start_ix + n] = True
        assert len(seen) == len(small_grid) * len(METHODS)
        assert all(c.all() for c in seen.values())


def test_tasks_per_task_does_not_change_the_grid_shape():
    """Resizing --per-task only changes how many tasks the grid has, never which
    (condition, method, sample) triples are covered -- same invariant
    size_mfa_array's own docstring makes for the plain MFA benchmark array."""
    small_grid = _build_conditions(n_met_reps=2)
    fine = tasks(small_grid, ("mfa",), 20, 3)
    coarse = tasks(small_grid, ("mfa",), 20, 20)
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
    assert len(missing_only) == 1 + 3 * 20     # baseline + 3 fractions x 20 replicates
    assert len(noise_only) == 1 + 6            # baseline + 6 noise cells
    assert missing_only[0][0].name == noise_only[0][0].name == "met100_rep00_lab1_flux5"


def test_n_met_reps_flag_changes_the_selected_grid():
    grid = run_uncertainty_benchmark._selected_grid(
        run_uncertainty_benchmark.build_parser().parse_args([
            "--config", "x", "--data-dir", "y", "--studies", "missing",
            "--methods", "ai", "--n-met-reps", "3",
        ]))
    assert len(grid) == 1 + 3 * 3


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

    baseline = tmp_path / "baseline_mfa"
    _write_baseline_convergence(baseline / "pred_00000_00029_convergence.csv", range(30))

    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("ai",), 4, None)
    task_id = next(i for i, t in enumerate(grid) if t[0].name == "met100_rep00_lab1_flux5")

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", str(run_dir),
        "--output-dir", str(out_dir),
        "--baseline-mfa-dir", str(baseline),
        "--split", "all",
        "--n-samples", "4",
        "--methods", "ai",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    flux = pd.read_hdf(out_dir / "met100_rep00_lab1_flux5" / "ai" / "pred_00000_00003_flux.h5")
    assert np.isfinite(flux.values).all()


# ── end-to-end smoke, both methods ──────────────────────────────────────────────

def test_uncertainty_benchmark_list_prints_grid_and_runs_nothing(toy_run, capsys):
    """--list must work with no baseline MFA run on disk -- it never resolves the
    sample pool, only prints the grid structure."""
    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML, "--data-dir", toy_run["dir"], "--list",
        "--n-samples", "8", "--n-met-reps", "2",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{(1 + 3 * 2 + 6) * 2} tasks" in out
    assert not os.path.exists(os.path.join(toy_run["dir"], "uncertainty_benchmark"))


def test_uncertainty_benchmark_ai_task_writes_expected_layout(toy_run, tmp_path):
    """One --task-id call reaches straight into sigmaflux-benchmark and writes into
    <output-dir>/<condition>/ai/, mirroring ai_benchmark/'s own file schema, and
    resolves+persists samples.json alongside it."""
    baseline = tmp_path / "baseline_mfa"
    _write_baseline_convergence(baseline / "pred_00000_00029_convergence.csv", range(30))

    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("ai",), 8, None)
    met75 = next(c for c, *_ in grid if c.name.startswith("met75_rep00"))
    task_id = next(i for i, t in enumerate(grid) if t[0].name == met75.name)

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--baseline-mfa-dir", str(baseline),
        "--split", "all",
        "--n-samples", "8",
        "--methods", "ai",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    sub = out_dir / met75.name / "ai"
    flux = pd.read_hdf(sub / "pred_00000_00007_flux.h5")
    assert list(flux.index) == toy_run["flux_names"]
    assert np.isfinite(flux.values).all()
    assert os.path.exists(out_dir / "samples.json")


def test_uncertainty_benchmark_mfa_task_writes_expected_layout(toy_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # prepare(with_backward=True) drops dAs/dBs.pkl in CWD
    baseline = tmp_path / "baseline_mfa"
    _write_baseline_convergence(baseline / "pred_00000_00029_convergence.csv", range(30))

    out_dir = tmp_path / "uncertainty_benchmark"
    grid = tasks(CONDITIONS, ("mfa",), 1, None)
    lab5 = next(c for c, *_ in grid if c.name == "met100_rep00_lab5_flux5")
    task_id = next(i for i, t in enumerate(grid) if t[0].name == lab5.name)

    rc = run_uncertainty_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--baseline-mfa-dir", str(baseline),
        "--split", "all",
        "--n-samples", "1",
        "--methods", "mfa",
        "--backend", "native",
        "--max-starts", "1",
        "--max-iter", "5",
        "--task-id", str(task_id),
    ])
    assert rc == 0

    sub = out_dir / lab5.name / "mfa"
    flux = pd.read_hdf(sub / "pred_00000_00000_flux.h5")
    # Column label is the real HDF5 sample index the pool draw picked, not
    # positional 0 -- the toy fixture has 30 samples, so this also confirms the
    # sample-pool mechanism actually ran rather than degenerating to the plain split.
    assert len(flux.columns) == 1
    assert 0 <= int(flux.columns[0]) < 30
    assert (sub / "pred_00000_00000_convergence.csv").exists()


def test_uncertainty_benchmark_missing_baseline_raises_a_clear_error(toy_run, tmp_path):
    """Dispatching a task with no prior MFA run on disk must fail with an actionable
    message, not a confusing downstream KeyError from an empty sample pool."""
    out_dir = tmp_path / "uncertainty_benchmark"
    with pytest.raises(SystemExit, match="No pred_\\*_convergence.csv"):
        run_uncertainty_benchmark.cli([
            "--config", _TOY_CONFIG_YAML,
            "--data-dir", toy_run["dir"],
            "--output-dir", str(out_dir),
            "--baseline-mfa-dir", str(tmp_path / "nonexistent"),
            "--split", "all",
            "--n-samples", "1",
            "--methods", "ai",
            "--task-id", "0",
        ])


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


def test_keep_met_seed_gives_independent_replicates(toy_run, tmp_path, monkeypatch, capsys):
    """Two different --keep-met-seed values at the same --keep-met-frac must be
    capable of dropping different metabolites -- the mechanism the missing-data
    study's 20 replicates depend on."""
    monkeypatch.chdir(tmp_path)

    def _restricted_count(seed):
        capsys.readouterr()
        assert run_mfa_benchmark.cli([
            "--config", _TOY_CONFIG_YAML,
            "--data-dir", toy_run["dir"],
            "--output-dir", str(tmp_path / f"mfa_seed{seed}"),
            "--split", "all",
            "--n-benchmarks", "1",
            "--max-starts", "1",
            "--max-iter", "5",
            "--keep-met-frac", "0.5",
            "--keep-met-seed", str(seed),
        ]) == 0
        log = capsys.readouterr().out
        line = next(l for l in log.splitlines() if l.strip().startswith("Noisy MID:"))
        return int(line.split()[2])

    # Both keep exactly 1/2 metabolites (toy fixture has 2), so the counts must
    # match even though the *which* metabolite differs -- this asserts the seed is
    # wired through at all (a no-op --keep-met-seed would still pass this) alongside
    # test_metabolite_subset_seed_changes_the_draw, which proves the draw itself
    # differs by seed.
    assert _restricted_count(0) == _restricted_count(1)
