"""
Integration tests for the accuracy-benchmark console scripts.

``sigmaflux-benchmark`` and ``sigmaflux-benchmark-mfa`` score two different
predictors on the same held-out samples and must write the same HDF5 schema —
``plot_test_performance.py`` loads both with one reader. These tests run each CLI
end-to-end on the ``toy_run`` fixture and assert that schema, plus the properties
the shared helpers exist to guarantee.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sigmaflux.pipeline import run_ai_benchmark, run_mfa_benchmark
from sigmaflux.pipeline.run_mfa_benchmark import acceptance_threshold, make_init_fluxes

from conftest import TOY_CONFIG_YAML as _TOY_CONFIG_YAML


def _read_pair(out_dir, tag="00000_00001"):
    """The ``(labeling, flux)`` HDF5 pair a benchmark task writes."""
    base = os.path.join(out_dir, f"pred_{tag}")
    return pd.read_hdf(f"{base}_labeling.h5"), pd.read_hdf(f"{base}_flux.h5")


# ── CLI end-to-end ────────────────────────────────────────────────────────────

def test_ai_benchmark_cli(toy_run, tmp_path):
    """An ensemble run writes every member's own raw (unprojected) prediction into
    the one H5 pair, as a (sample, member) column MultiIndex — mirroring the MFA
    benchmark's one-job-one-file layout rather than one file per member. Aggregating
    across members is left to evaluation (plot_test_performance.load_benchmark)."""
    out_dir = tmp_path / "ai_benchmark"
    assert run_ai_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--split", "all",
        "--n-benchmarks", "2",
    ]) == 0

    labeling, flux = _read_pair(out_dir)
    assert isinstance(flux.columns, pd.MultiIndex)
    assert flux.columns.names == ["sample", "member"]
    # toy_run's ensemble has 2 members; --n-benchmarks 2 selects samples 0 and 1.
    assert sorted(flux.columns.tolist()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(flux.index) == toy_run["flux_names"]
    assert np.isfinite(flux.values).all()
    assert flux.columns.equals(labeling.columns)
    assert labeling.index.names == ["Tracer", "Metabolite", "Isotopomer"]


def test_mfa_benchmark_cli(toy_run, tmp_path, monkeypatch):
    """The MFA benchmark writes the same schema, NaN where no fit was accepted."""
    monkeypatch.chdir(tmp_path)   # prepare(with_backward=True) drops dAs/dBs.pkl in CWD
    out_dir = tmp_path / "mfa_benchmark"
    assert run_mfa_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--split", "all",
        "--n-benchmarks", "1",
        "--max-starts", "1",
        "--max-iter", "5",
    ]) == 0

    labeling, flux = _read_pair(out_dir, tag="00000_00000")
    assert list(flux.columns) == [0]
    assert list(flux.index) == toy_run["flux_names"]
    assert len(labeling.columns) == 1
    # A rejected fit is written as all-NaN so it drops out downstream rather than
    # polluting the benchmark with a bad solution — never partially NaN.
    for col in (flux[0], labeling[labeling.columns[0]]):
        assert col.isna().all() or col.notna().all()


def test_mfa_benchmark_diagnose_writes_nothing(toy_run, tmp_path, monkeypatch, capsys):
    """--diagnose reports one fit and leaves no output directory behind."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "mfa_benchmark"
    assert run_mfa_benchmark.cli([
        "--config", _TOY_CONFIG_YAML,
        "--data-dir", toy_run["dir"],
        "--output-dir", str(out_dir),
        "--split", "all",
        "--diagnose",
        "--max-iter", "5",
    ]) == 0

    out = capsys.readouterr().out
    assert "Flux vs ground truth" in out
    assert "Labeling recovery" in out
    assert not out_dir.exists()


# ── Helper properties ─────────────────────────────────────────────────────────

def test_acceptance_threshold_tightens_with_information():
    """The bound approaches 1 as the effective df grows, and drops the flux block at n_F=0."""
    assert acceptance_threshold(100, 10) > acceptance_threshold(10_000, 1_000)
    assert acceptance_threshold(1_000_000, 100_000) == pytest.approx(1.0, abs=1e-2)
    # n_F == 0 → h_eff = n_L; the two-block formula would divide by zero.
    assert acceptance_threshold(50, 0, z=1.645) == pytest.approx(
        1.0 + 1.645 * np.sqrt(2.0 / 50)
    )


def test_make_init_fluxes_is_a_random_feasible_sample():
    """The init is another training sample verbatim, chosen without using the truth.

    Every training sample is a feasible point of the same polytope, so S·v_init = 0
    holds by construction and reversibilities stay non-negative. Taking one unmodified
    — rather than interpolating toward the ground truth — is what keeps the benchmark
    honest: a fit seeded near the answer measures polishing, not solving.
    """
    reaction_ids = [f"r{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    flux_df = pd.DataFrame(rng.uniform(1.0, 5.0, (6, 4)), index=reaction_ids)
    rev_df = pd.DataFrame(rng.uniform(0.0, 1.0, (6, 4)), index=reaction_ids)
    gt_flux = flux_df.iloc[:, 0]

    v_init, r_init, info = make_init_fluxes(flux_df, rev_df, gt_flux, 0, reaction_ids, rng)

    assert info["j"] != 0                                  # never starts from the sample itself
    assert v_init == pytest.approx(flux_df.iloc[:, info["j"]].values)
    assert r_init == pytest.approx(rev_df.iloc[:, info["j"]].values)
    assert (r_init >= 0).all()
    # `dev` is reported, not targeted: it is whatever distance the draw happened to land at.
    expected = np.linalg.norm(v_init - gt_flux.values) / np.linalg.norm(gt_flux.values)
    assert info["dev"] == pytest.approx(expected)
