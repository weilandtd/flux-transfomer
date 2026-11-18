"""
Integration tests for the accuracy-benchmark console scripts.

``sigmaflux-benchmark`` and ``sigmaflux-benchmark-mfa`` score two different
predictors on the same held-out samples and must write the same HDF5 schema —
``plot_test_performance.py`` loads both with one reader. These tests run each CLI
end-to-end on the ``toy_run`` fixture and assert that schema, plus the properties
the shared helpers exist to guarantee.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sigmaflux.pipeline import run_ai_benchmark, run_mfa_benchmark
from sigmaflux.pipeline._benchmark import build_mfa_context, load_benchmark_inputs, noisy_measurements
from sigmaflux.pipeline._paths import load_config
from sigmaflux.pipeline.run_mfa_benchmark import (
    EFFECTIVE_DOF_FILE, acceptance_threshold, build_noisy_experiments, chi2_acceptance,
    effective_parameter_count, fitted_parameter_count, make_init_fluxes,
    make_measured_flux_data, measured_flux_ids, rescore_acceptance,
)

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


def test_build_noisy_experiments_attaches_flux_once(toy_run):
    """A measured flux is a culture-level constraint, not a per-tracer one: it must
    land on exactly one Experiment. Attaching it to every tracer's Experiment (the
    old behaviour) makes fit_mfa_model's batched objective score it once per tracer
    instead of once, whenever a fit spans more than one batch."""
    cfg, root = load_config(toy_run["config"])
    cobra_model, _, _, flux_df, _ = load_benchmark_inputs(toy_run["dir"], cfg, root)
    mfa_model, _, experiments, _, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"], flux_df.iloc[:, 0], "test", with_backward=False)

    gt_flux = flux_df.iloc[:, 0]
    flux_ids = measured_flux_ids(toy_run["measured_fluxes"], gt_flux)
    gt_series = pd.Series(np.random.default_rng(0).uniform(0, 1, len(labeling_index)),
                          index=labeling_index)
    noisy_series, noisy_flux = noisy_measurements(
        gt_series, gt_flux.reindex(flux_ids).to_numpy(float), 0.01, 0.05, 42, 0)
    flux_data = make_measured_flux_data(mfa_model, flux_ids, gt_flux, 0.05, noisy_flux)

    # toy_run's config has one real tracer; duplicate the template Experiment to
    # stand in for a second one -- build_noisy_experiments only cares how many
    # Experiments it is given, not whether their tracers are physically distinct.
    noisy_exps = build_noisy_experiments(
        experiments + experiments, noisy_series, 0.01, mfa_model, flux_data)

    assert len(noisy_exps) == 2 * len(experiments)
    assert noisy_exps[0].fluxes is flux_data
    assert not noisy_exps[1].fluxes   # empty/None, not the same FluxData repeated


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


def test_chi2_acceptance_blends_the_block_scales():
    """dof is measurements minus parameters; the labeling scale is weighted by block size."""
    bound, dof, scale = chi2_acceptance(300, 100, 50, scale_labeling=0.8)
    assert dof == 350
    assert scale == pytest.approx((0.8 * 300 + 100) / 400)
    assert bound == pytest.approx(scale * stats.chi2.ppf(0.95, 350))


def _effective_count(toy_run, noise_std, flux_noise_std=0.05):
    cfg, root = load_config(toy_run["config"])
    cobra_model, mapped, _, flux_df, rev_df = load_benchmark_inputs(toy_run["dir"], cfg, root)
    mfa_model, _, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"], flux_df.iloc[:, 0], "test", with_backward=False)
    scored = labeling_index.get_level_values("Metabolite").isin(mapped)
    flux_ids = measured_flux_ids(toy_run["measured_fluxes"], flux_df.iloc[:, 0])
    gamma, lam = effective_parameter_count(
        mfa_model, labeling_to, scored, flux_df, rev_df, flux_ids, noise_std, flux_noise_std, 1.0,
        n_states=3)
    return gamma, lam, fitted_parameter_count(mfa_model, mapped)


def test_effective_parameter_count_is_bounded_by_the_structural_count(toy_run):
    """Each direction counts between 0 and 1, so γ never exceeds the directions fitted."""
    gamma, lam, structural = _effective_count(toy_run, 0.01)
    assert 0.0 <= gamma <= len(lam) <= structural
    assert (lam >= 0).all()


def test_effective_parameter_count_grows_as_the_noise_shrinks(toy_run):
    """Less noise determines more directions; noise swamping every signal determines none."""
    swamped = _effective_count(toy_run, 1e3, flux_noise_std=1e6)[0]
    counts = [_effective_count(toy_run, s)[0] for s in (1e-2, 1e-4)]
    assert swamped < 0.05
    assert swamped < counts[0] < counts[1]


def test_rescore_acceptance_rejudges_with_the_effective_count(tmp_path):
    """Each row keeps its scale; only the dof moves, so the bound loosens and fits between
    the old and new bound flip to accepted."""
    scale, dof = 0.8, 300
    old = scale * stats.chi2.ppf(0.95, dof)
    conv = pd.DataFrame({"sample": [0, 1, 2], "ssr_fit": [old - 1, old + 5, old + 500],
                         "reached_optimum": [True, False, False], "accept_rule": "chi2",
                         "chi2_dof": dof, "chi2_bound": old})
    assert rescore_acceptance(conv, str(tmp_path)).equals(conv)   # no record, no change

    (tmp_path / EFFECTIVE_DOF_FILE).write_text(json.dumps(
        {"p_eff": 20.0, "n_measurements": 350, "alpha": 0.05}))
    out = rescore_acceptance(conv, str(tmp_path))
    assert (out["chi2_dof"] == 330).all()
    assert out["chi2_bound"].iloc[0] == pytest.approx(scale * stats.chi2.ppf(0.95, 330))
    assert out["reached_optimum"].tolist() == [True, True, False]


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
