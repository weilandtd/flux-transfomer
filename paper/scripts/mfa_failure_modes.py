"""
MFA failure-mode analysis: where and why 13C-MFA restarts fail to find a
statistically acceptable fit.

The MFA benchmark gives each sample up to ``--max-starts`` random-restart attempts
and keeps the best fit found; not every sample clears the standard 13C-MFA chi-square
goodness-of-fit test even after exhausting all restarts. This script visualizes
whether that failure looks structural (unsolved samples occupy a distinct region of
the flux landscape) or scattered, and checks a simple structural predictor directly:
how many fluxes are actually measured relative to the size of the network.

  1. A t-SNE embedding of ground-truth flux vectors for the FULL sample population
     (not just the benchmarked subset), colored by attempts-to-solve, so the
     benchmarked samples are seen in the context of the whole flux landscape rather
     than in isolation.
  2. Resolved fraction vs. known/total flux ratio across organisms -- "known"
     fluxes are the ones directly measured (extracellular exchange rates, biomass
     components; ``training.measured_fluxes`` in the organism config) rather than
     inferred only from the labeling pattern, so this is a direct test of "is a
     model with proportionally fewer direct measurements harder to solve".

Ground truth (not the MFA fit) is used for the t-SNE embedding: the question is
whether the TRUE metabolic state predicts difficulty, and using the (partially
failed) MFA estimate would be circular.

``starts_used`` in *_convergence.csv is ``best["start"]`` -- the restart index that
produced the lowest-SSR fit, not a count of attempts made (see the restart loop in
run_mfa_benchmark.py). For a solved sample this genuinely is "attempts needed": the
loop breaks the instant a fit is accepted, and the accepting restart is provably
the run's best. For an *unsolved* sample every one of the --max-starts restarts was
already run regardless -- the recorded value is only which restart happened to have
the best-but-still-rejected SSR, and empirically lands anywhere from 1 to 10. It is
therefore never put on the continuous color scale; unsolved samples are always
drawn black.

The t-SNE embedding is fit on the full ~10,000-sample population per organism
(10x the ~1,000 that are actually benchmarked), so it is noticeably slower than a
1,000-point embedding and worth capping BLAS/OpenMP threads for on a shared node
(export OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=4 or similar --
sklearn's TSNE(n_jobs=-1) oversubscribing every core on a many-core node has caused
a multi-minute run to spin for 50+ CPU-minutes with no progress). Every scatter
point is drawn as a true vector path (no ``rasterized=True``), which makes these
SVGs meaningfully larger for ~10,000 points -- a deliberate tradeoff, not an
oversight.

Per model, in --out-dir:
  mfa_failure_tsne_<key>.svg   t-SNE of the full population, colored by attempts
  mfa_failure_tsne_<key>.csv   per-sample t-SNE coords, attempts, solved, benchmarked

Across all models:
  mfa_failure_modes_summary.csv              one row per model
  mfa_failure_resolved_vs_known_ratio.svg    resolved fraction vs. known/total flux ratio

A model with no MFA benchmark data yet is skipped with a warning rather than
failing the run.

Usage:
    python paper/scripts/mfa_failure_modes.py
    python paper/scripts/mfa_failure_modes.py --models m_musculus_t_cell
    python paper/scripts/mfa_failure_modes.py --run-suffix 20260807
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from _figure_common import (
    CONFIG_DIR, add_common_args, apply_style, run_dir_for, save_figure, selected_models)
from sigmaflux.ai.data import load_data
from sigmaflux.pipeline._paths import load_config

apply_style()

COLORS = {
    "unsolved":   "#000000",  # failed attempts -- always black, never on the attempts scale
    "population": "#DDDDDD",  # full population, not benchmarked -- background context
}

# Continuous scale for "attempts to solve" (solved samples only -- see plot_tsne_scatter).
ATTEMPTS_CMAP = sns.color_palette("dark:salmon_r", as_cmap=True)

# Duplicated from plot_uncertainty_vs_mfa.py / plot_uncertainty_validation.py: this
# palette isn't exported from _figure_common.py, so every script that needs a
# per-organism color keeps its own copy rather than adding a new shared dependency
# for one dict.
MODEL_COLORS = {
    "e_coli_core":       "#c17429",
    "s_cerevisiae":      "#2c9c8f",
    "i_orientalis":      "#7fb3d5",
    "m_musculus_t_cell": "#9b7fb3",
    "e_coli":            "#f3c47f",
    "r_toruloides":      "#777777",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--max-starts", type=int, default=10,
                   help="Restart budget the benchmark was run with (default: 10)")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="t-SNE perplexity (default: 30.0)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random state for t-SNE (default: 42)")
    return p.parse_args()


def load_convergence(bench_dir):
    """All *_convergence.csv rows for a benchmark dir, one row per sample.

    None when no convergence records exist yet. Deduplicated by sample, keeping
    the last file found -- defensive against a rerun SLURM shard overlapping an
    earlier one; none of the four current organism runs actually have duplicates.
    """
    records = sorted(bench_dir.glob("pred_*_convergence.csv"))
    if not records:
        return None
    df = pd.concat([pd.read_csv(f) for f in records], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset="sample", keep="last")
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} duplicate sample row(s) "
              f"(overlapping shard reruns)")
    df["sample"] = df["sample"].astype(int)
    df["starts_used"] = df["starts_used"].astype(int)
    df["crashed_starts"] = df["crashed_starts"].astype(int)
    df["reached_optimum"] = df["reached_optimum"].astype(bool)
    return df


def zscore_flux(flux_df):
    """(n_samples, n_reactions) population z-score of a reactions x samples frame.

    Same zero-variance guard as sigmaflux.ai.data.prepare_dataset: a constant
    reaction has (value - mean) == 0 for every sample, so flooring the divisor to
    1.0 rather than dropping the reaction is exact, not an approximation.
    """
    mat = flux_df.to_numpy(dtype=float).T
    mu = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (mat - mu) / std


def embed_tsne(z, perplexity, random_state):
    n = z.shape[0]
    eff_perplexity = min(perplexity, max(5, (n - 1) // 3))
    if eff_perplexity < perplexity:
        print(f"  perplexity clipped {perplexity:g} -> {eff_perplexity:g} for n={n} samples")
    tsne = TSNE(n_components=2, perplexity=eff_perplexity, max_iter=1000,
                random_state=random_state, n_jobs=-1)
    return tsne.fit_transform(z)


def known_flux_ratio(model, n_total):
    """(n_known, n_total, ratio) of directly-measured vs. total fluxes for a model.

    "Known" = ``training.measured_fluxes`` in the organism's YAML -- the
    extracellular/biomass-component fluxes measured directly and fed to the MFA fit
    as constraints (see ``measured_rxn_ids`` in run_mfa_benchmark.py), as opposed to
    fluxes inferred only from the labeling pattern. ``n_total`` is the benchmark's
    own reaction axis (``flux_df.shape[0]``), passed in since it is already loaded
    once per model by the caller.
    """
    cfg, _ = load_config(CONFIG_DIR / f"{model['key']}.yaml")
    n_known = len(cfg.get("training", {}).get("measured_fluxes", []))
    return n_known, n_total, (n_known / n_total if n_total else np.nan)


def plot_tsne_scatter(df, model, out_dir, max_starts):
    """t-SNE of the full population; benchmarked samples colored on top of it.

    Three layers, later drawn over earlier: the full population in light grey
    (context -- where do the benchmarked samples sit in the whole flux landscape),
    solved samples colored by attempts-to-solve, unsolved samples in solid black.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    population = df[~df["benchmarked"]]
    solved = df[df["benchmarked"] & df["reached_optimum"]]
    unsolved = df[df["benchmarked"] & ~df["reached_optimum"]]

    ax.scatter(population["tsne_1"], population["tsne_2"], color=COLORS["population"],
              s=3, alpha=0.4, linewidths=0, label=f"not benchmarked (n={len(population)})")
    sc = ax.scatter(solved["tsne_1"], solved["tsne_2"], c=solved["starts_used"],
                    cmap=ATTEMPTS_CMAP, vmin=1, vmax=max_starts, s=5, alpha=0.8, linewidths=0)
    ax.scatter(unsolved["tsne_1"], unsolved["tsne_2"], color=COLORS["unsolved"],
              s=5, alpha=0.8, linewidths=0, label=f"unsolved (n={len(unsolved)})")
    fig.colorbar(sc, ax=ax, label="Attempts to solve")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=8, loc="best", markerscale=2)
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_failure_tsne_{model['key']}.svg")


def plot_resolved_vs_known_ratio(summary_df, out_dir):
    """Resolved fraction vs. known/total flux ratio, one point per organism."""
    sub = summary_df.dropna(subset=["known_flux_ratio", "resolved_fraction"])
    if sub.empty:
        return
    palette = {row["label"]: MODEL_COLORS.get(row["model"], "#777777")
              for _, row in sub.iterrows()}
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(data=sub, x="known_flux_ratio", y="resolved_fraction", hue="label",
                    palette=palette, s=60, linewidth=0, legend=False, ax=ax)
    for _, row in sub.iterrows():
        ax.annotate(row["label"], xy=(row["known_flux_ratio"], row["resolved_fraction"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, sub["known_flux_ratio"].max() * 1.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Measured / total fluxes")
    ax.set_ylabel("Resolved fraction")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "mfa_failure_resolved_vs_known_ratio.svg")


def summarize_model(model, n_samples, n_solved, n_unsolved, n_known, n_total, known_ratio):
    return dict(
        model=model["key"], label=model["label"],
        n_samples=n_samples, n_solved=n_solved, n_unsolved=n_unsolved,
        unsolved_fraction=n_unsolved / n_samples if n_samples else np.nan,
        resolved_fraction=n_solved / n_samples if n_samples else np.nan,
        n_known_fluxes=n_known, n_total_fluxes=n_total, known_flux_ratio=known_ratio,
    )


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in selected_models(args):
        run = run_dir_for(model, args.run_suffix)
        bench_dir = run / "mfa_benchmark"
        conv = load_convergence(bench_dir)
        if conv is None:
            print(f"  {model['label']}: skipped — no MFA convergence records in {bench_dir}")
            continue

        _, flux_df, _ = load_data(str(run))
        flux_df.columns = range(flux_df.shape[1])

        n_samples = len(conv)
        n_solved = int(conv["reached_optimum"].sum())
        n_unsolved = n_samples - n_solved
        print(f"\n{model['label']}: {n_samples} benchmarked of {flux_df.shape[1]} "
              f"total samples ({n_unsolved} unsolved)")

        # t-SNE is fit on the FULL population so the embedding reflects the whole
        # flux landscape, not just the samples that happened to be benchmarked --
        # the benchmarked subset is then colored on top of that shared geometry.
        z = zscore_flux(flux_df)
        tsne_2d = embed_tsne(z, args.perplexity, args.random_state)

        sample_df = pd.DataFrame({
            "sample": flux_df.columns, "tsne_1": tsne_2d[:, 0], "tsne_2": tsne_2d[:, 1],
        }).merge(conv[["sample", "starts_used", "crashed_starts", "reached_optimum"]],
                 on="sample", how="left")
        # The left join leaves reached_optimum as NaN for non-benchmarked rows,
        # which coerces the whole column to object dtype and breaks `~` downstream
        # -- record "benchmarked" first, then collapse reached_optimum to a clean
        # bool (False for the non-benchmarked rows is never read as a real outcome,
        # only ever gated behind "benchmarked").
        sample_df["benchmarked"] = sample_df["reached_optimum"].notna()
        sample_df["reached_optimum"] = np.where(
            sample_df["benchmarked"], sample_df["reached_optimum"], False).astype(bool)
        sample_df.to_csv(out_dir / f"mfa_failure_tsne_{model['key']}.csv", index=False)
        plot_tsne_scatter(sample_df, model, out_dir, args.max_starts)

        n_known, n_total, known_ratio = known_flux_ratio(model, flux_df.shape[0])
        print(f"  known/total fluxes: {n_known}/{n_total} = {known_ratio:.3f}")

        summaries.append(summarize_model(model, n_samples, n_solved, n_unsolved,
                                         n_known, n_total, known_ratio))

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = out_dir / "mfa_failure_modes_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")
        plot_resolved_vs_known_ratio(summary_df, out_dir)
    else:
        print("No model had MFA convergence data — nothing written.")


if __name__ == "__main__":
    main()
