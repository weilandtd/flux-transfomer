"""
MFA failure-mode analysis: where and why 13C-MFA restarts fail to find a
statistically acceptable fit.

The MFA benchmark gives each sample up to ``--max-starts`` random-restart attempts
and keeps the best fit found; not every sample clears the standard 13C-MFA chi-square
goodness-of-fit test even after exhausting all restarts. This script asks whether
that failure is a random draw or a structural property of the sample's underlying
metabolic state, with two analyses per organism:

  1. A t-SNE embedding of ground-truth flux vectors, colored by attempts-to-solve.
  2. A clustering analysis testing whether unsolved samples concentrate in
     particular regions of flux space, plus a per-reaction test of what
     specifically differs between solved and unsolved samples.

Ground truth (not the MFA fit) is used throughout: the question is whether the
TRUE metabolic state predicts difficulty, and using the (partially failed) MFA
estimate would be circular.

``starts_used`` in *_convergence.csv is ``best["start"]`` -- the restart index that
produced the lowest-SSR fit, not a count of attempts made (see the restart loop in
run_mfa_benchmark.py). For a solved sample this genuinely is "attempts needed": the
loop breaks the instant a fit is accepted, and the accepting restart is provably
the run's best. For an *unsolved* sample every one of the --max-starts restarts was
already run regardless -- the recorded value is only which restart happened to have
the best-but-still-rejected SSR, and empirically lands anywhere from 1 to 10. It is
therefore never put on the continuous color scale; unsolved samples are always
drawn black.

Per model, in --out-dir:
  mfa_failure_tsne_<key>.svg          t-SNE of ground-truth flux, colored by attempts
  mfa_failure_tsne_<key>.csv          per-sample t-SNE coords, cluster, attempts, solved
  mfa_failure_clusters_<key>.svg      unsolved fraction per flux cluster
  mfa_failure_clusters_<key>.csv      the table behind that plot
  mfa_failure_cluster_kscan_<key>.csv silhouette + cluster/failure association vs k
  mfa_failure_reactions_<key>.csv     per-reaction Mann-Whitney U test, solved vs unsolved
  mfa_failure_top_reactions_<key>.svg the most significant reactions from that table

Across all models:
  mfa_failure_modes_summary.csv                one row per model
  mfa_failure_summary_unsolved_fraction.svg    unsolved fraction, one bar per model
  mfa_failure_summary_cramers_v.svg            cluster/failure association strength

A model with no MFA benchmark data yet is skipped with a warning rather than
failing the run. The per-reaction test and the cluster/failure association test
are skipped (with a warning) when the smaller of {solved, unsolved} falls below
--min-group -- too few samples in one group to say anything about it.

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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from _figure_common import (
    add_common_args, apply_style, run_dir_for, save_figure, selected_models)
from sigmaflux.ai.data import load_data

apply_style()

COLORS = {
    "unsolved":        "#000000",  # failed attempts -- always black, never on the RdBu scale
    "reference":       "#AAAAAA",  # baseline / threshold lines
    "bar":             "#c17429",  # per-cluster unsolved-fraction bars
    "higher_unsolved": "#B2182B",  # RdBu_r red end -- flux higher where unsolved
    "higher_solved":   "#2166AC",  # RdBu_r blue end -- flux higher where solved
}

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
                   help="Random state for t-SNE and the permutation test (default: 42)")
    p.add_argument("--k", type=int, default=5,
                   help="Cluster count for the headline cluster plot, fixed across "
                        "models for comparability (default: 5)")
    p.add_argument("--max-clusters", type=int, default=8,
                   help="Upper end of the diagnostic silhouette/association k-scan "
                        "(default: 8)")
    p.add_argument("--min-group", type=int, default=5,
                   help="Minimum size of the smaller of {solved, unsolved} required "
                        "to run the per-reaction test and cluster association test "
                        "(default: 5)")
    p.add_argument("--n-permutations", type=int, default=10000,
                   help="Permutation-test resamples when contingency cells are too "
                        "small for the chi-square approximation (default: 10000)")
    p.add_argument("--fdr", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR threshold for the per-reaction test "
                        "(default: 0.05)")
    p.add_argument("--top-reactions", type=int, default=15,
                   help="Number of reactions shown in the top-reactions plot "
                        "(default: 15)")
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


def zscore_flux(flux_sub):
    """(n_samples, n_reactions) population z-score of a reactions x samples frame.

    Same zero-variance guard as sigmaflux.ai.data.prepare_dataset: a constant
    reaction has (value - mean) == 0 for every sample, so flooring the divisor to
    1.0 rather than dropping the reaction is exact, not an approximation.
    """
    mat = flux_sub.to_numpy(dtype=float).T
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


def association_test(labels, reached_optimum, n_permutations, random_state):
    """Chi-square test of cluster membership vs. solved/unsolved, with a Cramer's V
    effect size and an automatic permutation-test fallback when expected cell
    counts are too small for the chi-square approximation (Cochran's rule: every
    expected count >= 5).
    """
    table = pd.crosstab(labels, reached_optimum).to_numpy()
    chi2, p_asymptotic, _, expected = chi2_contingency(table)
    n = table.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))

    if expected.min() >= 5:
        return dict(chi2_stat=float(chi2), p_value=float(p_asymptotic),
                    cramers_v=cramers_v, test_method="chi2")

    rng = np.random.default_rng(random_state)
    reached = np.asarray(reached_optimum)
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(reached)
        perm_stats[i] = chi2_contingency(pd.crosstab(labels, shuffled).to_numpy())[0]
    p_value = float((1 + np.sum(perm_stats >= chi2)) / (n_permutations + 1))
    return dict(chi2_stat=float(chi2), p_value=p_value, cramers_v=cramers_v,
                test_method="permutation")


def cluster_kscan(z, reached_optimum, max_k, n_permutations, random_state, test_association):
    """Ward-linkage hierarchical clustering; silhouette (+ association test, when
    the caller says there are enough samples of both outcomes) at every k from 2
    to max_k. Returns the linkage matrix (the caller cuts it at the fixed --k) and
    the scan table -- this makes the "structural, not random" claim robust to the
    choice of k rather than resting on a single value.
    """
    n = z.shape[0]
    link = linkage(pdist(z, metric="euclidean"), method="ward")
    rows = []
    for kk in range(2, min(max_k, n - 1) + 1):
        labels = fcluster(link, kk, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        row = dict(k=kk, silhouette=float(silhouette_score(z, labels)))
        if test_association:
            row.update(association_test(labels, reached_optimum, n_permutations, random_state))
        else:
            row.update(test_method=None, chi2_stat=np.nan, p_value=np.nan, cramers_v=np.nan)
        rows.append(row)
    return link, pd.DataFrame(rows)


def per_cluster_table(labels, reached_optimum):
    df = pd.DataFrame({"cluster": labels, "reached_optimum": reached_optimum})
    tbl = df.groupby("cluster")["reached_optimum"].agg(n="size", n_solved="sum")
    tbl["n_unsolved"] = tbl["n"] - tbl["n_solved"]
    tbl["unsolved_frac"] = tbl["n_unsolved"] / tbl["n"]
    return tbl.reset_index()


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        running = min(running, p[idx] * n / rank)
        q[idx] = running
    return q


def per_reaction_failure_test(flux_sub, reached_optimum, fdr):
    """Mann-Whitney U test per reaction: does its ground-truth flux differ between
    solved and unsolved samples? ``flux_sub`` columns must be in the same sample
    order as ``reached_optimum``'s index (both threaded from one canonical sample
    list by the caller).
    """
    reached = reached_optimum.to_numpy()
    rows = []
    for reaction, values in flux_sub.iterrows():
        v = values.to_numpy(dtype=float)
        solved_v, unsolved_v = v[reached], v[~reached]
        if len(solved_v) < 2 or len(unsolved_v) < 2 or len(np.unique(v)) < 2:
            rows.append(dict(
                reaction=reaction, n_solved=len(solved_v), n_unsolved=len(unsolved_v),
                median_solved=np.median(solved_v) if len(solved_v) else np.nan,
                median_unsolved=np.median(unsolved_v) if len(unsolved_v) else np.nan,
                cliffs_delta=np.nan, p_value=np.nan, direction=None))
            continue
        u_stat, p = mannwhitneyu(unsolved_v, solved_v, alternative="two-sided")
        delta = 2 * u_stat / (len(unsolved_v) * len(solved_v)) - 1
        rows.append(dict(
            reaction=reaction, n_solved=len(solved_v), n_unsolved=len(unsolved_v),
            median_solved=float(np.median(solved_v)), median_unsolved=float(np.median(unsolved_v)),
            cliffs_delta=float(delta), p_value=float(p),
            direction="higher_in_unsolved" if delta > 0 else "higher_in_solved"))

    tbl = pd.DataFrame(rows)
    valid = tbl["p_value"].notna()
    tbl["q_value"] = np.nan
    tbl.loc[valid, "q_value"] = bh_fdr(tbl.loc[valid, "p_value"].to_numpy())
    return tbl.sort_values("q_value", na_position="last").reset_index(drop=True)


def plot_tsne_scatter(df, model, out_dir, max_starts):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    solved = df[df["reached_optimum"]]
    unsolved = df[~df["reached_optimum"]]
    sc = ax.scatter(solved["tsne_1"], solved["tsne_2"], c=solved["starts_used"],
                    cmap="RdBu_r", vmin=1, vmax=max_starts, s=5, alpha=0.7,
                    linewidths=0, rasterized=True)
    ax.scatter(unsolved["tsne_1"], unsolved["tsne_2"], color=COLORS["unsolved"],
              s=5, alpha=0.7, linewidths=0, rasterized=True,
              label=f"unsolved (n={len(unsolved)})")
    fig.colorbar(sc, ax=ax, label="Attempts to solve")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_failure_tsne_{model['key']}.svg")


def plot_cluster_failure_rate(cluster_tbl, overall_unsolved_frac, model, out_dir):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.barplot(data=cluster_tbl, x="cluster", y="unsolved_frac", color=COLORS["bar"], ax=ax)
    ax.axhline(overall_unsolved_frac, color=COLORS["reference"], lw=0.8, ls="--", zorder=0)
    for i, row in cluster_tbl.iterrows():
        ax.annotate(f"n={int(row['n'])}", xy=(i, row["unsolved_frac"]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Flux cluster")
    ax.set_ylabel("Unsolved fraction")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_failure_clusters_{model['key']}.svg")


def plot_top_reactions(react_tbl, model, out_dir, top_n, fdr):
    tbl = react_tbl.dropna(subset=["q_value"]).head(top_n).copy()
    if tbl.empty:
        print(f"  {model['label']}: no reactions with a valid test -- "
              f"skipping top-reactions plot")
        return
    tbl["neglog10q"] = -np.log10(tbl["q_value"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(3.5, max(3.5, 0.22 * len(tbl) + 1)))
    palette = {"higher_in_unsolved": COLORS["higher_unsolved"],
              "higher_in_solved": COLORS["higher_solved"]}
    sns.barplot(data=tbl, y="reaction", x="neglog10q", hue="direction",
               palette=palette, dodge=False, order=tbl["reaction"], ax=ax)
    ax.axvline(-np.log10(fdr), color=COLORS["reference"], lw=0.8, ls="--", zorder=0)
    ax.set_xlabel(r"$-\log_{10}(q)$")
    ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_failure_top_reactions_{model['key']}.svg")


def plot_summary_comparison(summary_df, out_dir):
    for metric, ylabel, fname in [
            ("unsolved_fraction", "Unsolved fraction",
             "mfa_failure_summary_unsolved_fraction.svg"),
            ("cramers_v", "Cluster/failure association (Cramer's V)",
             "mfa_failure_summary_cramers_v.svg")]:
        sub = summary_df.dropna(subset=[metric])
        if sub.empty:
            continue
        colors = dict(zip(sub["label"], (MODEL_COLORS.get(k, "#777777") for k in sub["model"])))
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sns.barplot(data=sub, x="label", y=metric, order=sub["label"].tolist(),
                   hue="label", palette=colors, legend=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        sns.despine(ax=ax)
        save_figure(fig, out_dir / fname)


def summarize_model(model, n_samples, n_solved, n_unsolved, k, assoc, react_tbl, fdr):
    return dict(
        model=model["key"], label=model["label"],
        n_samples=n_samples, n_solved=n_solved, n_unsolved=n_unsolved,
        unsolved_fraction=n_unsolved / n_samples if n_samples else np.nan,
        k=k,
        cluster_test_method=assoc["test_method"] if assoc else None,
        cluster_p_value=assoc["p_value"] if assoc else np.nan,
        cramers_v=assoc["cramers_v"] if assoc else np.nan,
        n_reactions_tested=int(react_tbl["p_value"].notna().sum()) if react_tbl is not None else 0,
        n_reactions_fdr_significant=(int((react_tbl["q_value"] < fdr).sum())
                                     if react_tbl is not None else 0),
        fdr_threshold=fdr,
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

        # One canonical sample list threads through every array-producing step
        # below (z, reached, tsne_2d, labels) so they stay positionally aligned;
        # sample identity is only ever reattached via merge/reindex on "sample".
        samples = sorted(conv["sample"].unique())
        _, flux_df, _ = load_data(str(run))
        flux_df.columns = range(flux_df.shape[1])
        flux_sub = flux_df[samples]

        reached = conv.set_index("sample").reindex(samples)["reached_optimum"].to_numpy()
        n_samples = len(samples)
        n_solved = int(reached.sum())
        n_unsolved = n_samples - n_solved
        print(f"\n{model['label']}: {n_samples} samples ({n_unsolved} unsolved)")

        z = zscore_flux(flux_sub)
        tsne_2d = embed_tsne(z, args.perplexity, args.random_state)

        test_association = min(n_solved, n_unsolved) >= args.min_group
        if not test_association:
            print(f"  smaller group ({min(n_solved, n_unsolved)}) below --min-group "
                  f"({args.min_group}) — cluster association and per-reaction test skipped")

        link, kscan = cluster_kscan(z, reached, args.max_clusters, args.n_permutations,
                                    args.random_state, test_association)
        kscan.to_csv(out_dir / f"mfa_failure_cluster_kscan_{model['key']}.csv", index=False)

        k = min(args.k, n_samples - 1) if n_samples > 1 else 1
        labels = fcluster(link, k, criterion="maxclust")

        sample_df = pd.DataFrame({
            "sample": samples, "tsne_1": tsne_2d[:, 0], "tsne_2": tsne_2d[:, 1], "cluster": labels,
        }).merge(conv[["sample", "starts_used", "crashed_starts", "reached_optimum"]],
                 on="sample", how="left")
        sample_df.to_csv(out_dir / f"mfa_failure_tsne_{model['key']}.csv", index=False)
        plot_tsne_scatter(sample_df, model, out_dir, args.max_starts)

        cluster_tbl = per_cluster_table(labels, reached)
        cluster_tbl.to_csv(out_dir / f"mfa_failure_clusters_{model['key']}.csv", index=False)
        plot_cluster_failure_rate(cluster_tbl, n_unsolved / n_samples, model, out_dir)

        assoc, react_tbl = None, None
        if test_association:
            assoc = association_test(labels, reached, args.n_permutations, args.random_state)
            print(f"  cluster association ({assoc['test_method']}): "
                  f"p={assoc['p_value']:.4g}, Cramer's V={assoc['cramers_v']:.3f}")

            react_tbl = per_reaction_failure_test(
                flux_sub, pd.Series(reached, index=samples), args.fdr)
            react_tbl.to_csv(out_dir / f"mfa_failure_reactions_{model['key']}.csv", index=False)
            n_sig = int((react_tbl["q_value"] < args.fdr).sum())
            print(f"  {n_sig} of {int(react_tbl['p_value'].notna().sum())} reactions "
                  f"FDR-significant (q < {args.fdr})")
            plot_top_reactions(react_tbl, model, out_dir, args.top_reactions, args.fdr)

        summaries.append(summarize_model(model, n_samples, n_solved, n_unsolved, k,
                                         assoc, react_tbl, args.fdr))

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = out_dir / "mfa_failure_modes_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")
        plot_summary_comparison(summary_df, out_dir)
    else:
        print("No model had MFA convergence data — nothing written.")


if __name__ == "__main__":
    main()
