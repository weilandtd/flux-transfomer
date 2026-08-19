"""
Plot MFA vs SigmaFlux benchmark performance.

Loads ground-truth fluxes from the training H5 files in run_dir and compares
them to the benchmark predictions. Produces SVG figures:
  r2_cdf.svg          — cumulative distribution of per-reaction flux R² (MFA + SigmaFlux)
  r2_labeling_cdf.svg — cumulative distribution of per-sample labeling R²
  scatter.svg         — overlapping pred-vs-GT scatter (MFA + SigmaFlux)

Also writes output/identifiability.csv classifying each reaction as
identifiable by None / MFA / SF / Both (R² > R2_THRESHOLD).

**All R² are plain and unfiltered — do not add an outlier cut here.** The benchmark
already drops every sample whose fit is not statistically acceptable (fails the standard
¹³C-MFA χ² goodness-of-fit test, Antoniewicz et al. 2006 — see `converged_samples`), so
what reaches this script is the set of solutions each method actually stands behind. The
spread that remains is genuine flux non-identifiability, and filtering it measures how a
method does on the samples it happened to get right — which is exactly the selection bias
the benchmark exists to expose. Two such filters have been tried and removed: an NMAD cut,
and a Q-Q tail extrapolation.

Usage:
    python paper/analysis/plot_test_performance.py paper/.results/i_orientalis_20260614

Output is saved to <run_dir>/output/.
"""

import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
from sigmaflux.ai.data import load_data, _read_hdf  # noqa: E402

mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

R2_THRESHOLD = 0.8
# Clip floor sits on the 0.2 tick grid so every interval on the axis is 0.2 wide and the
# ticks are at their true data positions. The earlier −0.1 floor labelled "≤0" made the
# first interval 0.3 against 0.2 for the rest, and mislabelled its own position.
# (The predecessor histogram ticked bin *centres*, where "≤0" on the [−0.1, 0] catch-all bar
# was both correct and evenly spaced; that does not carry over to a CDF of the values.)
R2_CLIP = -0.2                 # R² at or below this piles up on the axis floor
X_TICKS  = [R2_CLIP, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
X_LABELS = ["≤−0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"]

COLORS = {
    "mfa": "#c17429",   # dark amber
    "ai":  "#1B6F6A",   # dark teal
}

LABEL = {
    "mfa": "MFA",
    "ai":  "SigmaFlux",
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot MFA vs SigmaFlux benchmark R²")
    p.add_argument("run_dir", type=pathlib.Path,
                   help="Training run directory (contains H5 training data and benchmark subdirs)")
    p.add_argument("--mfa-dir", type=pathlib.Path, default=None,
                   help="MFA benchmark directory (default: <run_dir>/mfa_benchmark)")
    p.add_argument("--ai-dir", type=pathlib.Path, default=None,
                   help="AI benchmark directory (default: <run_dir>/ai_benchmark)")
    return p.parse_args()


def converged_samples(bench_dir: pathlib.Path) -> set | None:
    """Samples whose fit is statistically acceptable, from the benchmark's own record.

    ``sigmaflux-benchmark-mfa`` writes ``*_convergence.csv`` holding, per sample, whether
    the best fit found passed the standard ¹³C-MFA goodness-of-fit test (Antoniewicz,
    Kelleher & Stephanopoulos 2006): SSR at or under the χ²(dof) bound at the run's
    ``--chi2-alpha`` (``reached_optimum``, see ``chi2_acceptance`` in
    ``run_mfa_benchmark.py``). This is the criterion MFA practice itself uses to call a
    fit good — not a benchmark-only heuristic — so scoring R² over these samples is what a
    working MFA pipeline would report.

    Returns None when no record exists (the AI arm, or an older MFA run), and the caller
    then scores everything.
    """
    records = sorted(bench_dir.glob("pred_*_convergence.csv"))
    if not records:
        return None
    df = pd.concat([pd.read_csv(f) for f in records], ignore_index=True)
    return set(df.loc[df["reached_optimum"], "sample"].astype(int))


def _sample_columns(pred_df):
    """Yield ``(sample_ix, prediction)`` for every column of a predictions frame.

    An ensemble AI benchmark's columns are a ``(sample, member)`` MultiIndex — every
    member's own raw prediction for a sample is a separate column. MFA and a
    single-checkpoint AI benchmark write one column per sample instead. Either way,
    this yields one ``(sample_ix, prediction)`` pair per column.
    """
    if isinstance(pred_df.columns, pd.MultiIndex):
        for col, ix in zip(pred_df.columns, pred_df.columns.get_level_values("sample")):
            yield int(ix), pred_df[col]
    else:
        for ix in pred_df.columns:
            yield int(ix), pred_df[ix]


def load_benchmark(bench_dir: pathlib.Path,
                   flux_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Load benchmark flux predictions and pair with GT from training H5 data.

    An ensemble AI benchmark's predictions are *raw* (unprojected): every member's
    own column for a sample (see :func:`_sample_columns`) is combined here by the
    per-reaction *median across whatever columns that sample has*, so aggregation
    happens at evaluation time and can be changed without rerunning inference. MFA
    and a single-checkpoint AI benchmark give each sample exactly one column, so this
    degenerates to "median of one", i.e. themselves.

    Returns the paired long frame and the number of samples the method failed to
    solve. A sample counts as failed when its fit did not pass the goodness-of-fit test
    (``reached_optimum`` is False, see :func:`converged_samples`) or every column's
    value for it was unusable. Either way it is excluded from R² and reported
    alongside it, so a method is never credited for samples it did not actually solve.
    """
    bench_dir = pathlib.Path(bench_dir)
    converged = converged_samples(bench_dir)

    by_sample = {}
    for h5_path in sorted(bench_dir.glob("pred_*_flux.h5")):
        pred_df = _read_hdf(h5_path, "fluxes")  # rows=reaction_ids
        for ix, pred in _sample_columns(pred_df):
            if ix >= flux_df.shape[1]:
                continue
            by_sample.setdefault(ix, []).append(pred)

    frames, n_failed = [], 0
    for ix, preds in sorted(by_sample.items()):
        if converged is not None and ix not in converged:
            n_failed += 1
            continue
        gt   = flux_df[ix]
        # median(axis=1) skips NaN per row, so one member's failed sample does not
        # drop the whole reaction for that sample as long as another member has it.
        pred = pd.concat(preds, axis=1).reindex(gt.index).median(axis=1)
        chunk = pd.DataFrame({
            "sample":   ix,
            "reaction": gt.index,
            "gt":       gt.values,
            "pred":     pred.values,
        }).dropna()
        if chunk.empty:
            n_failed += 1
        else:
            frames.append(chunk)
    if not frames:
        raise FileNotFoundError(f"No pred_*_flux.h5 files found in {bench_dir}")
    return pd.concat(frames, ignore_index=True), n_failed


def per_flux_r2(df: pd.DataFrame) -> pd.Series:
    """R² for each reaction across every sample the method solved.

    Reactions whose GT is constant across samples carry no variance to explain and
    yield NaN; they are dropped.
    """
    def _r2(g):
        gt   = g["gt"].to_numpy()
        pred = g["pred"].to_numpy()
        if len(gt) < 2:
            return np.nan
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return df.groupby("reaction").apply(_r2, include_groups=False).dropna()


def overall_r2(df: pd.DataFrame) -> float:
    ss_res = float(((df["pred"] - df["gt"]) ** 2).sum())
    ss_tot = float(((df["gt"] - df["gt"].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def identifiability_table(r2_mfa: pd.Series, r2_ai: pd.Series) -> pd.DataFrame:
    """Classify each reaction as identifiable by None/MFA/SF/Both (R² > R2_THRESHOLD)."""
    tbl = pd.DataFrame({"r2_mfa": r2_mfa, "r2_ai": r2_ai})
    mfa_id = tbl["r2_mfa"].gt(R2_THRESHOLD)
    ai_id  = tbl["r2_ai"].gt(R2_THRESHOLD)
    tbl["identifiable_by"] = np.select(
        [mfa_id & ai_id, mfa_id & ~ai_id, ~mfa_id & ai_id],
        ["Both", "MFA", "SF"],
        default="None",
    )
    # Explicit per-method flags for downstream selectors. The predicted-flux panels use
    # mfa_identifiable: those panels compare against the reference MFA fluxes, so where MFA
    # cannot determine a flux the reference value itself is unreliable and the comparison is
    # meaningless regardless of how SigmaFlux does.
    tbl["sf_identifiable"] = ai_id
    tbl["mfa_identifiable"] = mfa_id
    return tbl.sort_values(["identifiable_by", "r2_ai"], ascending=[True, False])


def plot_metric_cdf(series: dict, xlabel: str, ylabel: str, filename: str,
                    output_dir: pathlib.Path, xlim=(R2_CLIP, 1.0),
                    ticks=None, tick_labels=None,
                    threshold=R2_THRESHOLD, higher_is_better=True):
    """Cumulative distribution of a per-reaction metric for MFA and SigmaFlux.

    y = fraction of the population with metric ≤ x. The vertical gap between the two
    curves at `threshold` is the fraction of the population SigmaFlux moves onto the
    good side of that bar, and is annotated.
    """
    df = pd.concat([
        pd.DataFrame({"x": np.clip(vals.values, *xlim), "method": LABEL[key]})
        for key, vals in series.items() if not vals.empty
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.axvline(threshold, color="#AAAAAA", lw=0.8, ls="--", zorder=0)
    sns.ecdfplot(data=df, x="x", hue="method", ax=ax, lw=1.8,
                 palette={LABEL[k]: COLORS[k] for k in series})

    # Fraction of each population on the good side of the bar. Signed gap = what
    # SigmaFlux adds over MFA; negative for labeling R², which MFA optimises directly.
    good = {k: float((v > threshold).mean() if higher_is_better else (v < threshold).mean())
            for k, v in series.items() if not v.empty}
    if len(good) == 2:
        gap = good["ai"] - good["mfa"]
        ax.annotate(f"{100 * gap:+.0f}%",
                    xy=(threshold, 1.0 - (good["ai"] + good["mfa"]) / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10,
                    color=COLORS["ai"] if gap >= 0 else COLORS["mfa"])

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    if ticks is not None:
        ax.set_xticks(ticks)
        if tick_labels is not None:
            ax.set_xticklabels(tick_labels)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.move_legend(ax, "upper left", frameon=False, title=None)
    sns.despine(ax=ax)

    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / filename}")


def plot_scatter(df_mfa: pd.DataFrame, df_ai: pd.DataFrame, output_dir: pathlib.Path,
                 max_points: int = 2_000):
    """Single-axis overlapping log-log scatter of absolute flux values.

    Only `max_points` per series are drawn (subsampled) for a light, legible figure;
    the R² in the legend is always computed on the FULL set, not the subsample.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    all_abs = np.concatenate([
        np.abs(df_mfa["gt"].values), np.abs(df_mfa["pred"].values),
        np.abs(df_ai["gt"].values),  np.abs(df_ai["pred"].values),
    ])
    finite_pos = all_abs[all_abs > 0]
    vmin, vmax = finite_pos.min(), finite_pos.max()

    for df, key in [(df_mfa, "mfa"), (df_ai, "ai")]:
        r2 = overall_r2(df)   # full set, before subsampling for display
        gt_abs   = np.abs(df["gt"].values)
        pred_abs = np.abs(df["pred"].values)
        mask = (gt_abs > 0) & (pred_abs > 0)
        x, y = gt_abs[mask], pred_abs[mask]
        if len(x) > max_points:
            idx = rng.choice(len(x), size=max_points, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=4, color=COLORS[key], alpha=0.8, linewidths=0,
                   label=f"{LABEL[key]}  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([vmin, vmax], [vmin, vmax], color="#AAAAAA", lw=0.8, ls="--")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Simulated flux (mmol gDW⁻¹ h⁻¹)")
    ax.set_ylabel("Recovered flux (mmol gDW⁻¹ h⁻¹)")
    ax.legend(frameon=False, fontsize=10, markerscale=5)
    sns.despine(ax=ax)

    fig.savefig(output_dir / "scatter.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'scatter.svg'}")


def load_labeling_benchmark(bench_dir: pathlib.Path,
                            training_labeling_df: pd.DataFrame) -> pd.Series:
    """Per-sample R² between predicted labeling (benchmark H5) and GT (training H5).

    Row order of pred H5 matches training H5: both built by build_labeling_index
    on the same model, so positional comparison is safe.
    """
    r2_vals = {}
    for pred_path in sorted(pathlib.Path(bench_dir).glob("pred_*_labeling.h5")):
        pred_lab = _read_hdf(pred_path, "labeling")  # MultiIndex rows × absolute sample cols
        for ix in pred_lab.columns:
            ix = int(ix)
            if ix >= training_labeling_df.shape[1]:
                continue
            p_ser, g_ser = pred_lab[ix].align(training_labeling_df[ix], join="inner")
            p = p_ser.values.astype(float)
            g = g_ser.values.astype(float)
            # Score only observable labeling: components with non-zero GT (> 1e-6).
            # The labeling vector is mostly exact zeros (unpopulated M+n), which carry
            # no information and would distort R².
            mask = ~np.isnan(p) & ~np.isnan(g) & (g > 1e-6)
            if mask.sum() < 2:
                continue
            p, g = p[mask], g[mask]
            ss_res = ((p - g) ** 2).sum()
            ss_tot = ((g - g.mean()) ** 2).sum()
            r2_vals[ix] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return pd.Series(r2_vals).dropna()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    mfa_dir = args.mfa_dir or run_dir / "mfa_benchmark"
    ai_dir  = args.ai_dir  or run_dir / "ai_benchmark"
    output_dir = run_dir / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading ground-truth data from: {run_dir}")
    labeling_df, flux_df, _ = load_data(str(run_dir))
    flux_df.columns     = range(flux_df.shape[1])
    labeling_df.columns = range(labeling_df.shape[1])
    print(f"  {flux_df.shape[0]} reactions × {flux_df.shape[1]} samples")

    print(f"\nLoading MFA labeling benchmark from: {mfa_dir}")
    lab_r2_mfa = load_labeling_benchmark(mfa_dir, labeling_df)
    print(f"  {len(lab_r2_mfa)} samples")

    print(f"Loading AI labeling benchmark from: {ai_dir}")
    lab_r2_ai = load_labeling_benchmark(ai_dir, labeling_df)
    print(f"  {len(lab_r2_ai)} samples")

    plot_metric_cdf({"mfa": lab_r2_mfa, "ai": lab_r2_ai},
                    "Labeling R²", "Fraction of samples",
                    "r2_labeling_cdf.svg", output_dir,
                    ticks=X_TICKS, tick_labels=X_LABELS)

    print(f"Loading MFA benchmark from: {mfa_dir}")
    df_mfa, n_failed_mfa = load_benchmark(mfa_dir, flux_df)
    print(f"  {len(df_mfa):,} pairs from {df_mfa['reaction'].nunique()} reactions "
          f"({df_mfa['sample'].nunique()} samples, {n_failed_mfa} unsolved)")

    print(f"Loading AI benchmark from: {ai_dir}")
    df_ai, n_failed_ai = load_benchmark(ai_dir, flux_df)
    print(f"  {len(df_ai):,} pairs from {df_ai['reaction'].nunique()} reactions "
          f"({df_ai['sample'].nunique()} samples, {n_failed_ai} unsolved)")

    r2_mfa, r2_ai = per_flux_r2(df_mfa), per_flux_r2(df_ai)

    for name, r2, df, n_failed in [
            ("MFA", r2_mfa, df_mfa, n_failed_mfa),
            ("SigmaFlux", r2_ai, df_ai, n_failed_ai)]:
        n_total = df["sample"].nunique() + n_failed
        print(f"\n{name:9s} — median R²: {r2.median():+.3f}")
        print(f"            reactions R² > {R2_THRESHOLD}: "
              f"{int((r2 > R2_THRESHOLD).sum())} of {len(r2)}")
        print(f"            overall R² (pooled): {overall_r2(df):.3f}")
        print(f"            samples with no acceptable solution: "
              f"{n_failed} of {n_total} ({100 * n_failed / n_total:.1f}%)")

    tbl = identifiability_table(r2_mfa, r2_ai)
    counts = tbl["identifiable_by"].value_counts()
    print(f"\nIdentifiable reactions (R² > {R2_THRESHOLD}):")
    for cat in ["Both", "MFA", "SF", "None"]:
        print(f"  {cat:4s}: {counts.get(cat, 0):4d} reactions")
    csv_path = output_dir / "identifiability.csv"
    tbl.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    plot_metric_cdf({"mfa": r2_mfa, "ai": r2_ai},
                    "Per-reaction flux R²", "Fraction of reactions",
                    "r2_cdf.svg", output_dir,
                    ticks=X_TICKS, tick_labels=X_LABELS)
    # MFA on its own axis as well: in the overlay its curve is compressed into the left
    # edge, so the shape of the MFA distribution itself is hard to read.
    plot_metric_cdf({"mfa": r2_mfa},
                    "Per-reaction flux R²", "Fraction of reactions",
                    "r2_cdf_mfa.svg", output_dir,
                    ticks=X_TICKS, tick_labels=X_LABELS)
    plot_scatter(df_mfa, df_ai, output_dir)


if __name__ == "__main__":
    main()
