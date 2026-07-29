"""
Plot MFA vs SigmaFlux benchmark performance.

Loads ground-truth fluxes from the training H5 files in run_dir and compares
them to the benchmark predictions. Produces SVG figures:
  r2_cdf.svg          — cumulative distribution of per-reaction flux R² (MFA + SigmaFlux)
  r2_labeling_cdf.svg — cumulative distribution of per-sample labeling R²
  scatter.svg         — overlapping pred-vs-GT scatter (MFA + SigmaFlux)

Also writes output/identifiability.csv classifying each reaction as
identifiable by None / MFA / SF / Both (R² > R2_THRESHOLD).

All R² are plain (unfiltered). run_mfa_benchmark.py already rejects fits that fail
the reduced-χ² acceptance bound and writes them as NaN, so every MFA sample reaching
this script is a statistically acceptable fit; the spread that remains is genuine
flux non-identifiability, not numerical failure, and must not be filtered away.

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
from scipy.stats import norm

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
from sigmaflux.ai.data import load_data, _read_hdf  # noqa: E402

mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

R2_THRESHOLD = 0.8
QQ_ALPHA = 0.1                 # expected number of samples beyond the outlier cut under the
                               # bulk model, per reaction; smaller = more conservative cut
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


def load_benchmark(bench_dir: pathlib.Path,
                   flux_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Load benchmark flux predictions and pair with GT from training H5 data.

    Returns the paired long frame and the number of samples the method failed to
    solve. run_mfa_benchmark.py writes all-NaN fluxes for samples where no restart
    cleared the reduced-χ² acceptance bound; those samples drop out here and never
    count against the method's R², so the failure count must be reported alongside it.
    """
    bench_dir = pathlib.Path(bench_dir)
    frames, n_failed = [], 0
    for h5_path in sorted(bench_dir.glob("pred_*_flux.h5")):
        pred_df = _read_hdf(h5_path, "fluxes")  # rows=reaction_ids, cols=absolute sample_ix
        for ix in pred_df.columns:
            ix = int(ix)
            if ix >= flux_df.shape[1]:
                continue
            gt   = flux_df[ix]
            pred = pred_df[ix].reindex(gt.index)
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


def _robust_scale(resid: np.ndarray) -> float:
    """Scale of the bulk of a residual distribution, read off its central quantiles.

    IQR/1.349 is the σ a normal would need to produce the observed Q1..Q3, so it is
    set by the middle half of the samples and is unaffected by the tail.
    """
    q1, q3 = np.percentile(resid, [25, 75])
    return (q3 - q1) / 1.349


def qq_inlier_mask(resid: np.ndarray, alpha: float = QQ_ALPHA) -> np.ndarray:
    """Outlier cut obtained by extrapolating the central quantiles into the tail.

    Fit the bulk scale from Q1..Q3 (`_robust_scale`), then cut at the quantile where
    that fitted model expects only `alpha` samples out of n to remain — i.e. where the
    observed tail departs from the straight line of the normal Q-Q plot. Unlike a fixed
    n-MAD rule the cut widens with n, so it does not get more aggressive as samples are
    added. Degenerate spread (IQR == 0) keeps everything.
    """
    n = len(resid)
    scale = _robust_scale(resid)
    if scale <= 0 or n < 2:
        return np.ones(n, dtype=bool)
    z_cut = norm.ppf(1.0 - alpha / (2.0 * n))
    return np.abs(resid - np.median(resid)) <= z_cut * scale


def per_flux_r2(df: pd.DataFrame, exclude_outliers: bool = True) -> pd.Series:
    """R² for each reaction across samples.

    With `exclude_outliers`, samples outside the Q-Q extrapolated bulk (see
    `qq_inlier_mask`) are dropped first. This measures how well a method does when it
    lands on a sensible solution, and must always be reported next to the unfiltered R²
    and the exclusion rate from `outlier_fraction` — the excluded MFA fits are not
    numerical failures, they clear the benchmark's own reduced-χ² acceptance bound.

    Reactions whose GT is constant across samples carry no variance to explain and
    yield NaN; they are dropped.
    """
    def _r2(g):
        gt   = g["gt"].to_numpy()
        pred = g["pred"].to_numpy()
        if exclude_outliers:
            keep = qq_inlier_mask(pred - gt)
            gt, pred = gt[keep], pred[keep]
        if len(gt) < 2:
            return np.nan
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return df.groupby("reaction").apply(_r2, include_groups=False).dropna()


def outlier_fraction(df: pd.DataFrame) -> pd.Series:
    """Fraction of samples the Q-Q cut removes, per reaction."""
    def _frac(g):
        return float(1.0 - qq_inlier_mask(g["pred"].to_numpy() - g["gt"].to_numpy()).mean())
    return df.groupby("reaction").apply(_frac, include_groups=False).dropna()


def standardized_residuals(df: pd.DataFrame) -> np.ndarray:
    """Per-reaction residuals divided by that reaction's bulk scale, pooled.

    Standardising per reaction makes reactions with different flux magnitudes
    comparable, so a single Q-Q plot can show whether one common bulk model describes
    them all — and where it stops holding.
    """
    out = []
    for _, g in df.groupby("reaction"):
        r = g["pred"].to_numpy() - g["gt"].to_numpy()
        scale = _robust_scale(r)
        if scale > 0:
            out.append((r - np.median(r)) / scale)
    return np.concatenate(out) if out else np.empty(0)


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
    # meaningless regardless of how SigmaFlux does. Note both flags use the Q-Q-filtered R²
    # — on the unfiltered R² MFA clears the bar for no reaction at all and the panels would
    # be empty.
    tbl["sf_identifiable"] = ai_id
    tbl["mfa_identifiable"] = mfa_id
    return tbl.sort_values(["identifiable_by", "r2_ai"], ascending=[True, False])


def _cut_z(n_per_reaction: int, alpha: float = QQ_ALPHA) -> float:
    """The standardized-residual cut `qq_inlier_mask` applies to a reaction of n samples."""
    return float(norm.ppf(1.0 - alpha / (2.0 * n_per_reaction)))


def plot_qq(resid: dict, n_per_reaction: dict, output_dir: pathlib.Path,
            filename: str = "qq_residuals.svg", n_points: int = 600, ylim: float = 12.0):
    """Normal Q-Q plot of pooled standardized residuals, with the outlier cut marked.

    This is the audit for `qq_inlier_mask`: the bulk of each method should track the
    diagonal (the model fitted from its own central quantiles), and the cut sits where
    the observed tail lifts off that line. Everything outside the dashed lines is what
    the filtered R² drops. The cut is set by the per-reaction sample count, not the
    pooled count, so it matches what the mask actually applies.
    """
    probs = (np.arange(n_points) + 0.5) / n_points
    theo  = norm.ppf(probs)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lim = float(np.abs(theo).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#AAAAAA", lw=0.8, zorder=0)

    for key, vals in resid.items():
        if len(vals) == 0:
            continue
        z_cut = _cut_z(n_per_reaction[key])
        ax.plot(theo, np.quantile(vals, probs), color=COLORS[key], lw=1.8,
                label=f"{LABEL[key]} — {100 * np.mean(np.abs(vals) > z_cut):.1f}% cut")
        for s in (-1, 1):
            ax.axhline(s * z_cut, color=COLORS[key], lw=0.7, ls="--", alpha=0.6, zorder=0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("Normal quantile")
    ax.set_ylabel("Standardized residual\n(per-reaction, IQR-scaled)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    sns.despine(ax=ax)

    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / filename}")


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

    r2_mfa,   r2_ai   = per_flux_r2(df_mfa), per_flux_r2(df_ai)
    raw_mfa,  raw_ai  = (per_flux_r2(df_mfa, exclude_outliers=False),
                         per_flux_r2(df_ai,  exclude_outliers=False))
    out_mfa,  out_ai  = outlier_fraction(df_mfa), outlier_fraction(df_ai)

    # The filtered R² answers "how well does the method do when it lands on a sensible
    # solution". It is only interpretable next to the unfiltered R² and the exclusion
    # rate, because the two differ enormously for MFA and barely at all for SigmaFlux —
    # the excluded MFA fits still clear the benchmark's reduced-χ² acceptance bound.
    for name, r2, raw, out, df, n_failed in [
            ("MFA", r2_mfa, raw_mfa, out_mfa, df_mfa, n_failed_mfa),
            ("SigmaFlux", r2_ai, raw_ai, out_ai, df_ai, n_failed_ai)]:
        n_total = df["sample"].nunique() + n_failed
        print(f"\n{name:9s} — median R²: {r2.median():+.3f} after Q-Q outlier cut "
              f"(α={QQ_ALPHA:g}) | {raw.median():+.3f} unfiltered")
        print(f"            reactions R² > {R2_THRESHOLD}: {int((r2 > R2_THRESHOLD).sum())} "
              f"of {len(r2)} | {int((raw > R2_THRESHOLD).sum())} unfiltered")
        print(f"            samples cut as outliers: {100 * out.mean():.1f}% "
              f"(median {100 * out.median():.1f}% per reaction)")
        print(f"            overall R² (pooled, unfiltered): {overall_r2(df):.3f}")
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
    plot_qq({"mfa": standardized_residuals(df_mfa), "ai": standardized_residuals(df_ai)},
            {"mfa": df_mfa["sample"].nunique(), "ai": df_ai["sample"].nunique()},
            output_dir)
    plot_scatter(df_mfa, df_ai, output_dir)


if __name__ == "__main__":
    main()
