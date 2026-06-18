"""
Plot MFA vs SigmaFlux benchmark performance.

Loads ground-truth fluxes from the training H5 files in run_dir and compares
them to the benchmark predictions. Produces SVG figures:
  r2_histogram.svg         — overlapping R² histograms for flux (MFA + SigmaFlux)
  r2_labeling_histogram.svg — overlapping R² histograms for per-sample labeling
  scatter.svg              — overlapping pred-vs-GT scatter (MFA + SigmaFlux)

Also writes output/identifiability.csv classifying each reaction as
identifiable by None / MFA / SF / Both (R² > R2_THRESHOLD).

Usage:
    python paper/analysis/plot_test_performance.py paper/.results/i_orientalis_20260614

Output is saved to <run_dir>/output/.
"""

import argparse
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from sigmaflux.ai.data import load_data, _read_hdf  # noqa: E402

mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

R2_THRESHOLD = 0.2
LABELING_R2_THRESHOLD = None   # only samples whose labeling fit this well count toward flux R²;
                              # set to None to disable the filter (use all samples)
ROBUST_R2_NMAD = 2.0          # per-reaction R² excludes samples whose residual (pred−GT) has a
                              # modified z-score above this (outlier-robust); None = plain R²
BINS = np.concatenate([[-0.1], np.linspace(0, 1, 11)])   # ≤0 catch-all + 10 bins in [0,1]
# Tick at the center of each bar (bin), labelled by the bin's upper edge.
_BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2   # [-0.05, 0.05, 0.15, ..., 0.95]
HIST_TICKS  = _BIN_CENTERS[[0, 2, 4, 6, 8, 10]].tolist()  # ≤0 bin + bins → 0.2,0.4,0.6,0.8,1.0
HIST_LABELS = ["≤0", "0.2", "0.4", "0.6", "0.8", "1.0"]

COLORS = {
    "mfa": "#f3c47f",
    "ai":  "#cae9e6",
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


def load_benchmark(bench_dir: pathlib.Path, flux_df: pd.DataFrame) -> pd.DataFrame:
    """Load benchmark flux predictions and pair with GT from training H5 data."""
    bench_dir = pathlib.Path(bench_dir)
    frames = []
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
            frames.append(chunk)
    if not frames:
        raise FileNotFoundError(f"No pred_*_flux.h5 files found in {bench_dir}")
    return pd.concat(frames, ignore_index=True)


def _robust_inlier_mask(resid: np.ndarray, nmad: float) -> np.ndarray:
    """Samples within `nmad` modified z-scores (Iglewicz–Hoaglin) of the median
    residual. Degenerate spread (MAD == 0) → keep all."""
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad <= 0:
        return np.ones(len(resid), dtype=bool)
    mod_z = 0.6745 * (resid - med) / mad
    return np.abs(mod_z) <= nmad


def per_flux_r2(df: pd.DataFrame) -> pd.Series:
    """Outlier-robust R² for each reaction across samples.

    A few samples whose fit produced a wild flux dominate the sum of squares and tank
    the score. Samples whose residual (pred − GT) is an outlier (|modified z-score| >
    ROBUST_R2_NMAD) are excluded before computing R². Set ROBUST_R2_NMAD = None for the
    plain R². Reactions with constant GT (after exclusion) yield NaN and are dropped.
    """
    def _r2(g):
        gt   = g["gt"].to_numpy()
        pred = g["pred"].to_numpy()
        if ROBUST_R2_NMAD is not None:
            keep = _robust_inlier_mask(pred - gt, ROBUST_R2_NMAD)
            gt, pred = gt[keep], pred[keep]
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
    return tbl.sort_values(["identifiable_by", "r2_mfa"], ascending=[True, False])


def _cmap_from_color(hex_color):
    """Linear colormap from white to a solid hex color."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("", ["white", hex_color])


def plot_histogram(r2_mfa: pd.Series, r2_ai: pd.Series, output_dir: pathlib.Path):
    """Single-axis overlapping R² histograms for MFA and SigmaFlux."""
    fig, ax = plt.subplots(figsize=(3.5, 1.5))

    for r2_vals, key in [(r2_mfa, "mfa"), (r2_ai, "ai")]:
        vals = np.clip(r2_vals.values, -0.1, 1.0)
        ax.hist(vals, bins=BINS, color=COLORS[key], alpha=0.8, label=LABEL[key], edgecolor="none")

    ax.set_xticks(HIST_TICKS)
    ax.set_xticklabels(HIST_LABELS)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("Per-reaction flux R²")
    ax.set_ylabel("Reactions")
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    fig.savefig(output_dir / "r2_histogram.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'r2_histogram.svg'}")


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


def plot_labeling_histogram(r2_mfa: pd.Series, r2_ai: pd.Series, output_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    for r2_vals, key in [(r2_mfa, "mfa"), (r2_ai, "ai")]:
        if r2_vals.empty:
            continue
        vals = np.clip(r2_vals.values, -0.1, 1.0)
        # Normalise each series by its own sample count → bar height = fraction of samples.
        weights = np.ones(len(vals)) / len(vals)
        ax.hist(vals, bins=BINS, weights=weights, color=COLORS[key], alpha=0.8,
                label=LABEL[key], edgecolor="none")
    ax.set_xticks(HIST_TICKS)
    ax.set_xticklabels(HIST_LABELS)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Labeling R²")
    ax.set_ylabel("Fraction\nof samples")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.savefig(output_dir / "r2_labeling_histogram.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'r2_labeling_histogram.svg'}")


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

    plot_labeling_histogram(lab_r2_mfa, lab_r2_ai, output_dir)

    print(f"Loading MFA benchmark from: {mfa_dir}")
    df_mfa = load_benchmark(mfa_dir, flux_df)
    print(f"  {len(df_mfa):,} pairs from {df_mfa['reaction'].nunique()} reactions "
          f"({df_mfa['sample'].nunique()} samples)")

    print(f"Loading AI benchmark from: {ai_dir}")
    df_ai = load_benchmark(ai_dir, flux_df)
    print(f"  {len(df_ai):,} pairs from {df_ai['reaction'].nunique()} reactions "
          f"({df_ai['sample'].nunique()} samples)")

    # For per-flux R² and identifiability, optionally keep only samples whose labeling
    # was reproduced well (per method). LABELING_R2_THRESHOLD = None disables the filter.
    # The scatter plot always uses ALL samples (df_mfa/df_ai unfiltered).
    if LABELING_R2_THRESHOLD is None:
        df_mfa_id, df_ai_id = df_mfa, df_ai
        print("\nLabeling-R² filter: disabled (all samples used for R²/identifiability)")
    else:
        good_mfa = set(lab_r2_mfa[lab_r2_mfa > LABELING_R2_THRESHOLD].index)
        good_ai  = set(lab_r2_ai[lab_r2_ai  > LABELING_R2_THRESHOLD].index)
        df_mfa_id = df_mfa[df_mfa["sample"].isin(good_mfa)]
        df_ai_id  = df_ai[df_ai["sample"].isin(good_ai)]
        print(f"\nLabeling-R² > {LABELING_R2_THRESHOLD} filter (for R²/identifiability): "
              f"MFA {len(good_mfa)} samples, AI {len(good_ai)} samples retained")
        if df_mfa_id.empty or df_ai_id.empty:
            raise ValueError(
                f"No samples pass the labeling-R² > {LABELING_R2_THRESHOLD} filter "
                f"(MFA pairs={len(df_mfa_id)}, AI pairs={len(df_ai_id)})."
            )

    r2_mfa = per_flux_r2(df_mfa_id)
    r2_ai  = per_flux_r2(df_ai_id)

    print(f"\nMFA   R² — median: {r2_mfa.median():.3f}  mean: {r2_mfa.mean():.3f}  "
          f"overall: {overall_r2(df_mfa_id):.3f}")
    print(f"AI    R² — median: {r2_ai.median():.3f}  mean: {r2_ai.mean():.3f}  "
          f"overall: {overall_r2(df_ai_id):.3f}")

    tbl = identifiability_table(r2_mfa, r2_ai)
    counts = tbl["identifiable_by"].value_counts()
    print(f"\nIdentifiable reactions (R² > {R2_THRESHOLD}):")
    for cat in ["Both", "MFA", "SF", "None"]:
        print(f"  {cat:4s}: {counts.get(cat, 0):4d} reactions")
    csv_path = output_dir / "identifiability.csv"
    tbl.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    plot_histogram(r2_mfa, r2_ai, output_dir)
    plot_scatter(df_mfa, df_ai, output_dir)


if __name__ == "__main__":
    main()
