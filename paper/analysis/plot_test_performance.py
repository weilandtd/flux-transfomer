"""
Plot MFA vs SigmaFlux benchmark performance.

Loads ground-truth fluxes from the training H5 files in run_dir and compares
them to the benchmark predictions. Produces two SVG figures:
  r2_histogram.svg — overlapping R² histograms (MFA + SigmaFlux on same axes)
  scatter.svg      — overlapping pred-vs-GT scatter (MFA + SigmaFlux on same axes)

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
from sigmaflux.ai.data import load_data  # noqa: E402

mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

R2_THRESHOLD = 0.8
BINS = np.linspace(0, 1, 21)   # 20 uniform bins [0, 0.05, ..., 1.0]

COLORS = {
    "mfa": "#2C7BB6",
    "ai":  "#269d89",
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
    """Load benchmark predictions and pair with GT from training H5 data.

    flux_df has shape (n_reactions, n_total_samples); columns are positional
    (pd.concat of per-batch files resets nothing, so use iloc[:, ix]).
    """
    bench_dir = pathlib.Path(bench_dir)
    frames = []
    for csv_path in sorted(bench_dir.glob("fluxes_samples_*.csv")):
        df = pd.read_csv(csv_path).set_index("reaction")
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        for col in pred_cols:
            ix = int(col[5:])
            if ix >= flux_df.shape[1]:
                continue
            gt   = flux_df.iloc[:, ix]           # Series indexed by reaction ID
            pred = df[col].reindex(gt.index)     # align to gt reaction order
            chunk = pd.DataFrame({
                "reaction": gt.index,
                "gt":       gt.values,
                "pred":     pred.values,
            }).dropna()
            frames.append(chunk)
    if not frames:
        raise FileNotFoundError(f"No fluxes_samples_*.csv files found in {bench_dir}")
    return pd.concat(frames, ignore_index=True)


def per_flux_r2(df: pd.DataFrame) -> pd.Series:
    """R² for each reaction across all samples; drop NaN (constant GT)."""
    def _r2(g):
        ss_res = ((g["pred"] - g["gt"]) ** 2).sum()
        ss_tot = ((g["gt"] - g["gt"].mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return df.groupby("reaction").apply(_r2).dropna()


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
        ax.hist(r2_vals.clip(lower=0.0), bins=BINS,
                color=COLORS[key], alpha=0.6, label=LABEL[key], edgecolor="none")

    ax.set_xlabel("R²")
    ax.set_ylabel("Reactions")
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    fig.savefig(output_dir / "r2_histogram.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'r2_histogram.svg'}")


def plot_scatter(df_mfa: pd.DataFrame, df_ai: pd.DataFrame, output_dir: pathlib.Path,
                 max_points: int = 20_000):
    """Single-axis overlapping log-log scatter of absolute flux values."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    all_abs = np.concatenate([
        np.abs(df_mfa["gt"].values), np.abs(df_mfa["pred"].values),
        np.abs(df_ai["gt"].values),  np.abs(df_ai["pred"].values),
    ])
    finite_pos = all_abs[all_abs > 0]
    vmin, vmax = finite_pos.min(), finite_pos.max()

    for df, key in [(df_mfa, "mfa"), (df_ai, "ai")]:
        gt_abs   = np.abs(df["gt"].values)
        pred_abs = np.abs(df["pred"].values)
        mask = (gt_abs > 0) & (pred_abs > 0)
        x, y = gt_abs[mask], pred_abs[mask]
        if len(x) > max_points:
            idx = rng.choice(len(x), size=max_points, replace=False)
            x, y = x[idx], y[idx]
        r2 = overall_r2(df)
        ax.scatter(x, y, s=1, color=COLORS[key], alpha=0.3, linewidths=0,
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


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    mfa_dir = args.mfa_dir or run_dir / "mfa_benchmark"
    ai_dir  = args.ai_dir  or run_dir / "ai_benchmark"
    output_dir = run_dir / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading ground-truth fluxes from: {run_dir}")
    _, flux_df, _ = load_data(str(run_dir))
    print(f"  {flux_df.shape[0]} reactions × {flux_df.shape[1]} samples")

    print(f"Loading MFA benchmark from: {mfa_dir}")
    df_mfa = load_benchmark(mfa_dir, flux_df)
    print(f"  {len(df_mfa):,} (reaction × sample) pairs from "
          f"{df_mfa['reaction'].nunique()} reactions")

    print(f"Loading AI benchmark from: {ai_dir}")
    df_ai = load_benchmark(ai_dir, flux_df)
    print(f"  {len(df_ai):,} (reaction × sample) pairs from "
          f"{df_ai['reaction'].nunique()} reactions")

    r2_mfa = per_flux_r2(df_mfa)
    r2_ai  = per_flux_r2(df_ai)

    print(f"\nMFA   R² — median: {r2_mfa.median():.3f}  mean: {r2_mfa.mean():.3f}  "
          f"overall: {overall_r2(df_mfa):.3f}")
    print(f"AI    R² — median: {r2_ai.median():.3f}  mean: {r2_ai.mean():.3f}  "
          f"overall: {overall_r2(df_ai):.3f}")

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
