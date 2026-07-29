"""
How much explained variance does SigmaFlux add, per pathway?

This replaces a fold-enrichment view of the same question. Fold enrichment asks only whether
the *rescued* reactions are concentrated in a subsystem, which turned out to be a poor
summary: with 51 of 382 reactions rescued in I. orientalis, a subsystem in which every
reaction is rescued saturates at 382/51 = 7.5, so the metric hits its ceiling instead of
reporting how large the gain is, and subsystems with nothing rescued sit at 0 for a reason
that has nothing to do with quality (MFA already resolved them).

The effect size reported here is the mean per-reaction gain in explained variance,

    gain_i = clip(R²_SF, 0, 1) - clip(R²_MFA, 0, 1)

so it lives in [-1, 1] and reads directly as "SigmaFlux explains this much more of the
variance". **Clipping at 0 is essential, not cosmetic**: raw R² is unbounded below and the
raw mean gain reaches +4838 (IO purine metabolism) and -71.6 (SC Gly/Ser/Thr) purely from
reactions whose R² is hugely negative. Every R² below 0 means the same thing in practice —
no better than predicting the reaction's mean — so the degree of negativity carries no
information worth propagating into a pathway average.

Significance is a paired Wilcoxon signed-rank test on the clipped per-reaction pairs within
each subsystem, Benjamini-Hochberg corrected. Pathways with no rescued reaction at all are
dropped, since the question is about pathways SigmaFlux improves.

Usage:
    python paper/analysis/benchmark/pathway_improvement.py

Writes pathway_improvement.csv per organism plus a combined dot plot.
"""

import argparse
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

mpl.use("Agg")
mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "attention"))
from tca_attention import BLACK_GOLD  # noqa: E402  single source for the shared ramp

COLORS = {"cmap": BLACK_GOLD, "ref": "#AAAAAA", "err": "#666666"}
ORGANISMS = {"i_orientalis": "I. orientalis", "s_cerevisiae": "S. cerevisiae"}

R2_THRESHOLD = 0.8
MIN_REACTIONS = 4
N_BOOT = 2000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-suffix", default="20260614")
    p.add_argument("--level", default="subsystem", choices=["subsystem", "coarse"])
    p.add_argument("--min-reactions", type=int, default=MIN_REACTIONS)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--keep-unrescued", action="store_true",
                   help="Keep subsystems where no reaction is rescued (dropped by default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p.parse_args()


def bh_fdr(p: np.ndarray) -> np.ndarray:
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


def improvement(run_dir: pathlib.Path, level: str, min_n: int, keep_unrescued: bool,
                rng) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "output" / "pathway_assessment.csv").dropna(subset=[level])
    sf = df.r2_sf.clip(0, 1)
    mfa = df.r2_mfa.clip(0, 1)
    df = df.assign(gain=sf - mfa, sf_c=sf, mfa_c=mfa,
                   rescued=(df.r2_sf > R2_THRESHOLD) & (df.r2_mfa <= R2_THRESHOLD))

    rows = []
    for name, g in df.groupby(level):
        if len(g) < min_n:
            continue
        if not keep_unrescued and int(g.rescued.sum()) == 0:
            continue
        # Bootstrap the mean gain rather than assuming normality: pathways hold 4-57
        # reactions and the per-reaction gains are strongly skewed.
        boot = rng.choice(g.gain.values, size=(N_BOOT, len(g)), replace=True).mean(axis=1)
        try:
            p = float(wilcoxon(g.sf_c.values, g.mfa_c.values,
                               alternative="greater", zero_method="zsplit").pvalue)
        except ValueError:          # all pairs tied
            p = 1.0
        rows.append({
            level: name, "n": len(g), "k_rescued": int(g.rescued.sum()),
            "mean_gain": float(g.gain.mean()), "median_gain": float(g.gain.median()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "r2_mfa": float(g.mfa_c.median()), "r2_sf": float(g.sf_c.median()),
            "median_flux": float(g.med_gt.median()), "p": p,
        })
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out.p.values)
    return out.sort_values("mean_gain", ascending=False)


def plot(tables: dict, fdr: float, out_path: pathlib.Path):
    both = pd.concat([t.assign(organism=ORGANISMS[o]) for o, t in tables.items()],
                     ignore_index=True)
    level = [c for c in ("subsystem", "coarse") if c in both.columns][0]
    # Rank by the largest gain either organism shows, best at the top.
    order = (both.groupby(level)["mean_gain"].max()
                 .sort_values(ascending=True).index.tolist())
    both["neglogq"] = -np.log10(both.q.clip(lower=1e-12))

    fig, axes = plt.subplots(1, len(tables), figsize=(3.2 * len(tables) + 3.2,
                                                     0.34 * len(order) + 2.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    vmax = float(both.neglogq.max())
    for ax, org in zip(axes, [ORGANISMS[o] for o in tables]):
        s = both[both.organism == org].set_index(level).reindex(order)
        y = np.arange(len(order))
        ax.hlines(y, s.ci_lo, s.ci_hi, color=COLORS["err"], lw=1.0, zorder=2)
        edge = np.where(s.q.fillna(1) < fdr, "#000000", "#BBBBBB")
        sc = ax.scatter(s.mean_gain, y, s=20 + 20 * s.k_rescued.fillna(0),
                        c=s.neglogq, cmap=COLORS["cmap"], vmin=0, vmax=vmax,
                        edgecolor=edge, linewidth=1.0, zorder=3)
        ax.axvline(0.0, color=COLORS["ref"], lw=0.8, ls="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel("Mean gain in explained variance\n($R^2_{SigmaFlux} - R^2_{MFA}$)")
        ax.set_title(org, fontsize=11, style="italic")
        sns.despine(ax=ax)

    cb = fig.colorbar(sc, ax=axes, fraction=0.030, pad=0.02)
    cb.set_label("$-\\log_{10}$ q")
    for k in sorted({int(v) for v in np.percentile(both.k_rescued.dropna(), [10, 50, 95])}):
        axes[0].scatter([], [], s=20 + 20 * k, c="#777777", edgecolor="white",
                        linewidth=0.5, label=f"{k}")
    axes[0].legend(title="reactions rescued", frameon=False, fontsize=8, title_fontsize=8,
                   loc="lower right", labelspacing=1.0, handletextpad=0.4, borderpad=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tables = {}
    for org in ORGANISMS:
        run_dir = _REPO_ROOT / "paper" / ".results" / f"{org}_{args.run_suffix}"
        t = improvement(run_dir, args.level, args.min_reactions, args.keep_unrescued, rng)
        tables[org] = t
        t.to_csv(run_dir / "output" / "pathway_improvement.csv", index=False)
        print(f"\n=== {ORGANISMS[org]} === {len(t)} subsystems with >= {args.min_reactions} "
              f"reactions and >= 1 rescued")
        cols = [args.level, "n", "k_rescued", "r2_mfa", "r2_sf", "mean_gain",
                "ci_lo", "ci_hi", "q", "median_flux"]
        print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = args.out or (_REPO_ROOT / "paper" / ".results" / f"i_orientalis_{args.run_suffix}"
                       / "output" / "pathway_improvement.svg")
    plot(tables, args.fdr, out)


if __name__ == "__main__":
    main()
