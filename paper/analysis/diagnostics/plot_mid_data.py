"""
Plot the measured mass isotopomer distribution of one compound under one tracer.

Bars are the mean fraction over biological replicates, error bars their standard
deviation — this is the raw measurement uncertainty every flux estimate is fitted to, so
it sets the floor on how well any method can be expected to do.

The clean per-condition xlsx holds one sheet per tracer and one column per
`{Condition}{GrowthCondition}_Rep{BioRep}`, e.g. `C1_Rep1`. Growth conditions are shaded
by dilution rate with the same `dark:<colour>_r` ramp the predicted-flux panels use, so
the two figures are readable together.

Usage:
    python paper/analysis/diagnostics/plot_mid_data.py            # malate, U-13C glc (50%)
    python paper/analysis/diagnostics/plot_mid_data.py --compound Cit --tracer '1,2-13C glucose (50%)'

Output goes to <run_dir>/output/mid_<compound>_<condition>.svg
"""

import argparse
import pathlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sigmaflux.pipeline._paths import load_config, resolve_path

mpl.use("Agg")
mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Same per-condition base colours as the predicted-flux panels.
CONDITION_COLORS = {"C": "#f3c47f", "N": "#7fb3d5", "P": "#2c9c8f"}
COLORS = {"error": "#444444"}

CONFIG_DIR = _REPO_ROOT / "paper" / "configs"

#: Organism configs carrying an `experiment:` block, by config stem.
ORGANISM_KEYS = sorted(
    p.stem for p in CONFIG_DIR.glob("*.yaml")
    if "experiment:" in p.read_text()
)


def organism(key: str) -> dict:
    """Experimental-data paths for an organism, read from its YAML config."""
    cfg, root = load_config(CONFIG_DIR / f"{key}.yaml")
    exp = cfg["experiment"]
    return dict(prefix=cfg["name"], data=exp["data_dir"],
                reference=exp["reference_flux"], root=root)


def _opath(org: dict, *parts) -> pathlib.Path:
    """Resolve a config-relative path against the organism's project root."""
    return pathlib.Path(resolve_path("/".join(parts), org["root"]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organism", default="i_orientalis", choices=ORGANISM_KEYS)
    p.add_argument("--compound", default="Mal", help="Compound name as it appears in the xlsx")
    p.add_argument("--tracer", default="U-13C glucose (50%)", help="Sheet name of the tracer")
    p.add_argument("--condition", default="C", choices=["C", "N", "P"])
    p.add_argument("--mode", default="mid", choices=["mid", "noise"],
                   help="mid: one compound's MID; noise: replicate-SD distribution over "
                        "every measurement in every organism/condition/tracer")
    p.add_argument("--assumed-sigma", type=float, default=0.01,
                   help="MID noise_std the benchmark assumes, marked on the noise figure")
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p.parse_args()


def load_mid(org: dict, condition: str, tracer: str, compound: str) -> pd.DataFrame:
    """Tidy replicate-level MIDs: one row per (growth condition, replicate, M+n)."""
    xlsx = _opath(org, org["data"], f"{org['prefix']}_{condition}.xlsx")
    sheet = pd.read_excel(xlsx, sheet_name=tracer)

    sub = sheet[sheet["Compound"].astype(str) == compound]
    if sub.empty:
        raise ValueError(f"Compound '{compound}' not in {xlsx.name} / '{tracer}'. "
                         f"Available: {sorted(sheet['Compound'].astype(str).unique())}")

    pat = re.compile(rf"^{re.escape(condition)}(\d+)_Rep(\d+)$")
    rows = []
    for col in sub.columns:
        m = pat.match(str(col))
        if not m:
            continue
        for _, r in sub.iterrows():
            rows.append({"gc": int(m.group(1)), "rep": int(m.group(2)),
                         "mid": int(r["C_Label"]), "fraction": float(r[col])})
    return pd.DataFrame(rows)


def load_all_replicate_sd() -> pd.DataFrame:
    """Replicate mean and SD for every (organism, condition, tracer, compound, M+n).

    This is the measurement-noise model of the whole dataset: the SD across biological
    replicates at every single measured point. What a χ² objective sums is the *squared*
    residual, so the statistic to compare a single assumed σ against is the RMS of these
    SDs, not their median — the distribution is strongly right-skewed and the two differ
    by more than 3×.
    """
    rows = []
    for org_key in ORGANISM_KEYS:
        org = organism(org_key)
        for cond in ("C", "N", "P"):
            xlsx = _opath(org, org["data"], f"{org['prefix']}_{cond}.xlsx")
            if not xlsx.exists():
                continue
            for tracer, sheet in pd.read_excel(xlsx, sheet_name=None).items():
                pat = re.compile(rf"^{re.escape(cond)}(\d+)_Rep(\d+)$")
                groups = {}
                for col in sheet.columns:
                    m = pat.match(str(col))
                    if m:
                        groups.setdefault(m.group(1), []).append(col)
                for gc, cols in groups.items():
                    if len(cols) < 2:
                        continue
                    block = sheet[cols]
                    rows.append(pd.DataFrame({
                        "organism": org_key, "condition": cond, "tracer": tracer,
                        "gc": int(gc), "compound": sheet["Compound"].astype(str),
                        "mid": sheet["C_Label"],
                        "mean": block.mean(axis=1), "sd": block.std(axis=1),
                        "n_rep": len(cols),
                    }))
    return pd.concat(rows, ignore_index=True).dropna(subset=["sd"])


def plot_noise_distribution(df: pd.DataFrame, assumed_sigma: float,
                            out_path: pathlib.Path):
    """Distribution of the replicate SD over every measured MID point."""
    sd = df["sd"].to_numpy()
    sd = sd[sd > 0]
    median, rms = float(np.median(sd)), float(np.sqrt((sd ** 2).mean()))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    bins = np.logspace(np.log10(sd.min()), np.log10(sd.max()), 45)
    ax.hist(sd, bins=bins, color=CONDITION_COLORS["C"], edgecolor="none")
    ax.set_xscale("log")

    # RMS and the assumed sigma land within 2% of each other, so their labels are stacked
    # at different heights rather than overplotted on the same line.
    top = ax.get_ylim()[1]
    for x, lab, style, y in [(median, f"median {median:.4f}", ":", 0.98),
                             (rms, f"RMS {rms:.4f}", "-", 0.80),
                             (assumed_sigma, f"assumed σ {assumed_sigma:g}", "--", 0.62)]:
        ax.axvline(x, color=COLORS["error"], lw=1.0, ls=style)
        ax.annotate(lab, xy=(x, top * y), xytext=(3, 0), textcoords="offset points",
                    va="top", ha="left", fontsize=7, color=COLORS["error"])

    ax.set_xlabel("Replicate SD of MID fraction")
    ax.set_ylabel("Measurements")
    sns.despine(ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"  {len(sd):,} measured points, {int(df.n_rep.min())}–{int(df.n_rep.max())} replicates each")
    print(f"  median {median:.4f}   mean {sd.mean():.4f}   RMS {rms:.4f}"
          f"   q90 {np.percentile(sd, 90):.4f}   q99 {np.percentile(sd, 99):.4f}")
    print(f"  fraction at or above the assumed σ={assumed_sigma:g}: "
          f"{100 * np.mean(sd >= assumed_sigma):.1f}%")
    print(f"  assumed σ / RMS = {assumed_sigma / rms:.2f}x  (RMS is what a χ² sees)")


def growth_rates(org: dict, condition: str) -> dict:
    """Dilution rate per growth condition, from the reference biomass flux."""
    ref = pd.read_excel(_opath(org, org["data"], org["reference"])).set_index("id")
    pat = re.compile(rf"^{re.escape(condition)}(\d*)_mfa$")
    out = {}
    for c in ref.columns:
        m = pat.match(str(c))
        if m and "BIOMASS" in ref.index:
            gc = int(m.group(1)) if m.group(1) else 1
            out[gc] = round(abs(float(ref.loc["BIOMASS", c])), 2)
    return out


def plot_mid(df: pd.DataFrame, rates: dict, compound: str, tracer: str,
             condition: str, out_path: pathlib.Path):
    stat = (df.groupby(["gc", "mid"])["fraction"]
              .agg(mean="mean", sd="std", n="size").reset_index())
    gcs = sorted(stat.gc.unique())
    mids = sorted(stat.mid.unique())
    shades = sns.color_palette(f"dark:{CONDITION_COLORS[condition]}_r", n_colors=len(gcs))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    width = 0.8 / len(gcs)
    for k, gc in enumerate(gcs):
        s = stat[stat.gc == gc].set_index("mid").reindex(mids)
        offset = (k - (len(gcs) - 1) / 2) * width
        ax.bar(np.arange(len(mids)) + offset, s["mean"], width=width,
               color=shades[k], edgecolor="none",
               label=f"{rates.get(gc, gc):.2f}" if rates else str(gc))
        # SD over biological replicates; caps kept small so 5 groups stay legible.
        ax.errorbar(np.arange(len(mids)) + offset, s["mean"], yerr=s["sd"],
                    fmt="none", ecolor=COLORS["error"], elinewidth=0.7, capsize=1.5)

    ax.set_xticks(np.arange(len(mids)))
    ax.set_xticklabels([f"M+{m}" for m in mids])
    ax.set_xlabel(f"{compound} isotopomer")
    ax.set_ylabel("Fraction of pool")
    # Headroom for the legend: without it the entries sit on top of the tallest bars.
    ax.set_ylim(0, float((stat["mean"] + stat["sd"].fillna(0)).max()) * 1.45)
    ax.legend(title="D [h⁻¹]", frameon=False, fontsize=8, title_fontsize=8,
              loc="upper right", handlelength=1.0, ncol=2, columnspacing=0.8)
    sns.despine(ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    n_rep = int(stat.n.min())
    print(f"Saved: {out_path}  ({len(gcs)} growth conditions x {n_rep} replicates)")
    print(stat.pivot(index="mid", columns="gc", values="mean").round(3).to_string())
    print("\nSD over replicates:")
    print(stat.pivot(index="mid", columns="gc", values="sd").round(4).to_string())


def main():
    args = parse_args()
    org = organism(args.organism)
    default_dir = (_REPO_ROOT / "paper" / ".results" / f"{args.organism}_20260614" / "output")

    if args.mode == "noise":
        df = load_all_replicate_sd()
        out = args.out or (default_dir / "mid_noise_distribution.svg")
        plot_noise_distribution(df, args.assumed_sigma, out)
        return

    df = load_mid(org, args.condition, args.tracer, args.compound)
    rates = growth_rates(org, args.condition)
    out = args.out or (default_dir / f"mid_{args.compound}_{args.condition}.svg")
    print(f"{args.compound} | {args.tracer} | condition {args.condition}")
    plot_mid(df, rates, args.compound, args.tracer, args.condition, out)


if __name__ == "__main__":
    main()
