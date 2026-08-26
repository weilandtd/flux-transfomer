"""
Does our reported interval width mean the same thing as a published MFA confidence
interval? Per model:

  uncertainty_vs_mfa_<key>.svg    ECDF of the per-reaction (our width / MFA width) ratio
  uncertainty_vs_mfa.csv          every reaction's widths and ratio, not just the figure

An MFA CI is conditional on the observed measurement (parameter continuation: how far can
this flux move before the fit to *this* labeling data breaks). Ensemble disagreement is
not -- it is a marginal spread across the training distribution -- so a ratio close to 1
is the actual target here, not "narrower is better". Reported per reaction, signed, in
linear flux units, as a full distribution rather than one pooled or median number: fluxes
span many orders of magnitude, and a single summary would describe only the largest ones.

Restricted to points where BOTH sides give a finite bound: a published bound at or beyond
the +/-1e8 sentinel means MFA could not identify the flux at all (see
sigmaflux.utils.read_reference_flux's UNBOUNDED_FLUX handling), not a measurement to
compare against.

Usage:
    python paper/scripts/plot_uncertainty_vs_mfa.py
    python paper/scripts/plot_uncertainty_vs_mfa.py --models i_orientalis
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _figure_common import (
    COLORS, add_common_args, apply_style, build_reference_comparison,
    resolve_organism, run_dir_for, save_figure, selected_models)

apply_style()

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
    return p.parse_args()


def per_reaction_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """One row per reaction: median (our width / MFA width), both sides finite.

    Width is ``|hi - lo|`` rather than ``hi - lo``: ``pair_fluxes`` already reorders each
    interval's ends after orienting the reference (see
    ``sigmaflux.utils.read_reference_flux.pair_fluxes``), but taking the absolute value
    here is a cheap second guarantee that an inverted pair cannot silently flip the sign
    of the ratio.
    """
    d = df.dropna(subset=["ref_lo", "ref_hi", "pred_lo", "pred_hi"]).copy()
    d["mfa_width"] = (d["ref_hi"] - d["ref_lo"]).abs()
    d["our_width"] = (d["pred_hi"] - d["pred_lo"]).abs()
    d = d[d["mfa_width"] > 0]                    # a zero-width MFA bound has no ratio
    d["ratio"] = d["our_width"] / d["mfa_width"]

    return (d.groupby("reaction")
             .agg(ratio=("ratio", "median"), mfa_width=("mfa_width", "median"),
                 our_width=("our_width", "median"), n_points=("ratio", "size"))
             .reset_index())


def plot_ratio_ecdf(frames, out_dir):
    """ECDF of the per-reaction width ratio, one curve per model, log-scaled x-axis."""
    tidy = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.axvline(0.0, color=COLORS["reference"], lw=1.0, ls="--", zorder=1)

    sns.ecdfplot(data=tidy, x="log10_ratio", hue="model", ax=ax,
                 palette={m: MODEL_COLORS.get(k, "#777777")
                          for k, m in zip(tidy["key"], tidy["model"])},
                 lw=1.6, zorder=2)

    ax.set_xlabel("log$_{10}$(our width / MFA width)")
    ax.set_ylabel("Cumulative fraction of reactions")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "uncertainty_vs_mfa.svg")


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for model in selected_models(args):
        run = run_dir_for(model, args.run_suffix)
        if not list(run.glob("predictions/predictions_*.csv")):
            print(f"  {model['label']}: skipped — no predictions/ in {run.name}")
            continue
        try:
            org = resolve_organism(run)
            df = build_reference_comparison(run, org, None)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"  {model['label']}: skipped — {type(exc).__name__}: {exc}")
            continue

        tbl = per_reaction_ratio(df)
        if tbl.empty:
            print(f"  {model['label']}: skipped — no reaction has both a finite "
                  f"published bound and a finite predicted interval")
            continue
        tbl.insert(0, "model", model["label"])
        tbl.insert(1, "key", model["key"])
        frames.append(tbl)

        r = tbl["ratio"]
        print(f"  {model['label']:26s} {len(tbl):4d} reactions | "
              f"median ratio {r.median():7.2f}x | "
              f"IQR [{r.quantile(.25):.2f}, {r.quantile(.75):.2f}] | "
              f"ours narrower on {100 * (r < 1).mean():.0f}%")

    if not frames:
        print("No model had both predictions and a usable reference — nothing written.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["log10_ratio"] = np.log10(combined["ratio"].clip(lower=1e-12))
    plot_ratio_ecdf([combined], out_dir)

    path = out_dir / "uncertainty_vs_mfa.csv"
    combined.drop(columns="log10_ratio").to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
