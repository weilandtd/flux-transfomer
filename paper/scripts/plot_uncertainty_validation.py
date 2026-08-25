"""
Does the reported flux uncertainty actually cover the truth?

Per model:
  uncertainty_coverage_<key>.svg     CDF of per-reaction coverage, against nominal
  uncertainty_validation_<key>.csv   per-reaction coverage and spread components

The interval SigmaFlux reports is a propagation of the measured inputs and the ensemble's
own disagreement (see :mod:`sigmaflux.ai.uncertainty`). Nothing in it is fitted to make it
cover ground truth, so coverage is evidence rather than a restatement of a tuning step --
which is exactly why it is worth measuring, and why the number is reported here whichever
way it lands.

**What this can and cannot check.** Each held-out benchmark sample carries one already-noisy
input, not a distribution of them, so only the *ensemble* term can be scored against truth
here. The deployed interval adds the measured-input term on top, and is therefore wider:
the coverage below is a lower bound on the deployed interval's coverage, not an estimate of
it. The input term's size is reported alongside, from the same runs, so the gap is visible.

Coverage is scored **per reaction, signed, in linear flux units**. Pooling across reactions
would describe the largest fluxes and nothing else -- these span some eight orders of
magnitude -- and scoring magnitudes would let a flux recovered in the wrong direction count
as covered.

Usage:
    python paper/scripts/plot_uncertainty_validation.py
    python paper/scripts/plot_uncertainty_validation.py --models i_orientalis
"""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

from sigmaflux.ai.data import load_data, _read_hdf

from _figure_common import (
    COLORS, add_common_args, apply_style, run_dir_for, save_figure, selected_models)

apply_style()

MODEL_COLORS = {
    "e_coli_core":       "#c17429",   # burnt orange
    "s_cerevisiae":      "#2c9c8f",   # teal
    "i_orientalis":      "#7fb3d5",   # light blue
    "m_musculus_t_cell": "#9b7fb3",   # muted purple
    "e_coli":            "#f3c47f",   # golden
    "r_toruloides":      "#777777",   # grey
}

# Two-sided miss rate the reported interval targets, matching summarize_draws' default.
ALPHA = 0.05


def norm_or_t_quantile(n_members):
    """The quantile summarize_draws uses: Student-t on the member axis' degrees of freedom."""
    return t_dist.ppf(1.0 - ALPHA / 2.0, max(n_members - 1, 1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    return p.parse_args()


def member_predictions(run):
    """(reactions, samples, members) predictions and the matching ground truth.

    The AI benchmark writes every ensemble member's own raw prediction as its own column
    under a (sample, member) MultiIndex, so the member axis survives to here without
    rerunning inference.
    """
    files = sorted((run / "ai_benchmark").glob("pred_*_flux.h5"))
    if not files:
        return None, None, None
    _, flux_df, _ = load_data(str(run))
    flux_df.columns = range(flux_df.shape[1])

    frames = [_read_hdf(str(f), "fluxes") for f in files]
    pred = pd.concat(frames, axis=1)
    if not isinstance(pred.columns, pd.MultiIndex):
        return None, None, None

    samples = np.array(sorted({int(s) for s in pred.columns.get_level_values("sample")
                               if int(s) < flux_df.shape[1]}))
    reactions = pred.index
    by_sample = pred.columns.get_level_values("sample").astype(int)
    stack = np.stack([pred.to_numpy()[:, by_sample == s] for s in samples], axis=1)
    return reactions, stack, flux_df.reindex(reactions)[samples].to_numpy()


def per_reaction_validation(reactions, stack, truth):
    """Per-reaction coverage of the ensemble interval, and the spread behind it.

    Built the same way inference builds it -- ``point +/- t * sigma`` with the degrees of
    freedom taken from the member axis (see :func:`sigmaflux.ai.uncertainty.summarize_draws`)
    -- formed here over the member axis alone, the only axis a held-out sample carries. An
    empirical percentile over ~20 members would be a different, narrower interval than the
    one that ships, and would validate something nobody uses.
    """
    point = np.nanmedian(stack, axis=2)
    sigma = np.nanstd(stack, axis=2, ddof=1)
    half = norm_or_t_quantile(stack.shape[2]) * sigma

    covered = np.abs(point - truth) <= half
    out = pd.DataFrame({
        "reaction":     reactions,
        "coverage":     np.nanmean(covered, axis=1),
        "sigma_model":  np.nanmedian(np.nanstd(stack, axis=2, ddof=1), axis=1),
        "error_sd":     np.nanstd(point - truth, axis=1),
        "gt_sd":        np.nanstd(truth, axis=1),
        "n_samples":    np.sum(np.isfinite(truth), axis=1),
    })
    # A reaction whose truth never varies carries no information about whether an interval
    # tracks it, and its coverage is an artefact of a single value.
    return out[out["gt_sd"] > 0].reset_index(drop=True)


def plot_coverage(frames, out_dir):
    """CDF of per-reaction coverage, one curve per model, against the nominal rate."""
    nominal = 1.0 - ALPHA
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.axvline(nominal, color=COLORS["reference"], lw=1.0, ls="--", zorder=1)

    tidy = pd.concat(frames, ignore_index=True)
    sns.ecdfplot(data=tidy, x="coverage", hue="model", ax=ax,
                 palette={m: MODEL_COLORS.get(k, "#777777")
                          for k, m in zip(tidy["key"], tidy["model"])},
                 lw=1.6, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Per-reaction coverage of the\n{100 * nominal:.0f}% interval")
    ax.set_ylabel("Cumulative fraction of reactions")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / "uncertainty_coverage.svg")


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal = 1.0 - ALPHA
    frames = []
    for model in selected_models(args):
        run = run_dir_for(model, args.run_suffix)
        reactions, stack, truth = member_predictions(run)
        if stack is None:
            print(f"  {model['label']}: skipped — no ensemble ai_benchmark in {run.name}")
            continue

        tbl = per_reaction_validation(reactions, stack, truth)
        tbl.insert(0, "model", model["label"])
        tbl.insert(1, "key", model["key"])
        frames.append(tbl)

        # The ensemble is overconfident where the error outruns the spread it reports; the
        # ratio says by how much, and is a fairer summary than coverage alone because it
        # does not saturate at 1.
        ratio = (tbl["error_sd"] / tbl["sigma_model"].replace(0, np.nan)).median()
        print(f"  {model['label']:26s} {len(tbl):4d} reactions | "
              f"median coverage {tbl['coverage'].median():.3f} (nominal {nominal:.2f}) | "
              f"reactions at or above nominal {100 * (tbl['coverage'] >= nominal).mean():.0f}% | "
              f"error sd / ensemble sd {ratio:.2f}")

    if not frames:
        print("No model had an ensemble benchmark — nothing written.")
        return

    plot_coverage(frames, out_dir)
    path = out_dir / "uncertainty_validation.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
