"""
Plot the SSR ("local optima") landscape computed by mfa_local_optima_landscape.py.

For each organism with a completed sweep, plots the full-population t-SNE
embedding from mfa_failure_modes.py, colored by each candidate flux vector's SSR
when scored against one solved and one unsolved exemplar sample's fixed noisy
measurements (no optimization -- a forward-simulate-and-score pass per candidate,
see mfa_local_optima_landscape.py). Multiple disconnected low-SSR ("cold") regions
away from the exemplar's own position would be evidence of local optima a
restart-based fit can get trapped in; a single basin around the exemplar's own
point is evidence the failure is not this.

The color axis is SSR over the chi-square acceptance bound, linear and clipped at
--ssr-clip, rather than log10(SSR): the question this panel exists to answer is
"which candidates would pass the goodness-of-fit test", and a log axis spanning the
field's full range -- which runs to many decades -- compresses the whole decision
region into a hairline near the floor. On the ratio scale the threshold is 1.0 by
construction, so a reader does not have to locate it. Candidates at or below it are
ringed and counted: those are alternative flux states that explain the exemplar's
measurements as well as an accepted fit does.

Candidates are ground-truth flux states from the sampled population, not converged
optima of a fit, so a ringed candidate is an alternative *acceptable explanation*
rather than a minimum an optimizer demonstrably reaches; their absence is evidence
against multi-modality, not a proof of unimodality.

Requires mfa_failure_modes.py to already have been run for the same --run-suffix
(the t-SNE embedding is shared from its output, not recomputed here), and
paper/submit/submit_mfa_local_optima_landscape.sh's array to have written
<run-dir>/mfa_local_optima/landscape_*.csv + landscape_exemplars.json.

Per model, in --out-dir:
  mfa_local_optima_<key>_solved.svg      t-SNE colored by SSR vs. the solved exemplar
  mfa_local_optima_<key>_unsolved.svg    t-SNE colored by SSR vs. the unsolved exemplar
  mfa_local_optima_<key>.csv             the merged per-candidate table behind both
  mfa_local_optima_acceptable_<key>.csv  candidates at or below the chi2 bound, if any

Usage:
    python paper/scripts/plot_mfa_local_optima_landscape.py
    python paper/scripts/plot_mfa_local_optima_landscape.py --models m_musculus_t_cell
"""

import argparse
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from _figure_common import (
    add_common_args, apply_style, run_dir_for, save_figure, selected_models)

apply_style()

COLORS = {
    "reference":  "#AAAAAA",  # chi2 acceptance bound marker on the colorbar
    "exemplar":   "#000000",  # the exemplar sample's own position
    "acceptable": "#1B6F6A",  # other candidates that pass the goodness-of-fit test
}

# Diverging, centered on the acceptance bound: the two sides of the threshold are
# categorically different answers, not two ends of one intensity.
SSR_CMAP = "RdBu_r"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ssr-clip", type=float, default=3.0,
                   help="Top of the SSR/chi2-bound color axis; candidates above it "
                        "saturate (default: 3.0)")
    return p.parse_args()


def load_landscape(landscape_dir):
    """Merged per-candidate SSR table + exemplar metadata, or (None, None)."""
    exemplars_path = landscape_dir / "landscape_exemplars.json"
    shards = sorted(landscape_dir.glob("landscape_*.csv"))
    if not exemplars_path.exists() or not shards:
        return None, None
    exemplars = json.loads(exemplars_path.read_text())
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    df = df.drop_duplicates(subset="candidate_sample", keep="last")
    return df, exemplars


def landscape_frame(tsne_df, landscape_df, exemplars, tag):
    """Per-candidate embedding coordinates plus SSR and SSR/chi2-bound for one exemplar."""
    ssr_col = f"ssr_vs_{tag}_exemplar"
    df = tsne_df.merge(
        landscape_df[["candidate_sample", ssr_col]].rename(
            columns={"candidate_sample": "sample"}), on="sample", how="inner")
    df = df.dropna(subset=[ssr_col]).rename(columns={ssr_col: "ssr"})
    df["ssr_ratio"] = df["ssr"] / exemplars["chi2_bound"]
    df["exemplar_sample"] = exemplars[f"{tag}_sample"]
    df["tag"] = tag
    return df


def acceptable_candidates(df):
    """Candidates other than the exemplar itself whose SSR passes the chi2 bound.

    The exemplar's own ground truth is excluded because it is the target the field was
    scored against, not an alternative explanation of it.
    """
    return df[(df["ssr_ratio"] <= 1.0) & (df["sample"] != df["exemplar_sample"])]


def plot_landscape(df, exemplars, tag, model, out_dir, ssr_clip):
    """t-SNE embedding colored by SSR over the chi2 acceptance bound, linear and clipped."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    # Two-slope so 1.0 -- the acceptance bound -- sits at the colormap's midpoint
    # whatever the clip is, rather than wherever the data happens to put it.
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=ssr_clip)
    sc = ax.scatter(df["tsne_1"], df["tsne_2"], c=df["ssr_ratio"].clip(upper=ssr_clip),
                    cmap=SSR_CMAP, norm=norm, s=4, alpha=0.8, linewidths=0)

    accepted = acceptable_candidates(df)
    ax.scatter(accepted["tsne_1"], accepted["tsne_2"], s=36, facecolor="none",
               edgecolor=COLORS["acceptable"], linewidths=1.0,
               label=f"also acceptable (n={len(accepted)})")

    exemplar_sample = exemplars[f"{tag}_sample"]
    exemplar_row = df[df["sample"] == exemplar_sample]
    if not exemplar_row.empty:
        ax.scatter(exemplar_row["tsne_1"], exemplar_row["tsne_2"], s=60,
                  facecolor="none", edgecolor=COLORS["exemplar"], linewidths=1.2,
                  label=f"exemplar (sample {exemplar_sample})")

    cbar = fig.colorbar(sc, ax=ax, label=r"SSR / $\chi^2$ bound")
    cbar.ax.axhline(1.0, color=COLORS["reference"], lw=1.0, ls="--")
    ticks = list(np.linspace(0, ssr_clip, 4))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks[:-1]] + [f"$\geq${ticks[-1]:g}"])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"mfa_local_optima_{model['key']}_{tag}.svg")
    return accepted


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in selected_models(args):
        run = run_dir_for(model, args.run_suffix)
        tsne_path = out_dir / f"mfa_failure_tsne_{model['key']}.csv"
        if not tsne_path.exists():
            print(f"  {model['label']}: skipped — run mfa_failure_modes.py first "
                  f"({tsne_path} not found)")
            continue

        landscape_dir = run / "mfa_local_optima"
        landscape_df, exemplars = load_landscape(landscape_dir)
        if landscape_df is None:
            print(f"  {model['label']}: skipped — no completed sweep in {landscape_dir} "
                  f"(run submit_mfa_local_optima_landscape.sh first)")
            continue

        tsne_df = pd.read_csv(tsne_path)
        n_scored = landscape_df["candidate_sample"].nunique()
        print(f"\n{model['label']}: {n_scored} of {len(tsne_df)} population candidates "
              f"scored (solved exemplar={exemplars['solved_sample']}, "
              f"unsolved exemplar={exemplars['unsolved_sample']}, "
              f"chi2_bound={exemplars['chi2_bound']:.1f})")

        accepted = []
        for tag in ("solved", "unsolved"):
            df = landscape_frame(tsne_df, landscape_df, exemplars, tag)
            hits = plot_landscape(df, exemplars, tag, model, out_dir, args.ssr_clip)
            # The median says what --ssr-clip would have to be for the field's shape,
            # rather than only its acceptable corner, to survive the color axis.
            print(f"  {tag} exemplar: {len(hits)} other candidate(s) at or below the "
                  f"chi2 bound; SSR/bound in the field spans "
                  f"{df['ssr_ratio'].min():.2f} (min) to {df['ssr_ratio'].max():.3g} "
                  f"(max), median {df['ssr_ratio'].median():.3g}")
            accepted.append(hits[["tag", "sample", "exemplar_sample", "ssr", "ssr_ratio"]])

        # Written even when empty: "no alternative acceptable state was found" is the
        # result, and a missing file would read as "this was never checked".
        accepted_path = out_dir / f"mfa_local_optima_acceptable_{model['key']}.csv"
        pd.concat(accepted, ignore_index=True).assign(
            chi2_bound=exemplars["chi2_bound"]).to_csv(accepted_path, index=False)
        print(f"Saved: {accepted_path}")

        merged = tsne_df.merge(landscape_df, left_on="sample", right_on="candidate_sample",
                               how="inner").drop(columns=["candidate_sample"])
        merged_path = out_dir / f"mfa_local_optima_{model['key']}.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Saved: {merged_path}")


if __name__ == "__main__":
    main()
