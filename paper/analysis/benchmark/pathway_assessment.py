"""
Pathway-level assessment: which subsystems does MFA recover, and which does SigmaFlux?

Per-reaction R² is hard to read across 400 reactions. Aggregating to the GEM's own
subsystem annotation answers the question a biologist actually asks — *can I believe the
glycolytic fluxes? the TCA cycle? amino acid biosynthesis?* — and shows whether the two
methods fail in the same places or in different ones.

Subsystems (and KEGG reaction ids / EC numbers) come from the GEM Excel `Reactions` sheet,
which the prepared JSON model drops. The coarse grouping is imported from
`models/build_pathway_annotations.py` so the two stay consistent.

Usage:
    python paper/analysis/benchmark/pathway_assessment.py paper/.results/i_orientalis_20260614

Writes <run_dir>/output/pathway_assessment.csv and prints the ranked tables.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
sys.path.insert(0, str(_REPO_ROOT / "models"))
from build_pathway_annotations import SUBSYSTEM_TO_COARSE  # noqa: E402
from plot_test_performance import load_benchmark, per_flux_r2, R2_THRESHOLD  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402

GEMS = {
    "i_orientalis": "models/I_orientalis/imIsor_GEM.xlsx",
    "s_cerevisiae": "models/S_cerevisiae/scGEM.xlsx",
}

# A subsystem needs at least this many scored reactions before its median is worth
# reporting — below it the median is one or two reactions and swings wildly.
MIN_REACTIONS = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--min-reactions", type=int, default=MIN_REACTIONS)
    p.add_argument("--flux-floor", type=float, default=0.0,
                   help="Optionally drop reactions below this median |flux| (default: keep all)")
    return p.parse_args()


def reaction_annotation(run_dir: pathlib.Path) -> pd.DataFrame:
    """subsystem / coarse subsystem / KEGG id / EC number, per reaction."""
    organism = next((k for k in GEMS if run_dir.name.startswith(k)), None)
    if organism is None:
        raise ValueError(f"Cannot detect organism from '{run_dir.name}'. Known: {list(GEMS)}")
    gem = pd.read_excel(_REPO_ROOT / GEMS[organism], sheet_name="Reactions")
    ann = gem.set_index(gem["Rxn ID"].astype(str))[["subsystem", "kegg.reaction", "ec-code"]]
    ann = ann.rename(columns={"kegg.reaction": "kegg", "ec-code": "ec"})
    ann["coarse"] = ann["subsystem"].map(SUBSYSTEM_TO_COARSE).fillna("Other")
    return ann


def build(run_dir: pathlib.Path, flux_floor: float) -> pd.DataFrame:
    _, flux, _ = load_data(str(run_dir))
    flux.columns = range(flux.shape[1])
    df_mfa, _ = load_benchmark(run_dir / "mfa_benchmark", flux)
    df_ai, _ = load_benchmark(run_dir / "ai_benchmark", flux)

    gt = df_mfa.pivot_table(index="reaction", columns="sample", values="gt")
    tbl = pd.DataFrame({
        "r2_mfa": per_flux_r2(df_mfa),
        "r2_sf":  per_flux_r2(df_ai),
        "med_gt": np.median(np.abs(gt.values), axis=1),
    }, index=gt.index).dropna(subset=["r2_mfa", "r2_sf"])
    if flux_floor > 0:
        tbl = tbl[tbl.med_gt > flux_floor]
    return tbl.join(reaction_annotation(run_dir), how="left")


def summarise(tbl: pd.DataFrame, level: str, min_n: int) -> pd.DataFrame:
    """Per-pathway medians, ranked by how many more reactions SigmaFlux resolves.

    Ranking uses `delta_ok`, the difference in the fraction of reactions clearing
    R² > 0.8, rather than the difference in median R². A median R² difference is
    unbounded and a few pathways sit at R² ≈ −20, which would otherwise put them at the
    top of the list purely because their MFA number is catastrophic.
    """
    g = tbl.groupby(level).agg(
        n=("r2_sf", "size"),
        med_flux=("med_gt", "median"),
        r2_mfa=("r2_mfa", "median"),
        r2_sf=("r2_sf", "median"),
        mfa_ok=("r2_mfa", lambda s: (s > R2_THRESHOLD).mean()),
        sf_ok=("r2_sf", lambda s: (s > R2_THRESHOLD).mean()),
    )
    g["sf_better"] = tbl.groupby(level).apply(
        lambda d: (d.r2_sf > d.r2_mfa).mean(), include_groups=False)
    g["delta_ok"] = g.sf_ok - g.mfa_ok
    return g[g.n >= min_n].sort_values("delta_ok", ascending=False)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    tbl = build(run_dir, args.flux_floor)

    out = run_dir / "output"
    out.mkdir(exist_ok=True)
    tbl.to_csv(out / "pathway_assessment.csv")

    n_ann = int(tbl.subsystem.notna().sum())
    print(f"\n{len(tbl)} scored reactions, {n_ann} with a GEM subsystem "
          f"({tbl.subsystem.nunique()} distinct)")

    fmt = lambda v: f"{v:.3f}"
    for level, title in [("coarse", "COARSE PATHWAY GROUP"), ("subsystem", "SUBSYSTEM")]:
        s = summarise(tbl, level, args.min_reactions)
        print(f"\n=== {title} — ranked by extra reactions SigmaFlux resolves "
              f"(delta_ok, R² > {R2_THRESHOLD}) ===")
        print(s.to_string(float_format=fmt))

    s = summarise(tbl, "subsystem", args.min_reactions)
    print("\n=== Where MFA is already competitive (smallest gap) ===")
    print(s.tail(8).sort_values("delta_ok").to_string(float_format=fmt))

    # MFA's pathway-level accuracy tracks pathway flux magnitude, and inversely: the
    # pathways it recovers are the low-flux biosynthetic ones that biomass stoichiometry
    # already pins down, not the high-flux central carbon a 13C experiment is run to
    # measure. Reported on log flux because pathway medians span ~4 orders of magnitude.
    lf = np.log10(np.clip(s.med_flux, 1e-6, None))
    print("\n=== Does pathway flux magnitude predict whether a method works? ===")
    print(f"  Spearman(log median pathway flux, median R²_MFA) = "
          f"{pd.Series(lf).corr(s.r2_mfa, method='spearman'):+.3f}")
    print(f"  Spearman(log median pathway flux, median R²_SF ) = "
          f"{pd.Series(lf).corr(s.r2_sf, method='spearman'):+.3f}")
    hi, lo = s[s.med_flux > 1.0], s[s.med_flux < 0.1]
    for name, part in [("high-flux pathways (>1)", hi), ("low-flux pathways (<0.1)", lo)]:
        if len(part):
            print(f"  {name:26s} n={len(part):2d}  median R²: "
                  f"MFA {part.r2_mfa.median():+.3f}  SF {part.r2_sf.median():+.3f}  "
                  f"| resolved: MFA {100 * part.mfa_ok.median():.0f}%  SF {100 * part.sf_ok.median():.0f}%")

    print("\n=== Pathways neither method resolves (both medians below the bar) ===")
    dead = s[(s.r2_mfa < R2_THRESHOLD) & (s.r2_sf < R2_THRESHOLD)].sort_values("med_flux",
                                                                               ascending=False)
    print(dead.to_string(float_format=fmt) if len(dead) else "  (none)")
    print(f"\nSaved: {out / 'pathway_assessment.csv'}")


if __name__ == "__main__":
    main()
