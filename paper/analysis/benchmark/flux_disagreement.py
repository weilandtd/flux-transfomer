"""
Rank the fluxes where MFA and SigmaFlux disagree most, and decide which to trust.

On the simulation benchmark the ground truth is known, so a disagreement can be
adjudicated outright: for every reaction we report how far apart the two methods are and
which one is actually closer to the truth. That adjudication is then explained with three
structural covariates read off the model, and cross-checked against the ensemble's own
uncertainty — the only one of these signals that is also available on real data, where no
ground truth exists.

Covariates (all computed from model_prepared_for_mfa.json):
  has_atom_map  reaction carries an atom mapping at all. Without one it is invisible to a
                13C experiment (water, protons, phosphate, O2), so MFA has no information
                about it beyond stoichiometry.
  is_transport  the same metabolite appears on both sides in different compartments. Such
                an antiport swaps a species for itself and is close to label-neutral at
                steady state, so it too is weakly constrained by MIDs.
  nullspace     row norm of the reaction in an orthonormal basis of null(S) — how much of
                the reaction's flux lives in a direction stoichiometry cannot pin down.
  loop          same norm but for null(S) restricted to *internal* reactions: the
                thermodynamically infeasible cycles. A reaction on such a cycle can run at
                any rate without changing net metabolism, which is the strongest single
                predictor here — it is the only covariate that anti-correlates with both
                methods' R², whereas raw nullspace does not.

Usage:
    python paper/analysis/benchmark/flux_disagreement.py paper/.results/i_orientalis_20260614

Writes <run_dir>/output/flux_disagreement.csv and prints the top-N table.
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from plot_test_performance import load_benchmark, per_flux_r2, R2_THRESHOLD  # noqa: E402
from sigmaflux.ai.data import load_data  # noqa: E402

# Reactions below this typical |flux| are numerically negligible: their spread across
# samples is ~1e-3 mmol/gDW/h, both methods are pure noise there, and they otherwise
# dominate any disagreement ranking without carrying biological meaning.
FLUX_FLOOR = 0.1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--flux-floor", type=float, default=FLUX_FLOOR,
                   help=f"Ignore reactions with median |flux| below this (default {FLUX_FLOOR})")
    p.add_argument("--top", type=int, default=30, help="Rows to print (default 30)")
    return p.parse_args()


def structural_covariates(model_json: pathlib.Path) -> pd.DataFrame:
    """has_atom_map / is_transport / nullspace participation, per reaction."""
    mdl = json.load(open(model_json))
    rxs = mdl["reactions"]
    mets = sorted({m for r in rxs for m in r["metabolites"]})
    mi = {m: i for i, m in enumerate(mets)}

    S = np.zeros((len(mets), len(rxs)))
    for j, r in enumerate(rxs):
        for m, c in r["metabolites"].items():
            S[mi[m], j] = c

    def _null_row_norms(mat):
        _, sv, vt = np.linalg.svd(mat, full_matrices=True)
        rank = int((sv > max(mat.shape) * np.finfo(float).eps * sv.max()).sum())
        return np.linalg.norm(vt[rank:].T, axis=1)

    # null(S): flux directions with no net metabolite change, which stoichiometry alone
    # cannot determine. The row norm is how exposed a reaction is to that freedom.
    nullspace = _null_row_norms(S)

    # Restricting to internal reactions leaves only the closed cycles — a loop can carry
    # any flux without net turnover, so nothing in the data pins it down.
    is_exchange = np.array([r["id"].startswith("EX_") or len(r["metabolites"]) == 1
                            for r in rxs])
    loop = np.zeros(len(rxs))
    loop[~is_exchange] = _null_row_norms(S[:, ~is_exchange])

    def compartment_stripped(m):
        return re.sub(r"_[a-z]$", "", m)

    rows = []
    for j, r in enumerate(rxs):
        atom_map = (r.get("annotation") or {}).get("atom_mapping") or ""
        subs = {compartment_stripped(m) for m, c in r["metabolites"].items() if c < 0}
        prods = {compartment_stripped(m) for m, c in r["metabolites"].items() if c > 0}
        rows.append({
            "reaction":     r["id"],
            "has_atom_map": bool(atom_map),
            "is_transport": bool(subs & prods),
            "nullspace":    nullspace[j],
            "loop":         loop[j],
        })
    return pd.DataFrame(rows).set_index("reaction")


def build_table(run_dir: pathlib.Path, flux_floor: float) -> pd.DataFrame:
    labeling, flux, _ = load_data(str(run_dir))
    flux.columns = range(flux.shape[1])
    df_mfa, _ = load_benchmark(run_dir / "mfa_benchmark", flux)
    df_ai, _ = load_benchmark(run_dir / "ai_benchmark", flux)

    piv = lambda d, v: d.pivot_table(index="reaction", columns="sample", values=v)
    gt_m, pm = piv(df_mfa, "gt"), piv(df_mfa, "pred")
    pa = piv(df_ai, "pred")
    # MFA drops the samples it could not solve, so compare the two methods only where both
    # produced a prediction — otherwise the disagreement mixes in a different sample set.
    rxn = gt_m.index.intersection(pa.index)
    smp = gt_m.columns.intersection(pa.columns)
    G, PM, PA = gt_m.loc[rxn, smp].values, pm.loc[rxn, smp].values, pa.loc[rxn, smp].values

    med_gt = np.median(np.abs(G), axis=1)
    tbl = pd.DataFrame({
        "med_gt":       med_gt,
        "disagree_abs": np.median(np.abs(PM - PA), axis=1),
        "mfa_err":      np.median(np.abs(PM - G), axis=1),
        "sf_err":       np.median(np.abs(PA - G), axis=1),
    }, index=rxn)
    tbl["disagree_rel"] = tbl.disagree_abs / np.clip(med_gt, 1e-12, None)

    # R² uses the project-standard Q-Q outlier cut so these numbers match the benchmark
    # figures; the error columns above are medians and need no filtering.
    tbl["r2_mfa"] = per_flux_r2(df_mfa)
    tbl["r2_sf"] = per_flux_r2(df_ai)

    tbl = tbl.join(structural_covariates(run_dir / "model_prepared_for_mfa.json"))

    rel = run_dir / "ensemble" / "reliability_table.csv"
    if rel.exists():
        ens = pd.read_csv(rel).set_index("reaction")
        tbl["u_rel"] = ens["u_rel"]          # calibrated 95% half-width / flux range
        tbl["n_members"] = _n_members(run_dir)
    else:
        print(f"  (no {rel} — run ensemble_calibration.py for uncertainty columns)")

    tbl["winner"] = np.where(tbl.mfa_err < tbl.sf_err, "MFA", "SigmaFlux")
    good_mfa, good_sf = tbl.r2_mfa > R2_THRESHOLD, tbl.r2_sf > R2_THRESHOLD
    tbl["trust"] = np.select(
        [good_mfa & good_sf, good_sf & ~good_mfa, good_mfa & ~good_sf],
        ["either", "SigmaFlux", "MFA"], default="neither")

    # Carrying no carbon is tested FIRST, because it is the more fundamental property and
    # the two classes overlap: water and Pi transport sit on cycles too, yet SigmaFlux
    # recovers them (R² ≈ 0.94) since they are rigidly coupled to growth through the
    # biomass equation. Ordering cycle-first would file them under "cycle" and blur the
    # one class that is genuinely hopeless.
    #
    # That class is "carbon cycle": a reaction carrying carbon atoms *and* sitting on an
    # internal cycle. The cycle can run at any rate without changing net metabolism (the
    # three mitochondrial carriers in I. orientalis sum to an exact null vector), and
    # because it only permutes identical species it barely perturbs labeling either — so
    # nothing in the data constrains it and neither method recovers it.
    on_cycle = tbl.loop > tbl.loop.quantile(0.90)
    tbl["mechanism"] = np.select(
        [~tbl.has_atom_map, on_cycle, tbl.is_transport],
        ["13C-blind", "carbon cycle", "transport/antiport"],
        default="carbon-mapped internal")
    return tbl[tbl.med_gt > flux_floor].sort_values("disagree_abs", ascending=False)


def _n_members(run_dir: pathlib.Path) -> int:
    return len(list((run_dir / "ensemble").glob("model_*.pt")))


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC of `score` for separating `positive` — Mann-Whitney, ties averaged."""
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def validate_uncertainty(tbl: pd.DataFrame) -> None:
    """Can the ensemble spread alone flag the untrustworthy fluxes?

    This is the only trust signal in the table that survives on real data — ground truth,
    and therefore r2_sf and `winner`, do not exist there. So the question that matters is
    whether low spread actually implies low error. Scored with -u_rel so that higher =
    predicted-better, giving an AUC above 0.5 when the ensemble is informative.
    """
    print(f"\n=== ENSEMBLE UNCERTAINTY ({int(tbl.n_members.iloc[0])} members) ===")
    print("Rank correlation with the quantities it should track:")
    print(tbl[["u_rel", "sf_err", "r2_sf", "disagree_abs", "loop"]]
          .corr(method="spearman")["u_rel"].round(3).to_string())

    score = -tbl["u_rel"].to_numpy()
    print("\nAUC of ensemble spread alone (no ground truth) for predicting:")
    print(f"  SigmaFlux R² > {R2_THRESHOLD}          : {_auc(score, (tbl.r2_sf > R2_THRESHOLD).to_numpy()):.3f}")
    print(f"  reaction is trustworthy at all : {_auc(score, (tbl.trust != 'neither').to_numpy()):.3f}")
    print(f"  SigmaFlux beats MFA            : {_auc(score, (tbl.winner == 'SigmaFlux').to_numpy()):.3f}")

    print("\nMedian calibrated uncertainty by mechanism:")
    print(tbl.groupby("mechanism")["u_rel"].median().sort_values()
          .round(3).to_string())

    # The ensemble is *overconfident* on cycle reactions: every member is fit to the same
    # prior, so they agree on that prior's mean even where the flux is unidentifiable and
    # the agreement means nothing. The structural cycle flag catches exactly those cases,
    # which is why the two signals must be combined as a conjunction rather than averaged
    # into one score — the combined ranking AUC barely moves, but the filter below does.
    print("\nDeployable rule (both terms computable without ground truth):")
    low_u = tbl.u_rel < tbl.u_rel.median()
    for on_cycle in (False, True):
        m = low_u & ((tbl.mechanism == "carbon cycle") == on_cycle)
        if not m.any():
            continue
        print(f"  low ensemble spread & {'ON' if on_cycle else 'not on'} a carbon cycle: "
              f"n={int(m.sum()):3d}  median R²_SF={tbl[m].r2_sf.median():+.3f}  "
              f"R²>{R2_THRESHOLD} for {100 * (tbl[m].r2_sf > R2_THRESHOLD).mean():.0f}%")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    tbl = build_table(run_dir, args.flux_floor)

    out = run_dir / "output"
    out.mkdir(exist_ok=True)
    tbl.to_csv(out / "flux_disagreement.csv")

    cols = ["med_gt", "disagree_abs", "disagree_rel", "mfa_err", "sf_err",
            "r2_mfa", "r2_sf", "winner", "trust", "has_atom_map", "is_transport",
            "nullspace", "loop", "mechanism"]
    if "u_rel" in tbl:
        cols.append("u_rel")
    print(f"\n{len(tbl)} reactions with median |flux| > {args.flux_floor}\n")
    print(f"=== TOP {args.top} DISAGREEMENTS ===")
    print(tbl[cols].head(args.top).to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\nSigmaFlux closer to truth on {100 * (tbl.winner == 'SigmaFlux').mean():.0f}% "
          f"of these reactions; trust verdict: "
          + ", ".join(f"{k}={v}" for k, v in tbl.trust.value_counts().items()))

    print("\n=== WHY THEY DISAGREE ===")
    grp = tbl.groupby("mechanism").agg(n=("med_gt", "size"),
                                   disagree=("disagree_abs", "median"),
                                   r2_mfa=("r2_mfa", "median"),
                                   r2_sf=("r2_sf", "median"),
                                   sf_closer=("winner", lambda s: (s == "SigmaFlux").mean()),
                                   trust_sf=("trust", lambda s: (s == "SigmaFlux").mean()),
                                   trust_none=("trust", lambda s: (s == "neither").mean()))
    print(grp.sort_values("disagree", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))

    num = [c for c in ["nullspace", "loop", "disagree_abs", "mfa_err", "sf_err",
                       "r2_mfa", "r2_sf", "u_rel"] if c in tbl]
    print("\nSpearman correlations of the structural covariates:")
    print(tbl[num].corr(method="spearman")[["nullspace", "loop"]].round(3).to_string())

    hi = tbl.loop > tbl.loop.quantile(0.90)
    lo = tbl.loop < tbl.loop.quantile(0.50)
    print(f"\n  top-decile cycle reactions (n={int(hi.sum())}): "
          f"disagree={tbl[hi].disagree_abs.median():.3f}  "
          f"R2 MFA={tbl[hi].r2_mfa.median():+.3f}  SF={tbl[hi].r2_sf.median():+.3f}")
    print(f"  bottom-half cycle reactions (n={int(lo.sum())}): "
          f"disagree={tbl[lo].disagree_abs.median():.3f}  "
          f"R2 MFA={tbl[lo].r2_mfa.median():+.3f}  SF={tbl[lo].r2_sf.median():+.3f}")

    if "u_rel" in tbl:
        validate_uncertainty(tbl)
    print(f"\nSaved: {out / 'flux_disagreement.csv'}")


if __name__ == "__main__":
    main()
