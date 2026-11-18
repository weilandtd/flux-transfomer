"""
Labeling of a ¹³C refit constrained to the published MFA fluxes, for the experimental panels.

The MFA counterpart of paper/scripts/experimental_mid.py. The published fits report net
fluxes for part of the network only -- no exchange fluxes, no compartment pool fractions --
so their labeling cannot be simulated directly. Instead, for every growth condition
SigmaFlux was scored on, the network is refit to the measured MIDs with every reported net
flux pinned to the published value. Unreported net fluxes, exchange fluxes and pool
fractions are free, so the result is the best labeling the published flux state admits in
this network.

The published values were fit in other networks and are not exactly a steady state of this
one, so they are first moved to the closest steady state within the model bounds
(``bounded_moma``, unreported fluxes nearly free). Every reported flux is then pinned to
within ``PIN_REL`` (floor ``PIN_ABS``) of that state. ``fluxes.csv`` records how far each
one had to move.

Per model, in ``<run-dir>/experimental_mid_mfa/``:
  tokens.csv    SigmaFlux's ``experimental_mid/tokens.csv`` with ``sim`` replaced by the
                refit's labeling
  fluxes.csv    per growth condition and reported reaction: published value and 95%
                interval, the closest steady state, and the refit's net flux
  fit.csv       per growth condition: Ipopt status, SSR, and how far the reported fluxes
                moved from the published values

The fit starts from the closest steady state. It sees the same tokens SigmaFlux is scored
on, with the replicate std floored at the training labeling noise. Run paper/scripts/experimental_mid.py
first; the fits take minutes per condition, so submit
paper/submit/submit_experimental_mid_mfa.sh.

Usage:
    python paper/scripts/experimental_mid_mfa.py
    python paper/scripts/experimental_mid_mfa.py --models e_coli_core
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.core.data import MassIsotopomerData
from sigmaflux.core.experiment import Experiment
from sigmaflux.core.fbapro import bounded_moma
from sigmaflux.core.measurement_pools import POOL_DRAIN_PREFIX
from sigmaflux.core.mfa_ipopt import fit_mfa_model_ipopt
from sigmaflux.pipeline._benchmark import build_mfa_context, load_prepared_model, pin_threads
from sigmaflux.utils import read_reference_fluxes, reference_orientation

from _figure_common import (
    PREDICTION_SPECS, UNBOUNDED_FLUX, _REPO_ROOT, add_common_args, reference_path,
    resolve_organism, run_dir_for, selected_models)
from experimental_mid import tokens_path

sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from validate_inferred_fluxes import STD_FLOOR  # noqa: E402

TOKEN = ["Tracer", "Metabolite", "Isotopomer"]

#: Objective weight of an unreported flux, relative to a reported one, in the projection.
UNREPORTED_WEIGHT = 1e-4

#: Reactions the atomistic models renamed or split, as the published reactions whose
#: fluxes add up to them. Transketolase is split into half-reactions through ``tktc2_c``;
#: succinate dehydrogenase names its quinone. Published reactions without a counterpart
#: here (biomass, the carbon-free respiratory chain) stay unreported.
RENAMED = {
    "TKT1h_c": ["TKT1_c"], "TKT2h_c": ["TKT2_c"], "TKThalf_c": ["TKT1_c", "TKT2_c"],
    "SUCDq6_m": ["SUCDqx_m"], "SUCDq7_m": ["SUCDqx_m"], "SUCDq9_m": ["SUCDqx_m"],
}

#: Projection bound for a reported reaction allowed to run in a direction.
OPEN_BOUND = 1e5

#: Half-width of the band a reported flux is pinned to: relative, with an absolute floor.
PIN_REL = 0.01
PIN_ABS = 1e-3

#: Columns that describe SigmaFlux's prediction interval, not the refit.
SF_ONLY = ["Value_lo", "Value_hi", "sigma_input", "n_draws"]


def mfa_tokens_path(run):
    return run / "experimental_mid_mfa" / "tokens.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--max-iter", type=int, default=3000)
    return p.parse_args()


def published(org, cond, gc, rxns, orientation):
    """Signed published point estimate and 95% interval, reindexed onto ``rxns``."""
    path = reference_path(org)
    cols = {}
    for name, suffix in (("value", "_mfa"), ("lo", "_mfaLB"), ("hi", "_mfaUB")):
        ref = read_reference_fluxes(path, cond, suffix, unbounded=UNBOUNDED_FLUX)
        cols[name] = (with_renamed(ref[gc], rxns).reindex(rxns)
                      if ref is not None and gc in ref else None)
    if cols["value"] is None:
        return None
    sign = pd.Series(orientation, dtype=float).reindex(rxns).fillna(1.0)
    value = cols["value"] * sign
    lo, hi = (cols[k] * sign if cols[k] is not None else pd.Series(np.nan, index=rxns)
              for k in ("lo", "hi"))
    # A flipped interval swaps ends.
    return value, lo.where(sign > 0, hi), hi.where(sign > 0, lo)


def with_renamed(ref, rxns):
    """``ref`` plus the model's renamed reactions, summed from their published sources."""
    extra = {target: ref.reindex(sources).sum(min_count=len(sources))
             for target, sources in RENAMED.items()
             if target in rxns and target not in ref.index}
    return pd.concat([ref, pd.Series(extra, dtype=float)])


def model_bounds(reactions):
    """Bounds of ``reactions``, with pool drains pinned at 1 so the legs are fractions."""
    lb = np.array([r.lower_bound for r in reactions])
    ub = np.array([r.upper_bound for r in reactions])
    drain = np.array([r.id.startswith(POOL_DRAIN_PREFIX) for r in reactions])
    lb[drain] = ub[drain] = 1.0
    return lb, ub


def pinned_bounds(S, lb, ub, value):
    """Closest steady state to the published values, and bounds pinning the reported
    fluxes to a narrow band around it.

    A reported reaction keeps only its direction from the model bounds, which are the box
    the training data were sampled in, not a property of the network. Unreported reactions
    keep the box.

    Returns ``(closest, lb, ub)``.
    """
    reported = value.notna().to_numpy()
    lb = np.where(reported, np.where(lb >= 0, 0.0, -OPEN_BOUND), lb)
    ub = np.where(reported, np.where(ub <= 0, 0.0, OPEN_BOUND), ub)
    closest = bounded_moma(value.fillna(0.0).to_numpy(), lb, ub, S=S,
                           weights=np.where(reported, 1.0, UNREPORTED_WEIGHT))
    band = np.maximum(PIN_REL * np.abs(closest), PIN_ABS)
    new_lb = np.where(reported, np.maximum(lb, closest - band), lb)
    new_ub = np.where(reported, np.minimum(ub, closest + band), ub)
    return closest, new_lb, new_ub


def measured_experiments(mfa_model, experiments, tokens, std_floor):
    """One Experiment per prepared tracer, carrying that tracer's measured tokens."""
    out = []
    for experiment in experiments:
        tracer = "|".join(str(t) for t in experiment.tracers)
        g = tokens[tokens["Tracer"] == tracer]
        rows = [{"id": r.Metabolite, "mid": int(r.Isotopomer[2:]), "mean": r.measured,
                 "std": max(float(np.nan_to_num(r.measured_std)), std_floor)}
                for r in g.itertuples()]
        # M+0 lets MassIsotopomerData infer the EMU size.
        sums = g.groupby("Metabolite")["measured"].sum()
        rows += [{"id": m, "mid": 0, "mean": max(0.0, 1.0 - s), "std": std_floor}
                 for m, s in sums.items()]
        mid = (MassIsotopomerData(model=mfa_model, table=pd.DataFrame(rows))
               if rows else MassIsotopomerData())
        out.append(Experiment(mid, list(experiment.tracers)))
    return out


def refit(mfa_model, experiments, tokens, lb, ub, start, std_floor, max_iter):
    """Fit exchange, pool and unreported fluxes to the measured tokens within lb/ub."""
    for r, lo, hi in zip(mfa_model.reactions, lb, ub):
        r.bounds = (lo, hi)
    mfa_model.reset_initial_fluxes(start)
    mfa_model.experiments = measured_experiments(mfa_model, experiments, tokens, std_floor)
    return fit_mfa_model_ipopt(mfa_model, max_iter=max_iter)


def model_tokens(run, org, max_iter):
    """Refit labeling tokens and fit diagnostics for every (condition, gc) SigmaFlux was
    scored on."""
    sf = pd.read_csv(tokens_path(run))
    cobra_model, mapped = load_prepared_model(str(run), org["cfg"], org["root"])
    S = create_stoichiometric_matrix(cobra_model)
    rxns = [r.id for r in cobra_model.reactions]
    orientation = {} if org.get("reference_signed") else reference_orientation(org["media"])
    std_floor = org["cfg"].get("training", {}).get("labeling_noise_std") or 0.01

    mfa_model = None
    parts, fits, fluxes = [], [], []
    for (cond, gc), grp in sf.groupby(["condition", "gc"], sort=False):
        pub = published(org, cond, gc, rxns, orientation)
        if pub is None:
            print(f"  {cond}{gc}: skipped — no reference column")
            continue
        value, lo, hi = pub
        if mfa_model is None:
            mfa_model, _, experiments, labeling_to, labeling_index = build_mfa_context(
                cobra_model, org["cfg"]["labeling"]["tracers"], value.fillna(0.0),
                "published_refit", with_backward=True, restrict_targets_to=mapped)
            base_lb, base_ub = model_bounds(mfa_model.reactions)
        closest, lb, ub = pinned_bounds(S, base_lb, base_ub, value)
        reported = value.notna().to_numpy()
        start = pd.Series(closest, index=rxns)

        started = time.perf_counter()
        try:
            res = refit(mfa_model, experiments, grp, lb, ub, start, std_floor, max_iter)
            with torch.no_grad():
                sim = torch.take(mfa_model().reshape(-1), labeling_to).cpu().numpy()
        except RuntimeError as exc:
            print(f"  {cond}{gc}: skipped — {str(exc).splitlines()[0][:120]}")
            continue
        sim = pd.Series(sim, index=labeling_index, name="sim")

        v = mfa_model.fluxes.detach().cpu().numpy()
        pub = value.to_numpy()[reported]
        moved = np.abs(v[reported] - pub)
        rel = moved / np.maximum(np.abs(pub), PIN_ABS)
        out = (grp.drop(columns=["sim", "error", "z", *SF_ONLY], errors="ignore")
                  .merge(sim.reset_index(), on=TOKEN, how="inner"))
        out["error"] = out["sim"] - out["measured"]
        out["z"] = out["error"] / out["measured_std"].fillna(0.0).clip(lower=STD_FLOOR)
        fits.append(dict(condition=cond, gc=gc, converged=res["converged"], n_iter=res["n_iter"],
                         ssr=res["loss"], n_tokens=len(out), n_reported=int(reported.sum()),
                         median_abs_change=float(np.median(moved)),
                         max_abs_change=float(moved.max()),
                         median_rel_change=float(np.median(rel)),
                         frac_within_5pct=float(np.mean(rel <= 0.05)),
                         seconds=time.perf_counter() - started))
        print(f"  {cond}{gc}: {len(out)}/{len(grp)} tokens, SSR {res['loss']:.4g}, "
              f"converged {res['converged']}; reported fluxes vs published: median change "
              f"{np.median(moved):.3g} ({np.median(rel):.1%}), max {moved.max():.3g}, "
              f"{np.mean(rel <= 0.05):.0%} within 5% [{fits[-1]['seconds']:.0f}s]")
        parts.append(out)
        fluxes.append(pd.DataFrame({
            "condition": cond, "gc": gc, "reaction": rxns, "published": value.to_numpy(),
            "published_lo": lo.to_numpy(), "published_hi": hi.to_numpy(),
            "closest_steady_state": closest, "refit": v})[reported])
    if not parts:
        return None, None, None
    return (pd.concat(parts, ignore_index=True), pd.DataFrame(fits),
            pd.concat(fluxes, ignore_index=True))


def main():
    args = parse_args()
    pin_threads()
    models = [m for m in selected_models(args) if m["key"] in PREDICTION_SPECS]

    print("── Refitting to the measured MIDs with the published MFA fluxes pinned ──")
    for model in models:
        run = run_dir_for(model, args.run_suffix)
        if not tokens_path(run).is_file():
            print(f"  {model['label']}: skipped — no {tokens_path(run)}; "
                  f"run experimental_mid.py first")
            continue
        print(f"\n{model['label']}  ({run.name})")
        try:
            df, fits, fluxes = model_tokens(run, resolve_organism(run), args.max_iter)
        except Exception as exc:
            print(f"  {model['label']}: skipped — {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:160]}")
            continue
        if df is None:
            print(f"  {model['label']}: skipped — no token simulated")
            continue
        path = mfa_tokens_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.assign(model=model["label"], key=model["key"]).to_csv(path, index=False)
        fits.assign(model=model["label"], key=model["key"]).to_csv(
            path.with_name("fit.csv"), index=False)
        fluxes.assign(model=model["label"], key=model["key"]).to_csv(
            path.with_name("fluxes.csv"), index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
