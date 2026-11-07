"""
Close the loop on a real experiment: Data -> SigmaFlux -> fluxes -> simulation -> MIDs.

A flux prediction is only worth reporting if it is a flux state the network can actually
hold and if it reproduces the labeling it was derived from. Nothing in the pipeline checks
either, so this script does, against real measured data rather than held-out synthetic
samples (which is what ``sigmaflux-benchmark`` already covers).

Three families of check, in the order they can invalidate each other:

1. **Feasibility** -- is the prediction a flux state at all? Stoichiometric balance
   ``S.v = 0``, the FVA bounding box, and irreversible reactions pointing the way their
   bounds allow. Checked for every (draw, member) prediction and for both ensemble
   aggregates. Each member predicts ``v = y @ Z_full.T`` and so lies in the null space by
   construction; the **mean** the predictions CSV reports is a convex combination of those
   and stays there, while the **median** it used to report is not and can leave it
   entirely. Both are kept so the gap stays measured rather than assumed.

2. **Simulated MID validity** -- ``sum(M+1..M+N) <= 1`` per (tracer, metabolite), i.e. the
   implied M+0 is a real fraction. The EMU solver clamps negative contributions at zero
   (``core/calc_labeling``), so a group summing above one is a genuine defect, not noise.

3. **Agreement with the data** -- simulated vs measured MID per token: RMSE, worst offender,
   and a z-score against the measured replicate spread, written to
   ``simulated_vs_measured_<organism>.csv``. The parity scatter this used to draw is now
   paper/scripts/plot_experimental.py, which builds the paper figure from the same
   primitives.

Only the maximum-likelihood input (the replicate mean) is forward-simulated by default --
one EMU pass per ensemble member plus one each for the mean and median states. The
feasibility checks are cheap and run over every bootstrap draw. ``--sim-draws`` adds
simulations of individual draws.

Exit status is 0 only when every invariant holds; the per-check verdict is printed at the
end, so a partial failure names which one.

Usage:
    python paper/analysis/diagnostics/validate_inferred_fluxes.py \\
        --config     paper/configs/e_coli_core.yaml \\
        --checkpoint paper/.results/e_coli_core_lcms_20260827/ensemble \\
        --data-dir   data/E_coli_LC_MS \\
        --out-dir    paper/.results/e_coli_core_lcms_20260827/validation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.ai.uncertainty import DEFAULT_N_BOOTSTRAP, sample_inputs, _medoid_weights
from sigmaflux.pipeline._benchmark import (
    build_mfa_context, forward_labeling, load_prepared_model,
)
from sigmaflux.pipeline._paths import (
    add_root_argument, load_config, mapping_csv_for, resolve_path,
)
from sigmaflux.pipeline.check_mfa_model import measurable_metabolites, prepared_model
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions,
    load_condition_inputs, predict_draws,
)

#: Absolute slack on the bound and balance checks, matching tests/test_core_mfa_ipopt.py.
TOL = 1e-6
#: Floor on the measured replicate std when forming a z-score, so a metabolite whose
#: replicates happened to agree exactly does not produce an infinite score.
STD_FLOOR = 1e-3


# ── Inputs ────────────────────────────────────────────────────────────────────

def prepared_network(cfg, root, checkpoint, out_dir):
    """The prepared model and the metabolites the platform can see.

    Loads whichever run directory holds the ``model_prepared_for_mfa.json`` the checkpoint
    was trained against -- the checkpoint's own directory, or its parent for an
    ``ensemble/`` subdirectory -- via the same
    :func:`~sigmaflux.pipeline._benchmark.load_prepared_model` the benchmarks use. Falls
    back to re-deriving the model from the config, which re-solves FVA and writes a stray
    artifact into *out_dir*, only when no run directory carries one.
    """
    base = os.path.dirname(os.path.abspath(checkpoint))
    for run_dir in (base, os.path.dirname(base)):
        if os.path.isfile(os.path.join(run_dir, "model_prepared_for_mfa.json")):
            return load_prepared_model(run_dir, cfg, root)
    print("Prepared network: rebuilding from config (no model_prepared_for_mfa.json found)")
    model = prepared_model(cfg, root, out_dir)
    return model, measurable_metabolites(cfg, root)


# ── Prediction -> flux state ──────────────────────────────────────────────────

def split_predictions(raw, sf, reaction_ids):
    """Pivot raw predictions into (flux, reversibility) frames aligned to *reaction_ids*.

    ``predict_draws`` has already tagged each row flux vs reversibility -- a reaction can
    appear as both, so the name alone does not identify a quantity.
    ``reversibility_names`` is a strict subset of the reactions (15 of 79 on E. coli core),
    so reindexing leaves NaN at every reaction the model predicts no reversibility for.
    Those fill with zero, which is inert downstream because ``MFAModel.reversible_fluxes``
    multiplies by the reversibility mask first -- but they are *not* predictions, so
    :func:`feasibility` is told which positions are real via the returned mask rather than
    scoring a fill value as if the model had produced it.

    Args:
        raw: long predictions from :func:`predict_draws`, with a 'Type' column.
        sf: the predictor, for ``reversibility_names``.
        reaction_ids: model reaction order the columns are aligned to.

    Returns ``(flux_wide, rev_wide, rev_predicted)``.
    """
    raw = raw.copy()
    if "Member" not in raw.columns:
        raw["Member"] = 0
    keys = ["Sample", "Member"]

    def _pivot(sub):
        return (sub.pivot_table(index=keys, columns="Flux Name", values="Value")
                   .reindex(columns=reaction_ids).fillna(0.0))

    rev_predicted = np.isin(reaction_ids, list(sf.reversibility_names))
    return (_pivot(raw[raw["Type"] == "flux"]),
            _pivot(raw[raw["Type"] == "reversibility"]),
            rev_predicted)


def aggregate_state(flux_wide, rev_wide, sample, how):
    """Combine the ensemble members at one sample into a single flux state.

    ``how="mean"`` is the aggregate that stays a flux state: every member predicts
    ``v = y @ Z_full.T`` for its own basis coefficients, so an average of members is
    ``mean(y) @ Z_full.T`` -- still in the null space, hence still ``S.v = 0``, and still
    inside the (convex) bounding box whenever every member is -- which is why it is what
    ``EnsembleSigmaFlux.predict`` and the predictions CSV report. ``how="median"`` was the
    previous aggregate; a per-reaction median is not a convex combination, so it can leave
    the null space entirely. ``how="medoid"``/``"medoid_convex"`` (see
    :func:`sigmaflux.ai.uncertainty._medoid_weights`) are candidate replacements for the
    mean that stay closer to the "typical" member instead of blending uniformly across all
    of them. All are kept so the differences stay measured, not assumed.

    Args:
        flux_wide: per-(Sample, Member) fluxes, columns in model reaction order.
        rev_wide: matching reversibilities.
        sample: the input draw to aggregate over.
        how: "mean", "median", "medoid" or "medoid_convex".
    """
    if how in ("mean", "median"):
        return (getattr(flux_wide.xs(sample, level="Sample"), how)(axis=0),
                getattr(rev_wide.xs(sample, level="Sample"), how)(axis=0))
    if how in ("medoid", "medoid_convex"):
        fw = flux_wide.xs(sample, level="Sample")
        rw = rev_wide.xs(sample, level="Sample")
        w = _medoid_weights(fw.to_numpy(float), how=how)
        return (pd.Series(w @ fw.to_numpy(float), index=fw.columns),
                pd.Series(w @ rw.to_numpy(float), index=rw.columns))
    raise ValueError(f"Unknown how={how!r}")


def as_state_frames(flux_series, rev_series, sample):
    """Wrap one (flux, reversibility) pair in the (Sample, Member) frames :func:`feasibility` takes."""
    ix = pd.MultiIndex.from_tuples([(sample, -1)], names=["Sample", "Member"])
    return flux_series.to_frame().T.set_index(ix), rev_series.to_frame().T.set_index(ix)


# ── Feasibility ───────────────────────────────────────────────────────────────

def feasibility(mfa_model, flux_wide, rev_wide, rev_predicted):
    """Per-flux-state bound, direction, reversibility-floor and mass-balance violations.

    Returns a frame with one row per (Sample, Member) and the counts/worst magnitude of
    each violation class. Irreversible reactions get both checks: the bounding box already
    encodes their direction, but reporting the sign violation separately answers the
    question "does the prediction point the right way" directly. The two reversibility
    checks only look at positions the model actually predicts (*rev_predicted*), so the
    zero-fill for the other reactions is never scored as a violation.
    """
    lb, ub = (t.cpu().numpy() for t in mfa_model.get_flux_bounds())
    mask = mfa_model._reversibility_mask.cpu().numpy()
    S = mfa_model.stoichiometry.cpu().numpy()
    irrev = mask == 0
    fwd_only, rev_only = irrev & (lb >= 0), irrev & (ub <= 0)
    bad_rev = irrev & rev_predicted           # predicted reversibility where none is allowed
    floored = (mask > 0) & rev_predicted      # predicted reversibility that must clear the floor

    rows = []
    for key, v in flux_wide.iterrows():
        v = v.to_numpy(float)
        r = rev_wide.loc[key].to_numpy(float)
        below, above = lb - v, v - ub
        rows.append({
            "Sample": key[0], "Member": key[1],
            "n_below_lb":    int((below > TOL).sum()),
            "n_above_ub":    int((above > TOL).sum()),
            "worst_bound":   float(max(below.max(), above.max(), 0.0)),
            "n_wrong_sign":  int(((fwd_only & (v < -TOL)) | (rev_only & (v > TOL))).sum()),
            "n_irrev_rev":   int((np.abs(r[bad_rev]) > TOL).sum()),
            "n_rev_below_floor": int((r[floored] < mfa_model.min_rev_flux - TOL).sum()),
            "max_Sv":        float(np.abs(S @ v).max()),
        })
    return pd.DataFrame(rows)


def worst_bound_offenders(mfa_model, v, reaction_ids, top=10):
    """The *top* reactions furthest outside their bounds for one flux vector."""
    lb, ub = (t.cpu().numpy() for t in mfa_model.get_flux_bounds())
    v = np.asarray(v, dtype=float)
    excess = np.maximum(lb - v, v - ub)
    order = np.argsort(-excess)[:top]
    return pd.DataFrame({
        "reaction":  [reaction_ids[i] for i in order],
        "value":     v[order],
        "lb":        lb[order],
        "ub":        ub[order],
        "violation": excess[order],
    })


# ── Simulated MIDs ────────────────────────────────────────────────────────────

def simulate(mfa_model, flux_series, rev_series, labeling_to, labeling_index):
    """One EMU forward pass for a flux state, as a Series indexed by the labeling tokens."""
    sim = forward_labeling(mfa_model,
                           np.nan_to_num(flux_series.to_numpy(float), nan=0.0),
                           np.nan_to_num(rev_series.to_numpy(float), nan=1.0),
                           labeling_to)
    return pd.Series(sim, index=labeling_index, name="sim")


def composition(sim):
    """Per-(Tracer, Metabolite) ``sum(M+1..M+N)`` and the M+0 it implies."""
    grouped = sim.groupby(level=["Tracer", "Metabolite"]).sum().rename("sum_M1_MN")
    out = grouped.reset_index()
    out["implied_M0"] = 1.0 - out["sum_M1_MN"]
    return out


def measured_table(df_L, name_to_id):
    """Measured MIDs as mean/std per token, keyed by the internal tracer id.

    M+0 is dropped to match the simulated index, which carries M+1..M+N only.
    """
    df = df_L[df_L["Isotopomer"] != "M+0"].copy()
    df["Tracer"] = df["Tracer Name"].map(name_to_id)
    unknown = df.loc[df["Tracer"].isna(), "Tracer Name"].unique()
    if len(unknown):
        print(f"  WARNING: tracer name(s) not in the config: {sorted(unknown)}")
    agg = (df.dropna(subset=["Tracer"])
             .groupby(["Tracer", "Metabolite ID", "Isotopomer"], sort=False)["Fraction"]
             .agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"Metabolite ID": "Metabolite",
                              "mean": "measured", "std": "measured_std",
                              "count": "n_replicates"}))
    return agg


def agreement(sim, measured):
    """Join simulated against measured MIDs and score the difference per token."""
    df = measured.merge(sim.reset_index(), on=["Tracer", "Metabolite", "Isotopomer"], how="inner")
    df["error"] = df["sim"] - df["measured"]
    df["z"] = df["error"] / df["measured_std"].fillna(0.0).clip(lower=STD_FLOOR)
    return df.sort_values("error", key=np.abs, ascending=False)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report_feasibility(feas, label):
    """Summarize a feasibility frame; returns True when every state is a valid flux state.

    A reversibility below ``min_rev_flux`` is reported but does not fail the check:
    ``MFAModel.reversible_fluxes`` floors it before the EMU pass, so the forward
    simulation never sees the under-floor value. The other four are real defects -- a
    state outside the box, pointing the wrong way, carrying exchange on an irreversible
    reaction, or off the null space is not a flux state at all.
    """
    n = len(feas)
    counts = {c: int((feas[c] > 0).sum()) for c in
              ("n_below_lb", "n_above_ub", "n_wrong_sign", "n_irrev_rev", "n_rev_below_floor")}
    print(f"\n{label}  ({n} flux state{'s' if n != 1 else ''})")
    print(f"  outside bounding box : {counts['n_below_lb'] + counts['n_above_ub']:>5} states"
          f"   worst violation {feas['worst_bound'].max():.3e}")
    print(f"  wrong direction      : {counts['n_wrong_sign']:>5} states"
          f"   ({int(feas['n_wrong_sign'].sum())} reaction-instances)")
    print(f"  reversibility on an irreversible reaction : {counts['n_irrev_rev']:>5} states")
    print(f"  reversibility below min_rev_flux          : {counts['n_rev_below_floor']:>5} states"
          f"   (floored by the simulator, not a defect)")
    print(f"  mass balance max|S.v| : {feas['max_Sv'].max():.3e}")
    hard = sum(v for k, v in counts.items() if k != "n_rev_below_floor")
    return hard == 0 and feas["max_Sv"].max() < 1e-4


def report_composition(comp, label):
    """Report the sum(M+1..M+N) <= 1 invariant; returns True when it holds."""
    bad = comp[comp["sum_M1_MN"] > 1.0 + TOL]
    print(f"\n{label}")
    print(f"  groups                : {len(comp)}")
    print(f"  max sum(M+1..M+N)     : {comp['sum_M1_MN'].max():.6f}")
    print(f"  min implied M+0       : {comp['implied_M0'].min():.6f}")
    print(f"  groups summing > 1.0  : {len(bad)}")
    if len(bad):
        print(bad.nlargest(10, "sum_M1_MN").to_string(index=False))
    return len(bad) == 0


def report_agreement(df, label):
    """Report simulated-vs-measured MID agreement."""
    print(f"\n{label}")
    print(f"  tokens compared       : {len(df)}")
    if df.empty:
        print("  nothing to compare -- simulated and measured token sets do not overlap")
        return
    print(f"  RMSE                  : {np.sqrt((df['error'] ** 2).mean()):.4f}")
    print(f"  MAE                   : {df['error'].abs().mean():.4f}")
    print(f"  max |error|           : {df['error'].abs().max():.4f}")
    print(f"  median |z|            : {df['z'].abs().median():.2f}")
    print("\n  worst 12 tokens:")
    print(df.head(12)[["Tracer", "Metabolite", "Isotopomer",
                       "measured", "sim", "error", "z"]].to_string(index=False))
    per_met = (df.assign(sq=df["error"] ** 2).groupby("Metabolite")["sq"]
                 .mean().pow(0.5).sort_values(ascending=False).rename("rmse"))
    print("\n  per-metabolite RMSE (worst 15):")
    print(per_met.head(15).to_string())


def report_member_spread(sims, measured, label):
    """RMSE against the measured MIDs for each ensemble member individually.

    The aggregate can fit badly for two very different reasons: every member is wrong, or
    the members disagree and their average lands between two good answers. The spread of
    per-member RMSE separates those.
    """
    rmse = {k: np.sqrt((agreement(v, measured)["error"] ** 2).mean())
            for k, v in sims.items() if k.startswith("member_")}
    if not rmse:
        return
    vals = np.array(list(rmse.values()))
    best = min(rmse, key=rmse.get)
    print(f"\n{label}")
    print(f"  members               : {len(vals)}")
    print(f"  RMSE best / median / worst : {vals.min():.4f} / "
          f"{np.median(vals):.4f} / {vals.max():.4f}   (best: {best})")


# ── Driver ────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",     required=True, help="Organism YAML config")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint, ensemble dir or .sfx")
    parser.add_argument("--data-dir",   required=True, help="Directory containing experimental xlsx")
    parser.add_argument("--out-dir",    required=True, help="Directory for the CSV and plots")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to run (default: all discovered from --data-dir)")
    parser.add_argument("--growth-conditions", nargs="+", type=int, default=None,
                        help="Growth conditions to run (default: all found in the MID file)")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="Override experiment.n_bootstrap for the input draws")
    parser.add_argument("--sim-draws", type=int, default=0,
                        help="Also forward-simulate this many individual input draws "
                             "(costs one EMU pass per draw per member); the "
                             "maximum-likelihood input is always simulated")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the input draws")
    return add_root_argument(parser)


def main(args):
    cfg, root = load_config(args.config, args.root)
    config_path = cfg["_config_path"]
    organism = cfg.get("name", "organism")
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None
    n_bootstrap = (args.n_bootstrap
                   or cfg.get("experiment", {}).get("n_bootstrap", DEFAULT_N_BOOTSTRAP))

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mapping_path = resolve_path(mapping_csv_for(cfg), root)
    ext_rel = cfg.get("experiment", {}).get("ext_flux")
    ext_file = resolve_path(ext_rel, root) if ext_rel else _find_ext_flux_file(data_dir)
    if ext_file is None:
        sys.exit(f"ERROR: no *_ExtFlux.xlsx found in {data_dir}")

    all_files = _discover_conditions(data_dir)
    if not all_files:
        sys.exit(f"ERROR: no condition xlsx files (*_C.xlsx etc.) found in {data_dir}")
    conditions = args.conditions or sorted(all_files)
    missing = [c for c in conditions if c not in all_files]
    if missing:
        sys.exit(f"ERROR: conditions {missing} not in {data_dir}. Available: {sorted(all_files)}")

    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Data dir    : {data_dir}")
    print(f"Mapping     : {mapping_path}")
    print(f"Conditions  : {conditions}")
    print(f"Bootstrap   : {n_bootstrap} cross-tracer replicate draws")

    sf = load_predictor(config_path, args.checkpoint)
    members = getattr(sf, "members", [sf])
    name_to_id = sf.tracer_ids()
    print(f"Members     : {len(members)}")
    print(f"Tracers     : {list(name_to_id)}")

    cobra_model, targets = prepared_network(cfg, root, args.checkpoint, out_dir)
    print(f"EMU targets : {len(targets) if targets else 'all atom-mapped metabolites'}")

    rng = np.random.default_rng(args.seed)
    mfa_model = reaction_ids = labeling_to = labeling_index = None
    per_token, per_state, checks = [], [], []

    for condition in conditions:
        mid_file = all_files[condition]
        gcs = args.growth_conditions or _inspect_growth_conditions(mid_file)
        _rule(f"Condition {condition}  ({os.path.basename(mid_file)})   growth conditions {gcs}")

        for gc in gcs:
            # Inputs, draws and prediction all go through run_inference_workflow's own
            # helpers, so this validates the pipeline rather than a parallel copy of it.
            df_L, df_Vm, df_err = load_condition_inputs(
                mid_file, ext_file, condition, gc, mapping_path, flux_names,
                model=cobra_model)

            reps = df_L.groupby("Tracer Name")["Sample"].nunique()
            print(f"\n-- growth condition {gc} --")
            print(f"  replicates per tracer : {reps.to_dict()}")

            df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=rng,
                                                    n_bootstrap=n_bootstrap)
            n_draws = df_L_draws["Sample"].max()   # the MLE row sits at the top index
            raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))

            flux_wide, rev_wide, rev_predicted = split_predictions(
                raw, sf, [r.id for r in cobra_model.reactions])

            if mfa_model is None:
                v0 = flux_wide.iloc[0]
                mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
                    cobra_model, cfg["labeling"]["tracers"], v0,
                    "validate", with_backward=False, restrict_targets_to=targets)

            # ── 1. Feasibility, over every draw and member ────────────────────
            feas = feasibility(mfa_model, flux_wide, rev_wide, rev_predicted)
            feas.insert(0, "GrowthCondition", gc)
            feas.insert(0, "Condition", condition)
            per_state.append(feas)
            checks.append((f"{condition}{gc} per-member flux states feasible",
                           report_feasibility(feas, "[1] Feasibility of every per-member prediction")))

            states = {how: aggregate_state(flux_wide, rev_wide, n_draws, how)
                      for how in ("mean", "median")}
            for tag, label in (("mean",   "[1b] Ensemble MEAN   (what the predictions CSV reports)"),
                               ("median", "[1c] Ensemble MEDIAN (the previous aggregate, for contrast)")):
                v, r = states[tag]
                f = feasibility(mfa_model, *as_state_frames(v, r, n_draws), rev_predicted)
                passed = report_feasibility(f, label)
                # Only the mean is reported, so only the mean is held to the invariant;
                # the median is printed to keep the difference between them measured.
                if tag == "mean":
                    checks.append((f"{condition}{gc} ensemble mean feasible", passed))
                if f["worst_bound"].max() > TOL:
                    print(worst_bound_offenders(mfa_model, v.to_numpy(float),
                                                reaction_ids).to_string(index=False))

            # ── 2 & 3. Simulate the maximum-likelihood input ──────────────────
            sims = {tag: simulate(mfa_model, v, r, labeling_to, labeling_index)
                    for tag, (v, r) in states.items()}
            for m in sorted(flux_wide.index.get_level_values("Member").unique()):
                sims[f"member_{m}"] = simulate(mfa_model, flux_wide.loc[(n_draws, m)],
                                               rev_wide.loc[(n_draws, m)],
                                               labeling_to, labeling_index)
            for d in range(min(args.sim_draws, n_draws)):
                dv, dr = aggregate_state(flux_wide, rev_wide, d, "mean")
                sims[f"draw_{d}"] = simulate(mfa_model, dv, dr, labeling_to, labeling_index)

            measured = measured_table(df_L, name_to_id)
            report_composition(
                composition(measured.set_index(["Tracer", "Metabolite", "Isotopomer"])["measured"]
                                    .rename("sim")),
                "[2a] Measured MID (control -- should already sum to <= 1)")
            for tag, num in (("mean", "2b"), ("median", "2c")):
                ok_sum = report_composition(
                    composition(sims[tag]),
                    f"[{num}] Simulated MID from the ensemble-{tag.upper()} flux state")
                if tag == "mean":
                    checks.append((f"{condition}{gc} simulated MID sums to <= 1", ok_sum))

            joined = agreement(sims["mean"], measured)
            report_agreement(joined, "[3a] Simulated (ensemble-MEAN flux) vs measured MID")
            report_agreement(agreement(sims["median"], measured),
                             "[3b] Simulated (ensemble-MEDIAN flux) vs measured MID")
            report_member_spread(sims, measured, "[3c] Per-member fit -- can any member explain the data?")

            # 'sim'/'error'/'z' are the mean state; the median is carried alongside for
            # comparison, since that is the number the predictions CSV actually reports.
            key = ["Tracer", "Metabolite", "Isotopomer"]
            joined = joined.merge(
                agreement(sims["median"], measured)[key + ["sim"]]
                    .rename(columns={"sim": "sim_median"}), on=key, how="left")
            joined.insert(0, "GrowthCondition", gc)
            joined.insert(0, "Condition", condition)
            per_token.append(joined)

    tok_path = os.path.join(out_dir, f"simulated_vs_measured_{organism}.csv")
    pd.concat(per_token, ignore_index=True).to_csv(tok_path, index=False)
    st_path = os.path.join(out_dir, f"flux_state_checks_{organism}.csv")
    pd.concat(per_state, ignore_index=True).to_csv(st_path, index=False)

    _rule("VERDICT")
    print(f"per-token comparison -> {tok_path}")
    print(f"per-state checks     -> {st_path}\n")
    width = max(len(name) for name, _ in checks)
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}")
    failed = [name for name, passed in checks if not passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


def cli(argv=None):
    """Console-script entry point."""
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(cli())
