"""Fit classical MFA to *real* labeling data with x3cflux, and see which answer it supports.

Asks one question: fed the measured MIDs and the measured exchange fluxes, does an
independent MFA implementation land on the published fit or somewhere else? The
M. musculus naive T cell needs this because its measured glucose uptake (-11.93) and the
published fit (-0.975) disagree by 12x, and the AI prediction sits with the measurement.
Whichever of the two x3cflux reproduces is the one the data actually supports.

Unlike ``run_mfa_data.py`` this fits the run directory's ``model_prepared_for_mfa.json`` --
the model the AI predictions live in -- so all three flux vectors are directly comparable
and the already-validated ``model_facts.json`` conversion applies unchanged.

Usage:
    python paper/analysis/diagnostics/mfa_real_data_x3cflux.py \
        --config paper/configs/m_musculus_t_cell.yaml \
        --run    paper/.results/m_musculus_t_cell_20260831 \
        --out    paper/.results/mfa_data_mm_20260901 \
        --conditions N A --n-starts 8
"""

# cyipopt links libspral, which needs the conda env's libstdc++; importing torch first pins
# the system copy and the later x3cflux import fails. This must precede every other import
# -- see sigmaflux.x3cflux.__init__.
import sys

from sigmaflux.x3cflux import preimport

if not preimport():
    sys.exit("x3cflux is not installed: conda env create -f environment.yml")

import argparse
import json
import os

import numpy as np
import pandas as pd
from optlang.symbolics import Zero

from sigmaflux.pipeline._benchmark import (
    build_mfa_context, forward_labeling, load_prepared_model,
)
from sigmaflux.pipeline._paths import load_config, mapping_csv_for, resolve_path
from sigmaflux.utils import read_xlsx_extflux, read_xlsx_mid_data
from sigmaflux.x3cflux.facts import model_facts
from sigmaflux.x3cflux.fit import X3CFluxBackend

#: Floor on a measured flux's sigma, so a value reported as exactly 0 with zero error
#: (naive ``DNA_c``) does not become an infinitely stiff constraint.
FLUX_SIGMA_FLOOR = 1e-3


def tracer_keys(labeling_index, tracer_cfg):
    """Map each labeling-index tracer key onto its workbook sheet name.

    ``build_labeling_index`` keys a tracer by ``str(Tracer)``, which carries no sheet
    name; ``build_experiments`` builds one experiment per ``labeling.tracers`` entry in
    order, so the correspondence is positional.

    Args:
        labeling_index: the model's ``(tracer, metabolite, 'M+k')`` tuples.
        tracer_cfg: the config's ``labeling.tracers`` list.
    """
    seen = list(dict.fromkeys(t for t, _, _ in labeling_index))
    if len(seen) != len(tracer_cfg):
        raise ValueError(f"{len(seen)} tracers in the EMU index but {len(tracer_cfg)} "
                         f"in the config; the positional mapping would be wrong")
    return {key: cfg.get("name", key) for key, cfg in zip(seen, tracer_cfg)}


def measured_labeling(df_L, labeling_index, atom_counts, sheet_for):
    """Measured MID values aligned to a *restricted* labeling index.

    x3cflux scores a metabolite as a whole spectrum -- ``build_measurements`` demands
    M+1..M+n complete for every metabolite it is handed. Real data covers a metabolite
    partially or not at all, so the index is filtered to the (tracer, metabolite) pairs
    that are fully measured, and only those reach the fit.

    Args:
        df_L: measured MIDs from :func:`read_xlsx_mid_data`, replicates already averaged.
        labeling_index: the model's full ``(tracer, metabolite, 'M+k')`` tuples.
        atom_counts: metabolite id -> carbon count.
        sheet_for: labeling-index tracer key -> workbook sheet name.

    Returns ``(kept_index, values, kept_positions, dropped)``.
    """
    lookup = {(r["Tracer Name"], r["Metabolite ID"], r["Isotopomer"]): float(r["Fraction"])
              for _, r in df_L.iterrows()}

    by_pair = {}
    for pos, (trc, met, iso) in enumerate(labeling_index):
        by_pair.setdefault((trc, met), []).append((pos, iso))

    kept, values, kept_pos, dropped = [], [], [], []
    for (trc, met), entries in by_pair.items():
        n = atom_counts.get(met, 0)
        sheet = sheet_for[trc]
        found = {iso: lookup.get((sheet, met, iso)) for _, iso in entries}
        if n == 0 or len(entries) != n or any(v is None for v in found.values()):
            dropped.append((sheet, met))
            continue
        for pos, iso in sorted(entries, key=lambda e: int(e[1][2:])):
            kept.append((trc, met, iso))
            values.append(found[iso])
            kept_pos.append(pos)
    return kept, np.asarray(values, dtype=float), np.asarray(kept_pos, dtype=int), dropped


def reconciled_start(cobra_model, flux_table, reaction_ids):
    """Least-absolute-deviation reconciliation of the measured fluxes, as a feasible start.

    Minimises the sigma-weighted L1 deviation from the measurements subject to S.v = 0 and
    the model bounds. This is the one start that is both data-driven and at the right
    magnitude for a low-flux state. The published fit is deliberately never used as a
    start, so no start can leak the reference the fit is being compared against.

    Args:
        cobra_model: the prepared model.
        flux_table: rows of ``reaction``/``mean``/``std``.
        reaction_ids: reaction order of the returned vector.
    """
    with cobra_model as m:
        obj = Zero
        for _, row in flux_table.iterrows():
            rxn = m.reactions.get_by_id(row["reaction"])
            sigma = max(float(row["std"]), FLUX_SIGMA_FLOOR)
            pos = m.problem.Variable(f"dp_{rxn.id}", lb=0)
            neg = m.problem.Variable(f"dn_{rxn.id}", lb=0)
            m.add_cons_vars([pos, neg])
            m.add_cons_vars([m.problem.Constraint(
                rxn.flux_expression - pos + neg,
                lb=float(row["mean"]), ub=float(row["mean"]), name=f"dev_{rxn.id}")])
            obj += (pos + neg) / sigma
        m.objective = m.problem.Objective(obj, direction="min")
        sol = m.optimize()
        if sol.status != "optimal":
            return None, None
        return sol.fluxes.reindex(reaction_ids).to_numpy(float), float(m.objective.value)


def chi2(sim_lab, obs_lab, noise_std, sim_flux, obs_flux, flux_sigma):
    """Chi-square of a fitted state against the measurements it was fitted to."""
    lab = float((((sim_lab - obs_lab) / noise_std) ** 2).sum())
    flx = float((((sim_flux - obs_flux) / flux_sigma) ** 2).sum())
    return lab + flx, lab, flx


def run_condition(cond, cfg, root, run_dir, out_dir, args, ctx, samples):
    """Fit one condition from several starts and write the best."""
    cobra_model, mfa_model, reaction_ids, labeling_to, labeling_index, facts, sheet_for = ctx
    data_dir = resolve_path(cfg["experiment"]["data_dir"], root)
    mid_file = os.path.join(data_dir, f"{cfg['name']}_{cond}.xlsx")
    ext_file = os.path.join(data_dir, f"{cfg['name']}_ExtFlux.xlsx")

    print(f"\n{'=' * 70}\n  Condition {cond}\n{'=' * 70}")

    df_L = read_xlsx_mid_data(xlsx_path=mid_file, mapping_csv=resolve_path(mapping_csv_for(cfg), root),
                              growth_condition=args.gc)
    df_L = df_L.groupby(["Tracer Name", "Metabolite ID", "Isotopomer"],
                        as_index=False)["Fraction"].mean()

    df_Vm, df_err = read_xlsx_extflux(xlsx_path=ext_file, condition=cond,
                                      flux_names=cfg["training"]["measured_fluxes"],
                                      return_errors=True, growth_condition=args.gc)
    flux_table = pd.DataFrame({
        "reaction": df_Vm["Flux Name"].to_numpy(),
        "mean": df_Vm["Value"].to_numpy(float),
        "std": np.maximum(df_err["Value"].to_numpy(float), FLUX_SIGMA_FLOOR),
    })
    flux_table = flux_table[flux_table["reaction"].isin(set(reaction_ids))].reset_index(drop=True)

    atom_counts = {m["id"]: len(m["atom_pattern"]) for m in facts["metabolites"]}
    kept_index, obs_lab, kept_pos, dropped = measured_labeling(
        df_L, labeling_index, atom_counts, sheet_for)
    n_pairs = len({(t, m) for t, m, _ in kept_index})
    print(f"  MID tokens fitted : {len(kept_index)} over {n_pairs} (tracer, metabolite) pairs")
    print(f"  dropped pairs     : {len(dropped)} (absent or partially measured)")
    print(f"  fluxes fitted     : {len(flux_table)}")
    if not len(kept_index):
        print("  no fully-measured metabolite — skipping")
        return None

    flux_ids = flux_table["reaction"].tolist()
    obs_flux = flux_table["mean"].to_numpy(float)
    flux_sigma = flux_table["std"].to_numpy(float)

    backend = X3CFluxBackend(facts, kept_index, flux_ids, args.noise_std,
                             use_flux_bounds=not args.no_bounds, verbose=args.verbose)
    simulator, fluxmap = backend.simulator_for(obs_lab, obs_flux, flux_sigma)

    # Starts spanning scales, so the answer is not an artefact of where the fit began.
    starts = []
    recon, recon_dev = reconciled_start(cobra_model, flux_table, reaction_ids)
    if recon is not None:
        print(f"  L1 reconciliation of the measured fluxes: {recon_dev:.1f} sigma total")
        starts.append(("reconciled-L1", recon))
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(len(samples), size=min(args.n_starts, len(samples)), replace=False):
        starts.append((f"train-sample-{int(i)}",
                       samples.iloc[int(i)].reindex(reaction_ids).fillna(0.0).to_numpy(float)))

    results, best_fluxes = [], {}
    for label, v_init in starts:
        mfa_model.reset_initial_fluxes(
            initial_flux_distribution=pd.Series(v_init, index=reaction_ids))
        r_init = mfa_model.reversibility.detach().cpu().numpy().astype(float)
        v_fit, r_fit, info = backend.fit(simulator, fluxmap, v_init, r_init,
                                         max_iter=args.max_iter, isolate=not args.no_isolate)
        if v_fit is None:
            print(f"  start {label:<18s} CRASHED ({info.get('crashed')})")
            continue
        sim_lab = np.asarray(forward_labeling(mfa_model, v_fit, r_fit, labeling_to),
                             dtype=float)[kept_pos]
        sim_flux = pd.Series(v_fit, index=reaction_ids).reindex(flux_ids).to_numpy(float)
        total, lab, flx = chi2(sim_lab, obs_lab, args.noise_std, sim_flux, obs_flux, flux_sigma)
        print(f"  start {label:<18s} chi2={total:12.1f}  (labeling {lab:11.1f}, flux {flx:9.1f})"
              f"  status={info.get('status')}")
        results.append(dict(start=label, chi2=total, chi2_labeling=lab, chi2_flux=flx,
                            status=info.get("status")))
        best_fluxes[label] = pd.Series(v_fit, index=reaction_ids)

    if not results:
        print("  every start crashed — no fit for this condition")
        return None

    best = min(results, key=lambda d: d["chi2"])
    print(f"\n  best: {best['start']}  chi2={best['chi2']:.1f}")
    v_best = best_fluxes[best["start"]]
    pd.DataFrame({"Flux Name": reaction_ids, "Value": v_best.to_numpy(float)}).to_csv(
        os.path.join(out_dir, f"x3cflux_{cfg['name']}_{cond}.csv"), index=False)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"x3cflux_starts_{cfg['name']}_{cond}.csv"), index=False)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True, help="run directory holding the prepared model")
    p.add_argument("--out", required=True, help="output directory; never a run directory")
    p.add_argument("--conditions", nargs="+", default=["N", "A"])
    p.add_argument("--gc", type=int, default=1)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--noise-std", type=float, default=0.01, help="MID measurement sigma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bounds", action="store_true", help="drop the cobra FVA bounds")
    p.add_argument("--no-isolate", action="store_true", help="do not fork each start")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    cfg, root = load_config(args.config, None)
    run_dir = resolve_path(args.run, root)
    out_dir = resolve_path(args.out, root)
    os.makedirs(out_dir, exist_ok=True)

    cobra_model, mapped_met_ids = load_prepared_model(run_dir, cfg, root)
    samples = pd.read_csv(os.path.join(run_dir, "flux_samples.csv"), index_col=0)
    mfa_model, reaction_ids, _, labeling_to, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"], samples.iloc[0], "MFA_real",
        with_backward=False)
    labeling_index = [tuple(k) for k in labeling_index]
    facts = model_facts(cobra_model, mfa_model, mapped_met_ids, cfg)
    ctx = (cobra_model, mfa_model, reaction_ids, labeling_to, labeling_index, facts,
           tracer_keys(labeling_index, cfg["labeling"]["tracers"]))

    summary = {}
    for cond in args.conditions:
        best = run_condition(cond, cfg, root, run_dir, out_dir, args, ctx, samples)
        if best is not None:
            summary[cond] = best
    with open(os.path.join(out_dir, "x3cflux_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote → {out_dir}")


if __name__ == "__main__":
    main()
