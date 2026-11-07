"""
Check the GC/MS E. coli core model against the Long & Antoniewicz (2019) data it models.

For each strain, the published MFA fluxes are projected onto the prepared core model's
steady-state space (1/sigma^2-weighted least squares on the published net fluxes of
atom-mapped reactions and BIOMASS, subject to S.v = 0 and the prepared bounds), pushed
through the EMU pass, and the simulated fragment MIDs are compared with the measured ones
after natural-isotope correction. A curation error -- a wrong carbon set, a reversed atom
order, a wrong tracer -- shows up as one fragment, or one tracer, far off the rest.

Exchange fluxes are not published, so every reversible reaction is run at two settings --
at its floor, and at an exchange equal to its net flux plus one -- and the check is
approximate: a few mol% is the scale of agreement to expect, not machine precision. A
fragment that only agrees under exchange (ribose, the OAC-derived amino acids) is read
through a reversible node, not mis-curated.

The second sheet is scored as both [1,6-13C] and U-13C glucose, which is how it was found
to hold [1,6-13C]: the workbooks labelled it U-13C.

Usage:
    python paper/analysis/e_coli_gcms/check_fragments_at_reference.py
    python paper/analysis/e_coli_gcms/check_fragments_at_reference.py --strains 1 4 10 20
"""

import argparse
import os
import tempfile
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from cobra.io import load_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.core.model import MFAModel
from sigmaflux.pipeline._paths import load_config, resolve_path
from sigmaflux.pipeline.generate_training_data import build_experiments, build_targets
from sigmaflux.pipeline.sample_flux import prepare_for_mfa
from sigmaflux.utils import read_reference_fluxes
from sigmaflux.utils.read_xlsx_mid_data import read_xlsx_mid_data

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "paper/configs/e_coli_core_gc.yaml"
MAPPING = ROOT / "models/atomistic/E_coli_core/E_coli_core_fragment_mapping.csv"
DATA = ROOT / "data/E_coli_GC_MS"
STRAINS = pd.read_csv(DATA / "strain_index.csv").set_index("condition")["strain"]

SHEET_12 = "1,2-13C glucose"
SHEET_2 = "1,6-13C glucose"
HYPOTHESES = {
    "U-13C":   [1, 1, 1, 1, 1, 1],
    "1,6-13C": [1, 0, 0, 0, 0, 1],
}
EXCHANGE = ("floor", "moderate")


def published_sigma(value, lower, upper):
    """Standard deviation of each published flux, from its 95% interval.

    Where the interval does not bracket its own value (the workbook's BIOMASS row), 10%
    of the value stands in.
    """
    sigma = (upper - lower) / (2 * 1.96)
    broken = (lower > value) | (upper < value) | sigma.isna()
    sigma[broken] = 0.1 * value[broken].abs()
    return sigma.clip(lower=0.01)


def published_fluxes(model, value, sigma, ridge=0.01):
    """Steady-state flux vector closest to the published net fluxes, or None.

    Each published flux is weighted by 1/sigma^2; reactions the reference does not
    constrain carry a small ``ridge`` penalty, so the fit cannot invent side routes to
    absorb what the core network lacks.

    Returns ``(fluxes, RMS misfit over the fitted reactions, in published sigmas)``.
    """
    S = create_stoichiometric_matrix(model)
    ids = [r.id for r in model.reactions]
    lb = np.array([r.lower_bound for r in model.reactions])
    ub = np.array([r.upper_bound for r in model.reactions])
    mapped = {r.id for r in model.reactions if r.annotation.get("atom_mapping")} | {"BIOMASS"}
    known = [(ids.index(i), value[i], sigma[i]) for i in value.dropna().index
             if i in mapped]
    rows, target, sd = (np.array(x) for x in zip(*known))
    free = [i for i in range(len(ids)) if i not in set(rows)]
    v = cp.Variable(len(ids))
    objective = (cp.sum_squares(cp.multiply(1 / sd, v[rows] - target))
                 + ridge * cp.sum_squares(v[free]))
    problem = cp.Problem(cp.Minimize(objective), [S @ v == 0, v >= lb, v <= ub])
    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=400000)
    if v.value is None:
        return None
    return (pd.Series(v.value, index=ids),
            float(np.sqrt(np.mean(((v.value[rows] - target) / sd) ** 2))))


def simulate(model, fluxes, tracers, fragments, exchange):
    """``{(tracer name, fragment): MID over M+0..M+n}`` at *fluxes*.

    Args:
        exchange: ``'floor'`` for every exchange at its minimum, ``'moderate'`` for an
            exchange of ``|v| + 1`` on every reversible reaction.
    """
    mfa = MFAModel(model, "check", dtype=torch.float64, initial_flux_distribution=fluxes)
    experiments = build_experiments(mfa, tracers)
    targets, _, patterns = build_targets(mfa, restrict_to=set(fragments))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            mfa.prepare(experiments, targets=targets, with_backward=False)
        finally:
            os.chdir(cwd)
    with torch.no_grad():
        if exchange == "floor":
            mfa._reversibility.data.zero_()
        else:
            mfa._reversibility.data.copy_(mfa.fluxes.abs() + 1.0)
        out = mfa().cpu().numpy()
    if out.ndim == 2:
        out = out[None]
    return {(t["name"], f): out[e, :len(patterns[f]) + 1, mfa._metabolite_hash[f]]
            for e, t in enumerate(tracers) for f in fragments}


def measured(gc):
    """``{(sheet, fragment): corrected MID}`` for one strain."""
    df = read_xlsx_mid_data(str(DATA / "EColi_B.xlsx"), str(MAPPING), growth_condition=gc)
    out = {}
    for (sheet, frag), g in df.groupby(["Tracer Name", "Metabolite ID"]):
        g = g.assign(k=g["Isotopomer"].str[2:].astype(int)).sort_values("k")
        out[(sheet, frag)] = g["Fraction"].to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strains", type=int, nargs="+", default=list(range(1, 22)))
    parser.add_argument("--out", default=None, help="write the per-fragment table here")
    args = parser.parse_args()

    cfg, root = load_config(CONFIG)
    model = load_json_model(resolve_path(cfg["model_file"], root))
    prepare_for_mfa(model, cfg)
    fragments = sorted(set(pd.read_csv(MAPPING)["model_id"]))
    path = str(DATA / "E_coli_core_reference_flux.xlsx")
    reference = read_reference_fluxes(path, "B")
    lower = read_reference_fluxes(path, "B", "_mfaLB")
    upper = read_reference_fluxes(path, "B", "_mfaUB")

    rows = []
    for gc in args.strains:
        sigma = published_sigma(reference[gc], lower[gc], upper[gc])
        fit = published_fluxes(model, reference[gc], sigma)
        if fit is None:
            print(f"B{gc} ({STRAINS[f'B{gc}']}): published fluxes infeasible in the core")
            continue
        fluxes, misfit = fit
        data = measured(gc)
        for hyp, labeling in HYPOTHESES.items():
            tracers = [{"tracer_id": "Gluc_e", "labeling": [1, 1, 0, 0, 0, 0],
                        "enrichment": 1.0, "name": SHEET_12},
                       {"tracer_id": "Gluc_e", "labeling": labeling,
                        "enrichment": 1.0, "name": SHEET_2}]
            for exchange in EXCHANGE:
                sim = simulate(model, fluxes, tracers, fragments, exchange)
                for (sheet, frag), mid in sim.items():
                    if (sheet, frag) not in data:
                        continue
                    err = mid - data[(sheet, frag)]
                    rows.append({"gc": gc, "strain": STRAINS[f"B{gc}"],
                                 "flux_misfit": misfit, "hypothesis": hyp,
                                 "exchange": exchange, "sheet": sheet, "fragment": frag,
                                 "rmse": float(np.sqrt(np.mean(err ** 2))),
                                 "max_abs": float(np.abs(err).max())})

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\nFlux misfit of the projected published state (RMS, in published sigmas):")
    print(table.groupby("strain", sort=False)["flux_misfit"].first().round(3).to_string())
    print("\nMID RMSE by sheet and tracer hypothesis (median over strains and fragments):")
    print(table.groupby(["sheet", "hypothesis", "exchange"])["rmse"].median()
               .round(4).to_string())
    best = table[(table["hypothesis"] == "1,6-13C") | (table["sheet"] == SHEET_12)]
    best = best.drop_duplicates(["gc", "sheet", "exchange", "fragment"])
    print("\nPer-fragment RMSE (median over strains; second sheet as 1,6-13C):")
    print(best.pivot_table(index="fragment", columns=["sheet", "exchange"], values="rmse",
                           aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    main()
