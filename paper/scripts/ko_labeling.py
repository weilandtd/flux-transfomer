"""
Labeling implied by SigmaFlux's knockout-panel predictions, against the measured MIDs.

For every strain of the Long & Antoniewicz (2019) panel, the flux state inference
reported -- plain (``predictions/``) and FBApro-projected (``predictions_fbapro/``) -- is
pushed through the EMU pass with its predicted reversibilities, and the simulated
fragment MIDs are joined against the natural-isotope-corrected measured ones. FBApro
leaves reversibilities untouched, so the two methods differ only in net flux.

Writes ``<run>/ko_labeling/tokens.csv``: one row per (method, strain, tracer, fragment,
isotopologue) with the measured and simulated fraction. M+0 is excluded, as in
experimental_mid.py. Prints the per-strain RMSE (mol%) and R² of simulated vs measured.

Usage:
    python paper/scripts/ko_labeling.py
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions,
    load_condition_inputs)

from _figure_common import _REPO_ROOT, r2_linear, resolve_organism

sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from validate_inferred_fluxes import agreement, measured_table  # noqa: E402

RUN = _REPO_ROOT / "paper/.results/e_coli_core_gc_20260924"
CONDITION = "B"
#: Prediction directory per method.
METHODS = {"sigmaflux": "predictions", "fbapro": "predictions_fbapro"}
TOKEN = ["Tracer", "Metabolite", "Isotopomer"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=pathlib.Path, default=RUN)
    return parser.parse_args()


def reported_state(path, gc, sf):
    """(flux_arr, rev_arr) of one strain's reported state, in the predictor's order."""
    df = pd.read_csv(path)
    df = df[(df["Condition"] == CONDITION) & (df["GrowthCondition"] == gc)]
    block = {t: d.set_index("Flux Name")["Value"] for t, d in df.groupby("Type")}
    return (block["flux"].reindex(sf.flux_names).to_numpy()[None],
            block["reversibility"].reindex(sf.reversibility_names).to_numpy()[None])


def main():
    args = parse_args()
    run = args.run.resolve()
    org = resolve_organism(run)
    cfg = org["cfg"]
    data_dir = resolve_path(org["data"], org["root"])
    mid_file = _discover_conditions(data_dir)[CONDITION]
    ext_flux = cfg.get("experiment", {}).get("ext_flux")
    ext_file = resolve_path(ext_flux, org["root"]) if ext_flux else _find_ext_flux_file(data_dir)
    mapping_path = resolve_path(mapping_csv_for(cfg), org["root"])
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None

    sf = load_predictor(cfg["_config_path"], None, data_dir=str(run))
    member = sf.members[0]
    context = member.mfa_context()
    name_to_id = sf.tracer_ids()
    files = {m: sorted((run / d).glob("predictions_*.csv"))[-1] for m, d in METHODS.items()}

    parts = []
    for gc in _inspect_growth_conditions(mid_file):
        df_L, _, _ = load_condition_inputs(mid_file, ext_file, CONDITION, gc, mapping_path,
                                           flux_names)
        measured = measured_table(df_L, name_to_id)
        for method, path in files.items():
            mids = member.simulate_mid(*reported_state(path, gc, sf), [0], context=context)
            sim = (mids.rename(columns={"Metabolite ID": "Metabolite", "Fraction": "sim"})
                       .assign(Tracer=lambda d: d["Tracer Name"].map(name_to_id))
                       .set_index(TOKEN)["sim"])
            parts.append(agreement(sim, measured).assign(method=method, gc=gc))
    tokens = pd.concat(parts, ignore_index=True)
    out = run / "ko_labeling" / "tokens.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_csv(out, index=False)
    print(f"Saved: {out}")

    # A state whose EMU solve fails has no labeling; score only what simulated.
    summary = (tokens.dropna(subset=["sim"]).groupby(["gc", "method"])
                     .apply(lambda d: pd.Series({
                         "rmse_mol%": 100 * np.sqrt(np.mean(d["error"] ** 2)),
                         "r2": r2_linear(d["sim"], d["measured"])}))
                     .unstack("method")
                     .reindex(sorted(tokens["gc"].unique())))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print(summary.median().round(3).to_string())


if __name__ == "__main__":
    main()
