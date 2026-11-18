"""Interface-test inputs for one pretrained model: python build_interface_inputs.py <model>

Writes tests/data/interface_test/<model>/input/ from the model's training config:
labeling_measurements_<tracer name>.csv per tracer, measured_fluxes.csv and
compound_mapping.csv. MIDs come out as the model consumes them, natural-abundance
corrected where the mapping gives a fragment formula.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

from sigmaflux.pipeline._paths import load_config, mapping_csv_for, resolve_path
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions)
from sigmaflux.utils import read_xlsx_extflux, read_xlsx_mid_data

ROOT = Path(__file__).resolve().parent.parent
model = sys.argv[1]
cfg, root = load_config(ROOT / f"paper/configs/{model}.yaml", ROOT)
experiment = cfg["experiment"]
data_dir = resolve_path(experiment["data_dir"], root)
ext_file = (resolve_path(experiment["ext_flux"], root) if experiment.get("ext_flux")
            else _find_ext_flux_file(data_dir))
flux_names = cfg["training"].get("measured_fluxes") or None
mapping = pd.read_csv(resolve_path(mapping_csv_for(cfg), root))

out = ROOT / f"tests/data/interface_test/{model}/input"
out.mkdir(parents=True, exist_ok=True)
mapping[["experimental_id", "model_id"]].to_csv(out / "compound_mapping.csv", index=False)

formulas = mapping.drop_duplicates("experimental_id").set_index("experimental_id")
formulas = formulas.drop(columns="model_id")


def identity_mapping(mid_file, path):
    """Every compound the workbook reports, mapped onto itself so its name survives."""
    compounds = pd.concat(pd.read_excel(mid_file, sheet_name=None).values())["Compound"]
    ids = compounds.dropna().astype(str).unique()
    formulas.reindex(ids).rename_axis("experimental_id").reset_index().assign(
        model_id=ids).to_csv(path, index=False)


mids, fluxes = [], []
with tempfile.NamedTemporaryFile("w", suffix=".csv") as identity:
    for condition, mid_file in sorted(_discover_conditions(data_dir).items()):
        identity_mapping(mid_file, identity.name)
        for gc in _inspect_growth_conditions(mid_file):
            label = f"{condition}{gc}"
            df_L = read_xlsx_mid_data(mid_file, identity.name, growth_condition=gc)
            df_L["Sample"] = df_L["Sample"].rank(method="dense").astype(int)
            mids.append(df_L.assign(Condition=label))
            df_Vm, df_err = read_xlsx_extflux(ext_file, condition=condition,
                                              flux_names=flux_names, return_errors=True,
                                              growth_condition=gc)
            fluxes.append(df_Vm.merge(df_err, on=["Sample", "Flux Name"],
                                      suffixes=("", "_err")).assign(Condition=label))

mids = pd.concat(mids).rename(columns={"Sample": "Replicate", "Metabolite ID": "Compound"})
for tracer, df in mids.groupby("Tracer Name", sort=False):
    df[["Condition", "Replicate", "Compound", "Isotopomer", "Fraction"]].to_csv(
        out / f"labeling_measurements_{tracer}.csv", index=False)
fluxes = pd.concat(fluxes).rename(columns={"Value_err": "Std"})
fluxes[["Condition", "Flux Name", "Value", "Std"]].to_csv(out / "measured_fluxes.csv",
                                                          index=False)
print(f"Wrote {out.relative_to(ROOT)}: {mids['Tracer Name'].nunique()} tracer(s), "
      f"{mids['Condition'].nunique()} condition(s)")
