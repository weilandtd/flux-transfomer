"""Point estimates and 95% intervals for one condition of a pretrained model.

    python run_inference.py i_orientalis B1

Reads tests/data/interface_test/<model>/input/ and writes
tests/data/interface_test/<model>/output/predictions_<condition>.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sigmaflux.ai import load_model
from sigmaflux.ai.uncertainty import sample_inputs, summarize_draws
from sigmaflux.pipeline.run_inference_workflow import predict_draws

ROOT = Path(__file__).resolve().parent.parent
N_BOOTSTRAP = 200

model, condition = sys.argv[1:3]
data = ROOT / "tests/data/interface_test" / model
sf = load_model(ROOT / f"models/pretrained/{model}-v1.4.1.sfx")

# One CSV per tracer, named labeling_measurements_<tracer name>.csv.
PREFIX = "labeling_measurements_"
mids = pd.concat(pd.read_csv(f).assign(**{"Tracer Name": f.stem.removeprefix(PREFIX)})
                 for f in sorted((data / "input").glob(PREFIX + "*.csv")))
mids = mids[mids["Condition"] == condition]

# Resolver stand-in: experimental compound name -> model metabolite id.
names = pd.read_csv(data / "input/compound_mapping.csv").rename(
    columns={"experimental_id": "Compound", "model_id": "Metabolite ID"})
df_L = mids.merge(names, on="Compound").rename(columns={"Replicate": "Sample"})

fluxes = pd.read_csv(data / "input/measured_fluxes.csv")
fluxes = fluxes[fluxes["Condition"] == condition]
df_Vm = fluxes[["Flux Name", "Value"]]
df_err = fluxes[["Flux Name", "Std"]].rename(columns={"Std": "Value"})

# Bootstrap replicates across tracers and resample fluxes from their Std, then
# summarize the ensemble predictions into a point estimate and interval.
df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=np.random.default_rng(0),
                                        n_bootstrap=N_BOOTSTRAP, bounds=sf.flux_bounds())
raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))
result = summarize_draws(raw, n_draws=N_BOOTSTRAP, group_cols=["Type", "Flux Name"])
out = data / "output"
out.mkdir(exist_ok=True)
result.to_csv(out / f"predictions_{condition}.csv", index=False)
