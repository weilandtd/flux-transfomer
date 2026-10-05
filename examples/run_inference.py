"""Point estimates and 95% intervals for one condition: python run_inference.py B1"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sigmaflux.ai import load_model
from sigmaflux.ai.uncertainty import sample_inputs, summarize_draws
from sigmaflux.pipeline.run_inference_workflow import predict_draws

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "models/pretrained/i_orientalis-v1.4.0.sfx"
NAME_MAP = ROOT / "models/atomistic/I_orientalis/I_orientalis_experiment_mapping.csv"
N_BOOTSTRAP = 200

condition = sys.argv[1]
sf = load_model(MODEL)

# One CSV per tracer, named labeling_measurements_<tracer name>.csv.
PREFIX = "labeling_measurements_"
mids = pd.concat(pd.read_csv(f).assign(**{"Tracer Name": f.stem.removeprefix(PREFIX)})
                 for f in sorted((HERE / "data").glob(PREFIX + "*.csv")))
mids = mids[mids["Condition"] == condition]

# Resolver stand-in: experimental compound name -> model metabolite id.
names = pd.read_csv(NAME_MAP).rename(columns={"experimental_id": "Compound",
                                              "model_id": "Metabolite ID"})
df_L = mids.merge(names, on="Compound").rename(columns={"Replicate": "Sample"})

fluxes = pd.read_csv(HERE / "data/measured_fluxes.csv")
fluxes = fluxes[fluxes["Condition"] == condition]
df_Vm = fluxes[["Flux Name", "Value"]]
df_err = fluxes[["Flux Name", "Std"]].rename(columns={"Std": "Value"})

# Bootstrap replicates across tracers and resample fluxes from their Std, then
# summarize the ensemble predictions into a point estimate and interval.
df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=np.random.default_rng(0),
                                        n_bootstrap=N_BOOTSTRAP, bounds=sf.flux_bounds())
raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))
result = summarize_draws(raw, n_draws=N_BOOTSTRAP, group_cols=["Type", "Flux Name"])
result.to_csv(f"predictions_{condition}.csv", index=False)
