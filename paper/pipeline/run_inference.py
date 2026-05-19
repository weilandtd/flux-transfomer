"""
Predict fluxes from experimental MID measurements.

Inputs
------
  df_L.csv   — labeling data (one row per isotopomer measurement)
  df_Vm.csv  — measured exchange fluxes (optional)

Output
------
  predictions.csv — Flux Name, Value (one row per reaction)

Input CSV formats
-----------------
df_L:
    Tracer Name,Metabolite ID,Isotopomer,Fraction
    U-13C glucose (50%),pyr,M+0,0.452
    U-13C glucose (50%),pyr,M+1,0.301
    U-13C glucose (50%),pyr,M+2,0.183
    U-13C glucose (50%),pyr,M+3,0.064
    1,2-13C glucose (50%),pyr,M+0,0.210
    ...

    Tracer names must match names listed in the config YAML (labeling.tracers[*].name).
    Metabolite IDs must match the COBRA model (e.g. pyr, akg, oaa, g6p, ...).
    Isotopomers: M+0, M+1, M+2, ...

    Metabolites absent from this file are automatically masked — the model
    predicts without seeing their labeling data (robust behaviour trained via
    met_mask_prob augmentation).

df_Vm (optional — but use it if these were measured):
    Flux Name,Value
    EX_glc__D_e,-10.5
    EX_etoh_e,3.2
    EX_ac_e,0.1
    EX_glyc_e,0.05
    EX_succ_e,0.0
    BIOMASS,0.15

    Flux names must match the measured_fluxes list in the config YAML.
    Use sf.expected_measured_fluxes() to see the full list.

Multiple samples
----------------
Add a 'Sample' column (integer) to both CSVs.  The output will also have
a 'Sample' column.  All samples are predicted in one forward pass.

Usage
-----
python paper/pipeline/run_inference.py \
    --config paper/configs/i_orientalis.yaml \
    --checkpoint <output_dir>/pretrained_model/imisor_transformer.pt \
    --labeling <output_dir>/df_L_<organism>_<condition>_<timestamp>.csv \
    --measured-fluxes <output_dir>/df_Vm_<organism>_<condition>_<timestamp>.csv \
    --data-dir <output_dir>/pretrained_model \
    --out <output_dir>/predictions.csv

    # Show expected tracer names:
    python paper/pipeline/run_inference.py ... --show-tracers

    # Show measured flux IDs to provide in df_Vm:
    python paper/pipeline/run_inference.py ... --show-measured-fluxes
"""

import argparse
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.inference import SigmaFlux


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config',      required=True,  help='Path to config YAML')
    parser.add_argument('--checkpoint',  required=True,  help='Path to .pt checkpoint')
    parser.add_argument('--labeling',    default=None,   help='df_L CSV file')
    parser.add_argument('--measured-fluxes', default=None, metavar='CSV',
                        help='df_Vm CSV file (optional)')
    parser.add_argument('--out',         default=None,   help='Output CSV (default: stdout)')
    parser.add_argument('--data-dir',    default=None,
                        help='Path to HDF5 batch files dir (overrides checkpoint data_dir)')
    parser.add_argument('--show-tracers', action='store_true',
                        help='Print expected tracer names and exit')
    parser.add_argument('--show-measured-fluxes', action='store_true',
                        help='Print measured flux IDs to provide in df_Vm and exit')
    args = parser.parse_args()

    sf = SigmaFlux(args.config, args.checkpoint, data_dir=args.data_dir)

    if args.show_tracers:
        print("Expected tracer names (use exactly these strings in df_L 'Tracer Name' column):")
        for t in sf.expected_tracers():
            print(f"  {t}")
        return

    if args.show_measured_fluxes:
        ids = sf.expected_measured_fluxes()
        if ids:
            print("Measured flux IDs to provide in df_Vm 'Flux Name' column:")
            for r in ids:
                print(f"  {r}")
        else:
            print("This checkpoint has no measured_fluxes (df_Vm not needed).")
        return

    if args.labeling is None:
        parser.error("--labeling is required (unless using --show-* flags)")

    df_L = pd.read_csv(args.labeling)
    df_Vm = pd.read_csv(args.measured_fluxes) if args.measured_fluxes else None

    result = sf.predict(df_L, df_Vm=df_Vm)

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Saved → {args.out}  ({len(result)} rows)")
    else:
        print(result.to_string(index=False))


if __name__ == '__main__':
    main()
