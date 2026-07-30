"""
Build metabolite name mapping CSVs for each organism.

Run this once per organism (or whenever the model or compound list changes).
Edit the USER CONFIGURATION section below with your actual file paths, then run:

    .venv/bin/python models/build_metabolite_mapping_files.py

Each entry produces one CSV mapping experimental compound names (from the MID
xlsx) to model metabolite IDs (from the AMM xlsx).  The output CSV is used by
prepare_experiment_data.py and run_inference.py via --mapping-csv.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sigmaflux.utils import build_data_name_maps


def _measured_compound_names(*mapping_csvs):
    """Compound names measured for organisms that already have a mapping CSV.

    R. toruloides has no MID data yet, so its measurable token space is taken from
    the compounds the same LC/MS platform resolves for the other two yeasts.
    """
    names = set()
    for path in mapping_csvs:
        with open(os.path.join(os.path.dirname(__file__), '..', '..', path)) as f:
            names.update(row["experimental_id"] for row in csv.DictReader(f))
    return sorted(names)

# ============================================================
# USER CONFIGURATION — edit paths for your local setup
# ============================================================

ORGANISMS = [
    {
        "name": "I_orientalis",
        "mid_xlsx": [
            "data/I_orientalis/I_orientalis_C.xlsx",
            "data/I_orientalis/I_orientalis_N.xlsx",
            "data/I_orientalis/I_orientalis_P.xlsx",
            "data/I_orientalis/I_orientalis_batch.xlsx",
        ],
        "amm_xlsx":   "models/atomistic/I_orientalis/imIsor_AMM.xlsx",
        "output_csv": "models/atomistic/I_orientalis/I_orientalis_mapping.csv",
    },
    {
        "name": "S_cerevisiae",
        "mid_xlsx": [
            "data/S_cervesiae/S_cerevisiae_C.xlsx",
            "data/S_cervesiae/S_cerevisiae_N.xlsx",
            "data/S_cervesiae/S_cerevisiae_P.xlsx",
            "data/S_cervesiae/S_cerevisiae_batch.xlsx",
        ],
        "amm_xlsx":   "models/atomistic/S_cerevisiae/scAMM.xlsx",
        "output_csv": "models/atomistic/S_cerevisiae/S_cerevisiae_mapping.csv",
    },
    {
        # No data/R_toruloides yet — the compound list comes from the other two yeasts.
        "name": "R_toruloides",
        "mid_xlsx": None,
        "compound_names": _measured_compound_names(
            "models/atomistic/I_orientalis/I_orientalis_mapping.csv",
            "models/atomistic/S_cerevisiae/S_cerevisiae_mapping.csv",
        ),
        "amm_xlsx":   "models/atomistic/R_toruloides/imRt_AMM.xlsx",
        "output_csv": "models/atomistic/R_toruloides/R_toruloides_mapping.csv",
    },
]

# ============================================================
# END USER CONFIGURATION
# ============================================================

if __name__ == "__main__":
    for org in ORGANISMS:
        print(f"\n=== {org['name']} ===")
        build_data_name_maps(
            experimental_xlsx  = org["mid_xlsx"],
            experimental_names = org.get("compound_names"),
            model_amm_xlsx     = org["amm_xlsx"],
            output_csv         = org["output_csv"],
            overrides          = org.get("overrides"),
        )
    print("\nDone.")
