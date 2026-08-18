"""
Build metabolite name mapping CSVs for each organism.

Run this once per organism (or whenever the model or compound list changes).
Edit the USER CONFIGURATION section below with your actual file paths, then run:

    .venv/bin/python models/build_metabolite_mapping_files.py

Each entry produces one CSV mapping experimental compound names (from the MID
xlsx) to model metabolite IDs (from the AMM xlsx).  The output CSV is used by
prepare_experiment_data.py and run_inference.py via --mapping-csv.

An entry may also carry `lcms_knowns`, the LC/MS platform's compound library.  Its
matches are unioned into the output, so the mapping describes everything the
platform *can* measure rather than only what one experiment did.  The measured
short names still have to be present or the MID workbooks would not load — the
sheets say `Gly` and `aKG`, the library says `Glycine` and `Oxoglutaric acid`.
"""

import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sigmaflux.utils import build_data_name_maps, match_lcms_knowns


def _compound_names_all_sheets(*mid_xlsx):
    """Every compound measured across every tracer sheet of the given MID workbooks
    (a single sheet per file would miss compounds unique to one tracer).

    Args:
        mid_xlsx: paths to MID workbooks, each scanned across all its sheets.
    """
    import pandas as pd

    names = set()
    for path in mid_xlsx:
        full = os.path.join(os.path.dirname(__file__), '..', '..', path)
        for sheet in pd.ExcelFile(full).sheet_names:
            df = pd.read_excel(full, sheet)
            if "Compound" in df.columns:
                names.update(df["Compound"].dropna().astype(str))
    return sorted(names)


def _measured_compound_names(*mapping_csvs):
    """Compound names measured for organisms that already have a mapping CSV
    (used for R. toruloides, which has no MID data of its own yet).

    Args:
        mapping_csvs: paths to already-built mapping CSVs to pool names from.
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
    {
        "name": "M_musculus_T_Cell",
        "mid_xlsx": None,
        "compound_names": _compound_names_all_sheets(
            "data/M_musculus_T_Cell/M_musculus_T_Cell_A.xlsx",
            "data/M_musculus_T_Cell/M_musculus_T_Cell_N.xlsx",
        ),
        "amm_xlsx":   "models/atomistic/M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx",
        "output_csv": "models/atomistic/M_musculus_T_Cell/M_musculus_T_Cell_mapping.csv",
        # This model has no cytosolic succinate and writes aconitate with a
        # stereochemistry tag, so the shared table's assumptions do not hold.
        "overrides": {
            "Suc":  ["succ_m"],
            "Acon": ["acon_C_m"],
        },
        "lcms_knowns": "data/HILIC_LCMS_knowns.csv",
        "model_json":  "models/atomistic/M_musculus_T_Cell/mmTcell_annotated_hierarchical.json",
        "gem_xlsx":    "models/atomistic/M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx",
    },
]

# ============================================================
# END USER CONFIGURATION
# ============================================================

def _add_lcms_knowns(org):
    """Union the LC/MS measurable space into an already-written mapping CSV.

    Args:
        org: this organism's config dict from ORGANISMS.
    """
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    from cobra.io import load_json_model

    model = load_json_model(os.path.join(root, org["model_json"]))
    knowns = match_lcms_knowns(
        gem_xlsx=os.path.join(root, org["gem_xlsx"]),
        knowns_path=os.path.join(root, org["lcms_knowns"]),
        model_metabolite_ids=[m.id for m in model.metabolites],
    )
    output = os.path.join(root, org["output_csv"])
    measured = pd.read_csv(output)
    merged = (pd.concat([measured, knowns], ignore_index=True)
                .drop_duplicates()
                .sort_values(["experimental_id", "model_id"]))
    merged.to_csv(output, index=False)
    print(f"  + LC/MS knowns: {knowns['model_id'].nunique()} measurable metabolites, "
          f"{len(merged)} mappings total ({len(measured)} measured)")


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
        if org.get("lcms_knowns"):
            _add_lcms_knowns(org)
    print("\nDone.")
