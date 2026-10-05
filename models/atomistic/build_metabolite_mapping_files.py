"""
Build metabolite name mapping CSVs for each organism.

Run this once per organism (or whenever the model changes). Edit the USER
CONFIGURATION section below with your actual file paths, then run:

    .venv/bin/python models/build_metabolite_mapping_files.py

Each entry produces one CSV mapping LC/MS platform compound names (from
data/HILIC_LCMS_knowns.csv) to model metabolite IDs, filtered to rows whose atom
map tracks the compound's full carbon count (see
sigmaflux.utils.carbon_coverage.filter_by_carbon_coverage). This CSV is the
LC/MS-platform-measurable subset of the model -- what one experiment happened to
report plays no part in it. It is used by prepare_experiment_data.py and
run_inference.py via --mapping-csv.

An entry whose GEM xlsx carries no metabolite names (E_coli, E_coli_core) supplies
`name_fallback`, a BiGG-id -> name table from _bigg_name_fallback.py, used to fill
the gap before matching.

**The `*_fragment_mapping.csv` files are NOT built here.** GC/MS measures derivatised
*fragments* -- Ala232, Glu432, a named ion carrying a known subset of a molecule's carbons
-- not whole molecules, so they cannot be matched against `data/HILIC_LCMS_knowns.csv`.
E_coli_core's is built from `data/GCMS_knowns.csv` by build_gcms_fragment_model.py, onto
fragment pools. E_coli's is still hand-written and maps fragments onto whole molecules.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sigmaflux.utils import match_lcms_knowns
from sigmaflux.utils.carbon_coverage import filter_by_carbon_coverage, formulas_from_knowns
from _bigg_name_fallback import E_COLI_BIGG_NAMES, E_COLI_CORE_NAMES, fill_missing_names

LCMS_KNOWNS = "data/HILIC_LCMS_knowns.csv"

# ============================================================
# USER CONFIGURATION — edit paths for your local setup
# ============================================================

# `model_json` names the atom-mapping-annotated model (the one paper/configs/*.yaml
# points `model_file` at); `gem_xlsx` is that model's sibling GEM workbook.
ORGANISMS = [
    {
        "name": "I_orientalis",
        "output_csv": "models/atomistic/I_orientalis/I_orientalis_mapping.csv",
        "model_json": "models/atomistic/I_orientalis/imIsor_annotated_hierarchical.json",
        "gem_xlsx":   "models/atomistic/I_orientalis/imIsor_GEM.xlsx",
    },
    {
        "name": "S_cerevisiae",
        "output_csv": "models/atomistic/S_cerevisiae/S_cerevisiae_mapping.csv",
        "model_json": "models/atomistic/S_cerevisiae/sc_annotated_hierarchical.json",
        "gem_xlsx":   "models/atomistic/S_cerevisiae/scGEM.xlsx",
    },
    {
        "name": "R_toruloides",
        "output_csv": "models/atomistic/R_toruloides/R_toruloides_mapping.csv",
        "model_json": "models/atomistic/R_toruloides/imRt_annotated_hierarchical.json",
        "gem_xlsx":   "models/atomistic/R_toruloides/imRt_GEM.xlsx",
    },
    {
        "name": "M_musculus_T_Cell",
        "output_csv": "models/atomistic/M_musculus_T_Cell/M_musculus_T_Cell_mapping.csv",
        "model_json": "models/atomistic/M_musculus_T_Cell/mmTcell_annotated_hierarchical.json",
        "gem_xlsx":   "models/atomistic/M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx",
    },
    {
        # iAF1260_GEM.xlsx's metname column is entirely empty, so name matching needs
        # the BiGG-id fallback table.
        "name": "E_coli",
        "output_csv": "models/atomistic/E_coli/E_coli_mapping.csv",
        "model_json": "models/atomistic/E_coli/iAF1260_annotated.json",
        "gem_xlsx":   "models/atomistic/E_coli/iAF1260_GEM.xlsx",
        "name_fallback": E_COLI_BIGG_NAMES,
        "strip_compartment": True,
    },
    {
        # Same gap as E_coli, but example_GEM.xlsx's ids carry no compartment suffix
        # and don't follow BiGG spelling, hence the separate table.
        "name": "E_coli_core",
        "output_csv": "models/atomistic/E_coli_core/E_coli_core_mapping.csv",
        "model_json": "models/atomistic/E_coli_core/e_coli_core_annotated.json",
        "gem_xlsx":   "models/atomistic/E_coli_core/example_GEM.xlsx",
        "name_fallback": E_COLI_CORE_NAMES,
        "strip_compartment": False,
    },
]

# ============================================================
# END USER CONFIGURATION
# ============================================================


def _build_platform_mapping(org, root):
    """Write org's `output_csv`: model metabolites matched to data/HILIC_LCMS_knowns.csv by name.

    Args:
        org: this organism's config dict from ORGANISMS.
        root: project root, for resolving org's relative paths.
    """
    from cobra.io import load_json_model

    gem_xlsx = os.path.join(root, org["gem_xlsx"])
    metabolites = pd.read_excel(gem_xlsx, sheet_name='Metabolites', usecols=['metid', 'metname'])
    if org.get("name_fallback"):
        metabolites = fill_missing_names(
            metabolites, 'metid', 'metname', org["name_fallback"], org["strip_compartment"])

    model = load_json_model(os.path.join(root, org["model_json"]))
    mapping = match_lcms_knowns(
        gem_xlsx=gem_xlsx,
        knowns_path=os.path.join(root, LCMS_KNOWNS),
        model_metabolite_ids=[m.id for m in model.metabolites],
        metabolites_df=metabolites,
    )
    if mapping.empty:
        print("  WARNING: 0 LC/MS knowns matched by name -- no row can be confirmed "
              "against the platform")

    output = os.path.join(root, org["output_csv"])
    mapping.sort_values(["experimental_id", "model_id"]).to_csv(output, index=False)
    print(f"  {mapping['model_id'].nunique()} measurable metabolites, {len(mapping)} mappings -> {output}")


def _filter_carbon_coverage(org, root):
    """Drop mapping rows whose model_id does not track every carbon of its real molecule.

    Args:
        org: this organism's config dict from ORGANISMS.
        root: project root, for resolving org's relative paths.
    """
    from cobra.io import load_json_model

    model = load_json_model(os.path.join(root, org["model_json"]))
    formula_by_name = formulas_from_knowns(os.path.join(root, LCMS_KNOWNS))

    output = os.path.join(root, org["output_csv"])
    mapping = pd.read_csv(output)
    filtered = filter_by_carbon_coverage(mapping, model, formula_by_name)
    filtered.to_csv(output, index=False)
    print(f"  Carbon coverage: {len(mapping)} -> {len(filtered)} mappings")


def _collapse_to_measurement_tokens(org, root):
    """Rewrite ``model_id`` to the id the platform actually reports.

    A metabolite the model pools is reported as its pool -- naming one compartment would
    assert a labeling the extract cannot distinguish -- and a metabolite in an excluded
    compartment is not in the extract at all. Doing it here is what keeps a re-run of
    this script from reverting the mapping files to per-compartment ids.

    Args:
        org: this organism's config dict from ORGANISMS.
        root: project root, for resolving org's relative paths.
    """
    from cobra.io import load_json_model

    from sigmaflux.core.measurement_pools import measurement_token

    model = load_json_model(os.path.join(root, org["model_json"]))
    output = os.path.join(root, org["output_csv"])
    mapping = pd.read_csv(output)
    mapping["model_id"] = [measurement_token(model, m) for m in mapping["model_id"]]
    collapsed = (mapping.dropna(subset=["model_id"])
                        .drop_duplicates(["experimental_id", "model_id"])
                        .sort_values(["experimental_id", "model_id"]))
    collapsed.to_csv(output, index=False)
    print(f"  Measurement tokens: {len(mapping)} -> {len(collapsed)} mappings")


if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), '..', '..')
    for org in ORGANISMS:
        print(f"\n=== {org['name']} ===")
        _build_platform_mapping(org, _root)
        _filter_carbon_coverage(org, _root)
        _collapse_to_measurement_tokens(org, _root)
    print("\nDone.")
