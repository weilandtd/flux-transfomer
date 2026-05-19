"""
Prepare experimental data CSVs for a single organism/condition.

Orchestrates build_data_name_maps, read_xlsx_mid_data, and
read_xlsx_extflux in sequence, auto-discovering the required files from
the data and model directories.

Usage:
    python paper/pipeline/prepare_experiment_data.py \
        <data_dir> \
        <model_dir> \
        <condition> \
        [--organism ORGANISM] \
        [--output_dir OUTPUT_DIR]

Example:
    python paper/pipeline/prepare_experiment_data.py \\
        SigmaFlux/data/I_orientalis \\
        SigmaFlux/models/I_orientalis \\
        P
"""

import os
import glob
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sigmaflux.utils import (
    build_data_name_maps,
    read_xlsx_mid_data,
    read_xlsx_extflux,
)


# ---- file discovery helpers ----

def _find_mid_xlsx(data_dir: str, condition: str) -> str:
    if condition.upper() == 'B':
        pattern = os.path.join(data_dir, '*batch*.xlsx')
    else:
        pattern = os.path.join(data_dir, f'*_{condition}_13C.xlsx')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No MID xlsx found for condition='{condition}' in {data_dir}. "
            f"Pattern tried: {os.path.basename(pattern)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple MID xlsx found for condition='{condition}': {matches}. "
            f"Specify a more specific data_dir."
        )
    return matches[0]


def _find_external_xlsx(data_dir: str) -> str:
    candidates = [
        f for f in glob.glob(os.path.join(data_dir, '*.xlsx'))
        if 'external' in os.path.basename(f).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No xlsx with 'external' in name found in {data_dir}."
        )
    return candidates[0]


def _find_amm_xlsx(model_dir: str) -> str:
    candidates = [
        f for f in glob.glob(os.path.join(model_dir, '*.xlsx'))
        if 'AMM' in os.path.basename(f)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No AMM xlsx found in {model_dir}. "
            f"Expected a file with 'AMM' in its name."
        )
    return candidates[0]


def _find_json_model(model_dir: str) -> str:
    matches = glob.glob(os.path.join(model_dir, '*.json'))
    if not matches:
        raise FileNotFoundError(f"No JSON model found in {model_dir}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple JSON models found in {model_dir}: {matches}. "
            f"Ensure only one .json file is present."
        )
    return matches[0]


# ---- main ----

def prepare_experiment_data(
    data_dir:   str,
    model_dir:  str,
    condition:  str,
    organism:   str = None,
    output_dir: str = None,
) -> None:
    """
    Prepare experimental data CSVs for a given organism and condition.

    Parameters
    ----------
    data_dir : str
        Path to organism data folder (e.g. .../data/I_orientalis).
    model_dir : str
        Path to organism model folder (e.g. .../models/I_orientalis).
    condition : str
        Condition code: 'P', 'N', 'C', or 'B' (batch).
    organism : str, optional
        Organism name for output filenames. Defaults to basename of data_dir.
    output_dir : str, optional
        Output directory. Defaults to data_dir.
    """
    if organism is None:
        organism = os.path.basename(os.path.normpath(data_dir))
    if output_dir is None:
        output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- discover files ----
    mid_xlsx   = _find_mid_xlsx(data_dir, condition)
    ext_xlsx   = _find_external_xlsx(data_dir)
    amm_xlsx   = _find_amm_xlsx(model_dir)
    json_model = _find_json_model(model_dir)

    print(f"MID xlsx:        {mid_xlsx}")
    print(f"External xlsx:   {ext_xlsx}")
    print(f"AMM xlsx:        {amm_xlsx}")
    print(f"JSON model:      {json_model}")
    print(f"Organism:        {organism}")
    print(f"Condition:       {condition}")
    print(f"Output dir:      {output_dir}")
    print()

    mapping_csv = os.path.join(output_dir, 'metabolite_name_mapping.csv')

    # ---- step 1: build metabolite name mapping ----
    print("=== Step 1: build_data_name_maps ===")
    build_data_name_maps(
        experimental_xlsx = mid_xlsx,
        model_amm_xlsx    = amm_xlsx,
        output_csv        = mapping_csv,
    )
    print()

    # ---- step 2: read MID data ----
    print("=== Step 2: read_xlsx_mid_data ===")
    read_xlsx_mid_data(
        xlsx_path   = mid_xlsx,
        mapping_csv = mapping_csv,
        organism    = organism,
        condition   = condition,
        output_dir  = output_dir,
    )
    print()

    # ---- step 3: read external fluxes ----
    print("=== Step 3: read_xlsx_extflux ===")
    read_xlsx_extflux(
        xlsx_path  = ext_xlsx,
        model_json = json_model,
        organism   = organism,
        condition  = condition,
        output_dir = output_dir,
    )
    print()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare experimental data CSVs for a single organism/condition."
    )
    parser.add_argument(
        "data_dir",
        help="Path to organism data folder (e.g. SigmaFlux/data/I_orientalis)",
    )
    parser.add_argument(
        "model_dir",
        help="Path to organism model folder (e.g. SigmaFlux/models/I_orientalis)",
    )
    parser.add_argument(
        "condition",
        help="Condition code: P, N, C, or B (batch)",
    )
    parser.add_argument(
        "--organism",
        default=None,
        help="Organism name for output filenames (default: basename of data_dir)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: data_dir)",
    )
    args = parser.parse_args()

    prepare_experiment_data(
        data_dir   = args.data_dir,
        model_dir  = args.model_dir,
        condition  = args.condition,
        organism   = args.organism,
        output_dir = args.output_dir,
    )
