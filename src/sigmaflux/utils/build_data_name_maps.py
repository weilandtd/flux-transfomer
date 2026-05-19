"""
Build metabolite name mapping from experimental xlsx to model IDs.
"""

import re
import pandas as pd


# Hardcoded fallback mappings for experimental names that cannot be resolved.
# Add new entries here as new datasets are encountered.
MANUAL_OVERRIDES: dict[str, list[str]] = {
    "PG":     ["3pg_c"],
    "FBP":    ["fdp_c"],
    "IPPM":   ["2ippm_c"],
    "Suc":    ["succ_c", "succ_m"],
    "Citrl":  ["citr__L_c"],
    "glucose": ["glc__D_c"],
}


def build_data_name_maps(
    experimental_xlsx: str,
    model_amm_xlsx: str,
    output_csv: str = "metabolite_name_mapping.csv",
    compound_col: str = "Compound",
    amm_sheet: str = "Atom Transition",
    met_col: str = "metAbbreviation",
    compartments: tuple = ("_c", "_m", "_e"),
) -> pd.DataFrame:
    """
    Generate metabolite_name_mapping.csv mapping experimental compound names
    to model metabolite IDs.

    Parameters
    ----------
    experimental_xlsx : str
        Path to experimental MID xlsx. The first sheet matching [A-Z]+_[0-9]+
        (e.g. NU_1, C12_1) is used to extract unique compound names.
    model_amm_xlsx : str
        Path to model atom mapping xlsx. Sheet 'Atom Transition', column
        metAbbreviation supplies all model metabolite IDs.
    output_csv : str
        Output path for the two-column CSV (experimental_id, model_id).
    compound_col : str
        Column name in experimental sheet containing metabolite names.
    amm_sheet : str
        Sheet name in model AMM xlsx.
    met_col : str
        Column name in AMM sheet containing model metabolite IDs.
    compartments : tuple
        Compartment suffixes to include in output (default: _c, _m, _e).

    Returns
    -------
    pd.DataFrame with columns [experimental_id, model_id].
    Unmatched experimental names are printed as warnings.
    """

    # ---- 1. Find a data sheet in experimental xlsx ----
    # Supports both old format (sheet names like CU_1, N12_3) and new clean
    # format (sheet names are tracer names, e.g. 'U-13C glucose (50%)').
    xl = pd.ExcelFile(experimental_xlsx)
    condition_sheet = next(
        (s for s in xl.sheet_names if re.match(r'^[A-Z]+_[0-9]+$', s)),
        None
    )
    if condition_sheet is None:
        # New clean format: use first sheet that contains compound_col
        for s in xl.sheet_names:
            df_peek = pd.read_excel(experimental_xlsx, sheet_name=s, nrows=1)
            if compound_col in df_peek.columns:
                condition_sheet = s
                break
    if condition_sheet is None:
        raise ValueError(
            f"No usable data sheet found in {experimental_xlsx}. "
            f"Expected a sheet with a '{compound_col}' column. "
            f"Available sheets: {xl.sheet_names}"
        )

    # ---- 2. Extract unique experimental metabolite names ----
    df_exp = pd.read_excel(experimental_xlsx, sheet_name=condition_sheet)
    if compound_col not in df_exp.columns:
        raise ValueError(
            f"Column '{compound_col}' not found in sheet '{condition_sheet}'. "
            f"Available columns: {list(df_exp.columns)}"
        )
    exp_names = df_exp[compound_col].dropna().unique().tolist()

    # ---- 3. Build base-name -> model IDs lookup from AMM xlsx ----
    df_amm = pd.read_excel(model_amm_xlsx, sheet_name=amm_sheet)
    if met_col not in df_amm.columns:
        raise ValueError(
            f"Column '{met_col}' not found in sheet '{amm_sheet}'. "
            f"Available columns: {list(df_amm.columns)}"
        )
    model_ids = df_amm[met_col].dropna().unique().tolist()

    # Strip compartment suffix and stereoisomer to get base name
    def get_base(model_id):
        base = re.sub(r'_out$', '', model_id)
        base = re.sub(r'_[a-z]$', '', base)       # strip single-letter compartment
        base = re.sub(r'__[LDld]$', '', base)     # strip stereoisomer
        return base.lower()

    base_to_ids: dict = {}
    for mid in model_ids:
        base = get_base(mid)
        base_to_ids.setdefault(base, []).append(mid)

    # ---- 4. Match each experimental name -> model IDs ----
    records = []
    unmatched = []

    for exp in exp_names:
        candidates = []

        for query in [
            exp.lower(),                  # direct
            exp.lower() + '__l',          # L-stereoisomer (amino acids)
            exp.lower() + '__d',          # D-stereoisomer
        ]:
            if query in base_to_ids:
                candidates = base_to_ids[query]
                break

        # Keep only requested compartments
        matched = [m for m in candidates if any(m.endswith(c) for c in compartments)]

        if matched:
            for model_id in matched:
                records.append({"experimental_id": exp, "model_id": model_id})
        elif exp in MANUAL_OVERRIDES:
            for model_id in MANUAL_OVERRIDES[exp]:
                records.append({"experimental_id": exp, "model_id": model_id})
        else:
            unmatched.append(exp)

    # ---- 5. Report unmatched ----
    if unmatched:
        print("WARNING: No model match found for the following experimental metabolites:")
        for name in unmatched:
            print(f"  {name}")
        print("Add entries to MANUAL_OVERRIDES in build_data_name_maps.py to resolve.")

    # ---- 6. Write CSV ----
    df_out = pd.DataFrame(records, columns=["experimental_id", "model_id"])
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote {len(df_out)} mappings ({len(exp_names) - len(unmatched)}/{len(exp_names)} "
          f"experimental metabolites matched) -> {output_csv}")

    return df_out

