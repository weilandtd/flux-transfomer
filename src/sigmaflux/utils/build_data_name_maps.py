"""
Build metabolite name mapping from experimental xlsx to model IDs.
"""

import re
import pandas as pd


# Hardcoded fallback mappings for experimental names that cannot be resolved.
# Add new entries here as new datasets are encountered.
MANUAL_OVERRIDES: dict[str, list[str]] = {
    "PG":    ["3pg_c"],
    "FBP":   ["fdp_c"],
    "IPPM":  ["2ippm_c"],
    "Suc":   ["succ_c", "succ_m"],
    "Citrl": ["citr__L_c"],
    "glucose": ["glc__D_c"],
    "GAP":   ["g3p_c"],
    "PG6":   ["6pgc_c"],
    "aKV":   ["3mob_c", "3mob_m"],
}


def build_data_name_maps(
    experimental_xlsx: str | list[str] | None,
    model_amm_xlsx: str,
    output_csv: str = "metabolite_name_mapping.csv",
    compound_col: str = "Compound",
    amm_sheet: str = "Atom Transition",
    met_col: str = "metAbbreviation",
    compartments: tuple = ("_c", "_m", "_e"),
    experimental_names: list[str] | None = None,
    overrides: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Generate metabolite_name_mapping.csv mapping experimental compound names to model metabolite IDs, and return it as a DataFrame.

    Pass exactly one of experimental_xlsx or experimental_names. Compound names are
    matched against model IDs by base name (compartment/stereoisomer stripped), falling
    back to MANUAL_OVERRIDES merged with `overrides`; unmatched names are printed as
    warnings.
    """
    if (experimental_xlsx is None) == (experimental_names is None):
        raise ValueError("pass exactly one of experimental_xlsx or experimental_names")

    manual_overrides = {**MANUAL_OVERRIDES, **(overrides or {})}

    # ---- 1 & 2. Extract unique compound names across all xlsx files ----
    xlsx_paths = ([] if experimental_xlsx is None
                  else [experimental_xlsx] if isinstance(experimental_xlsx, str)
                  else experimental_xlsx)
    exp_names_set: set = set(experimental_names or [])

    for xlsx_path in xlsx_paths:
        xl = pd.ExcelFile(xlsx_path)
        condition_sheet = next(
            (s for s in xl.sheet_names if re.match(r'^[A-Z]+_[0-9]+$', s)),
            None
        )
        if condition_sheet is None:
            for s in xl.sheet_names:
                df_peek = pd.read_excel(xlsx_path, sheet_name=s, nrows=1)
                if compound_col in df_peek.columns:
                    condition_sheet = s
                    break
        if condition_sheet is None:
            raise ValueError(
                f"No usable data sheet found in {xlsx_path}. "
                f"Expected a sheet with a '{compound_col}' column. "
                f"Available sheets: {xl.sheet_names}"
            )
        df_exp = pd.read_excel(xlsx_path, sheet_name=condition_sheet)
        if compound_col not in df_exp.columns:
            raise ValueError(
                f"Column '{compound_col}' not found in sheet '{condition_sheet}'. "
                f"Available columns: {list(df_exp.columns)}"
            )
        exp_names_set.update(df_exp[compound_col].dropna().unique().tolist())

    exp_names = sorted(exp_names_set)

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
    model_id_set = set(model_ids)

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
        elif exp in manual_overrides:
            # An override may name a metabolite this organism does not carry, e.g. when
            # the compound list is shared across organisms; drop those silently.
            present = [m for m in manual_overrides[exp] if m in model_id_set]
            for model_id in present:
                records.append({"experimental_id": exp, "model_id": model_id})
            if not present:
                unmatched.append(exp)
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

