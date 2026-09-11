"""
Read MID fraction data from clean xlsx into long-format df_L DataFrame.

Expected xlsx format (one sheet per tracer, sheet name = tracer name):
    Columns: Compound, C_Label, C1_Rep1, C1_Rep2, C1_Rep3, C2_Rep1, ...
    Column naming: {ConditionPrefix}{GrowthCondition}_Rep{BioRep}
    e.g. C1_Rep1, N3_Rep2, B1_Rep1

Output df_L columns: Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction
"""

import re
import warnings

import pandas as pd

_GC_REP_PAT = re.compile(r'^([A-Z]+)(\d+)_Rep(\d+)$')


def _load_mapping(mapping_csv: str) -> dict:
    """``{experimental_id: [token ids]}`` for the workbook's compound names.

    The CSV already names the token the platform reports -- a whole-cell pool where the
    compartments cannot be resolved, a plain compartment id where they can -- so this is
    a straight read with no reinterpretation.

    Args:
        mapping_csv: experimental-id -> model-id CSV.
    """
    df = pd.read_csv(mapping_csv)
    return df.groupby('experimental_id')['model_id'].apply(list).to_dict()


def read_xlsx_mid_data(
    xlsx_path:        str,
    mapping_csv:      str,
    growth_condition: int | None = None,
) -> pd.DataFrame:
    """Read MID fraction data from a clean per-condition xlsx (one tracer per sheet) into df_L [Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction], mapping compound names to model IDs via mapping_csv.

    Args:
        xlsx_path: path to the clean per-condition xlsx.
        mapping_csv: CSV mapping experimental compound names to model metabolite ids.
        growth_condition: if given, restricts to columns matching that GrowthCondition number.
    """
    met_map = _load_mapping(mapping_csv)
    _warn_on_unresolved_compartments(met_map, mapping_csv)
    xl      = pd.ExcelFile(xlsx_path)
    records = []

    for sheet in xl.sheet_names:
        tracer_name = sheet
        df = xl.parse(sheet).dropna(subset=['Compound'])

        gc_rep_cols = []
        for c in df.columns:
            m = _GC_REP_PAT.match(str(c))
            if m:
                gc = int(m.group(2))
                rep = int(m.group(3))
                if growth_condition is None or gc == growth_condition:
                    gc_rep_cols.append((c, rep))

        for col, bio_rep in gc_rep_cols:
            for _, row in df.iterrows():
                exp_name = str(row['Compound'])
                if exp_name not in met_map:
                    continue
                isotopomer = f"M+{int(row['C_Label'])}"
                fraction   = float(row[col])
                for model_id in met_map[exp_name]:
                    records.append({
                        'Tracer Name':   tracer_name,
                        'Sample':        bio_rep,
                        'Metabolite ID': model_id,
                        'Isotopomer':    isotopomer,
                        'Fraction':      fraction,
                    })

    return pd.DataFrame(
        records,
        columns=['Tracer Name', 'Sample', 'Metabolite ID', 'Isotopomer', 'Fraction'],
    )


def _warn_on_unresolved_compartments(met_map, mapping_csv):
    """Flag compounds that still land in more than one model metabolite.

    Writing one measured MID into several compartments does not model a mixture, it
    asserts the compartments are identically labeled. A compound whose model carries a
    whole-cell pool should name that pool here instead, so this firing means the mapping
    file has fallen behind the model.
    """
    split = {k: v for k, v in met_map.items() if len(v) > 1}
    if not split:
        return
    detail = ", ".join(f"{k} -> {v}" for k, v in sorted(split.items())[:5])
    if len(split) > 5:
        detail += ", ..."
    warnings.warn(
        f"{mapping_csv}: {len(split)} compound(s) map to several model metabolites and "
        f"are written into each one, which asserts the compartments are identically "
        f"labeled rather than mixing them ({detail}). Name the whole-cell pool "
        f"instead, or confirm the model has no pool for them.",
        RuntimeWarning, stacklevel=3)
