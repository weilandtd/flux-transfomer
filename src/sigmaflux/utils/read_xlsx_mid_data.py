"""
Read MID fraction data from clean xlsx into long-format df_L DataFrame.

Expected xlsx format (one sheet per tracer, sheet name = tracer name):
    Columns: Compound, C_Label, C1_Rep1, C1_Rep2, C1_Rep3, C2_Rep1, ...
    Column naming: {ConditionPrefix}{GrowthCondition}_Rep{BioRep}
    e.g. C1_Rep1, N3_Rep2, B1_Rep1

Output df_L columns: Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction
"""

import re
import pandas as pd

_GC_REP_PAT = re.compile(r'^([A-Z]+)(\d+)_Rep(\d+)$')


def _load_mapping(mapping_csv: str) -> dict:
    df = pd.read_csv(mapping_csv)
    return df.groupby('experimental_id')['model_id'].apply(list).to_dict()


def read_xlsx_mid_data(
    xlsx_path:        str,
    mapping_csv:      str,
    growth_condition: int | None = None,
) -> pd.DataFrame:
    """
    Read MID fraction data from a clean per-condition xlsx into a DataFrame.

    Parameters
    ----------
    xlsx_path : str
        Path to clean MID xlsx.  Each sheet is one tracer; the sheet name is
        used directly as 'Tracer Name' in the output.  Data columns follow the
        naming convention ``{ConditionPrefix}{GrowthCondition}_Rep{BioRep}``
        (e.g. C1_Rep1, N3_Rep2, B1_Rep1).
    mapping_csv : str
        Path to metabolite_name_mapping.csv mapping experimental compound names
        to model metabolite IDs.
    growth_condition : int or None
        If given, only columns whose GrowthCondition number matches are used.
        If None, all growth conditions are included.

    Returns
    -------
    pd.DataFrame with columns [Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction].
    The Sample value is the bio-replicate number (1-based).
    """
    met_map = _load_mapping(mapping_csv)
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
