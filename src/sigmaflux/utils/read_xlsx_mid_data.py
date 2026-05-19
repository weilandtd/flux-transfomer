"""
Read MID fraction data from clean xlsx into long-format df_L DataFrame.

Expected xlsx format (one sheet per tracer, sheet name = tracer name):
    Columns: Compound, C_Label, Rep_1, Rep_2, ..., Rep_N

Output df_L columns: Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction
"""

import pandas as pd


def _load_mapping(mapping_csv: str) -> dict:
    df = pd.read_csv(mapping_csv)
    return df.groupby('experimental_id')['model_id'].apply(list).to_dict()


def read_xlsx_mid_data(
    xlsx_path:   str,
    mapping_csv: str,
) -> pd.DataFrame:
    """
    Read MID fraction data from a clean per-condition xlsx into a DataFrame.

    Parameters
    ----------
    xlsx_path : str
        Path to clean MID xlsx.  Each sheet is one tracer; the sheet name is
        used directly as 'Tracer Name' in the output.  Columns must be
        Compound, C_Label, Rep_1, Rep_2, ... (bio-replicates as columns).
    mapping_csv : str
        Path to metabolite_name_mapping.csv mapping experimental compound names
        to model metabolite IDs.

    Returns
    -------
    pd.DataFrame with columns [Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction].
    """
    met_map = _load_mapping(mapping_csv)
    xl      = pd.ExcelFile(xlsx_path)
    records = []

    for sheet in xl.sheet_names:
        tracer_name = sheet
        df = xl.parse(sheet).dropna(subset=['Compound'])
        rep_cols = [c for c in df.columns if str(c).startswith('Rep_')]

        for col in rep_cols:
            bio_rep = int(str(col).split('_')[1])
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
