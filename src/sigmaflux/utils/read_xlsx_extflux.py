"""
Read external flux measurements from clean ExtFlux xlsx into df_Vm DataFrame.

Expected xlsx format — single sheet named 'Constraints':
    Columns: Condition, BioReplicate, BIOMASS, EX_glc__D_e, EX_etoh_e, ...
    Flux columns are model reaction IDs written during data conversion.
"""

import pandas as pd


def read_xlsx_extflux(
    xlsx_path:  str,
    condition:  str,
    flux_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read external flux measurements from clean ExtFlux xlsx into a DataFrame.

    Parameters
    ----------
    xlsx_path : str
        Path to clean ExtFlux xlsx.  Must contain a sheet named 'Constraints'
        with columns: Condition, BioReplicate, <reaction_id>, ...
        Flux column names are model reaction IDs (set during data conversion).
    condition : str
        Condition code to filter (e.g. 'C', 'N', 'P', 'B').
    flux_names : list[str] | None
        Reaction IDs to read (e.g. from config ``training.measured_fluxes``).
        If None, every non-metadata column is read.

    Returns
    -------
    pd.DataFrame with columns [Sample, Flux Name, Value].
    """
    xl = pd.ExcelFile(xlsx_path)
    if 'Constraints' not in xl.sheet_names:
        raise ValueError(
            f"Sheet 'Constraints' not found in {xlsx_path}. "
            f"Available: {xl.sheet_names}"
        )
    df = xl.parse('Constraints')

    rows = df[df['Condition'] == condition].copy()
    if rows.empty:
        raise ValueError(
            f"No rows for condition '{condition}' in {xlsx_path}. "
            f"Available: {df['Condition'].unique().tolist()}"
        )

    meta_cols = {'Condition', 'BioReplicate'}
    if flux_names is not None:
        missing = [f for f in flux_names if f not in rows.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Flux columns {missing} not found in {xlsx_path} and will be "
                f"treated as unknown (0) by the model. "
                f"Available: {[c for c in rows.columns if c not in meta_cols]}"
            )
        flux_cols = [f for f in flux_names if f in rows.columns]
    else:
        flux_cols = [c for c in rows.columns if c not in meta_cols]

    records = []
    for _, row in rows.iterrows():
        bio_rep = int(row['BioReplicate'])
        for col in flux_cols:
            records.append({
                'Sample':    bio_rep,
                'Flux Name': col,
                'Value':     float(row[col]),
            })

    df_out = pd.DataFrame(records, columns=['Sample', 'Flux Name', 'Value'])
    return df_out
