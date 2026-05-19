"""
Read external flux measurements from clean ExtFlux xlsx into df_Vm DataFrame.

Expected xlsx format — single sheet named 'Constraints':
    Columns: Condition, BIOMASS, EX_glc__D_e, EX_glc__D_e_err, ...
    Condition encodes nutrient limitation and growth condition as a single label,
    e.g. C1, C2, N3, P5, B1.
    Error columns carry an '_err' suffix (e.g. EX_glc__D_e_err).
"""

import re
import pandas as pd

_COND_PAT = re.compile(r'^([A-Z]+)(\d+)$')


def read_xlsx_extflux(
    xlsx_path:        str,
    condition:        str,
    flux_names:       list[str] | None = None,
    return_errors:    bool = False,
    growth_condition: int | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read external flux measurements from clean ExtFlux xlsx into a DataFrame.

    Parameters
    ----------
    xlsx_path : str
        Path to clean ExtFlux xlsx.  Must contain a sheet named 'Constraints'
        with columns: Condition, <reaction_id>, ...
        Condition values are combined labels such as C1, N3, B1.
        Error columns have an '_err' suffix (e.g. EX_glc__D_e_err).
    condition : str
        Nutrient-limitation prefix to filter (e.g. 'C', 'N', 'P', 'B').
        Matches all rows whose Condition starts with this prefix.
    flux_names : list[str] | None
        Reaction IDs to read (without '_err' suffix).  If None, every
        non-metadata non-error column is read.
    return_errors : bool
        If True, also return df_err with the same structure as df_Vm but
        containing the measurement errors for each flux.
    growth_condition : int | None
        If given, only the row with the matching numeric suffix is returned
        (e.g. condition='C', growth_condition=3 → row 'C3').

    Returns
    -------
    df_Vm : pd.DataFrame with columns [Sample, Flux Name, Value].
            Sample is the growth-condition number extracted from the label.
    df_err : pd.DataFrame (only when return_errors=True).
    """
    xl = pd.ExcelFile(xlsx_path)
    if 'Constraints' not in xl.sheet_names:
        raise ValueError(
            f"Sheet 'Constraints' not found in {xlsx_path}. "
            f"Available: {xl.sheet_names}"
        )
    df = xl.parse('Constraints')

    mask = df['Condition'].apply(
        lambda v: bool(_COND_PAT.match(str(v))) and
                  _COND_PAT.match(str(v)).group(1) == condition
    )
    rows = df[mask].copy()
    if rows.empty:
        raise ValueError(
            f"No rows for condition '{condition}' in {xlsx_path}. "
            f"Available: {df['Condition'].unique().tolist()}"
        )

    if growth_condition is not None:
        target = f"{condition}{growth_condition}"
        rows = rows[rows['Condition'] == target].copy()
        if rows.empty:
            raise ValueError(
                f"No row for '{target}' in {xlsx_path}."
            )

    all_flux_cols = [c for c in rows.columns
                     if c != 'Condition' and not str(c).endswith('_err')]
    err_cols_map  = {c: c + '_err' for c in all_flux_cols
                     if c + '_err' in rows.columns}

    if flux_names is not None:
        missing = [f for f in flux_names if f not in rows.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Flux columns {missing} not found in {xlsx_path} and will be "
                f"treated as unknown (0) by the model. "
                f"Available: {all_flux_cols}"
            )
        flux_cols = [f for f in flux_names if f in rows.columns]
    else:
        flux_cols = all_flux_cols

    records     = []
    err_records = []
    for _, row in rows.iterrows():
        m      = _COND_PAT.match(str(row['Condition']))
        sample = int(m.group(2)) if m else 1
        for col in flux_cols:
            records.append({'Sample': sample, 'Flux Name': col, 'Value': float(row[col])})
            if return_errors and col in err_cols_map:
                err_records.append({
                    'Sample':    sample,
                    'Flux Name': col,
                    'Value':     float(row[err_cols_map[col]]),
                })

    df_Vm = pd.DataFrame(records, columns=['Sample', 'Flux Name', 'Value'])

    if return_errors:
        df_err = pd.DataFrame(err_records, columns=['Sample', 'Flux Name', 'Value'])
        return df_Vm, df_err

    return df_Vm
