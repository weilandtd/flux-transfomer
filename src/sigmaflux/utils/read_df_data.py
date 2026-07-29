"""
Convert utils staging DataFrames to MFA pipeline-ready tables (in-memory, no files written).
"""

import pandas as pd


def mid_df_to_mass_isotopomer_table(
    df_L: pd.DataFrame,
    tracer_name: str | None = None,
) -> pd.DataFrame:
    """
    Convert df_L from read_xlsx_mid_data to a MassIsotopomerData-ready table.

    Parameters
    ----------
    df_L : pd.DataFrame
        Output of read_xlsx_mid_data.
        Columns: [Tracer Name, Sample, Metabolite ID, Isotopomer, Fraction].
    tracer_name : str or None
        If given, filter to rows whose 'Tracer Name' matches this value.

    Returns
    -------
    pd.DataFrame with columns [id, mid, mean, std].
        id  — model metabolite ID
        mid — integer isotopomer index (0, 1, 2, …)
        mean/std — aggregated across bio-replicates (Sample dimension)
    """
    df = df_L if tracer_name is None else df_L[df_L['Tracer Name'] == tracer_name]
    df = df.copy()
    df['mid'] = df['Isotopomer'].str.removeprefix('M+').astype(int)
    agg = (
        df.groupby(['Metabolite ID', 'mid'])['Fraction']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'Metabolite ID': 'id'})
    )
    return agg[['id', 'mid', 'mean', 'std']]


def extflux_df_to_flux_table(
    df_Vm: pd.DataFrame,
    df_err: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert df_Vm (and optional df_err) from read_xlsx_extflux to a FluxData-ready table.

    Parameters
    ----------
    df_Vm : pd.DataFrame
        Output of read_xlsx_extflux.
        Columns: [Sample, Flux Name, Value].
    df_err : pd.DataFrame or None
        Optional error DataFrame with the same structure as df_Vm.
        When provided, its Value column becomes the std column.

    Returns
    -------
    pd.DataFrame with columns [reaction, mean, std].
        reaction — model reaction ID
        mean/std — flux value and its uncertainty
    """
    if df_err is not None:
        merged = df_Vm.merge(
            df_err[['Sample', 'Flux Name', 'Value']],
            on=['Sample', 'Flux Name'],
            how='left',
            suffixes=('_mean', '_std'),
        )
        result = (
            merged.groupby('Flux Name', sort=False)
            .agg(mean=('Value_mean', 'mean'), std=('Value_std', 'mean'))
            .reset_index()
            .rename(columns={'Flux Name': 'reaction'})
        )
    else:
        result = (
            df_Vm.groupby('Flux Name', sort=False)['Value']
            .agg(['mean', 'std'])
            .reset_index()
            .rename(columns={'Flux Name': 'reaction'})
        )
    return result[['reaction', 'mean', 'std']]


def extract_tracer_list(df_L: pd.DataFrame) -> set[str]:
    """Return the set of unique tracer names from df_L for use with get_tracers_from_data."""
    return set(df_L['Tracer Name'].unique())
