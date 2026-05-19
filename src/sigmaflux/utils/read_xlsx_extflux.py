"""
Read external flux measurements from experimental xlsx into long-format df_Vm CSV.
"""

import re
import json
import pandas as pd
from datetime import datetime


# ---- constants ----

# Maps xlsx column prefix to model metabolite base when they differ.
# Add new entries here as new datasets are encountered.
COLUMN_OVERRIDES: dict[str, str] = {
    'ace': 'ac',  # Ace_EF (acetate)
}


# ---- helpers ----

def _get_base(met_id: str) -> str:
    base = re.sub(r'_out$', '', met_id)
    base = re.sub(r'_[a-z]$', '', base)
    base = re.sub(r'__[LDld]$', '', base)
    return base.lower()


def _build_exchange_lookup(model_json_path: str) -> dict:
    with open(model_json_path) as f:
        model = json.load(f)
    lookup = {}
    for rxn in model['reactions']:
        if not rxn['id'].startswith('EX_'):
            continue
        met_ids = list(rxn['metabolites'].keys())
        if len(met_ids) != 1:
            continue
        base = _get_base(met_ids[0])
        lookup[base] = rxn['id']
    return lookup


def _get_constrain_df(xl: pd.ExcelFile) -> pd.DataFrame:
    sheet = next((s for s in xl.sheet_names if 'constrain' in s.lower()), None)
    if sheet is None:
        raise ValueError(
            f"No sheet with 'constrain' in name found. "
            f"Available sheets: {xl.sheet_names}"
        )
    return xl.parse(sheet)


def _get_condition_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    id_col = df.columns[0]
    pattern = re.compile(rf'^{re.escape(condition)}\d*$')
    mask = df[id_col].astype(str).apply(lambda v: bool(pattern.match(v)))
    rows = df[mask].copy()
    if rows.empty:
        raise ValueError(
            f"No rows found for condition '{condition}'. "
            f"Row identifiers available: {df[id_col].tolist()}"
        )
    def _extract_rep(label):
        m = re.search(r'\d+$', str(label))
        return int(m.group()) if m else 1
    rows['bio_replicate'] = rows[id_col].apply(_extract_rep)
    return rows


def _get_flux_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.endswith('_IN') or c.endswith('_EF')]


def _get_biomass_col(df: pd.DataFrame) -> str:
    for col in ('DR', 'gr'):
        if col in df.columns:
            return col
    raise ValueError("Neither 'DR' nor 'gr' column found in constrain sheet.")


def _build_records(rows, flux_cols, biomass_col, ex_lookup):
    records = []
    for _, row in rows.iterrows():
        bio_rep = int(row['bio_replicate'])
        for col in flux_cols:
            prefix     = col.split('_')[0].lower()
            lookup_key = COLUMN_OVERRIDES.get(prefix, prefix)
            if lookup_key not in ex_lookup:
                print(f"WARNING: No exchange reaction found for column '{col}' "
                      f"(lookup key '{lookup_key}'). Skipping.")
                continue
            records.append({
                'Sample':    bio_rep,
                'Flux Name': ex_lookup[lookup_key],
                'Value':     float(row[col]),
            })
        records.append({
            'Sample':    bio_rep,
            'Flux Name': 'BIOMASS',
            'Value':     float(row[biomass_col]),
        })
    return records


# ---- main ----

def read_xlsx_extflux(
    xlsx_path:  str,
    model_json: str,
    organism:   str,
    condition:  str,
    output_dir: str = '.',
) -> pd.DataFrame:
    """
    Read external flux measurements from xlsx into long-format CSV.

    Parameters
    ----------
    xlsx_path : str
        Path to external measurements xlsx.
    model_json : str
        Path to COBRApy JSON model (e.g. sc_annotated.json, imIsor_annotated.json).
    organism : str
        Organism name used in the output filename (e.g. 'S_cervesiae').
    condition : str
        Row prefix identifying the condition (e.g. 'P', 'N', 'B'), also used
        in the output filename.
    output_dir : str
        Directory for the output CSV (default: current directory).

    Returns
    -------
    pd.DataFrame with columns [Sample, Flux Name, Value].
        Exchange flux rows map to model reaction IDs (e.g. 'EX_glc__D_e').
        Growth rate is stored as Flux Name 'BIOMASS'.
    """
    ex_lookup   = _build_exchange_lookup(model_json)
    xl          = pd.ExcelFile(xlsx_path)
    df          = _get_constrain_df(xl)
    rows        = _get_condition_rows(df, condition)
    flux_cols   = _get_flux_columns(df)
    biomass_col = _get_biomass_col(df)
    records     = _build_records(rows, flux_cols, biomass_col, ex_lookup)

    df_out = pd.DataFrame(records, columns=['Sample', 'Flux Name', 'Value'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f"df_Vm_{organism}_{condition}_{timestamp}.csv"
    out_path  = f"{output_dir}/{filename}"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {len(df_out)} rows ({rows['bio_replicate'].nunique()} bio replicates, "
          f"{df_out['Flux Name'].nunique()} fluxes) -> {out_path}")

    return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read external flux measurements xlsx into long-format df_Vm CSV."
    )
    parser.add_argument("xlsx_path",  help="Path to external measurements xlsx")
    parser.add_argument("model_json", help="Path to COBRApy JSON model")
    parser.add_argument("organism",   help="Organism name (e.g. S_cervesiae)")
    parser.add_argument("condition",  help="Condition row prefix (e.g. P, N, B)")
    parser.add_argument("--output_dir", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    read_xlsx_extflux(
        xlsx_path  = args.xlsx_path,
        model_json = args.model_json,
        organism   = args.organism,
        condition  = args.condition,
        output_dir = args.output_dir,
    )
