"""
Read MID fraction data from experimental xlsx into long-format df_L CSV.
"""

import re
import pandas as pd
from datetime import datetime


# ---- constants ----

TRACER_MAP = {
    'U':  'U-13C glucose (50%)',
    '12': '1,2-13C glucose (50%)',
}

COMPOUND_COL = 'Compound'
LABEL_COL    = 'C_Label'


# ---- helpers ----

def _load_mapping(mapping_csv: str) -> dict:
    df = pd.read_csv(mapping_csv)
    return df.groupby('experimental_id')['model_id'].apply(list).to_dict()


def _detect_mode(sheet_names: list) -> str:
    if any(re.match(r'^[A-Z]+(U|12)_\d+$', s) for s in sheet_names):
        return 'non_batch'
    if any(re.match(r'^[A-Z]+_(U|12)$', s) for s in sheet_names):
        return 'batch'
    raise ValueError(
        f"Cannot detect non-batch or batch mode from sheet names: {sheet_names}"
    )


def _get_non_batch_sheets(sheet_names: list, condition: str) -> list:
    results = []
    for s in sheet_names:
        m = re.match(r'^([A-Z]+)(U|12)_(\d+)$', s)
        if m and m.group(1) == condition:
            results.append((s, m.group(2), int(m.group(3))))
    return results


def _get_batch_sheets(sheet_names: list, prefix: str) -> list:
    results = []
    for s in sheet_names:
        m = re.match(r'^([A-Z]+)_(U|12)$', s)
        if m and m.group(1) == prefix:
            results.append((s, m.group(2)))
    return results


def _build_records(df: pd.DataFrame, tracer_name: str, bio_rep: int, met_map: dict) -> list:
    df = df.dropna(subset=[COMPOUND_COL])
    records = []
    for _, row in df.iterrows():
        exp_name = str(row[COMPOUND_COL])
        if exp_name not in met_map:
            continue
        isotopomer = f"M+{int(row[LABEL_COL])}"
        fraction   = float(row['fraction'])
        for model_id in met_map[exp_name]:
            records.append({
                'tracer_name':   tracer_name,
                'bio_replicate': bio_rep,
                'metabolite_id': model_id,
                'isotopomer':    isotopomer,
                'fraction':      fraction,
            })
    return records


def _parse_non_batch_sheet(
    xl: pd.ExcelFile,
    sheet_name: str,
    tracer_name: str,
    bio_rep: int,
    met_map: dict,
) -> list:
    df = xl.parse(sheet_name)
    data_cols = [c for c in df.columns if c not in (COMPOUND_COL, LABEL_COL)]
    df = df.copy()
    df['fraction'] = df[data_cols].mean(axis=1)
    return _build_records(df, tracer_name, bio_rep, met_map)


def _parse_batch_sheet(
    xl: pd.ExcelFile,
    sheet_name: str,
    tracer_name: str,
    met_map: dict,
) -> list:
    df = xl.parse(sheet_name)
    data_cols = [c for c in df.columns if c not in (COMPOUND_COL, LABEL_COL)]
    records = []
    for col in data_cols:
        m = re.search(r'\d+$', str(col))
        if m is None:
            continue
        bio_rep = int(m.group())
        sub = df[[COMPOUND_COL, LABEL_COL, col]].rename(columns={col: 'fraction'})
        records.extend(_build_records(sub, tracer_name, bio_rep, met_map))
    return records


# ---- main ----

def read_xlsx_mid_data(
    xlsx_path:    str,
    mapping_csv:  str,
    organism:     str,
    condition:    str,
    output_dir:   str = '.',
    batch_prefix: str = None,
) -> pd.DataFrame:
    """
    Read MID fraction data from experimental xlsx into long-format CSV.

    Parameters
    ----------
    xlsx_path : str
        Path to experimental MID xlsx.
    mapping_csv : str
        Path to metabolite_name_mapping.csv (from build_metabolite_name_mapping).
    organism : str
        Organism name used in the output filename (e.g. 'S_cervesiae').
    condition : str
        For non-batch: sheet prefix identifying the condition (e.g. 'P', 'N').
        For batch: label used only in the output filename (e.g. 'batch').
    output_dir : str
        Directory for the output CSV (default: current directory).
    batch_prefix : str, optional
        Organism code in batch sheet names (e.g. 'IO', 'SC').
        If None, inferred as first letter of each word in organism
        (e.g. 'I_orientalis' -> 'IO', 'S_cervesiae' -> 'SC').

    Returns
    -------
    pd.DataFrame with columns [tracer_name, bio_replicate, metabolite_id,
        isotopomer, fraction].
    """
    met_map = _load_mapping(mapping_csv)
    xl      = pd.ExcelFile(xlsx_path)
    mode    = _detect_mode(xl.sheet_names)

    records = []

    # ---- non-batch: each sheet = one bio replicate ----
    if mode == 'non_batch':
        sheets = _get_non_batch_sheets(xl.sheet_names, condition)
        if not sheets:
            raise ValueError(
                f"No sheets found for condition '{condition}' in {xlsx_path}. "
                f"Available sheets: {xl.sheet_names}"
            )
        for sheet_name, tracer_key, bio_rep in sheets:
            records.extend(
                _parse_non_batch_sheet(xl, sheet_name, TRACER_MAP[tracer_key], bio_rep, met_map)
            )

    # ---- batch: bio replicates are columns within each tracer sheet ----
    else:
        if batch_prefix is None:
            batch_prefix = ''.join(p[0].upper() for p in organism.split('_') if p)
        sheets = _get_batch_sheets(xl.sheet_names, batch_prefix)
        if not sheets:
            raise ValueError(
                f"No batch sheets found for prefix '{batch_prefix}' in {xlsx_path}. "
                f"Available sheets: {xl.sheet_names}"
            )
        for sheet_name, tracer_key in sheets:
            records.extend(
                _parse_batch_sheet(xl, sheet_name, TRACER_MAP[tracer_key], met_map)
            )

    df = pd.DataFrame(records, columns=[
        'tracer_name', 'bio_replicate', 'metabolite_id', 'isotopomer', 'fraction'
    ])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f"df_L_{organism}_{condition}_{timestamp}.csv"
    out_path  = f"{output_dir}/{filename}"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['bio_replicate'].nunique()} bio replicates, "
          f"{df['metabolite_id'].nunique()} metabolites) -> {out_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read MID xlsx into long-format df_L CSV.")
    parser.add_argument("xlsx_path",   help="Path to experimental MID xlsx")
    parser.add_argument("mapping_csv", help="Path to metabolite_name_mapping.csv")
    parser.add_argument("organism",    help="Organism name (e.g. S_cervesiae)")
    parser.add_argument("condition",   help="Condition prefix or label (e.g. P, N, batch)")
    parser.add_argument("--output_dir",   default=".", help="Output directory (default: .)")
    parser.add_argument("--batch_prefix", default=None,
                        help="Organism code in batch sheet names (e.g. IO); auto-inferred if omitted")
    args = parser.parse_args()

    read_xlsx_mid_data(
        xlsx_path    = args.xlsx_path,
        mapping_csv  = args.mapping_csv,
        organism     = args.organism,
        condition    = args.condition,
        output_dir   = args.output_dir,
        batch_prefix = args.batch_prefix,
    )
