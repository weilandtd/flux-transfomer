"""
Sanity-check a metabolite mapping against how many carbons the model actually tracks.

A name match alone (`build_data_name_maps`, `match_lcms_knowns`) says nothing about
whether the model's atom map covers the whole molecule. A cofactor like CoA or NAD is
often carried with zero tracked carbons -- present, mass-balanced, but with no
`atom_pattern` annotation at all, because nothing in the model's chemistry labels its
ring carbons. Acetyl-CoA is the partial case: the model tracks the acetyl unit's 2
carbons, not the 23 that belong to the full molecule. Either way, the model's
predicted MID for that pool is not a simulation of what an LC-MS instrument would see
for the real molecule, so it should not be trusted as a measurement -- only a
metabolite whose tracked carbon count matches its real carbon count can be.
"""
import re

import pandas as pd

from .lcms_knowns import normalize_compound, read_lcms_knowns


def carbon_count(formula) -> int | None:
    """Number of carbon atoms in a Hill-notation formula string, or None if unparseable.

    Args:
        formula: chemical formula, e.g. "C21H36N7O16P3S".
    """
    if not isinstance(formula, str) or not formula.strip():
        return None
    total, found = 0, False
    for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if not element:
            continue
        found = True
        if element == 'C':
            total += int(count) if count else 1
    return total if found else None


def tracked_carbon_count(model, model_id: str) -> int:
    """Carbons *model_id*'s atom map actually tracks, via its `atom_pattern` annotation; 0 if the metabolite is absent from *model* or carries no atom map."""
    try:
        met = model.metabolites.get_by_id(model_id)
    except KeyError:
        return 0
    return len(met.annotation.get('atom_pattern', ''))


def formulas_from_knowns(knowns_path) -> dict:
    """Normalized compound name -> chemical formula, from the HILIC LC/MS knowns table.

    Args:
        knowns_path: path to the LC/MS knowns table (data/HILIC_LCMS_knowns.csv).
    """
    knowns = read_lcms_knowns(knowns_path)
    formulas = {}
    for _, row in knowns.iterrows():
        formulas.setdefault(normalize_compound(row['compound']), row['formula'])
    return formulas


def filter_by_carbon_coverage(mapping_df: pd.DataFrame, model, formula_by_name: dict) -> pd.DataFrame:
    """Drop mapping rows whose model_id does not track every carbon of the real molecule.

    A row is kept unchecked (and so kept) when its experimental_id has no known real
    formula to check against -- there is nothing to disagree with.

    Args:
        mapping_df: DataFrame with `experimental_id`/`model_id` columns to filter.
        model: cobra Model, atom_mapping-annotated (its metabolites carry `atom_pattern`).
        formula_by_name: normalized compound name -> chemical formula, e.g. from
            `formulas_from_knowns`.
    """
    keep, dropped = [], []
    for _, row in mapping_df.iterrows():
        real_c = carbon_count(formula_by_name.get(normalize_compound(row['experimental_id'])))
        if real_c is None:
            keep.append(True)
            continue
        mapped_c = tracked_carbon_count(model, row['model_id'])
        ok = mapped_c == real_c
        keep.append(ok)
        if not ok:
            dropped.append((row['experimental_id'], row['model_id'], mapped_c, real_c))

    if dropped:
        detail = ", ".join(f"{eid}/{mid} ({got}/{want} C)" for eid, mid, got, want in dropped)
        print(f"  Carbon coverage filter: dropped {len(dropped)} of {len(mapping_df)} "
              f"row(s) whose tracked carbons don't match the real molecule: {detail}")
    return mapping_df[pd.Series(keep, index=mapping_df.index)].reset_index(drop=True)
