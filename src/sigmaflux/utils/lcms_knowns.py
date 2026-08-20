"""
Match a model's metabolites against the compounds an LC/MS platform can detect.

The knowns table (data/HILIC_LCMS_knowns.csv) lists every compound the HILIC method
resolves.  Intersected with a model it gives the *measurable* metabolite space, which
is a larger and more stable thing than whichever subset a single experiment happened
to quantify: feature selection should be free to choose any token the platform could
supply, not only the ones this run did.

Matching is on name only.  Formula matching looks attractive but is not usable here:
136 of the 551 HILIC formulas are shared between isomers, and GEM formulas are not
reliable either — the M. musculus GEM gives succinate C4H4O4, which is fumarate's, so
a formula match sends succinate to the wrong compound.  Both formula-only hits found
in that model were wrong, whereas name matching resolved 96 metabolites cleanly.
"""
import re

import pandas as pd

# Metabolic models name acids as anions, LC/MS libraries as free acids: the model says
# succinate, the library says Succinic acid.  Folding both to a common stem recovers
# 20 further metabolites in the M. musculus model, including the whole TCA cycle.
_ACID_SUFFIXES = ('icacid', 'ate')


def normalize_compound(name: str) -> str:
    """Fold a metabolite name to a form comparable across naming conventions."""
    text = str(name).lower().strip()
    text = re.sub(r'^(l|d|dl)[-\s]', '', text)
    text = re.sub(r'[^a-z0-9]', '', text)
    for suffix in _ACID_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            return text[:-len(suffix)]
    return text


def read_lcms_knowns(knowns_path) -> pd.DataFrame:
    """Read the knowns table.  It is tab-separated despite the .csv extension."""
    knowns = pd.read_csv(knowns_path, sep='\t')
    missing = {'compound', 'formula'} - set(knowns.columns)
    if missing:
        raise ValueError(f"{knowns_path}: missing columns {sorted(missing)}")
    return knowns


def match_lcms_knowns(gem_xlsx, knowns_path, model_metabolite_ids=None,
                      gem_sheet='Metabolites', id_col='metid', name_col='metname',
                      metabolites_df=None):
    """Map LC/MS compound names onto model metabolite IDs by normalized name, returning a DataFrame [experimental_id, model_id] with one row per matching compartment.

    Args:
        gem_xlsx: GEM xlsx to read model metabolites from.
        knowns_path: path to the LC/MS knowns table.
        model_metabolite_ids: restrict matches to these ids, e.g. ones surviving conversion into the COBRA model.
        gem_sheet: sheet in gem_xlsx holding the metabolites.
        id_col: column in gem_sheet holding metabolite ids.
        name_col: column in gem_sheet holding metabolite names.
        metabolites_df: already-loaded [id_col, name_col] DataFrame, used instead of
            reading gem_xlsx -- lets a caller patch in fallback names first, for a
            GEM whose name_col is empty.
    """
    knowns = read_lcms_knowns(knowns_path)
    by_name = {}
    ambiguous = set()
    for compound in knowns['compound']:
        key = normalize_compound(compound)
        if key in by_name and by_name[key] != compound:
            ambiguous.add(key)
        by_name.setdefault(key, compound)

    if ambiguous:
        import warnings
        warnings.warn(
            f"LC/MS knowns contains {len(ambiguous)} ambiguous compound name(s) after normalization "
            f"(e.g. {sorted(list(ambiguous))[:5]}). Mappings may be incorrect for those names."
        )
    metabolites = (metabolites_df if metabolites_df is not None
                   else pd.read_excel(gem_xlsx, sheet_name=gem_sheet, usecols=[id_col, name_col]))
    if model_metabolite_ids is not None:
        keep = set(model_metabolite_ids)
        metabolites = metabolites[metabolites[id_col].astype(str).isin(keep)]

    rows = []
    for _, metabolite in metabolites.iterrows():
        compound = by_name.get(normalize_compound(metabolite[name_col]))
        if compound is not None:
            rows.append({'experimental_id': compound,
                         'model_id': str(metabolite[id_col])})
    return pd.DataFrame(rows, columns=['experimental_id', 'model_id'])
