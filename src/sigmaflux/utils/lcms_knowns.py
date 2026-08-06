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
                      gem_sheet='Metabolites', id_col='metid', name_col='metname'):
    """Map LC/MS compound names onto model metabolite IDs.

    The GEM supplies the metabolite names; `model_metabolite_ids`, when given, keeps
    only metabolites that survived conversion into the COBRA model.

    Returns a DataFrame [experimental_id, model_id] in the same shape as the mapping
    CSVs, where experimental_id is the LC/MS compound name.  One compound maps to
    every matching compartment, so cytosolic and mitochondrial pools both appear.
    """
    knowns = read_lcms_knowns(knowns_path)
    by_name = {}
    for compound in knowns['compound']:
        by_name.setdefault(normalize_compound(compound), compound)

    metabolites = pd.read_excel(gem_xlsx, sheet_name=gem_sheet, usecols=[id_col, name_col])
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
