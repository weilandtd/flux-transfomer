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

Names that the two conventions spell irreconcilably go in ``_GEM_NAME_ALIASES`` rather
than being reached by loosening ``normalize_compound``.  Loosening is what silently lost
malate: the GEM writes ``(S)-malate`` and the library ``L-Malic acid``, and the leading
stereodescriptor is not one of the prefixes the normalizer strips, so the two folded to
``smal`` and ``mal`` and never met.  The obvious repair — strip the parenthesised
descriptor — maps ``(R)-lactate`` onto ``L-Lactic acid``, which is the wrong enantiomer,
and stripping leading locants collides 2-phosphoglycerate with 3-phosphoglycerate.  An
explicit table cannot do either.
"""
import re
import warnings

import pandas as pd

# Metabolic models name acids as anions, LC/MS libraries as free acids: the model says
# succinate, the library says Succinic acid.  Folding both to a common stem recovers
# 20 further metabolites in the M. musculus model, including the whole TCA cycle.
_ACID_SUFFIXES = ('icacid', 'ate')

#: GEM metabolite name (lowercased, whitespace-collapsed) -> compound in the knowns table.
#: Only for names the normalizer cannot bridge without also creating a wrong match.
_GEM_NAME_ALIASES = {
    '(s)-malate': 'L-Malic acid',
    '2-oxoglutarate': 'Oxoglutaric acid',
}

#: Chirality markers, as a *rejection* rule only.  Applied one way: two names that both
#: declare a configuration and disagree are not the same compound.  A name with no marker
#: constrains nothing.  This can only ever remove a match, never invent one.
_CHIRALITY_PREFIXES = (
    (re.compile(r'^\(\s*s\s*\)[-\s]'), 'L'),
    (re.compile(r'^\(\s*r\s*\)[-\s]'), 'D'),
    (re.compile(r'^l[-\s]'), 'L'),
    (re.compile(r'^d[-\s]'), 'D'),
)

_PHOSPHATE = re.compile(r'phosph')


def normalize_compound(name: str) -> str:
    """Fold a metabolite name to a form comparable across naming conventions."""
    text = str(name).lower().strip()
    text = re.sub(r'^(l|d|dl)[-\s]', '', text)
    text = re.sub(r'[^a-z0-9]', '', text)
    for suffix in _ACID_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            return text[:-len(suffix)]
    return text


def _alias_key(name: str) -> str:
    """Key a GEM name into ``_GEM_NAME_ALIASES``."""
    return re.sub(r'\s+', ' ', str(name).lower().strip())


def _chirality(name: str):
    """'L', 'D', or None -- the configuration a name declares, if it declares one."""
    text = str(name).lower().strip()
    for pattern, label in _CHIRALITY_PREFIXES:
        if pattern.match(text):
            return label
    return None


def _conflicts(gem_name: str, compound: str) -> str:
    """Reason these two names cannot be the same compound, or '' if none is provable."""
    a, b = _chirality(gem_name), _chirality(compound)
    if a is not None and b is not None and a != b:
        return f"chirality {a} vs {b}"
    if bool(_PHOSPHATE.search(str(gem_name).lower())) != bool(_PHOSPHATE.search(str(compound).lower())):
        return "one name is phosphorylated and the other is not"
    return ""


def read_lcms_knowns(knowns_path) -> pd.DataFrame:
    """Read the knowns table.  It is tab-separated despite the .csv extension."""
    knowns = pd.read_csv(knowns_path, sep='\t')
    missing = {'compound', 'formula'} - set(knowns.columns)
    if missing:
        raise ValueError(f"{knowns_path}: missing columns {sorted(missing)}")
    return knowns


def _validate_aliases(compounds) -> None:
    """Fail loudly on an alias that names a compound the platform does not list.

    An alias whose target has been renamed or removed would otherwise degrade silently
    back into the un-matched state this table exists to fix.
    """
    unknown = sorted(set(_GEM_NAME_ALIASES.values()) - set(compounds))
    if unknown:
        raise ValueError(
            f"_GEM_NAME_ALIASES names {len(unknown)} compound(s) absent from the knowns "
            f"table: {unknown}. Either the library changed or the alias is misspelled."
        )
    for gem_name, compound in _GEM_NAME_ALIASES.items():
        reason = _conflicts(gem_name, compound)
        if reason:
            raise ValueError(f"alias {gem_name!r} -> {compound!r} is inconsistent: {reason}")


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
    _validate_aliases(knowns['compound'])
    by_name = {}
    ambiguous = set()
    for compound in knowns['compound']:
        key = normalize_compound(compound)
        if key in by_name and by_name[key] != compound:
            ambiguous.add(key)
        by_name.setdefault(key, compound)

    if ambiguous:
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
    rejected = []
    for _, metabolite in metabolites.iterrows():
        name = metabolite[name_col]
        compound = (_GEM_NAME_ALIASES.get(_alias_key(name))
                    or by_name.get(normalize_compound(name)))
        if compound is None:
            continue
        reason = _conflicts(name, compound)
        if reason:
            rejected.append(f"{name!r} -> {compound!r} ({reason})")
            continue
        rows.append({'experimental_id': compound, 'model_id': str(metabolite[id_col])})

    if rejected:
        warnings.warn(
            f"Rejected {len(rejected)} name match(es) as provably different compounds: "
            f"{rejected[:5]}"
        )
    mapping = pd.DataFrame(rows, columns=['experimental_id', 'model_id'])
    _check_injective(mapping)
    return mapping


def _check_injective(mapping: pd.DataFrame) -> None:
    """Warn when one compound claims two metabolites, or one metabolite two compounds.

    A metabolite legitimately appears once per compartment, so the check is on the
    compartment-stripped id.  Either collision means the platform cannot tell the two
    apart and the mapping is asserting a measurement that does not exist.
    """
    if mapping.empty:
        return
    base = mapping['model_id'].str.rsplit('_', n=1).str[0]
    pairs = pd.DataFrame({'compound': mapping['experimental_id'], 'base': base}).drop_duplicates()

    for key, other, label in [('compound', 'base', 'compound'), ('base', 'compound', 'metabolite')]:
        counts = pairs.groupby(key)[other].nunique()
        clashing = counts[counts > 1]
        if len(clashing):
            warnings.warn(
                f"{len(clashing)} {label}(s) map to more than one counterpart, e.g. "
                f"{clashing.head(3).to_dict()}. The platform cannot resolve these separately."
            )
