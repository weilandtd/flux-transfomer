"""
Normalise the E. coli GEM/AMM workbooks into the canonical Maranas layout.

The two E. coli workbooks describe the same thing as the yeast and mouse ones but
spell it differently, in three ways that `convert_mfa_maranas` does not accept:

    Reactions sheet   'Rxn Formula'        instead of  'Reaction Formula'
    Atom Transition   'metabAbbreviation'  instead of  'metAbbreviation'
    metabolite ids    'ala-L[c]'           instead of  'ala-L_c'

The last one is the one that matters. The converter recognises the boundary
compartments by suffix — `_out` metabolites are collapsed into proper exchange
reactions and `_e` ones decide which exchanges get flipped to the secretion-positive
convention. Against bracket notation both tests silently return nothing, so an
unconverted iAF1260 keeps its 39 `[out]` metabolites and every exchange in its
original direction, with no error to say so.

Curating the workbooks rather than teaching the converter a second notation keeps
the organism-specific spelling in one place, next to the model it belongs to.

The workbooks are rewritten in place. The transform is idempotent — a second run
finds no brackets left to convert and no columns left to rename — so re-running is
a no-op and there is no parallel set of files to keep straight.

Run from the repository root:

    python models/atomistic/curate_e_coli_workbooks.py
"""
import pathlib
import re

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

CONFIGS = [
    dict(
        name='E_coli_core',
        gem='models/atomistic/E_coli_core/example_GEM.xlsx',
        amm='models/atomistic/E_coli_core/example_AMM.xlsx',
    ),
    dict(
        name='E_coli',
        gem='models/atomistic/E_coli/iAF1260_GEM.xlsx',
        amm='models/atomistic/E_coli/iAF1260_AMM.xlsx',
    ),
]

# Column headings that differ from the Maranas workbooks.
RENAME_REACTIONS = {'Rxn Formula': 'Reaction Formula', 'Rxn name': 'Rxn Name'}
RENAME_ATOM_TRANSITION = {'metabAbbreviation': 'metAbbreviation'}

_BRACKET = re.compile(r'\[([a-zA-Z]+)\]')

# The dilution sub-network, which SigmaFlux does not model. iAF1260 files it under
# an 'auxiliary: dilution' subsystem so `drop_subsystems` reaches it, but the core
# workbook puts everything in one subsystem, so it has to go by name.
#
# It has to go for a concrete reason, not tidiness. A dilution step is written
# `(0) Ala --> (1) Ala_d`: the zero coefficient means mass balance never sees it,
# while the atom mapping still says Ala(1,2,3) -> Ala_d(1,2,3), so labeling flows
# through. That is how the measured pool `Ala_d` gets diluted by `Ala_x`. COBRA
# discards the zero-coefficient reactant, leaving an atom mapping that points at a
# metabolite no longer in the reaction — exactly what convert_mfa_maranas warns of.
_DILUTION = re.compile(r'^dil_\d+$|_dil\d*$|^sink')

# Ad-hoc boundary reactions to drop once a proper exchange exists for the metabolite
# they served. `gluc_pre` additionally feeds off an orphan (`glc-D_pre` has no
# producer), so it can never carry flux.
DROP_REACTIONS = {'gluc_pre'}

# Removal of unidentifiable reactions without connection to carbon metabolism.
#
# These carry no carbon and cannot be identified from labeling data, yet each one
# adds a degree of freedom to the free-flux basis. iAF1260 offers two routes for the
# same job — two Mg transporters, dihydroorotate isozymes, duplicate phospholipid,
# FAD-reductase and siderophore routes — so biomass demand fixes the *sum* of a pair
# while nothing fixes the split between its members. A 13C experiment cannot resolve
# that split either, because no carbon passes through it.
#
# The consequence is not merely an unidentifiable flux, it is a broken selector. The
# sampler never explores those splits, so 11 of the 121 null-space directions arrive
# with no sampled variance at all (cov(theta) eigenvalues 1e-11 down to -1.5e-13).
# `pinv` and the prior ridge then read numerical noise out of the empty directions,
# giving an effective rank of 1.08, a 32.7-nat permutation null floor and *zero*
# selected tokens — shuffled data scores just as well in a direction that contains
# nothing, so no real measurement can clear the bar. Every other organism satisfies
# rank == d exactly; iAF1260 was the first that did not.
#
# Dropping the least-used member of each pair removes the free dimension and takes
# the deficit to 0. All eleven are trace-level (flux sd 0 to 6.6e-03) and carbon-free,
# so nothing observable is lost, and growth is unchanged at 0.475884 after every
# single removal.
UNIDENTIFIABLE_ROUTES = {
    'E_coli_core': set(),
    'E_coli': {'MG2uabcpp', 'GLYCTO3', 'PGP161abcpp', 'FADRx', 'PGPP181pp',
               'MLDCP1Bpp', 'PGPP160pp', 'DHORD5', 'FADRx2', 'FE3tex', 'NACODA'},
}

# Reactions that are bare sources/sinks rather than chemistry, and so become the
# exchange for the metabolite they feed. This is per model on purpose: the two
# workbooks use the same reaction names for different things. iAF1260's `gluc_up` is
# `--> glc-D_e`, a pure source, while the core model's `gluc_up` is
# `ATP + Gluc_e --> G6P` — the hexokinase step, which must be kept. Applying one
# map to both silently overwrote the core reaction.
SOURCES_AS_EXCHANGE = {
    'E_coli_core': {},
    'E_coli': {'gluc_up': 'glc-D_e', 'o2_in': 'o2_e'},
}

# Every model uses a different sentinel for "unbounded" (1e3, 1e8). Rescale to
# something a sampler can work with; the media block sets the real limits anyway.
BOUND_LIMIT = 100.0

# Metabolites that BIOMASS needs but the model cannot make, so biomass is pinned at
# zero flux. Both workbooks are MFA-purposed subsets rather than complete networks:
#   core     BIOMASS produces a `Biomass` metabolite that nothing consumes
#   iAF1260  four cofactors have no complete biosynthetic route in the subset
# Giving each an exchange lets biomass carry flux. None of them is carbon-tracked in
# the atom mapping, so this does not touch the labeling.
SUPPLIED = {
    'E_coli_core': ['Biomass'],
    'E_coli': ['mql8_c', 'q8h2_c', 'sheme_c', 'thmpp_c'],
}


def atom_map_rows(rxn_id, met, n_atoms):
    """AMM rows describing a depletion exchange `met -->`.

    An exchange needs an atom mapping like any other reaction. Without one the EMU
    expansion cannot follow the tracer's carbon out of the boundary metabolite and
    prunes the whole network — the labeling targets then come back empty.
    """
    return [{'rxnAbbreviation': rxn_id, 'metAbbreviation': met,
             'ReactantProductFlag': 'reactant',
             'maps': ','.join(str(i) for i in range(1, n_atoms + 1)),
             'startNodeSymbol': 'C', 'source': 'added by curation'}]


def canonicalise_boundaries(name, reactions, metabolites):
    """Give every extracellular metabolite a proper `EX_<met>` exchange.

    The two workbooks handle boundaries three different ways: iAF1260 names most of
    them `EX_ac(e)` but calls acetate `ac_out`, feeds glucose from a bare source
    `gluc_up -->  glc-D_e`, and the core model has no exchanges whatsoever. That last
    case is not merely untidy — `Gluc_e` has no producer, so steady state pins the
    whole model at zero flux.

    Exchanges are written in the depletion direction (`met_e -->`), which is already
    SigmaFlux's secretion-positive convention, so the converter leaves them alone.
    """
    reactions = reactions.copy()
    sources = SOURCES_AS_EXCHANGE[name]
    ids = set(reactions['Rxn ID'].astype(str))

    # EX_ac(e) -> EX_ac_e, so the exchange and its metabolite share a spelling
    renamed = {}
    for rxn_id in list(ids):
        m = re.fullmatch(r'EX_(.+)\((\w)\)', rxn_id)
        if m:
            renamed[rxn_id] = f"EX_{m.group(1)}_{m.group(2)}"
    for old, new in sources.items():
        if old in ids:
            renamed[old] = f"EX_{new}"
    reactions['Rxn ID'] = reactions['Rxn ID'].replace(renamed)

    # A source written `--> met_e` becomes the exchange `met_e -->`
    for old, met in sources.items():
        mask = reactions['Rxn ID'] == f"EX_{met}"
        reactions.loc[mask, 'Reaction Formula'] = f"(1) {met} -->"
        reactions.loc[mask, ['Lower bound', 'Upper bound']] = [-BOUND_LIMIT, BOUND_LIMIT]

    reactions = reactions[~reactions['Rxn ID'].isin(DROP_REACTIONS)]
    redundant = UNIDENTIFIABLE_ROUTES[name] & set(reactions['Rxn ID'].astype(str))
    if redundant:
        print(f"    dropped {len(redundant)} unidentifiable carbon-free routes: {sorted(redundant)}")
        reactions = reactions[~reactions['Rxn ID'].isin(redundant)]
    dilution = reactions['Rxn ID'].astype(str).str.contains(_DILUTION)
    if dilution.any():
        print(f"    dropped {dilution.sum()} dilution reactions")
        reactions = reactions[~dilution]

    ids = set(reactions['Rxn ID'].astype(str))
    external = [m for m in metabolites['metid'].astype(str) if m.endswith('_e')]
    external += [m for m in SUPPLIED[name] if m in set(metabolites['metid'].astype(str))]
    added = []
    for met in external:
        if f"EX_{met}" in ids:
            continue
        added.append({'Rxn ID': f"EX_{met}", 'Rxn Name': f"{met} exchange",
                      'Reaction Formula': f"(1) {met} -->",
                      'Lower bound': -BOUND_LIMIT, 'Upper bound': BOUND_LIMIT,
                      'subsystem': 'Exchange'})
    if added:
        reactions = pd.concat([reactions, pd.DataFrame(added)], ignore_index=True)

    # Collapse the sentinel "infinity" values onto a workable magnitude.
    for col, sign in (('Lower bound', -1), ('Upper bound', 1)):
        big = reactions[col].abs() > BOUND_LIMIT
        reactions.loc[big, col] = sign * BOUND_LIMIT * (reactions.loc[big, col] != 0)
    print(f"    boundaries: {len(renamed)} renamed, {len(added)} exchanges added, "
          f"{len(DROP_REACTIONS & set(ids))} dropped")
    return reactions


def to_canonical(text):
    """Rewrite bracket compartments as suffixes: 'ala-L[c]' -> 'ala-L_c'.

    Applied to whole reaction strings as well as bare ids, so it must leave
    everything else — coefficients, arrows, '+' — untouched. The hyphenated stereo
    tag is left alone: the fragment map in data/E_coli refers to these ids.
    """
    if not isinstance(text, str):
        return text
    return _BRACKET.sub(r'_\1', text)


def curate(name, gem, amm):
    gem_path, amm_path = REPO_ROOT / gem, REPO_ROOT / amm
    print(f"\n=== {name} ===")

    sheets = pd.read_excel(gem_path, sheet_name=None)
    reactions = sheets['Reactions'].rename(columns=RENAME_REACTIONS)
    reactions['Reaction Formula'] = reactions['Reaction Formula'].map(to_canonical)
    metabolites = sheets['Metabolites'].copy()
    metabolites['metid'] = metabolites['metid'].map(to_canonical)
    reactions = canonicalise_boundaries(name, reactions, metabolites)
    sheets['Reactions'], sheets['Metabolites'] = reactions, metabolites

    with pd.ExcelWriter(gem_path) as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    n_out = metabolites['metid'].str.endswith('_out').sum()
    print(f"  {gem_path.name}: {len(reactions)} reactions, "
          f"{len(metabolites)} metabolites ({n_out} boundary '_out')")

    sheets = pd.read_excel(amm_path, sheet_name=None)
    transitions = sheets['Atom Transition'].rename(columns=RENAME_ATOM_TRANSITION)
    transitions['metAbbreviation'] = transitions['metAbbreviation'].map(to_canonical)
    # The GEM renames have to reach the atom mapping too, or every renamed exchange
    # silently loses its mapping and the tracer cannot enter the network.
    # Derived from the ids themselves rather than from what this run happened to
    # rename, so it is correct even on a workbook that was partly curated already.
    transitions['rxnAbbreviation'] = transitions['rxnAbbreviation'].astype(str).apply(
        lambda r: re.sub(r'^EX_(.+)\((\w)\)$', r'EX_\1_\2', r))
    transitions['rxnAbbreviation'] = transitions['rxnAbbreviation'].replace(
        {old: f"EX_{met}" for old, met in SOURCES_AS_EXCHANGE[name].items()})
    # A route dropped from the GEM must not keep its atom mapping, or the mapping
    # describes chemistry the model no longer contains. Keyed off the final id set,
    # so re-running on an already-curated workbook is a no-op.
    orphaned = transitions['rxnAbbreviation'].isin(UNIDENTIFIABLE_ROUTES[name])
    if orphaned.any():
        print(f"    dropped {int(orphaned.sum())} atom transition(s) for the dropped routes")
        transitions = transitions[~orphaned]
    # Carbon count per metabolite, from the mappings that already exist.
    atoms = {}
    for met, grp in transitions.groupby('metAbbreviation'):
        atoms[met] = max(len(str(m).split(',')) for m in grp['maps'])
    # Any exchange in the model that carries carbon but has no mapping yet.
    mapped = set(transitions['rxnAbbreviation'])
    new_rows = []
    for rxn_id in reactions.loc[reactions['Rxn ID'].astype(str).str.startswith('EX_'), 'Rxn ID']:
        met = str(rxn_id)[len('EX_'):]
        if met in atoms and rxn_id not in mapped:
            new_rows += atom_map_rows(str(rxn_id), met, atoms[met])
    if new_rows:
        transitions = pd.concat([transitions, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"    atom mappings added for {len(new_rows)} exchange(s): "
              f"{[r['rxnAbbreviation'] for r in new_rows]}")
    if 'OriginalEquation' in transitions:
        transitions['OriginalEquation'] = transitions['OriginalEquation'].map(to_canonical)
    sheets['Atom Transition'] = transitions
    if 'Symmetry' in sheets and 'Metab_name' in sheets['Symmetry']:
        sheets['Symmetry'] = sheets['Symmetry'].copy()
        sheets['Symmetry']['Metab_name'] = sheets['Symmetry']['Metab_name'].map(to_canonical)

    with pd.ExcelWriter(amm_path) as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    print(f"  {amm_path.name}: {len(transitions)} atom transitions, "
          f"{transitions['rxnAbbreviation'].nunique()} reactions mapped")


if __name__ == '__main__':
    for cfg in CONFIGS:
        curate(**cfg)
    print("\nDone — workbooks normalised in place.")
