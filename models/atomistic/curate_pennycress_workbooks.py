"""
Normalise the Pennycress (T. arvense) GEM/AMM workbooks and GC/MS data into the canonical layout.

The workbooks transcribe the influx_s FTBL of Shi et al. (WT-G, WT-A1, WT-A2), which spells
the model the FTBL way, not the way `convert_mfa_maranas` and the eukaryotic models do:

    compartments    bare suffixes (HPc, HPp, AcCoAp)   instead of  HP_c, HP_h, AcCoA_h
    inputs          dead-end pools (GLCU, Ala_l)        instead of  EX_ exchanges on _e pools
    outputs         dead-end pools (Ala_ex)             instead of  drains
    natural 13C     C<n>nat inputs into each *_bm pool  instead of  corrected MIDs
    symmetry        tied scrambling pairs (vKDHa/b)     instead of  symmetric metabolites

Plastid is ``_h`` (BiGG's plastid/chloroplast), since ``_p`` means periplasm to the
measurement-pool code. Pools the FTBL lumps across compartments (OAA = OAA + malate, Cit,
KG, Succ, the amino acids) stay suffix-free.

The FTBL simulates natural 13C by feeding every measured ``*_bm`` pool from a ``C<n>nat``
input; here the MIDs are natural-abundance corrected instead (the fragment mapping), so those
legs are dropped and each ``*_bm`` pool is a pass-through of its metabolite.

Each FTBL scrambling pair is the same reaction written in two atom orientations and tied
50/50 by an equality. That is a symmetric molecule, so the pair becomes one reaction and the
molecule is declared symmetric: succinate (KDH, FM, ICL), glycerol (a new ``Glyc_c`` ahead
of ``Glyc_bm``, so the partial Glyc253 fragment is still cut from a non-symmetric pool),
diaminopimelate (a new ``DAP`` between Asp + Pyr and Lys) and shikimate, whose two
orientations differ by reversing carbons 3-7. ``vGlyc_out`` (measured) is the glycerol
synthesis step.

The GC/MS data are rewritten to the loader's ``<Prefix><n>_Rep<k>`` / ``<Prefix><n>``
convention: P1 is WT, P2 is the WD3 line (strain_index.csv). The published wild-type
fluxes (Shi et al. S15) become the reference workbook, under the curated reaction ids;
an S15 sum over a scrambling pair is the merged reaction's flux.

The source workbooks are left untouched; the canonical copies are written next to them.

Run from the repository root:

    python models/atomistic/curate_pennycress_workbooks.py
"""
import pathlib
import re

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
MODEL_DIR = REPO_ROOT / 'models/atomistic/T_arvense_L'
DATA_DIR = REPO_ROOT / 'data/T_arvense_L_GC_MS'

GEM_IN, AMM_IN = MODEL_DIR / 'Pennycress_GEM.xlsx', MODEL_DIR / 'Pennycress_AMM.xlsx'
GEM_OUT, AMM_OUT = MODEL_DIR / 'taGEM.xlsx', MODEL_DIR / 'taAMM.xlsx'

#: FTBL condition -> loader condition id.
CONDITIONS = {'WT': 'P1', 'WD3': 'P2'}

RENAME = {
    'HPc': 'HP_c', 'HPp': 'HP_h', 'TPc': 'TP_c', 'TPp': 'TP_h',
    'PGAc': 'PGA_c', 'PGAp': 'PGA_h', 'PEPc': 'PEP_c', 'PEPp': 'PEP_h',
    'RuPc': 'RuP_c', 'RuPp': 'RuP_h', 'XPc': 'XP_c', 'XPp': 'XP_h',
    'RPp': 'RP_h', 'EPp': 'EP_h', 'SPp': 'SP_h',
    'AcCoAc': 'AcCoA_c', 'AcCoAp': 'AcCoA_h', 'Pyr_p': 'Pyr_h',
    'Alac': 'Ala_c', 'Ac_p_bm': 'Ac_h_bm',
    'GLCU': 'Glc_e', 'Ala_l': 'Ala_e',
}
NATURAL = re.compile(r'C\d+nat')

#: The FTBL feeds glutamine from C5nat; as a medium component it gets its own exchange.
GLN_UPTAKE = 'vuptGln'

#: (kept, dropped, merged id) per scrambling pair.
MERGES = [('vKDHa', 'vKDHb', 'vKDH'), ('vFMa', 'vFMb', 'vFM'), ('vICLa', 'vICLb', 'vICL'),
          ('vAra', 'vArb', 'vAr'), ('vGlyc_outa', 'vGlyc_outb', 'vGlycP')]

#: (metabolite, default mapping, symmetry mapping) for the Symmetry sheet.
SYMMETRIES = [('Succ', '1 2 3 4', '4 3 2 1'), ('Glyc_c', '1 2 3', '3 2 1'),
              ('DAP', '1 2 3 4 5 6 7', '7 6 5 4 3 2 1'),
              ('Shikimat', '1 2 3 4 5 6 7', '1 2 7 6 5 4 3')]

#: New reactions: id -> (name, formula, [(metabolite, side, maps)]).
NEW_REACTIONS = {
    'vGlyc_out': ('glycerol synthesis', 'TP_c --> Glyc_c',
                  [('TP_c', 'reactant', '1,2,3'), ('Glyc_c', 'product', '1,2,3')]),
    'vDAP': ('diaminopimelate synthesis (lumped)', 'Asp + Pyr_h --> DAP',
             [('Asp', 'reactant', '1,2,3,4'), ('Pyr_h', 'reactant', '5,6,7'),
              ('DAP', 'product', '1,2,3,4,7,6,5')]),
    'vLys': ('diaminopimelate decarboxylase', 'DAP --> Lys + CO2',
             [('DAP', 'reactant', '1,2,3,4,5,6,7'), ('Lys', 'product', '1,2,3,4,5,6'),
              ('CO2', 'product', '7')]),
}
REPLACED = {'vLys_a', 'vLys_b'}
NEW_METABOLITES = [('Glyc_c', 'glycerol (cytosolic)', 'C3'),
                   ('DAP', 'diaminopimelate', 'C7'),
                   ('Gln_e', 'glutamine, medium', 'C5')]

#: The workbook files the reactions after vLys (glycerol, starch, sucrose, cell wall, fatty
#: acids, citrate) under "Val, Leu, Ile metabolism", a fill-down slip; keyed by product.
SUBSYSTEMS = {
    'Glyc': 'Glycerol synthesis', 'St': 'Starch, sucrose and cell wall',
    'GLC': 'Starch, sucrose and cell wall', 'HPc': 'Starch, sucrose and cell wall',
    'Ac': 'Fatty acid synthesis', 'Cit': 'TCA, Glyoxylate cycle',
}

#: (id, name, carbons) of every medium pool, each opened by an ``EX_<id>`` exchange.
INPUTS = [('Glc_e', 'glucose, medium', 6), ('Ala_e', 'alanine, medium', 3),
          ('Gln_e', 'glutamine, medium', 5)]


def _rename_formula(formula, mapping):
    """*formula* with each whole metabolite token renamed through *mapping*."""
    return re.sub(r'[A-Za-z0-9_]+', lambda m: mapping.get(m.group(0), m.group(0)), formula)


def _amm_rows(rxn_id, entries, source):
    return [dict(maps=maps, startNodeSymbol='C', metAbbreviation=met,
                 ReactantProductFlag=side, rxnAbbreviation=rxn_id, Source=source)
            for met, side, maps in entries]


def curate(gem, amm):
    """Canonical GEM and AMM sheets, from the workbooks as transcribed."""
    rxns, mets = gem['Reactions'].copy(), gem['Metabolites'].copy()
    at = amm['Atom Transition'].copy()
    outputs = set(mets.loc[mets['metid'].str.endswith('_ex'), 'metid'])

    # Natural-13C legs: the reactions fed by a C<n>nat input, bar glutamine uptake.
    natural_in = {r['Rxn ID'] for _, r in rxns.iterrows() if r['Rxn ID'] != GLN_UPTAKE
                  and NATURAL.fullmatch(r['Reaction Formula'].split('-->')[0].strip())}
    dropped = natural_in | REPLACED | {b for _, b, _ in MERGES}
    rxns = rxns[~rxns['Rxn ID'].isin(dropped)].copy()
    at = at[~at['rxnAbbreviation'].isin(dropped) & ~at['metAbbreviation'].isin(outputs)].copy()

    gln = rxns['Rxn ID'] == GLN_UPTAKE
    rxns.loc[gln, 'Reaction Formula'] = rxns.loc[gln, 'Reaction Formula'].str.replace(
        'C5nat', 'Gln_e')
    at.loc[(at['rxnAbbreviation'] == GLN_UPTAKE) & (at['metAbbreviation'] == 'C5nat'),
           'metAbbreviation'] = 'Gln_e'

    def formula(f):
        f = _rename_formula(f, RENAME)
        for out in outputs:
            f = re.sub(rf'\b{out}\b', '', f)
        return f.strip()

    rxns['Reaction Formula'] = rxns['Reaction Formula'].map(formula)
    at['metAbbreviation'] = at['metAbbreviation'].map(lambda m: RENAME.get(m, m))

    merged = {a: new for a, _, new in MERGES}
    rxns['Rxn ID'] = rxns['Rxn ID'].replace(merged)
    at['rxnAbbreviation'] = at['rxnAbbreviation'].replace(merged)
    for rid in [new for a, _, new in MERGES]:
        rxns.loc[rxns['Rxn ID'] == rid, 'notes'] = \
            'Merged FTBL scrambling pair; the scrambling is a symmetry annotation'
    glycp = rxns['Rxn ID'] == 'vGlycP'
    rxns.loc[glycp, 'Reaction Formula'] = 'Glyc_c --> Glyc_bm'
    rxns.loc[glycp, 'macro/bm'] = 0
    at.loc[(at['rxnAbbreviation'] == 'vGlycP') & (at['ReactantProductFlag'] == 'reactant'),
           'metAbbreviation'] = 'Glyc_c'

    template = gem['Reactions'].set_index('Rxn ID')
    new = []
    for rid, (name, f, entries) in NEW_REACTIONS.items():
        like = 'vGlyc_outa' if rid == 'vGlyc_out' else 'vLys_a'
        row = template.loc[like].copy()
        row['Rxn Name'], row['Reaction Formula'] = name, f
        row['notes'] = 'Curated: replaces the FTBL scrambling pair ' + \
            ('vGlyc_outa/b' if rid == 'vGlyc_out' else 'vLys_a/b')
        new.append(row.rename(rid))
        at = pd.concat([at, pd.DataFrame(_amm_rows(rid, entries, 'curated | symmetry'))],
                       ignore_index=True)
    rxns = pd.concat([rxns, pd.DataFrame(new).rename_axis('Rxn ID').reset_index()],
                     ignore_index=True)

    for product, subsystem in SUBSYSTEMS.items():
        block = rxns['Reaction Formula'].str.contains(rf'\b{product}_[a-z_]*bm\b')
        rxns.loc[block, 'subsystem'] = subsystem

    exchanges = pd.DataFrame([{
        'Rxn ID': f'EX_{met}', 'Rxn Name': f'{name} exchange',
        'Reaction Formula': f'{met[:-2]}_out --> {met}',
        'Lower bound': 0, 'Upper bound': 1000, 'Reversibility': 0, 'macro/bm': 0,
        'subsystem': 'Exchange'} for met, name, _ in INPUTS])
    rxns = pd.concat([rxns, exchanges], ignore_index=True)
    for met, _, n in INPUTS:
        carbons = ','.join(str(i) for i in range(1, n + 1))
        at = pd.concat([at, pd.DataFrame(_amm_rows(
            f'EX_{met}', [(met, 'reactant', carbons), (f'{met[:-2]}_out', 'product', carbons)],
            'curated | exchange'))], ignore_index=True)
    at = at.drop(columns='OriginalEquation')

    mets = mets[~mets['metid'].isin(outputs) & ~mets['metid'].str.fullmatch(NATURAL)].copy()
    mets['metid'] = mets['metid'].map(lambda m: RENAME.get(m, m))
    added = [{'metid': m, 'metname': name, 'metformula': formula}
             for m, name, formula in NEW_METABOLITES]
    added += [{'metid': f'{met[:-2]}_out', 'metname': f'{name} (boundary)', 'metformula': ''}
              for met, name, _ in INPUTS]
    mets = pd.concat([mets, pd.DataFrame(added)], ignore_index=True)

    symmetry = pd.DataFrame(SYMMETRIES, columns=['Metab_name', 'default mapping',
                                                 'symmetry mapping'])
    terminal = amm['Terminal'][~amm['Terminal']['Metab_name'].isin(outputs)]
    return ({'Reactions': rxns, 'Metabolites': mets},
            {'Atom Transition': at, 'Symmetry': symmetry, 'Terminal': terminal},
            outputs, natural_in)


def curate_data():
    """Loader-convention copies of the labeling and ExtFlux workbooks, plus strain_index."""
    columns = {f'{ftbl}_Rep1': f'{cond}_Rep1' for ftbl, cond in CONDITIONS.items()}
    labeling = pd.read_excel(DATA_DIR / 'Pennycress_Labeling.xlsx', sheet_name=None)
    with pd.ExcelWriter(DATA_DIR / 'Pennycress_P.xlsx') as writer:
        for sheet, df in labeling.items():
            df.rename(columns=columns).to_excel(writer, sheet_name=sheet, index=False)

    ext = pd.read_excel(DATA_DIR / 'Pennycress_ExtFlux.xlsx', sheet_name='Constraints')
    ext['Condition'] = ext['Condition'].map(CONDITIONS)
    with pd.ExcelWriter(DATA_DIR / 'T_arvense_L_ExtFlux.xlsx') as writer:
        ext.to_excel(writer, sheet_name='Constraints', index=False)

    pd.DataFrame({'condition': list(CONDITIONS.values()), 'strain': list(CONDITIONS)}) \
        .to_csv(DATA_DIR / 'strain_index.csv', index=False)


def curate_reference():
    """Wild-type Shi et al. S15 net fluxes as the reference workbook (P1 only)."""
    shi = pd.read_excel(DATA_DIR / 'Shi_et_al_flux_data.xlsx').set_index('id')
    ref = shi[['WT_mfa', 'WT_mfaLB', 'WT_mfaUB']].set_axis(['P1_mfa', 'P1_mfaLB', 'P1_mfaUB'], axis=1)
    # DAP and Glyc_c carry the same flux onwards.
    aliases = ref.loc[['vLys', 'vGlyc_out']].set_axis(['vDAP', 'vGlycP'])
    pd.concat([ref, aliases]).rename_axis('id').reset_index().to_excel(
        DATA_DIR / 'T_arvense_L_reference_flux.xlsx', index=False)
    return len(ref) + len(aliases)


def main():
    gem, amm, outputs, natural_in = curate(pd.read_excel(GEM_IN, sheet_name=None),
                                           pd.read_excel(AMM_IN, sheet_name=None))
    for path, sheets in ((GEM_OUT, gem), (AMM_OUT, amm)):
        with pd.ExcelWriter(path) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    print(f"  {len(gem['Reactions'])} reactions, {len(gem['Metabolites'])} metabolites; "
          f"dropped {len(outputs)} output pools and {len(natural_in)} natural-13C legs, "
          f"merged {len(MERGES) + 1} scrambling pairs")
    curate_data()
    n_ref = curate_reference()
    print(f"Wrote {DATA_DIR.relative_to(REPO_ROOT)}/Pennycress_P.xlsx, "
          f"T_arvense_L_ExtFlux.xlsx, strain_index.csv, "
          f"T_arvense_L_reference_flux.xlsx ({n_ref} wild-type fluxes)")


if __name__ == '__main__':
    main()
