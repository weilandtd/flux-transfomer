"""
Build *_annotated_hierarchical.json from the flat *_annotated.json models.

The aggregate BIOMASS reaction is split into the five macromolecule component
reactions (PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c), each producing a bm_*_c
pseudo-metabolite that BIOMASS then consumes at the macromolecule mass fraction.
Energy is redistributed out of BIOMASS into the components afterwards. This is what
the organism configs point `model_file` at, because the component fluxes are
separately measurable while the aggregate is not.

Only R. toruloides and the M. musculus T cell are regenerated here. The S. cerevisiae
and I. orientalis hierarchical models back the two released .sfx ensembles and must not
be rewritten; their published mass fractions are recorded in PUBLISHED_FRACTIONS for
reference, and tests/test_biomass_split.py pins the split against the S. cerevisiae
artifact.

Not every model carries all five macromolecules — the T-cell biomass has no
carbohydrate, so CARB_c is skipped and the energy fractions are renormalised over the
four components that do exist.

Usage:
    python models/atomistic/build_hierarchical_models.py
"""
import json
import pathlib
import sys
import warnings

import pandas as pd
from cobra.io import load_json_model, save_json_model

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'src'))

from sigmaflux.core.utils import (  # noqa: E402
    BIOMASS_COMPONENT_IDS, apply_terminal_annotations, derive_biomass_fractions,
    read_terminal_sheet, split_biomass_hierarchical,
)

# Mass fractions of the two shipped hierarchical models, for reference only.
PUBLISHED_FRACTIONS = {
    'S_cerevisiae': dict(PROTEIN_c=0.4888, RNA_c=0.0874, DNA_c=0.00354,
                         LIPID_c=0.02529, CARB_c=0.3949),
    'I_orientalis': dict(PROTEIN_c=0.47226, RNA_c=0.11098, DNA_c=0.00857,
                         LIPID_c=0.04562, CARB_c=0.36257),
}

CONFIGS = [
    dict(
        flat_json='models/atomistic/R_toruloides/imRt_annotated.json',
        gem_xlsx='models/atomistic/R_toruloides/imRt_GEM.xlsx',
        amm_xlsx='models/atomistic/R_toruloides/imRt_AMM.xlsx',
        out_json='models/atomistic/R_toruloides/imRt_annotated_hierarchical.json',
        fractions=None,   # derived from the BIOMASS stoichiometry
    ),
    dict(
        flat_json='models/atomistic/M_musculus_T_Cell/mmTcell_annotated.json',
        gem_xlsx='models/atomistic/M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
        amm_xlsx='models/atomistic/M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
        out_json='models/atomistic/M_musculus_T_Cell/mmTcell_annotated_hierarchical.json',
        # Derived (0.9396/0.0434/0.0142/0.0028) agrees with the measured activated
        # composition in MM_TCELL_exterinal_measurements.csv (0.9406/0.0458/0.0104/0.0033),
        # so keep the model's own stoichiometry as the basis.
        fractions=None,
    ),
]

# The two shipped models are not rebuilt, but curation still has to reach them: what the
# biomass split freezes is their stoichiometry, not the atom maps and terminal
# declarations the workbook owns. Both are annotations, so re-applying them leaves every
# stoichiometry byte untouched while the curation lands on the model the configs actually
# point `model_file` at. Without this, a recovered atom map (d9d7492) or a Terminal row
# reaches the flat model and silently stops there.
ANNOTATE_ONLY = [
    dict(hierarchical_json='models/atomistic/S_cerevisiae/sc_annotated_hierarchical.json',
         flat_json='models/atomistic/S_cerevisiae/sc_annotated.json',
         amm_xlsx='models/atomistic/S_cerevisiae/scAMM.xlsx'),
    dict(hierarchical_json='models/atomistic/I_orientalis/imIsor_annotated_hierarchical.json',
         flat_json='models/atomistic/I_orientalis/imIsor_annotated.json',
         amm_xlsx='models/atomistic/I_orientalis/imIsor_AMM.xlsx'),
]

# Annotations the flat model owns; everything else in a shipped model stays as shipped.
_SYNCED = (('reactions', 'atom_mapping'),
           ('metabolites', 'atom_pattern'),
           ('metabolites', 'symmetries'))

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def build_hierarchical(flat_json, gem_xlsx, amm_xlsx, out_json, fractions=None):
    flat_json = _REPO_ROOT / flat_json
    gem_xlsx  = _REPO_ROOT / gem_xlsx
    amm_xlsx  = _REPO_ROOT / amm_xlsx
    out_json  = _REPO_ROOT / out_json

    model = load_json_model(flat_json)

    if fractions is None:
        mets = pd.read_excel(gem_xlsx, sheet_name='Metabolites',
                             usecols=['metid', 'metformula'])
        formulas = dict(zip(mets['metid'].astype(str), mets['metformula']))
        fractions = derive_biomass_fractions(model, formulas)
        print("  derived mass fractions:")
        for cid in BIOMASS_COMPONENT_IDS:
            print(f"    {cid:<10} {fractions[cid]:.5f}")

    split_biomass_hierarchical(model, fractions)
    # Only now do the bm_* pools exist for the Terminal sheet to name.
    apply_terminal_annotations(model, amm_xlsx)
    save_json_model(model, str(out_json))

    biomass = model.reactions.get_by_id('BIOMASS')
    print(f"  {out_json.name}: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites")
    print(f"  BIOMASS consumes: {sorted(m.id for m in biomass.metabolites)}")


def resync_annotations(hierarchical_json, flat_json, amm_xlsx):
    """Refresh a shipped model's annotations from the workbook, changing nothing else.

    Patches the JSON directly rather than round-tripping through cobra: these two files
    were written out-of-band, each with its own serializer settings, so
    `save_json_model` would reformat all ~10k lines and bury the handful that actually
    change. The layout is read off the file itself and written back unchanged.
    """
    path = _REPO_ROOT / hierarchical_json
    raw = path.read_text()
    doc = json.loads(raw)

    opening = raw.split('\n', 2)[1]
    indent = len(opening) - len(opening.lstrip(' '))
    separators = (',', ': ' if '": ' in opening else ':')

    # The five biomass component reactions exist only in the hierarchical model, so a
    # missing id means the split created it, not that the annotation was lost.
    flat = json.loads((_REPO_ROOT / flat_json).read_text())
    for section, key in _SYNCED:
        source = {e['id']: e.get('annotation', {}).get(key) for e in flat[section]}
        for entry in doc[section]:
            fresh = source.get(entry['id'])
            stale = entry.get('annotation', {}).get(key)
            if entry['id'] not in source or fresh == stale:
                continue
            entry.setdefault('annotation', {})[key] = fresh
            print(f"    {section[:-1]} {entry['id']}: {key} refreshed from the workbook")

    pool_ids, rxn_ids = read_terminal_sheet(_REPO_ROOT / amm_xlsx)
    for idents, section, what in ((pool_ids, 'metabolites', 'pool'),
                                  (rxn_ids, 'reactions', 'reaction')):
        by_id = {entry['id']: entry for entry in doc[section]}
        for ident in idents:
            if ident not in by_id:
                warnings.warn(f"Terminal sheet names {what} {ident}, which is not in "
                              f"{path.name}", RuntimeWarning)
                continue
            by_id[ident].setdefault('annotation', {})['terminal'] = True

    path.write_text(json.dumps(doc, indent=indent, separators=separators))
    print(f"  {path.name}: terminal pools {pool_ids}, reactions {rxn_ids}")


if __name__ == '__main__':
    for cfg in CONFIGS:
        print(cfg['out_json'])
        build_hierarchical(**cfg)
    for cfg in ANNOTATE_ONLY:
        print(cfg['hierarchical_json'])
        resync_annotations(**cfg)
