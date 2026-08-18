"""
Build *_annotated_hierarchical.json from the flat *_annotated.json models.

The aggregate BIOMASS reaction is split into the five macromolecule component
reactions (PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c), each producing a bm_*_c
pseudo-metabolite that BIOMASS then consumes at the macromolecule mass fraction.
Energy is redistributed out of BIOMASS into the components afterwards. This is what
the organism configs point `model_file` at, because the component fluxes are
separately measurable while the aggregate is not.

All four hierarchical models are regenerated here. S. cerevisiae and I. orientalis
used to be exempt -- only their annotations were resynced, because their stoichiometry
backed the released v1.0.0 .sfx ensembles -- but the freeze was hiding a defect rather
than protecting anything: in both models the sterol/lipid branch is pruned by x3cflux
as unreachable, which takes bm_lipid_c with it, and LIPID_c is a measured flux in both
configs. So `sigmaflux-benchmark-mfa --backend x3cflux` could not run on either.

Their published mass fractions are passed in explicitly from
`PUBLISHED_BIOMASS_FRACTIONS` rather than derived, so rebuilding picks up every
workbook fix while the composition stays exactly what was published and trained on.
tests/test_biomass_split.py pins the result against tests/data/biomass_split_v1.json,
captured from the shipped artifacts before this rebuild.

Not every model carries all five macromolecules — the T-cell biomass has no
carbohydrate, so CARB_c is skipped and the energy fractions are renormalised over the
four components that do exist.

Usage:
    python models/atomistic/build_hierarchical_models.py
"""
import pathlib
import sys

import pandas as pd
from cobra.io import load_json_model, save_json_model

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'src'))

from sigmaflux.core.utils import (  # noqa: E402
    BIOMASS_COMPONENT_IDS, PUBLISHED_BIOMASS_FRACTIONS, apply_terminal_annotations,
    derive_biomass_fractions, split_biomass_hierarchical,
)

CONFIGS = [
    # The two published yeasts. `fractions` is passed rather than derived so the
    # composition stays bit-for-bit what the v1.0.0 ensembles were trained on; see
    # PUBLISHED_BIOMASS_FRACTIONS.
    dict(
        flat_json='models/atomistic/S_cerevisiae/sc_annotated.json',
        gem_xlsx='models/atomistic/S_cerevisiae/scGEM.xlsx',
        amm_xlsx='models/atomistic/S_cerevisiae/scAMM.xlsx',
        out_json='models/atomistic/S_cerevisiae/sc_annotated_hierarchical.json',
        fractions=PUBLISHED_BIOMASS_FRACTIONS['S_cerevisiae'],
    ),
    dict(
        flat_json='models/atomistic/I_orientalis/imIsor_annotated.json',
        gem_xlsx='models/atomistic/I_orientalis/imIsor_GEM.xlsx',
        amm_xlsx='models/atomistic/I_orientalis/imIsor_AMM.xlsx',
        out_json='models/atomistic/I_orientalis/imIsor_annotated_hierarchical.json',
        fractions=PUBLISHED_BIOMASS_FRACTIONS['I_orientalis'],
    ),
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


if __name__ == '__main__':
    for cfg in CONFIGS:
        print(cfg['out_json'])
        build_hierarchical(**cfg)
