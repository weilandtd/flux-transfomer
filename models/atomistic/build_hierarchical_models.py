"""
Build *_annotated_hierarchical.json from the flat *_annotated.json models.

The aggregate BIOMASS reaction is split into the five macromolecule component
reactions (PROTEIN_c, RNA_c, DNA_c, LIPID_c, CARB_c), each producing a bm_*_c
pseudo-metabolite that BIOMASS then consumes at the macromolecule mass fraction.
Energy is redistributed out of BIOMASS into the components afterwards. This is what
the organism configs point `model_file` at, because the component fluxes are
separately measurable while the aggregate is not.

Only R. toruloides is regenerated here. The S. cerevisiae and I. orientalis
hierarchical models back the two released .sfx ensembles and must not be rewritten;
their published mass fractions are recorded in PUBLISHED_FRACTIONS for reference, and
tests/test_biomass_split.py pins the split against the S. cerevisiae artifact.

Usage:
    python models/atomistic/build_hierarchical_models.py
"""
import pathlib
import sys

import pandas as pd
from cobra.io import load_json_model, save_json_model

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'src'))

from sigmaflux.core.utils import (  # noqa: E402
    BIOMASS_COMPONENT_IDS, derive_biomass_fractions, split_biomass_hierarchical,
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
        out_json='models/atomistic/R_toruloides/imRt_annotated_hierarchical.json',
        fractions=None,   # derived from the BIOMASS stoichiometry
    ),
]

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def build_hierarchical(flat_json, gem_xlsx, out_json, fractions=None):
    flat_json = _REPO_ROOT / flat_json
    gem_xlsx  = _REPO_ROOT / gem_xlsx
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
    save_json_model(model, str(out_json))

    biomass = model.reactions.get_by_id('BIOMASS')
    print(f"  {out_json.name}: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites")
    print(f"  BIOMASS consumes: {sorted(m.id for m in biomass.metabolites)}")


if __name__ == '__main__':
    for cfg in CONFIGS:
        print(cfg['out_json'])
        build_hierarchical(**cfg)
