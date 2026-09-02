"""
Build the E. coli core reference-flux workbook from Park et al. (2016).

Park et al., Nat Chem Biol 2016 (nchembio.2077) is the source of the LC-MS labeling in
data/E_coli_LC_MS/E_coli_B.xlsx, so its published fit is the like-for-like reference for
predictions made from that data -- unlike the Long & Antoniewicz GC-MS panel
(build_e_coli_reference_flux.py), which is a different experiment on a different platform.

Park_et_al_2016_Flux_data.xlsx already carries the crosswalk: its `core_id` column is the
model's own reaction id. What it does not carry is the column naming
``read_reference_fluxes`` reads -- `id` plus ``<condition><n>_mfa`` and its LB/UB
siblings -- so this script only reshapes and filters:

  1. Keeps rows whose `mapping` class is a defensible correspondence: `direct` (one Park
     row onto one core reaction) and `derived` (inferred from a node balance, with the
     reasoning on the row in `mapping_note`). Dropped are `not mapped` (no flux anyway)
     and `biomass drain (approximate)`, which the workbook's own README calls approximate
     -- R30 in particular is Park's acetate secretion plus the AcCoA drain to fatty
     acids, so it overstates pure acetate overflow -- and which publish no bounds.
  2. Renames the fluxes to `B1_mfa` / `B1_mfaLB` / `B1_mfaUB`. Park is a single
     chemostat state, and the LC-MS workbook has a single growth condition, so it is
     growth condition 1 of the 'B' (batch) condition letter that
     run_inference_workflow.py already names the matching prediction column with.

A bound Park does not report stays NaN rather than being filled. That is the same
statement the Long & Antoniewicz sentinel makes after ``UNBOUNDED_FLUX`` masking -- "this
fit does not bound the flux here" -- and the figure already draws it as an open-ended
error bar and excludes it from the interval-overlap count.

Fluxes are taken in mmol/gDW/h, the model's own units; the workbook's mM/min columns are
the same numbers before its documented x0.12 conversion.

Run this whenever Park_et_al_2016_Flux_data.xlsx changes:

    python models/atomistic/build_e_coli_core_park_reference.py
"""

import os

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

RAW = os.path.join(_ROOT, 'data/E_coli_LC_MS/Park_et_al_2016_Flux_data.xlsx')
OUT = os.path.join(_ROOT, 'data/E_coli_LC_MS/E_coli_core_park_reference_flux.xlsx')

#: Mapping classes that are a defensible 1:1 or node-balance correspondence.
SAFE_MAPPING = {'direct', 'derived'}

#: Source column -> reference-workbook column. Park is one state, so it is B1.
COLUMNS = {
    'Park_flux (mmol/gDW/h)': 'B1_mfa',
    'Park_LB (mmol/gDW/h)':   'B1_mfaLB',
    'Park_UB (mmol/gDW/h)':   'B1_mfaUB',
}


def main():
    park = pd.read_excel(RAW, sheet_name='fluxes')
    value_col = next(iter(COLUMNS))

    keep = park[park['mapping'].isin(SAFE_MAPPING) & park[value_col].notna()]
    out = (keep[['core_id', *COLUMNS]]
           .rename(columns={'core_id': 'id', **COLUMNS})
           .sort_values('id', key=lambda s: s.str.lower())
           .reset_index(drop=True))

    assert not out['id'].duplicated().any(), "core_id should be unique in the source sheet"
    out.to_excel(OUT, index=False)

    dropped = park[park[value_col].notna() & ~park['mapping'].isin(SAFE_MAPPING)]
    print(f"E. coli core (Park 2016): {len(out)} reactions -> {OUT}")
    print(f"  kept    : {keep['mapping'].value_counts().to_dict()}")
    print(f"  dropped : {dropped['mapping'].value_counts().to_dict()} "
          f"(plus {int(park[value_col].isna().sum())} rows with no Park flux)")
    print(f"  no published LB: {int(out['B1_mfaLB'].isna().sum())}, "
          f"no published UB: {int(out['B1_mfaUB'].isna().sum())}")


if __name__ == '__main__':
    main()
