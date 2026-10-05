"""
Build the reference-flux workbooks `_ref_by_gc` (analysis_predicted_fluxes.py) expects
for E. coli core and E. coli genome-scale, from the raw Long & Antoniewicz (2019)
workbook.

Long_Antoniewicz_2019_fluxresults.xlsx is keyed by core_id (row) and knockout strain
(column, `<strain>_mfa`) — neither matches `_ref_by_gc`'s `id`-indexed,
`<condition><n>_mfa`-columned expectation. This script:
  1. Reindexes rows from core_id to each organism's own reaction id, using
     E_coli_core_reaction_mapping.csv's `our_id` (core) / `genomescale_id`, translated
     into iAF1260's own naming (genome-scale) -- restricted to the unambiguous
     exact/exact-synonym/exact-fwd/exact-context rows (65 of 79; the other 14 need
     many-to-many or hand-verification this crosswalk does not attempt).
  2. Renames strain columns to `B<n>_mfa` (and, alongside it, `B<n>_mfaLB`/`B<n>_mfaUB`
     for the published confidence bounds) via data/E_coli/strain_index.csv, so they
     read as growth conditions of the 'B' (batch) condition letter, matching how
     run_inference_workflow.py already names the corresponding prediction columns.
  3. Replaces the BIOMASS row with the real specific growth rate (h^-1) from
     data/E_coli/EColi_ExtFlux_raw_by_strain.xlsx. core_id 75 ("biomass_out") is a
     Cmol-basis flux -- a different physical quantity from the model's BIOMASS
     reaction (which *is* the growth rate, by the standard FBA normalization) --
     matched into this crosswalk only by name/context (match_type exact-context, ~37x
     off numerically). The growth-rate column is what already feeds the model's own
     BIOMASS measured-flux input at inference, so reusing it here keeps the reference
     and the model's own conditioning value consistent instead of introducing an
     unverified Cmol-to-growth-rate conversion factor.

Run this whenever Long_Antoniewicz_2019_fluxresults.xlsx or
E_coli_core_reaction_mapping.csv changes:

    .venv/bin/python models/atomistic/build_e_coli_reference_flux.py
"""

import json
import os

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

RAW_MFA        = os.path.join(_ROOT, 'data/E_coli/Long_Antoniewicz_2019_fluxresults.xlsx')
REACTION_MAP   = os.path.join(_ROOT, 'models/atomistic/E_coli_core/E_coli_core_reaction_mapping.csv')
STRAIN_INDEX   = os.path.join(_ROOT, 'data/E_coli/strain_index.csv')
GROWTH_RATES   = os.path.join(_ROOT, 'data/E_coli/EColi_ExtFlux_raw_by_strain.xlsx')
IAF1260_MODEL  = os.path.join(_ROOT, 'models/atomistic/E_coli/iAF1260_annotated.json')
OUT_CORE       = os.path.join(_ROOT, 'data/E_coli/E_coli_core_reference_flux.xlsx')
OUT_GENOMESCALE = os.path.join(_ROOT, 'data/E_coli/E_coli_reference_flux.xlsx')

# Only these match_types are unambiguous 1:1 crosswalks; the rest (different-decomposition,
# exchange-pair, near-match, no-match, structural) need many-to-many or hand-verified
# stoichiometry this script does not attempt -- see E_coli_core_reaction_mapping.csv's
# own `note` column for what's missing and why.
SAFE_MATCH_TYPES = {"exact", "exact-synonym", "exact-fwd", "exact-context"}


def _to_genomescale_id(gs: str) -> str:
    """L&A's own genome-scale id, translated into iAF1260's naming where it differs."""
    if gs == "biomass_out":
        return "BIOMASS"
    if gs.startswith("EX_") and gs.endswith("(e)"):
        return gs[:-3] + "_e"
    return gs


def _reshape(mfa: pd.DataFrame, id_col: str, strain_index: pd.DataFrame) -> pd.DataFrame:
    """Reindex by `id_col`, rename `<strain>_mfa`/`_mfaLB`/`_mfaUB` columns to
    `B<n>_mfa`/`_mfaLB`/`_mfaUB` -- carries the published confidence bounds through
    alongside the point value, for figure2.py's error bars."""
    out = mfa.set_index(id_col)
    strain_to_b = {s.lower(): c for c, s in
                   zip(strain_index['condition'], strain_index['strain'])}
    rename = {}
    for col in out.columns:
        for suffix in ('_mfa', '_mfaLB', '_mfaUB'):
            if col.endswith(suffix):
                strain = col[:-len(suffix)].lower()
                if strain in strain_to_b:
                    rename[col] = f"{strain_to_b[strain]}{suffix}"
                break
    out = out[list(rename)].rename(columns=rename)
    out.index.name = 'id'
    return out.reset_index()


def _fix_biomass_row(out: pd.DataFrame, strain_index: pd.DataFrame) -> pd.DataFrame:
    """Replace the BIOMASS row's values with the real growth rate (h^-1) per strain."""
    rates = pd.read_excel(GROWTH_RATES, sheet_name='Constraints')[['strain', 'growth_rate_/h']]
    strain_to_b = {s.lower(): c for c, s in
                   zip(strain_index['condition'], strain_index['strain'])}
    rates['id'] = 'BIOMASS'
    rates['B'] = rates['strain'].str.lower().map(strain_to_b)
    by_b = rates.set_index('B')['growth_rate_/h']

    mask = out['id'] == 'BIOMASS'
    for col in out.columns:
        if not col.endswith('_mfa'):
            continue
        b = col[:-len('_mfa')]
        if b in by_b.index:
            out.loc[mask, col] = by_b[b]
    return out


def main():
    mfa = pd.read_excel(RAW_MFA, sheet_name='fluxes_absolute')
    mapping = pd.read_csv(REACTION_MAP)
    strain_index = pd.read_csv(STRAIN_INDEX)
    iaf1260_ids = {r['id'] for r in json.load(open(IAF1260_MODEL))['reactions']}

    safe = mapping[mapping['match_type'].isin(SAFE_MATCH_TYPES)].copy()
    assert not safe['our_id'].duplicated().any(), "our_id should be unique among safe rows"

    # ── E. coli core ──
    core_mfa = mfa.merge(safe[['core_id', 'our_id']], on='core_id', how='inner')
    core_mfa = core_mfa.drop(columns='core_id').rename(columns={'our_id': 'id'})
    core_out = _reshape(core_mfa, 'id', strain_index)
    core_out = _fix_biomass_row(core_out, strain_index)
    core_out.to_excel(OUT_CORE, index=False)
    print(f"E. coli core:    {len(core_out)} reactions -> {OUT_CORE}")

    # ── E. coli genome-scale ── only rows whose translated genomescale_id is a real,
    # single iAF1260 reaction (drops 'lumped:'/';'-combined ids, which don't have one
    # clean genome-scale counterpart).
    safe = safe.assign(gs_id=safe['genomescale_id'].map(_to_genomescale_id))
    gs_safe = safe[safe['gs_id'].isin(iaf1260_ids)]
    assert not gs_safe['gs_id'].duplicated().any()
    gs_mfa = mfa.merge(gs_safe[['core_id', 'gs_id']], on='core_id', how='inner')
    gs_mfa = gs_mfa.drop(columns='core_id').rename(columns={'gs_id': 'id'})
    gs_out = _reshape(gs_mfa, 'id', strain_index)
    gs_out = _fix_biomass_row(gs_out, strain_index)
    gs_out.to_excel(OUT_GENOMESCALE, index=False)
    print(f"E. coli genome-scale: {len(gs_out)} reactions -> {OUT_GENOMESCALE}")


if __name__ == '__main__':
    main()
