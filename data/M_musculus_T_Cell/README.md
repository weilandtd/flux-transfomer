# M. musculus primary T cell

Primary murine T cells isolated from spleen, cultured in RPMI1640 + 10% dFBS + 1% pen&strep.
Two states, which are this organism's condition letters:

| Letter | State | Culture |
|---|---|---|
| `A` | activated T cell | activated by aCD3/aCD28, maintained in IL2 at 1E6/mL for 24 h |
| `N` | naive cell | maintained in IL7 at 1E6/mL for 24 h |

Both were then switched to tracer media — naive at 3E6/mL, activated at 1E6/mL — labeled for
5 h, and 1 mL of culture extracted with 50 uL 40:40:20 + FA, neutralized.

## Tracers

Three separate cultures, each at 100% enrichment. The sheet names in the MID workbooks are the
tracer names the config's `labeling.tracers[].name` must match.

| Sheet | Tracer |
|---|---|
| `U-13C glucose` | 100% [U-13C6]glucose |
| `U-13C glutamine` | 100% [U-13C5]glutamine |
| `1,2-13C glucose` | 100% [1,2-13C2]glucose |

## Files

| File | Contents |
|---|---|
| `M_musculus_T_Cell_A.xlsx` | activated MIDs, one sheet per tracer, `A1_Rep1..A1_Rep3` |
| `M_musculus_T_Cell_N.xlsx` | naive MIDs, `N1_Rep1..N1_Rep3` |
| `M_musculus_T_Cell_ExtFlux.xlsx` | sheet `Constraints`, conditions `A1` and `N1` |
| `MFA_flux_data_MM.xlsx` | the published MFA fluxes, on converted-model reaction IDs |

All fluxes are nmol/1E6 cells/h in SigmaFlux's secretion-positive convention.

## How these were derived

The sources were the published MFA input table (umol/1E6 cells/h, uniformly
uptake-positive) and the published flux table (nmol/1E6 cells/h, in the raw Maranas
per-reaction direction, which differs between exchanges). Conversion to the model's
uniformly secretion-positive convention was therefore:

| | rule |
|---|---|
| measurements → `ExtFlux` | `value = -1000 x mean` — one rule, the source is uniformly signed |
| published fit → `MFA_flux_data_MM` | negate the 21 exchanges that conversion flips |

Two mappings were not one-to-one. `o2 [m]` and `o2 [c]` sum to `EX_o2_e`, errors in
quadrature; `o2 [c]` is deliberately *not* mapped to `O2USE1_c`, because the published fit
puts that reaction at 0.51 against a measured 16.43 ± 2.45, i.e. it was left unconstrained.
The published table predates the hierarchical biomass split, so its lumped `BIOMASS`
rows were expanded into `PROTEIN_c`/`RNA_c`/`DNA_c`/`LIPID_c`, and lumped transketolase
(`TKT1_c`, `TKT2_c`) was mapped onto the ping-pong half reactions the GEM uses, where the
carrier balance gives `TKThalf_c = TKT1_c + TKT2_c`.

Cross-check on the signs: activated `EX_ala__L_e` is `-1000 x 0.004153 = -4.153` from the
measurements and `-1 x 4.1375` from the published fit.

**Validation.** The curated activated flux vector balances in the converted model to
`||S.v||inf = 4.7e-07` against a peak flux of 701 — machine precision, which checks the
dropped dilution network, the `_out` removal, the 21 flipped exchanges and the
transketolase mapping in one go.

## Data excluded by the original authors

Recorded in the source workbook's `note` sheet, before it was curated away. These exclusions
are already applied to the MID workbooks here.

| Compound | Reason |
|---|---|
| `Cit` | From UGln the major MID is M+2, some M+3, but no M+4 found; cannot be explained, raw data checked and the peak was okay |
| `Acon` | From UGln the major MID is M+2, but no M+4 found; cannot be explained |
| `PEP` | Manually corrected M+2 to 0 from UGln |
| `GAP` | Removed due to poor data quality |
| `Ser` | Deleted in naive cells: no labeling detected in serine, yet downstream glycine is labeled |
| `OROT` | Manually re-picked peaks from UGln because M+4 was missed by automatic peak picking |

## Known inconsistencies in the published fit

- `protonleak_m` does not balance in the naive flux vector (off by 12.84). `ATPS_m` produces
  1.0 `protonleak_m` and `Ht_c_m` consumes 0.2663 per unit; the activated vector satisfies this
  exactly, the naive one does not. Left as published.
- 19 reactions carry `NA` for naive — deoxyribonucleotide synthesis (`RNDR*`, `TMDS_c`,
  `DTMPK_c`, ...) and de novo lipogenesis (`ACITL_c`, `SK_accoa_c`). Naive T cells neither
  divide nor run lipogenesis, so these were absent from the naive model; curated as blank.
- The GEM's exchange bounds disagree with its own fit: `EX_ala__L_e` is written uptake-only,
  yet naive alanine is fit as secreted. The config's `media` block opens both directions.
- `succ_m` carries the formula `C4H4O4` in the GEM, which is fumarate's; succinate is C4H6O4.
  Metabolite identification therefore matches on name, never on formula.
