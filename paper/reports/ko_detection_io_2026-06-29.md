# KO-detection experiment — I. orientalis (2026-06-29)

**Question:** after training on KO-enabled data, does the model predict ~0 flux for
reactions that were actually knocked out in validation samples?

**Setup:** 22,000 samples = 10k WT (reused) + 12k KO (30 top-WT-variance non-essential
reactions × 400, knocked to 0 during sampling). KO-aware split: WT + 25 "seen" KO
reactions → 80/20; 5 held-out reactions (`PFK_c, FBA3_c, MDH_c, GLUSx_c, AKGCITta_m`)
→ validation only. Trained `FluxDecoder` (KO config, `max_samples: null`), val flux
R² median 0.964 (healthy). Metric: `rel = |v_pred[R]| / WT_std[R]` per KO val sample
(0 = perfectly zeroed, ~1 = predicted at normal magnitude). n = 3,997 KO val samples.

## Result — hypothesis NOT confirmed
Fraction of KO val samples whose knocked-out reaction is predicted below X× its normal WT magnitude:

| threshold | ALL | seen KO | held-out KO |
|---|---|---|---|
| < 0.10× (strict "≈0") | 0.06 | 0.08 | 0.05 |
| < 0.25× | 0.18 | 0.22 | 0.13 |
| < 0.50× | 0.34 | 0.43 | 0.25 |
| < 1.00× (any reduction) | 0.53 | 0.64 | 0.41 |
| **median rel** | **0.92** | **0.61** | **1.18** |

- The model predicts the knocked-out reaction at ~**92% of its normal magnitude** (median) — i.e. business-as-usual, not zero. Only **6%** are driven near zero.
- ~half the time (rel ≥ 1) it predicts **more** flux than the reaction normally carries — completely missing the KO.
- **Seen > held-out** (median rel 0.61 vs 1.18): what little works is partly memorization of seen KO reactions; **generalization to unseen knockouts is poor**.

## Which reactions worked / failed
- **Partially detected** (median predicted ~20% of normal): `SBP_c, PC_c, PPCK_c, TALA_c` (~25–30% strict hits) — KO leaves a detectable labeling/flux signature.
- **Completely missed** (predicts >normal): `GLUDxi_c, GLUSx_c, FBP_c, Ht_c_m, CO2t_c_m` (median rel 1.7–2.1).

## Why it fails (consistent with the biomass-bias / observability findings)
1. **All 30 chosen KO reactions are *dependent* reactions** (none are free/identity null-space coordinates). `v[R]` is a linear combination that must *cancel* to reach 0 — structurally hard; the model can't set it cleanly.
2. **Measured fluxes still pin the distribution.** Even with KO-affected biomass/exchange inputs, an internal KO compensated by parallel pathways barely shifts them, so the model predicts the "typical" value for R rather than recognizing it's the zeroed one.
3. **Labeling carries little KO signal** for parallel-pathway reactions — same reason MIDs were weak throughout this project.

## Takeaway
"Predict zero for KO'd reactions" is **not** an emergent property here. The model is
measured-flux-driven; it reconstructs a plausible WT-like distribution and does not
localize internal knockouts from labeling. A few low-throughput / signature-leaving
reactions are partially caught. To make KO detection work would likely require:
choosing KO reactions that are *free* coordinates, a KO-aware objective/penalty, or a
dedicated KO head — not just enabling KO data.

## Files
- data: `paper/.results/i_orientalis_ko_20260628/` (flux_samples.csv, ko_labels.csv, split.json, imisor_transformer_ko.pt, ko_eval.csv)
- scripts: `paper/pipeline/data_generation/sample_ko_flux.py`, `paper/pipeline/data_generation/build_ko_split.py`, `paper/analysis/ko/ko_eval.py`; `--split-file` added to `paper/pipeline/training/train_ai_model.py`
