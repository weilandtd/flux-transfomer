# Ensemble reliability layer

Turns the point-prediction SigmaFlux model into a **per-reaction reliability tool**: an honest
error bar on every flux, a validated *believe* HIGH/LOW decision, and — when a flux is shaky —
whether the bottleneck is the **measurement** or the **model**.

Full write-up (intuition + math): [`../../reports/reliability_ensemble_methods.md`](../../reports/reliability_ensemble_methods.md).

## Pipeline

```
1. train ensemble      train_ai_model.py --ensemble-size 20     -> <run>/ensemble/model_{0..19}.pt
2. calibrate (sim)     ensemble_calibration.py                  -> factor c, believe threshold
3. attribute variance  variance_decomposition[_3way].py         -> model vs MID vs flux
4. run on experiment   believe.py                               -> believe_<cond>.csv
5. validate vs MFA     plot_believe_validation.py               -> believe_vs_mfa_*.svg
```

Steps 2–3 need ground truth, so they run on the **held-out validation set** (simulation). Step 4 is
the tool a biologist runs on **real data** — no ground truth needed; the calibration factor `c` and
tolerance `tau` from step 2 are shipped constants. Step 5 checks the shipped rule against the
independent experimental MFA reference.

## Scripts

| script | data | what it computes | key outputs |
|---|---|---|---|
| `ensemble_calibration.py` | val (sim) | coverage, calibration factor `c`, per-reaction relative uncertainty, the believe threshold (R²>0.9 guarantee) | `reliability_diagram.svg`, `decision_map.svg`, `reliability_table.csv` |
| `variance_decomposition.py` | val (sim) | 2-way random-effects ANOVA: **model** vs **MID-measurement**; injection sweep validates it reads injected noise | `variance_decomp_sweep.svg`, `variance_decomp_*.csv` |
| `variance_decomposition_3way.py` | val (sim) | 3-way ANOVA: **model** vs **MID** vs **external-flux**; isolation check attributes each channel correctly | `variance_decomp_3way_configs.svg`, `variance_decomp_3way_perrxn.svg` |
| `believe.py` | **real experiment** | one-stop tool: median flux ± calibrated 95% bar, believe HIGH/LOW, bottleneck (measurement/model) per reaction | `believe_<cond>.csv` |
| `plot_believe_validation.py` | real + MFA | validation figures: believe-coloured SigmaFlux-vs-MFA scatter, HIGH/LOW counts | `believe_vs_mfa_panels.svg`, `believe_vs_mfa_pooled.svg`, `believe_summary.svg` |

## Headline result (*I. orientalis*)

The believe rule selects the trustworthy reactions from **ensemble spread alone** (no MFA). Checked
against independent experimental MFA with the field-standard metric (|flux|, pooled over growth
conditions): believe-HIGH agrees at **R² ≈ 0.82–0.91**, believe-LOW is the −70ish garbage the rule
correctly rejects. Variance decomposition says the LOW reactions are **model-limited (~95%)**, not
measurement-limited — a better tracer, not more replicates, is the lever.
