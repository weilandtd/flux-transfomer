# Version comparison — old vs current model on I. orientalis (2026-06-27)

Head-to-head of the **initial** model (`LabelingToFluxTransformer`, from
`Codes/mammalian_flux_atlas/.../whole_body_model_ai`) vs the **current** model
(`FluxDecoder`) on the IO dataset (`i_orientalis_20260620/`). Both use the IDENTICAL
data and the same train/val split stored in the current checkpoint
(train=8000, val=2000). Per-reaction R² is computed on the held-out val split.

## Architecture differences (summary)
| | Old `LabelingToFluxTransformer` | Current `FluxDecoder` |
|---|---|---|
| Input unit | whole experiment (flattened MID vector, 2×2268) | per-MID token (196 Fisher-selected) + flux tokens |
| Identity info | tracer multi-hot only | radial embeddings (compound/label/tracer/reaction/type) |
| Decoder | self-attn over experiments → attention-pool → dense | encoder → cross-attention with learnable queries |
| Output | full v∈ℝ³⁸⁷, then project (I−S⁺S) | 44 null-space coords y → v = Z·y (exact Sv=0) |
| Measured fluxes | none (no mechanism) | 10 bypass tokens (first-class input) |
| Interpretability | none (attn over 2 experiments) | reaction×metabolite cross-attention maps |

## Compute / size / convergence (matched width d=128, 4 layers, single model)
| | Old | Current |
|---|---|---|
| Parameters | 1,178,956 | **913,259** |
| Epochs to converge | ~4,380 (early-stop) | **~144** |
| Train time | 1,435 s | far less (~mins) |
| Converged with shipped recipe? | **No** — needed a variance floor added | yes |

The old var-norm loss has **no variance floor**; near-constant reactions blew up the
objective (initial val loss 445,178) and it learned almost nothing at 1000 epochs
(median R² −0.92, 0 reactions >0.9). Adding the current version's variance floor
(=0.1) + ~4,380 epochs was required just to make it trainable.

## Accuracy — as designed (each version's real capability)
| | Old (labeling-only) | Current (with bypass fluxes) |
|---|---|---|
| R² median | 0.394 | **0.973** |
| # reactions R²>0.9 | 7 | **250** |
| # reactions R²>0.95 | 0 | **230** |

## Accuracy — labeling-only (controlled, no measured fluxes)
| | R² median | #R²>0.9 |
|---|---|---|
| Old (labeling-only by design) | 0.394 | 7 |
| Current, flux-trained, eval w/o fluxes (OOD) | 0.253 | 0 |
| Current, trained w/o fluxes (dropout=1.0), eval w/o fluxes | 0.008* | 0 |

*The dropout=1.0 labeling-only training looks degenerate (collapses toward the mean);
a cleaner `measured_fluxes=[]` retrain is the proper way to get this number.

**Honest read:** labeling alone resolves little in this system for *either* architecture
(consistent with the biomass-bias finding). The current model's large advantage is its
ability to **integrate measured fluxes**, which the old architecture cannot do at all.

## Files
- `paper/.results/version_benchmark/old_model_io.json` (old, fair var-floored run)
- `paper/.results/version_benchmark/current_labelonly_io.json` (current labeling-only)
- current as-designed R²: stored in `imisor_transformer.pt` `val_correlations`
- Figures: `cmp_r2_scatter.{svg,png}`, `cmp_count_bar.{svg,png}`, `cmp_old_convergence.{svg,png}`
- Scripts: `paper/analysis/benchmark/benchmark_old_model.py`, `benchmark_current_eval.py`, `benchmark_plots.py`

## Caveats
- Old run is single-model (not the ensemble it was designed as) and at matched width
  128 (its native default was 512), per the agreed comparison scope.
- Interpretability comparison is qualitative/structural (old has no per-metabolite attention).
