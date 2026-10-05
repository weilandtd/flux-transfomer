# Paper — Training & Synthesis Workflow

## Quick start

Run the entire pipeline for an organism by passing its config:

```bash
sigmaflux-workflow --config paper/configs/i_orientalis.yaml
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--config` | — | Organism YAML; sampling/labeling/training settings all live here |
| `--data-dir` | `<name>_<YYYYMMDD>` | Directory holding the HDF5 training data |
| `--root` | nearest `pyproject.toml`/`.git` | Root that relative config paths resolve against |
| `--skip-sampling` | — | Reuse existing `flux_samples.csv` |
| `--skip-generate` | — | Reuse existing HDF5 labeling files |
| `--skip-features` | — | Reuse existing `features.csv` |
| `--skip-train` | — | Skip training, keep data only |

Sample count, epochs and device are config keys (`sampling:`, `training:`,
`device:`), not command-line flags.

## Organisms

| Config | Model | Condition letters |
|---|---|---|
| `i_orientalis.yaml` | `imIsor_annotated_hierarchical.json` | `C`/`N`/`P` nutrient limitation, `B` batch |
| `s_cerevisiae.yaml` | `sc_annotated_hierarchical.json` | same |
| `r_toruloides.yaml` | `imRt_annotated_hierarchical.json` | no experimental data yet |
| `m_musculus_t_cell.yaml` | `mmTcell_annotated_hierarchical.json` | `A` activated, `N` naive |

Condition letters are scoped per organism: `N` is nitrogen limitation for the yeasts
and the naive cell state for the T cell.

The T cell is the one mammalian model. Fluxes are nmol/1E6 cells/h rather than
mmol/gDW/h, one biomass covers both cell states through the macromolecule component
reactions, and it carries no carbohydrate, so `CARB_c` is absent.

---

## Individual steps

The pipeline stages ship with the package as console scripts. Each takes the same
organism YAML; per-stage settings live in the config, not on the command line.

### 1. Sample flux distributions — `sigmaflux-sample-flux`

Prepares a COBRA sampling problem from an annotated model and
draws flux samples via COBRApy's `optgp` sampler.

```bash
sigmaflux-sample-flux \
    --config paper/configs/i_orientalis.yaml \
    --output-dir training_data
```

Outputs `flux_samples.csv` and a copy of the prepared model JSON to `--output-dir`.

### 2. Simulate isotope labeling — `sigmaflux-generate-data`

For each sampled flux distribution, runs `MFAModel` forward to compute
simulated mass isotopomer distributions (MIDs) and saves them to HDF5.

```bash
sigmaflux-generate-data \
    --config paper/configs/i_orientalis.yaml \
    --output-dir training_data \
    --start-idx 0 \
    --end-idx 500
```

Outputs `wb_model_labeling_data_*.h5`, `wb_model_flux_data_*.h5` and
`wb_model_reversibility_data_*.h5`. Run in parallel slices
(`--start-idx` / `--end-idx`) to distribute across jobs.

### 3. Select informative features — `sigmaflux-select-features`

Fisher-information significance token selection over the generated HDF5, writing
`features.csv`. A token is included iff its standalone information about the free
fluxes exceeds a permutation-calibrated null floor — no redundancy trimming.

```bash
sigmaflux-select-features \
    --config paper/configs/i_orientalis.yaml \
    --data-dir training_data
```

The config key `features.token_selection_method` picks how the underlying
sensitivity Gram is estimated — `fisher` (exact, cross-fit, bias-corrected) or
`fisher_fast` (approximate, single-shot, cheaper) — both run the identical
significance test on top, so `features.csv`'s schema and downstream training are
unaffected by the choice; `fisher_fast` just costs less.

### 4. Train the decoder — `sigmaflux-train`

Trains `FluxDecoder` on the HDF5 files produced above.

```bash
sigmaflux-train \
    --config paper/configs/i_orientalis.yaml \
    --data-dir training_data \
    --output imisor_transformer.pt
```

Saves the trained model to `--output` (`.pt` file).

### 5. (optional) Train a reliability ensemble — `sigmaflux-train --ensemble-size 20`

Adding `--ensemble-size 20` trains 20 seed-varied models on a shared split, written to
`<output_dir>/ensemble/model_{0..19}.pt`.

This is the recommended way to train, because ensembles are the **default at
prediction time**: `sigmaflux.ai.load_predictor` returns an `EnsembleSigmaFlux`
whenever it finds an `ensemble/` directory next to the checkpoint, and reports the
per-reaction median over members. No extra flag is needed — point inference at the
run directory and it picks the ensemble up automatically.

### 6. Predict fluxes from experimental data — `sigmaflux-infer`

Discovers the per-condition MID spreadsheets in the experimental data directory
and writes `predictions_*.csv` per condition.

```bash
sigmaflux-infer \
    --config     paper/configs/i_orientalis.yaml \
    --checkpoint <run_dir>/imisor_transformer.pt \
    --data-dir   data/I_orientalis \
    --out-dir    <run_dir>/predictions
```

For programmatic use, `sigmaflux.ai.load_predictor` returns the same predictor and
exposes `.predict(df_L, df_Vm)`.

### 7. Package the model for release — `sigmaflux-export`

A trained run is 20 `.pt` files (252 MB for I. orientalis), and each one only
loads next to its config YAML, COBRA JSON and run directory. `sigmaflux-export`
packages them into one self-contained file:

```bash
sigmaflux-export \
    --source <run_dir>/ensemble \
    --config paper/configs/i_orientalis.yaml \
    --model-version 1.2.0
```

| | |
|---|---|
| 20 `.pt` members | 376.2 MB |
| `i_orientalis-v1.2.0.sfx` (fp16) | **129.5 MB — 2.9×** |
| `--precision fp32` | predictions bit-identical to the source, ~2x the fp16 size |

Most of what remains is the stored validation split (1000 samples with their inputs, so
R² is recomputable from the artifact alone); `--no-val-inputs` drops it.

The export also carries the run's prepared COBRA network (~30 KB packed), so a
consumer can rebuild the stoichiometry the predictions belong to:

```python
from sigmaflux.ai import load_model, read_stoichiometry

S, metabolite_ids, cobra_model = read_stoichiometry("i_orientalis-v1.2.0.sfx")
S @ predicted_fluxes      # ≈ 0 — columns are already in flux_names order
```

The saving comes from storing the shared payload once instead of 20×, dropping
training diagnostics nothing reads, regenerating the 14 frozen embedding tables
on load, and float16 weights. fp16 costs nothing that matters: on the held-out
split the ensemble median R² moves 0.981874 → 0.981816, and the count of
reactions above R² 0.9 (250) and 0.95 (239) is unchanged. Only reactions that
were already unidentifiable move, and the null-space basis stays float32 so
`S·v ≈ 0` still holds exactly.

Inspect one without loading any weights — or without torch installed:

```bash
sigmaflux-inspect i_orientalis-v1.2.0.sfx           # name, version, provenance
sigmaflux-inspect i_orientalis-v1.2.0.sfx --verify  # checksums for every entry
```

An export is **always an ensemble** (a single model is an ensemble of one), and
anything that accepts a checkpoint accepts it:

```bash
sigmaflux-infer --config paper/configs/i_orientalis.yaml \
                --checkpoint i_orientalis-v1.2.0.sfx \
                --data-dir data/I_orientalis --out-dir <run_dir>/predictions
```

### 8. Benchmark accuracy — `sigmaflux-benchmark`, `sigmaflux-benchmark-mfa`

Both benchmarks score a predictor on the samples withheld from AI training
(`--split val`), from the same noisy inputs, and write the same HDF5 schema
(`pred_<start>_<end>_{labeling,flux}.h5`) so their results load identically in
`paper/analysis/diagnostics/plot_test_performance.py`.

```bash
sigmaflux-benchmark \
    --config paper/configs/i_orientalis.yaml \
    --data-dir <run_dir> --n-benchmarks 2000        # → <run_dir>/ai_benchmark/

sigmaflux-benchmark-mfa \
    --config paper/configs/i_orientalis.yaml \
    --data-dir <run_dir> --n-benchmarks 2 --start-ix 0   # → <run_dir>/mfa_benchmark/

sigmaflux-benchmark-mfa --backend x3cflux \
    --config paper/configs/i_orientalis.yaml \
    --data-dir <run_dir> --n-benchmarks 2 --start-ix 0   # 13CFLUX/Ipopt, same schema
```

`--backend x3cflux` runs the identical fits through x3cflux 3.0.4 — an independent,
published implementation — on the same data with the same noise realisation, so the
MFA arm measures MFA rather than our L-BFGS-B. It needs the conda environment
(`conda env create -f environment.yml`), and it is ~two orders of magnitude faster
per fit because it uses a Gauss-Newton Hessian.

`--start-ix` / `--n-benchmarks` are an offset window into the split, not raw HDF5
columns, so a SLURM array tiles it deterministically. `paper/submit/submit_mfa_benchmark.sh`
derives the array's shape from a measured seconds-per-sample per backend rather than
hardcoding it — see `PILOT=` in that script. MFA is fit with up to `--max-starts`
restarts from a feasible near-GT init; the best fit is always written, never NaN-ed
for missing a bar, and `pred_<start>_<end>_convergence.csv` carries `ssr_ratio`,
`reached_optimum` and `seconds` so a reader can filter on "did this reach the global
optimum" and still see what that drops.

To tune the noise and optimiser settings before launching an array, fit a single
sample with full diagnostics (GT feasibility, init agreement, per-block χ² split,
measured-flux recovery, flux/labeling R²) and no HDF5 output:

```bash
sigmaflux-benchmark-mfa --diagnose \
    --config paper/configs/i_orientalis.yaml \
    --data-dir <run_dir> --start-ix 0 --max-iter 1000
```

## Notes

- Install the package once with `pip install -e .`; the `sigmaflux-*` console
  scripts and the `paper/analysis/` scripts then work from any directory.
- The `.h5` and `.pkl` training outputs are gitignored; only the final `.pt`
  model is committed.
