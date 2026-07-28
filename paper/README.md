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

Fisher/D-optimal token scoring over the generated HDF5, writing `features.csv`.

```bash
sigmaflux-select-features \
    --config paper/configs/i_orientalis.yaml \
    --data-dir training_data
```

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

## Notes

- Install the package once with `pip install -e .`; the `sigmaflux-*` console
  scripts and the `paper/analysis/` scripts then work from any directory.
- The `.h5` and `.pkl` training outputs are gitignored; only the final `.pt`
  model is committed.
