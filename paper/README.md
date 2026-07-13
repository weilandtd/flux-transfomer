# Paper — Training & Synthesis Workflow

## Quick start

Run the entire pipeline for a model by passing its folder name:

```bash
python paper/pipeline/workflows/run_training_workflow.py I_orientalis
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--n-samples` | 10000 | Flux samples to draw |
| `--epochs` | 100 | Training epochs |
| `--device` | auto | `cpu`, `cuda`, or `auto` |
| `--output-dir` | `<folder>_<YYYYMMDD>` | Where to store all outputs |
| `--output-model` | `<output_dir>/<folder>_transformer.pt` | Output checkpoint path |
| `--skip-sampling` | — | Reuse existing `flux_samples.csv` |
| `--skip-generate` | — | Reuse existing HDF5 labeling files |
| `--skip-train` | — | Skip training, keep data only |

---

## Individual steps

Three scripts implement the end-to-end pipeline for generating training data and
fitting the labeling-to-flux transformer used in the paper.

## Steps

### 1. Sample flux distributions — `sample_flux.py`

Prepares a COBRA sampling problem from an annotated model and
draws flux samples via COBRApy's `optgp` sampler.

```bash
python paper/pipeline/data_generation/sample_flux.py \
    --model-file models/I_orientalis/imIsor_annotated.json \
    --output-dir training_data \
    --n-samples 10000 \
    --processes 5
```

Outputs `flux_samples.csv` and a copy of the model JSON to `--output-dir`.

### 2. Simulate isotope labeling — `generate_training_data.py`

For each sampled flux distribution, runs `MFAModel` forward to compute
simulated mass isotopomer distributions (MIDs) and saves them to HDF5.

```bash
python paper/pipeline/data_generation/generate_training_data.py \
    --flux_samples training_data/flux_samples.csv \
    --start_idx 0 \
    --end_idx 500 \
    --output_dir training_data \
    --device cuda
```

Outputs `labeling.h5`, `fluxes.h5`, and `reversibilities.h5`.  Run in
parallel slices (`--start_idx` / `--end_idx`) to distribute across jobs.

### 3. Train the transformer — `train_ai_model.py`

Trains `LabelingToFluxTransformer` on the HDF5 files produced above.

```bash
python paper/pipeline/training/train_ai_model.py \
    --data-dir training_data \
    --model-json models/I_orientalis/imIsor_annotated.json \
    --output imisor_transformer.pt \
    --device cuda
```

Saves the trained model to `--output` (`.pt` file).

### 4. (optional) Train a reliability ensemble — `train_ai_model.py --ensemble-size 20`

Adding `--ensemble-size 20` trains 20 seed-varied models on a shared split, written to
`<output_dir>/ensemble/model_{0..19}.pt`. This powers the **reliability layer** (per-reaction
error bars + the *believe* HIGH/LOW rule + measurement-vs-model diagnosis). See
[`analysis/ensemble/README.md`](analysis/ensemble/README.md) and the write-up in
[`reports/reliability_ensemble_methods.md`](reports/reliability_ensemble_methods.md).

## Notes

- Run all scripts from the repository root with `PYTHONPATH=src` if the
  package is not installed in editable mode.
- The `.h5` and `.pkl` training outputs are gitignored; only the final `.pt`
  model is committed.
