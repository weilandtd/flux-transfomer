# SigmaFlux

[![Python application](https://github.com/weilandtd/sigmaflux/actions/workflows/python-app.yml/badge.svg)](https://github.com/weilandtd/sigmaflux/actions/workflows/python-app.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Metabolic flux analysis from stable-isotope labeling data. SigmaFlux pairs a
classical EMU (Elementary Metabolite Unit) ¹³C-MFA implementation with a
transformer that predicts fluxes directly from mass-isotopomer distributions,
so a fit that takes minutes of optimisation becomes a forward pass.

Atom-mapped network models for *Issatchenkia orientalis*, *Saccharomyces cerevisiae*,
*Rhodotorula toruloides*, *Escherichia coli* core and *Mus musculus* T cells ship
with the repository, with pretrained ensembles for all except *R. toruloides*.

## Installation

Requires Python ≥ 3.10 and [Git LFS](https://git-lfs.com) — the pretrained
models and Excel network files are LFS objects, and without it you get text
pointers instead of models.

```bash
git lfs install
git clone https://github.com/weilandtd/sigmaflux.git
cd sigmaflux

pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, ruff, mypy
pip install -e ".[docs]"    # + sphinx
```

For the full stack, including the optional x3cflux MFA backend, use conda instead —
x3cflux needs Ipopt, and pip's `cyipopt` is an sdist that wants Ipopt headers at
build time, so it has to come from conda-forge:

```bash
conda env create -f environment.yml   # from the repo root: `-e .` resolves against cwd
conda activate sigmaflux
```

Verify:

```bash
python -c "import sigmaflux; print(sigmaflux.__version__)"
sigmaflux-inspect models/pretrained/i_orientalis-v1.4.1.sfx
```

If `sigmaflux-inspect` reports a format error, the LFS objects were not fetched —
run `git lfs pull`.

Once published to PyPI, the library alone installs with `pip install sigmaflux`.
The pretrained models live in this repository, so clone it if you want those.

## Quick start — predict fluxes with a pretrained model

A packaged `.sfx` model is self-contained: no config file, training data or
network JSON required. Example inputs for every pretrained model ship in
`tests/data/interface_test/<model>/input/`, and
[`examples/run_inference.py`](examples/run_inference.py) predicts one condition
from them, with 95% intervals:

```bash
python examples/run_inference.py i_orientalis B1
# -> tests/data/interface_test/i_orientalis/output/predictions_B1.csv
```

### Inputs

| File | Columns | Notes |
|---|---|---|
| `labeling_measurements_<tracer>.csv`, one per tracer | `Condition, Replicate, Compound, Isotopomer, Fraction` | `<tracer>` must be one of `sf.expected_tracers()` |
| `compound_mapping.csv` | `experimental_id, model_id` | maps your compound names to `sf.expected_metabolites()` |
| `measured_fluxes.csv` | `Condition, Flux Name, Value, Std` | reactions from `sf.expected_measured_fluxes()` |

Measured fluxes are optional, but they set the scale of the prediction: the
labeling alone fixes only ratios.

### Prediction with intervals

This is what the example does, for one condition:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from sigmaflux.ai import load_model
from sigmaflux.ai.uncertainty import sample_inputs, summarize_draws
from sigmaflux.pipeline.run_inference_workflow import predict_draws

sf = load_model("models/pretrained/i_orientalis-v1.4.1.sfx")
data = Path("tests/data/interface_test/i_orientalis/input")
condition = "B1"

# Labeling: one CSV per tracer, named labeling_measurements_<tracer name>.csv.
prefix = "labeling_measurements_"
mids = pd.concat(pd.read_csv(f).assign(**{"Tracer Name": f.stem.removeprefix(prefix)})
                 for f in sorted(data.glob(prefix + "*.csv")))
mids = mids[mids["Condition"] == condition]

# Experimental compound names -> model metabolite IDs; replicates become samples.
names = pd.read_csv(data / "compound_mapping.csv").rename(
    columns={"experimental_id": "Compound", "model_id": "Metabolite ID"})
df_L = mids.merge(names, on="Compound").rename(columns={"Replicate": "Sample"})

# Measured fluxes and their standard deviations.
fluxes = pd.read_csv(data / "measured_fluxes.csv")
fluxes = fluxes[fluxes["Condition"] == condition]
df_Vm = fluxes[["Flux Name", "Value"]]
df_err = fluxes[["Flux Name", "Std"]].rename(columns={"Std": "Value"})

# Resample the replicates across tracers and the fluxes from their Std, predict
# every draw with every ensemble member, and summarize.
n_bootstrap = 200
df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=np.random.default_rng(0),
                                        n_bootstrap=n_bootstrap, bounds=sf.flux_bounds())
raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))
result = summarize_draws(raw, n_draws=n_bootstrap, group_cols=["Type", "Flux Name"])
```

`result` has one row per predicted quantity:

| Column | Meaning |
|---|---|
| `Type` | `flux` or `reversibility` |
| `Flux Name` | reaction ID |
| `Value` | the prediction at the replicate means and the measured fluxes |
| `Value_lo`, `Value_hi` | 95% interval: 2.5th–97.5th percentile over the bootstrap draws |
| `n_draws` | draws that reached the interval |
| `sigma_input` | standard deviation of the prediction across draws |
| `sigma_model` | disagreement between ensemble members at the replicate means |

`load_model` always returns an **ensemble** — a single model is an ensemble of
one — and reports the **ensemble medoid**: the member closest to the
per-reaction median, so every prediction is a flux state the model actually
produced. For a prediction without an interval, `sf.predict(df_L, df_Vm)`
returns one per `Sample`.

For results that are bit-identical across CPU types, set the environment
variables at the top of `examples/run_inference.py` before importing numpy or
torch.

To inspect a model without loading any weights (no torch required):

```bash
sigmaflux-inspect models/pretrained/i_orientalis-v1.4.1.sfx --verify
```

Reads the name, version, output dimensions, tracers, and provenance — the
sigmaflux version, git commit and training config that produced it.

## Training your own model

Everything is driven by one organism YAML (see `paper/configs/`). Four stages,
each an installed console script:

```bash
sigmaflux-sample-flux     --config <config>.yaml --output-dir <run>   # sample the flux polytope
sigmaflux-generate-data   --config <config>.yaml --output-dir <run>   # simulate labeling (EMU)
sigmaflux-select-features --config <config>.yaml --data-dir  <run>    # Fisher token selection
sigmaflux-train           --config <config>.yaml --data-dir  <run>    # train the FluxDecoder
```

Or all four at once:

```bash
sigmaflux-workflow --config paper/configs/i_orientalis.yaml
```

Train an ensemble and package it for release:

```bash
sigmaflux-train  --config <config>.yaml --data-dir <run> --ensemble-size 20
sigmaflux-export --source <run>/ensemble --config <config>.yaml --model-version 1.0.0
```

Ensembles are the recommended target: `load_model` and `load_predictor` use them
by default and report per-reaction medians. Exporting compresses ~250 MB of
checkpoints into a single ~35 MB file.

Full details, including the classical MFA entry points and the paper analyses,
are in [paper/README.md](paper/README.md). API reference: `doc/`
(`pip install -e ".[docs]" && sphinx-build -b html doc doc/_build`).

## Citation

If you use SigmaFlux in your work, please cite:

<!-- TODO: replace with the published reference on acceptance -->
```bibtex
@article{sigmaflux,
  title   = {TBD},
  author  = {Weilandt, Daniel R. and others},
  journal = {TBD},
  year    = {2026},
  doi     = {TBD},
}
```

A preprint and DOI will be linked here when available.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
