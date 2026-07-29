# SigmaFlux

Metabolic flux analysis from stable-isotope labeling data. SigmaFlux pairs a
classical EMU (Elementary Metabolite Unit) ¹³C-MFA implementation with a
transformer that predicts fluxes directly from mass-isotopomer distributions,
so a fit that takes minutes of optimisation becomes a forward pass.

Pretrained ensembles for *Issatchenkia orientalis* and *Saccharomyces cerevisiae*
ship with the repository.

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

Verify:

```bash
python -c "import sigmaflux; print(sigmaflux.__version__)"
sigmaflux-inspect models/pretrained/i_orientalis-v1.0.0.sfx
```

If `sigmaflux-inspect` reports a format error, the LFS objects were not fetched —
run `git lfs pull`.

Once published to PyPI, the library alone installs with `pip install sigmaflux`.
The pretrained models live in this repository, so clone it if you want those.

## Quick start — predict fluxes with a pretrained model

A packaged `.sfx` model is self-contained: no config file, training data or
network JSON required.

```python
import pandas as pd
from sigmaflux.ai import load_model

sf = load_model("models/pretrained/i_orientalis-v1.0.0.sfx")

# What the model expects
sf.expected_tracers()          # tracer names to use in df_L
sf.expected_metabolites()      # measurable metabolite IDs
sf.expected_measured_fluxes()  # reaction IDs to supply in df_Vm

# Measured mass-isotopomer distributions (one row per isotopomer).
# M+0 is excluded — fractions are determined by M+1…M+N.
df_L = pd.DataFrame([
    {"Tracer Name": "U-13C glucose (50%)", "Metabolite ID": "pyr_c",
     "Isotopomer": "M+1", "Fraction": 0.13},
    {"Tracer Name": "U-13C glucose (50%)", "Metabolite ID": "pyr_c",
     "Isotopomer": "M+2", "Fraction": 0.07},
    # ...
])

# Known exchange fluxes (optional, but sharply improves identifiability)
df_Vm = pd.DataFrame([
    {"Flux Name": "EX_glc__D_e", "Value": -10.0},
    {"Flux Name": "EX_etoh_e",   "Value": 6.2},
])

fluxes = sf.predict(df_L, df_Vm)   # columns: Flux Name, Value
```

`load_model` always returns an **ensemble** — a single model is an ensemble of
one — and reports the per-reaction median across members. Add a `Sample` column
to `df_L`/`df_Vm` to predict many samples in one call.

To inspect a model without loading any weights (no torch required):

```bash
sigmaflux-inspect models/pretrained/i_orientalis-v1.0.0.sfx --verify
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
