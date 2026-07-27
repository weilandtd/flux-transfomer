"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.generate_training_data`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-generate-data --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.generate_training_data import cli

warnings.warn(
    "paper/pipeline/data_generation/generate_training_data.py has moved into the sigmaflux package. "
    "Use `sigmaflux-generate-data` (or `python -m sigmaflux.pipeline.generate_training_data`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
