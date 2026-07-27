"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.extract_informative_features`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-select-features --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.extract_informative_features import cli

warnings.warn(
    "paper/pipeline/training/extract_informative_features.py has moved into the sigmaflux package. "
    "Use `sigmaflux-select-features` (or `python -m sigmaflux.pipeline.extract_informative_features`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
