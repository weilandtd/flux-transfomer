"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.train_ai_model`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-train --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.train_ai_model import cli

warnings.warn(
    "paper/pipeline/training/train_ai_model.py has moved into the sigmaflux package. "
    "Use `sigmaflux-train` (or `python -m sigmaflux.pipeline.train_ai_model`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
