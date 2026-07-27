"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.run_mfa_data`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-mfa-data --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.run_mfa_data import cli

warnings.warn(
    "paper/pipeline/inference/run_mfa_data.py has moved into the sigmaflux package. "
    "Use `sigmaflux-mfa-data` (or `python -m sigmaflux.pipeline.run_mfa_data`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
