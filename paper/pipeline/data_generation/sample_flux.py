"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.sample_flux`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-sample-flux --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.sample_flux import cli

warnings.warn(
    "paper/pipeline/data_generation/sample_flux.py has moved into the sigmaflux package. "
    "Use `sigmaflux-sample-flux` (or `python -m sigmaflux.pipeline.sample_flux`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
