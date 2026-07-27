"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.build_ko_split`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-build-ko-split --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.build_ko_split import cli

warnings.warn(
    "paper/pipeline/data_generation/build_ko_split.py has moved into the sigmaflux package. "
    "Use `sigmaflux-build-ko-split` (or `python -m sigmaflux.pipeline.build_ko_split`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
