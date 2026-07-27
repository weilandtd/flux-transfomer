"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.run_training_workflow`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-workflow --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.run_training_workflow import cli

warnings.warn(
    "paper/pipeline/workflows/run_training_workflow.py has moved into the sigmaflux package. "
    "Use `sigmaflux-workflow` (or `python -m sigmaflux.pipeline.run_training_workflow`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
