"""Deprecated shim — this stage now lives in :mod:`sigmaflux.pipeline.run_inference_workflow`.

Kept so existing submit scripts keep working. Prefer the console script:

    sigmaflux-infer --config <config.yaml>
"""

import warnings

from sigmaflux.pipeline.run_inference_workflow import cli

warnings.warn(
    "paper/pipeline/inference/run_inference_workflow.py has moved into the sigmaflux package. "
    "Use `sigmaflux-infer` (or `python -m sigmaflux.pipeline.run_inference_workflow`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    cli()
