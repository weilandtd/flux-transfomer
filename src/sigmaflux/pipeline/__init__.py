"""Executable pipeline stages, driven by a YAML config.

Each module exposes a ``cli(argv=None)`` entry point and is also installed as a
console script:

===============================  ==========================================
``sigmaflux-sample-flux``        :mod:`sigmaflux.pipeline.sample_flux`
``sigmaflux-generate-data``      :mod:`sigmaflux.pipeline.generate_training_data`
``sigmaflux-select-features``    :mod:`sigmaflux.pipeline.extract_informative_features`
``sigmaflux-train``              :mod:`sigmaflux.pipeline.train_ai_model`
``sigmaflux-workflow``           :mod:`sigmaflux.pipeline.run_training_workflow`
``sigmaflux-infer``              :mod:`sigmaflux.pipeline.run_inference_workflow`
``sigmaflux-mfa-data``           :mod:`sigmaflux.pipeline.run_mfa_data`
``sigmaflux-sample-ko-flux``     :mod:`sigmaflux.pipeline.sample_ko_flux`
``sigmaflux-build-ko-split``     :mod:`sigmaflux.pipeline.build_ko_split`
===============================  ==========================================

Relative paths inside a config resolve against that config's project root
rather than the sigmaflux installation, so a downstream project can keep its
own models, data and configs and still drive these stages. See
:mod:`sigmaflux.pipeline._paths`.

The submodules are intentionally *not* imported here: several pull in torch and
cobra, which is far too expensive for ``import sigmaflux.pipeline``.
"""

__all__ = [
    "build_ko_split",
    "extract_informative_features",
    "generate_training_data",
    "run_inference_workflow",
    "run_mfa_data",
    "run_training_workflow",
    "sample_flux",
    "sample_ko_flux",
    "train_ai_model",
]
