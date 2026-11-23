"""Executable pipeline stages, driven by a YAML config.

Each module exposes a ``cli(argv=None)`` entry point and is also installed as a
console script:

===============================  ==========================================
``flux-transformer-sample-flux``        :mod:`flux_transformer.pipeline.sample_flux`
``flux-transformer-generate-data``      :mod:`flux_transformer.pipeline.generate_training_data`
``flux-transformer-select-features``    :mod:`flux_transformer.pipeline.extract_informative_features`
``flux-transformer-train``              :mod:`flux_transformer.pipeline.train_ai_model`
``flux-transformer-workflow``           :mod:`flux_transformer.pipeline.run_training_workflow`
``flux-transformer-infer``              :mod:`flux_transformer.pipeline.run_inference_workflow`
``flux-transformer-mfa-data``           :mod:`flux_transformer.pipeline.run_mfa_data`
``flux-transformer-benchmark``          :mod:`flux_transformer.pipeline.run_ai_benchmark`
``flux-transformer-benchmark-mfa``      :mod:`flux_transformer.pipeline.run_mfa_benchmark`
``flux-transformer-sample-ko-flux``     :mod:`flux_transformer.pipeline.sample_ko_flux`
``flux-transformer-export``            :mod:`flux_transformer.pipeline.export_model`
``flux-transformer-inspect``           :mod:`flux_transformer.pipeline.inspect_model`
``flux-transformer-build-ko-split``     :mod:`flux_transformer.pipeline.build_ko_split`
===============================  ==========================================

Relative paths inside a config resolve against that config's project root
rather than the flux_transformer installation, so a downstream project can keep its
own models, data and configs and still drive these stages. See
:mod:`flux_transformer.pipeline._paths`.

The submodules are intentionally *not* imported here: several pull in torch and
cobra, which is far too expensive for ``import flux_transformer.pipeline``.
"""

__all__ = [
    'export_model',
    'inspect_model',
    "build_ko_split",
    "extract_informative_features",
    "generate_training_data",
    "run_ai_benchmark",
    "run_inference_workflow",
    "run_mfa_benchmark",
    "run_mfa_data",
    "run_training_workflow",
    "sample_flux",
    "sample_ko_flux",
    "train_ai_model",
]
