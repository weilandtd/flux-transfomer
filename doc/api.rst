API Reference
=============

Core
----

.. automodule:: flux_transformer.core.model
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.mfa
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.experiment
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.data
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.results
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.stoichiometry
   :members:
   :undoc-members:

.. automodule:: flux_transformer.core.kernel
   :members:
   :undoc-members:

EMU Expansion
-------------

.. automodule:: flux_transformer.expansion.emu_expansion
   :members:
   :undoc-members:

.. automodule:: flux_transformer.expansion.emu
   :members:
   :undoc-members:

.. automodule:: flux_transformer.expansion.atom_mapping
   :members:
   :undoc-members:

AI
--

Three modules handle a trained model, each with a distinct job:

- :mod:`flux_transformer.ai.inference` — turn experimental MID measurements into fluxes.
  Start here.
- :mod:`flux_transformer.ai.checkpoint` — rebuild a model together with the dataset it
  was scored on, from a **training checkpoint**. Needs the run directory.
- :mod:`flux_transformer.ai.export` — read and write a packaged **exported model**
  (``.sfx``): self-contained, compressed and versioned, loadable with nothing but
  the file. Unlike a ``.pt`` it contains no pickle, so loading one cannot execute
  arbitrary code.

.. automodule:: flux_transformer.ai.inference
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.export
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.checkpoint
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.model
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.train
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.predict
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.data
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.utils
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.features
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.fisher_selection
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.boundary_basis
   :members:
   :undoc-members:

.. automodule:: flux_transformer.ai.diagnostics
   :members:
   :undoc-members:

Pipeline
--------

The training and inference stages, each installed as a ``flux-transformer-*`` console
script. :mod:`flux_transformer.pipeline._paths` defines how relative config paths are
resolved, which downstream projects need in order to keep their own configs and
models in their own repository.

.. automodule:: flux_transformer.pipeline
   :members:
   :undoc-members:

.. automodule:: flux_transformer.pipeline._paths
   :members:
   :undoc-members:

Analysis
--------

.. automodule:: flux_transformer.analysis.mfa_logger
   :members:
   :undoc-members:

.. automodule:: flux_transformer.analysis.visualization
   :members:
   :undoc-members:

Visualization
-------------

.. automodule:: flux_transformer.viz.experiments
   :members:
   :undoc-members:

Generate
--------

.. automodule:: flux_transformer.generate.core
   :members:
   :undoc-members:

.. automodule:: flux_transformer.generate.tissue
   :members:
   :undoc-members:

.. automodule:: flux_transformer.generate.utils
   :members:
   :undoc-members:

Utilities
---------

Readers for the experimental spreadsheet formats
:meth:`flux_transformer.ai.inference.FluxTransformer.predict` consumes.

.. automodule:: flux_transformer.utils.read_xlsx_mid_data
   :members:
   :undoc-members:

.. automodule:: flux_transformer.utils.read_xlsx_extflux
   :members:
   :undoc-members:

.. automodule:: flux_transformer.utils.read_df_data
   :members:
   :undoc-members:
