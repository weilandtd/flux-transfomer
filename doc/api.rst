API Reference
=============

Core
----

.. automodule:: sigmaflux.core.model
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.mfa
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.experiment
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.data
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.results
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.stoichiometry
   :members:
   :undoc-members:

.. automodule:: sigmaflux.core.kernel
   :members:
   :undoc-members:

EMU Expansion
-------------

.. automodule:: sigmaflux.expansion.emu_expansion
   :members:
   :undoc-members:

.. automodule:: sigmaflux.expansion.emu
   :members:
   :undoc-members:

.. automodule:: sigmaflux.expansion.atom_mapping
   :members:
   :undoc-members:

AI
--

Start with :mod:`sigmaflux.ai.inference` — it turns experimental MID measurements
into fluxes. :mod:`sigmaflux.ai.checkpoint` is the research-side counterpart: it
rebuilds a trained model together with the dataset it was scored on.

.. automodule:: sigmaflux.ai.inference
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.checkpoint
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.model
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.train
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.predict
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.data
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.utils
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.features
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.fisher_selection
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.boundary_basis
   :members:
   :undoc-members:

.. automodule:: sigmaflux.ai.diagnostics
   :members:
   :undoc-members:

Pipeline
--------

The training and inference stages, each installed as a ``sigmaflux-*`` console
script. :mod:`sigmaflux.pipeline._paths` defines how relative config paths are
resolved, which downstream projects need in order to keep their own configs and
models in their own repository.

.. automodule:: sigmaflux.pipeline
   :members:
   :undoc-members:

.. automodule:: sigmaflux.pipeline._paths
   :members:
   :undoc-members:

Analysis
--------

.. automodule:: sigmaflux.analysis.mfa_logger
   :members:
   :undoc-members:

.. automodule:: sigmaflux.analysis.visualization
   :members:
   :undoc-members:

Visualization
-------------

.. automodule:: sigmaflux.viz.experiments
   :members:
   :undoc-members:

Generate
--------

.. automodule:: sigmaflux.generate.core
   :members:
   :undoc-members:

.. automodule:: sigmaflux.generate.tissue
   :members:
   :undoc-members:

.. automodule:: sigmaflux.generate.utils
   :members:
   :undoc-members:

Utilities
---------

Readers for the experimental spreadsheet formats
:meth:`sigmaflux.ai.inference.SigmaFlux.predict` consumes.

.. automodule:: sigmaflux.utils.read_xlsx_mid_data
   :members:
   :undoc-members:

.. automodule:: sigmaflux.utils.read_xlsx_extflux
   :members:
   :undoc-members:

.. automodule:: sigmaflux.utils.read_df_data
   :members:
   :undoc-members:
