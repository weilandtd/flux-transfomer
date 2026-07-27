Quick start
===========

Library API
-----------

.. code-block:: python

   from sigmaflux.core import MFAModel, Experiment, Tracer, fit_mfa_model
   from sigmaflux.expansion import EMUExpansion
   from sigmaflux.ai import FluxDecoder, FluxDecoderConfig
   from sigmaflux.ai.inference import load_predictor

Fitting a model to measured data:

.. code-block:: python

   from cobra.io import load_json_model

   cobra_model = load_json_model("models/toy/one_carbon_annotated.json")
   mfa_model = MFAModel(cobra_model, "TOY")

   experiment = Experiment(mid_data, [Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1])],
                           fluxes=flux_data)
   mfa_model.prepare([experiment])
   result = fit_mfa_model(mfa_model, max_iter=50, tol=1e-3)

.. note::

   ``fit_mfa_model`` minimises a null-space reduced chi-squared with L-BFGS-B and
   returns a dict. The older penalized-Lagrangian optimizer is still available as
   ``sigmaflux.core.fit_mfa_model_penalized`` for reproducing earlier analyses.

Pipeline
--------

The training and inference stages are installed as console scripts. Relative
paths in a config resolve against that config's project root, so a downstream
project can keep its own models and data:

.. code-block:: bash

   sigmaflux-workflow --config paper/configs/i_orientalis.yaml

   # or one stage at a time
   sigmaflux-sample-flux     --config <config.yaml>
   sigmaflux-generate-data   --config <config.yaml>
   sigmaflux-select-features --config <config.yaml> --data-dir <run_dir>
   sigmaflux-train           --config <config.yaml> --data-dir <run_dir>
   sigmaflux-infer           --config <config.yaml> --checkpoint <model.pt> \
                             --data-dir <xlsx_dir> --out-dir <results_dir>
