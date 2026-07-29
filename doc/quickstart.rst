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

Using a released model
----------------------

A packaged ``.sfx`` model is self-contained — no config YAML, no run directory,
no COBRA JSON:

.. code-block:: python

   from sigmaflux.ai import load_model

   sf = load_model("i_orientalis-v1.0.0.sfx")
   predictions = sf.predict(df_L, df_Vm)

``load_model`` always returns an ensemble (a single model is an ensemble of one)
and reports the per-reaction median over its members. To see what is inside a
file without loading any weights — or without torch installed at all:

.. code-block:: bash

   sigmaflux-inspect i_orientalis-v1.0.0.sfx
   sigmaflux-inspect i_orientalis-v1.0.0.sfx --verify   # check every checksum

To package your own trained run, see :ref:`exporting-a-model` below.

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

.. _exporting-a-model:

Exporting a model
-----------------

Training writes one ``.pt`` per ensemble member, each carrying a full copy of the
vocabularies, null-space basis and training diagnostics. ``sigmaflux-export``
packages them into a single file that stores the shared payload once and keeps
only the trained weights per member:

.. code-block:: bash

   sigmaflux-export --source <run_dir>/ensemble \
                    --config paper/configs/i_orientalis.yaml \
                    --model-version 1.0.0

On the released I. orientalis ensemble that is 252 MB of ``.pt`` files down to
about 34 MB — roughly 7x — with no identifiable reaction affected. Weights
default to ``fp16``; pass ``--precision fp32`` for an artifact whose predictions
are bit-identical to the source checkpoints.

Anything that takes a checkpoint takes an ``.sfx`` too, so inference is unchanged:

.. code-block:: bash

   sigmaflux-infer --config <config.yaml> --checkpoint i_orientalis-v1.0.0.sfx \
                   --data-dir <xlsx_dir> --out-dir <results_dir>

Versioning is three independent things: ``schema_version`` is the file format and
is bumped only when the layout changes (older files keep loading);
``model_version`` is the semver you give the weights; and the manifest's
``provenance`` block records the sigmaflux version, git commit, training config
and its checksum.
