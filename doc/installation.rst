Installation
============

.. code-block:: bash

   # Development install
   pip install -e ".[dev,docs]"

   # Runtime only
   pip install flux_transformer

The optional x3cflux MFA backend needs Ipopt, and pip's ``cyipopt`` is an sdist that
wants Ipopt headers at build time — so it has to come from conda-forge. Run this from
the repo root, since the environment file's ``-e .`` resolves against the working
directory:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate flux_transformer
