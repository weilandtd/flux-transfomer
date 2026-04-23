import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "sigmaflux"
author = "Daniel R. Weilandt"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # NumPy/Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
