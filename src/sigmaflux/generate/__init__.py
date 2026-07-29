"""Generic model-building workflow: assemble reduced models from metabolic tasks.

Organism-specific model construction — experimental-database readers, anatomy
(e.g. adding a heart compartment), and hard-coded dummy reactions — lives in the
downstream project that owns those models, not here.
"""

from sigmaflux.generate.core import add_reactions_from_tasks
from sigmaflux.generate.tissue import (
    add_input_reaction,
    add_output_reaction,
    get_organ_model_from_model,
    get_tissue_model_from_model,
    label_model,
    merge_models,
    restrict_exchange,
)
from sigmaflux.generate.utils import read_subnet

__all__ = [
    "add_reactions_from_tasks",
    "add_input_reaction",
    "add_output_reaction",
    "get_organ_model_from_model",
    "get_tissue_model_from_model",
    "label_model",
    "merge_models",
    "read_subnet",
    "restrict_exchange",
]
