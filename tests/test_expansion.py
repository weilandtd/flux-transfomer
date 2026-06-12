import pytest
import pandas as pd
import os
from collections import defaultdict
from cobra.io.json import load_json_model
from sigmaflux.core.model import MFAModel
from sigmaflux.core.experiment import Tracer, Experiment
from sigmaflux.core.data import MassIsotopomerData, FluxData
from test_models import make_test_data, make_test_flux_data

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))


def assert_no_redundant_edges(emu_network):
    for size, graph in emu_network.subnetworks.items():
        seen = defaultdict(set)
        for u, v, data in graph.edges(data=True):
            sig = (data['reaction'].id, data['direction'])
            assert sig not in seen[(u, v)], (
                f"Redundant edge in subnetwork size={size}: "
                f"{u} → {v} has duplicate (reaction, direction) pair: {sig}"
            )
            seen[(u, v)].add(sig)


@pytest.mark.parametrize("path,name,dataset", [
    (os.path.join(PROJECT_DIR, 'models', 'toy', 'one_carbon_annotated.json'), 'EC', 'toy'),
    (os.path.join(PROJECT_DIR, 'models', 'I_orientalis', 'imIsor_annotated.json'), 'IO', 'yeast'),
    (os.path.join(PROJECT_DIR, 'models', 'S_cerevisiae', 'sc_annotated.json'), 'SC', 'yeast'),
])
def test_no_redundant_edges(path, name, dataset):
    model = load_json_model(path)

    initial_fluxes = pd.Series(1.0, index=[r.id for r in model.reactions]) if dataset == 'yeast' else None
    mfa_model = MFAModel(model, name, initial_flux_distribution=initial_fluxes)

    data = MassIsotopomerData(mfa_model, make_test_data(dataset=dataset))
    flux_data = FluxData(mfa_model, make_test_flux_data(model, dataset=dataset))

    if dataset == 'yeast':
        tracer = Tracer(mfa_model.metabolites.glc__D_e, 0.1, [1, 1, 1, 1, 1, 1])
    else:
        tracer = Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1])

    mfa_model.prepare([Experiment(data, [tracer], fluxes=flux_data)])

    assert_no_redundant_edges(mfa_model.expansion.emu_network)
