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


def assert_edge_weights_match_stoichiometry(emu_network):
    """A reaction delivers its whole stoichiometry to every EMU it produces.

    The weights of one (EMU, reaction, direction) group must add up to the
    reaction's coefficient for that metabolite. Collecting origin sets into a
    ``set`` used to break this whenever two mapped copies drew from the same educt
    EMU -- CBPS_c's two adp both come from atp_c, so the reaction delivered half
    its flux; lumpFACOA160_c delivered a seventh of its CO2 and lumpIPC_c a
    quarter, which pulled every carboxylation product away from x3cflux.
    """
    totals, coefficients = defaultdict(float), {}
    for graph in emu_network.subnetworks.values():
        for _, target, data in graph.edges(data=True):
            key = (repr(target), data['reaction'].id, data['direction'])
            totals[key] += data['weight']
            coefficients[key] = abs(data['reaction'].get_coefficient(target.id))
    for key, total in totals.items():
        assert total == pytest.approx(coefficients[key], abs=1e-9), (
            f"{key[0]} <- {key[1]} ({key[2]}) weights sum to {total}, but the "
            f"stoichiometry moves {coefficients[key]}"
        )


MODELS = [
    (os.path.join(PROJECT_DIR, 'models', 'atomistic', 'toy', 'one_carbon_annotated.json'), 'EC', 'toy'),
    (os.path.join(PROJECT_DIR, 'models', 'atomistic', 'I_orientalis', 'imIsor_annotated.json'), 'IO', 'yeast'),
    (os.path.join(PROJECT_DIR, 'models', 'atomistic', 'S_cerevisiae', 'sc_annotated.json'), 'SC', 'yeast'),
    (os.path.join(PROJECT_DIR, 'models', 'atomistic', 'R_toruloides', 'imRt_annotated.json'), 'RT', 'yeast'),
    (os.path.join(PROJECT_DIR, 'models', 'atomistic', 'M_musculus_T_Cell', 'mmTcell_annotated.json'), 'MM', 'yeast'),
]


@pytest.fixture(scope="module", params=MODELS, ids=[m[1] for m in MODELS])
def expanded(request):
    """One expansion per model, shared by every assertion below.

    Module-scoped because expanding the yeast networks is the expensive part and
    every test here reads the same finished graph.
    """
    path, name, dataset = request.param
    model = load_json_model(path)

    initial_fluxes = pd.Series(1.0, index=[r.id for r in model.reactions]) if dataset == 'yeast' else None
    mfa_model = MFAModel(model, name, initial_flux_distribution=initial_fluxes)

    data = MassIsotopomerData(mfa_model, make_test_data(model, dataset=dataset))
    flux_data = FluxData(mfa_model, make_test_flux_data(model, dataset=dataset))

    if dataset == 'yeast':
        tracer = Tracer(mfa_model.metabolites.glc__D_e, 0.1, [1, 1, 1, 1, 1, 1])
    else:
        tracer = Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1])

    mfa_model.prepare([Experiment(data, [tracer], fluxes=flux_data)])
    return mfa_model.expansion.emu_network


def test_no_redundant_edges(expanded):
    assert_no_redundant_edges(expanded)


def test_edge_weights_match_stoichiometry(expanded):
    assert_edge_weights_match_stoichiometry(expanded)
