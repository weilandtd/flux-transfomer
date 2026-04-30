import torch
from cobra.io.json import load_json_model
from sigmaflux.core.model import MFAModel
from sigmaflux.core.experiment import Tracer, Experiment
from sigmaflux.core.data import MassIsotopomerData, FluxData


import pandas as pd
import numpy as np
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))


def make_test_data():
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', 'test_labeling.csv'))
    data = df.groupby(['id', 'mid']).mean()[['enrichment', ]]
    data['std'] = df.groupby(['id', 'mid']).std()['enrichment']
    data.columns = ['mean', 'std']
    return data.reset_index()


def make_test_flux_data(model):
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', 'test_fluxes.csv'))
    data = df.groupby(['id', ]).mean()[['flux', ]]
    data['std'] = df.groupby(['id', ]).std()['flux']
    data.columns = ['mean', 'std']
    data = data.reset_index().rename(columns={'id': 'reaction'})

    # Keep only fluxes that exist in the loaded model.
    reaction_ids = {reaction.id for reaction in model.reactions}
    return data[data['reaction'].isin(reaction_ids)].reset_index(drop=True)


# Test model initialization of IO model
def run_model_test(path, name):
    torch.autograd.set_detect_anomaly(True)

    model = load_json_model(path)
    mfa_model = MFAModel(model, name )

    assert len(mfa_model.reactions) > 0
    assert {r.id for r in mfa_model.reactions} == {r.id for r in model.reactions}

    # Make a test experiment
    data = MassIsotopomerData(mfa_model, make_test_data())
    flux_data = FluxData(mfa_model, make_test_flux_data(model))

    tracer_1 = Tracer(mfa_model.metabolites.glc__D_e, 0.1, [1, 1, 1, 1, 1, 1])

    experiment = Experiment(data, [tracer_1, ], fluxes=flux_data)

    # Test model
    mfa_model.prepare([experiment, ])

    # Assert network expansion
    assert len(mfa_model.expansion.emu_network.subnetworks[1].edges()) > 0

    # TODO add more tests for model expansion and labeling prediction

    labeling = mfa_model()

    assert (labeling[:, mfa_model._metabolite_hash['pyr_c']].detach().numpy() \
            != np.array([0.0, 0.0, 0.0, 0.0, 0.0], )).any()



def test_models():
    run_model_test(os.path.join(PROJECT_DIR, 'models', 'I_orientalis', 'imIsor_annotated.json'), 'IO')
    run_model_test(os.path.join(PROJECT_DIR, 'models', 'S_cerevisiae', 'sc_annotated.json'), 'SC')
