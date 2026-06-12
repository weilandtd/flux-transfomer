import pytest
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

print(f"TEST_DIR: {TEST_DIR}")
print(f"PROJECT_DIR: {PROJECT_DIR}")


def _dataset_prefix(dataset):
    if dataset == 'yeast':
        return 'yeast_test'
    if dataset == 'toy':
        return 'toy_test'
    raise ValueError(f"Unknown dataset '{dataset}'. Expected 'yeast' or 'toy'.")


def make_test_data(dataset='yeast'):
    prefix = _dataset_prefix(dataset)
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', f'{prefix}_labeling.csv'))
    data = df.groupby(['id', 'mid']).mean()[['enrichment', ]]
    data['std'] = df.groupby(['id', 'mid']).std()['enrichment']
    data.columns = ['mean', 'std']
    return data.reset_index()


def make_test_flux_data(model, dataset='yeast'):
    prefix = _dataset_prefix(dataset)
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', f'{prefix}_fluxes.csv'))
    data = df.groupby(['id', ]).mean()[['flux', ]]
    data['std'] = df.groupby(['id', ]).std()['flux']
    data.columns = ['mean', 'std']
    data = data.reset_index().rename(columns={'id': 'reaction'})

    # Keep only fluxes that exist in the loaded model.
    reaction_ids = {reaction.id for reaction in model.reactions}
    return data[data['reaction'].isin(reaction_ids)].reset_index(drop=True)


# Test model initialization of IO model
def run_model_test(path, name, dataset='yeast'):
    torch.autograd.set_detect_anomaly(True)

    model = load_json_model(path)

    # Maranas-format models have _out metabolites that are not mass-balanced in COBRA,
    # so COBRApy sampling gives zero glucose fluxes. Use uniform unit fluxes instead.
    if dataset == 'yeast':
        initial_fluxes = pd.Series(1.0, index=[r.id for r in model.reactions])
    else:
        initial_fluxes = None

    mfa_model = MFAModel(model, name, initial_flux_distribution=initial_fluxes)

    assert len(mfa_model.reactions) > 0
    assert {r.id for r in mfa_model.reactions} == {r.id for r in model.reactions}

    # Make a test experiment
    data = MassIsotopomerData(mfa_model, make_test_data(dataset=dataset))
    flux_data = FluxData(mfa_model, make_test_flux_data(model, dataset=dataset))

    if dataset == 'yeast':
        tracer_1 = Tracer(mfa_model.metabolites.glc__D_e, 0.5, [1, 1, 1, 1, 1, 1])
    elif dataset == 'toy':
        tracer_1 = Tracer(mfa_model.metabolites.Ser_in, 0.1, [1, 1, 1, ])
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'yeast' or 'toy'.")

    experiment = Experiment(data, [tracer_1, ], fluxes=flux_data)

    # Test model
    mfa_model.prepare([experiment, ])

    # Assert network expansion
    assert len(mfa_model.expansion.emu_network.subnetworks[1].edges()) > 0

    # TODO add more tests for model expansion and labeling prediction

    labeling = mfa_model()

    if dataset == 'yeast':
        assert labeling[:, mfa_model._metabolite_hash['pyr_c']].detach().numpy().any()
    elif dataset == 'toy':
        assert labeling[:, mfa_model._metabolite_hash['Ser']].detach().numpy().any()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'yeast' or 'toy'.")


@pytest.mark.parametrize("path,name,dataset", [
    (os.path.join(PROJECT_DIR, 'models', 'toy', 'one_carbon_annotated.json'), 'EC', 'toy'),
    (os.path.join(PROJECT_DIR, 'models', 'I_orientalis', 'imIsor_annotated.json'), 'IO', 'yeast'),
    (os.path.join(PROJECT_DIR, 'models', 'S_cerevisiae', 'sc_annotated.json'), 'SC', 'yeast'),
])
def test_models(path, name, dataset):
    run_model_test(path, name, dataset=dataset)


if __name__ == '__main__':
    run_model_test(os.path.join(PROJECT_DIR, 'models', 'toy', 'one_carbon_annotated.json'), 'EC', dataset='toy')
    run_model_test(os.path.join(PROJECT_DIR, 'models', 'I_orientalis', 'imIsor_annotated.json'), 'IO', dataset='yeast')
    run_model_test(os.path.join(PROJECT_DIR, 'models', 'S_cerevisiae', 'sc_annotated.json'), 'SC', dataset='yeast')

