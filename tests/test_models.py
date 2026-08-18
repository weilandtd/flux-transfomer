"""Every shipped network builds an EMU model and runs a forward pass.

One dataset entry per model family. The per-family specifics -- what the workbook
calls glucose, what it calls pyruvate, which reactions carry a measurable flux --
used to be if/elif chains inside the test body, which is why the two E. coli models
were never covered: neither uses BiGG ids the way the yeasts do (the core model
writes `Gluc_e`/`Pyr`/`Mal` and numbered reactions, iAF1260 writes `glc-D_e` but
`mal-L_c` rather than `mal__L_c`), so adding them meant a third and fourth branch in
four places. As a table it is one row each.
"""
import os

import pandas as pd
import pytest
import torch
import yaml
from cobra.io.json import load_json_model

from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.model import MFAModel
from sigmaflux.pipeline.sample_flux import apply_media, apply_reaction_bounds

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
MODELS = os.path.join(PROJECT_DIR, 'models', 'atomistic')

#: prefix        stem of the two data/test CSVs
#: tracer        metabolite the label enters on, and its enrichment/labeling
#: target        metabolite whose simulated MID must come out non-zero
#: unit_fluxes   Maranas-format models have `_out` metabolites that COBRA does not
#:               mass-balance, so sampling returns zero glucose flux; start from
#:               uniform unit fluxes instead. The toy model is balanced and does not.
#: config        organism YAML whose media block to apply first, when the model needs
#:               one. Both E. coli workbooks ship every exchange as secretion-only
#:               (`EX_Gluc_e` is bounded (0, 1000)), so on the raw JSON no carbon can
#:               enter and the EMU expansion is legitimately empty -- 0 edges at every
#:               size. The media block is what opens glucose uptake, so applying it is
#:               not test scaffolding: it is the difference between testing the model
#:               the pipeline uses and testing one with no carbon source.
DATASETS = {
    'toy': dict(prefix='toy_test', tracer='Ser_in', enrichment=0.1,
                labeling=[1, 1, 1], target='Ser', unit_fluxes=False, config=None),
    'yeast': dict(prefix='yeast_test', tracer='glc__D_e', enrichment=0.5,
                  labeling=[1] * 6, target='pyr_c', unit_fluxes=True, config=None),
    'ecoli': dict(prefix='ecoli_test', tracer='glc-D_e', enrichment=0.5,
                  labeling=[1] * 6, target='pyr_c', unit_fluxes=True,
                  config='paper/configs/e_coli.yaml'),
    'ecoli_core': dict(prefix='ecoli_core_test', tracer='Gluc_e', enrichment=0.5,
                       labeling=[1] * 6, target='Pyr', unit_fluxes=True,
                       config='paper/configs/e_coli_core.yaml'),
}

MODEL_CASES = [
    (os.path.join(MODELS, 'toy', 'one_carbon_annotated.json'), 'EC', 'toy'),
    (os.path.join(MODELS, 'I_orientalis', 'imIsor_annotated.json'), 'IO', 'yeast'),
    (os.path.join(MODELS, 'S_cerevisiae', 'sc_annotated.json'), 'SC', 'yeast'),
    (os.path.join(MODELS, 'R_toruloides', 'imRt_annotated.json'), 'RT', 'yeast'),
    (os.path.join(MODELS, 'M_musculus_T_Cell', 'mmTcell_annotated.json'), 'MM', 'yeast'),
    (os.path.join(MODELS, 'E_coli_core', 'e_coli_core_annotated.json'), 'ECC', 'ecoli_core'),
    (os.path.join(MODELS, 'E_coli', 'iAF1260_annotated.json'), 'ECO', 'ecoli'),
]


def make_test_data(model, dataset='yeast'):
    """Measured MIDs for *dataset*, restricted to metabolites the model has.

    The filter mirrors :func:`make_test_flux_data`. Without it a stale id passes
    straight through to ``MassIsotopomerData``, which does not raise on a miss, so
    the test would go on to assert against a silently empty measurement set.
    """
    prefix = DATASETS[dataset]['prefix']
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', f'{prefix}_labeling.csv'))
    data = df.groupby(['id', 'mid']).mean()[['enrichment']]
    data['std'] = df.groupby(['id', 'mid']).std()['enrichment']
    data.columns = ['mean', 'std']
    data = data.reset_index()

    metabolite_ids = {m.id for m in model.metabolites}
    kept = data[data['id'].isin(metabolite_ids)].reset_index(drop=True)
    assert not kept.empty, f"no {dataset} labeling ids present in the model"
    return kept


def make_test_flux_data(model, dataset='yeast'):
    prefix = DATASETS[dataset]['prefix']
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'test', f'{prefix}_fluxes.csv'))
    data = df.groupby(['id']).mean()[['flux']]
    data['std'] = df.groupby(['id']).std()['flux']
    data.columns = ['mean', 'std']
    data = data.reset_index().rename(columns={'id': 'reaction'})

    reaction_ids = {reaction.id for reaction in model.reactions}
    kept = data[data['reaction'].isin(reaction_ids)].reset_index(drop=True)
    assert not kept.empty, f"no {dataset} flux ids present in the model"
    return kept


def run_model_test(path, name, dataset='yeast'):
    torch.autograd.set_detect_anomaly(True)
    spec = DATASETS[dataset]

    model = load_json_model(path)
    if spec['config']:
        with open(os.path.join(PROJECT_DIR, spec['config'])) as fh:
            sampling = yaml.safe_load(fh)['sampling']
        apply_media(model, sampling['media'], close_unlisted=True)
        apply_reaction_bounds(model, sampling.get('reaction_bounds'))

    initial_fluxes = (pd.Series(1.0, index=[r.id for r in model.reactions])
                      if spec['unit_fluxes'] else None)

    mfa_model = MFAModel(model, name, initial_flux_distribution=initial_fluxes)

    assert len(mfa_model.reactions) > 0
    assert {r.id for r in mfa_model.reactions} == {r.id for r in model.reactions}

    data = MassIsotopomerData(mfa_model, make_test_data(model, dataset=dataset))
    flux_data = FluxData(mfa_model, make_test_flux_data(model, dataset=dataset))
    tracer = Tracer(mfa_model.metabolites.get_by_id(spec['tracer']),
                    spec['enrichment'], spec['labeling'])
    experiment = Experiment(data, [tracer], fluxes=flux_data)

    mfa_model.prepare([experiment])

    assert len(mfa_model.expansion.emu_network.subnetworks[1].edges()) > 0

    labeling = mfa_model()
    target = mfa_model._metabolite_hash[spec['target']]
    assert labeling[:, target].detach().numpy().any()


@pytest.mark.parametrize("path,name,dataset", MODEL_CASES)
def test_models(path, name, dataset):
    run_model_test(path, name, dataset=dataset)


if __name__ == '__main__':
    for path, name, dataset in MODEL_CASES:
        run_model_test(path, name, dataset=dataset)
