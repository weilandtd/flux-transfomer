from cobra.io.json import load_json_model

import pandas as pd
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))

# TODO Make one load data function so these become 2 liners!!!

def make_test_data():
    # Raw file
    df = pd.read_csv(PROJECT_DIR + '/../data/test/test_labeling.csv')
    # Process the file
    data = df.groupby(['id', 'mid']).mean()[['enrichment', ]]
    data['std'] = df.groupby(['id', 'mid']).std()['enrichment']
    data.columns = ['mean', 'std']

    data = data.reset_index()

    return data


def make_test_flux_data():
    # Raw file
    df = pd.read_csv(PROJECT_DIR + '/../data/test/test_fluxes.csv')
    # Process the file
    data = df.groupby(['id',]).mean()[['flux', ]]
    data['std'] = df.groupby(['id',]).std()['flux']
    data.columns = ['mean', 'std']

    data = data.reset_index()
    data = data.rename(columns={'id': 'reaction'})  # Rename to match expected column name

    return data


def make_toy_test_data(csv = None):
    # Raw file
    if csv is not None: 
        df = pd.read_csv(csv)
    else:
        df = pd.read_csv(PROJECT_DIR + '/../data/test/toy_test_labeling.csv')
    # Process the file
    data = df.groupby(['id', 'mid']).mean()[['enrichment', ]]
    data['std'] = df.groupby(['id', 'mid']).std()['enrichment']
    data.columns = ['mean', 'std']

    data = data.reset_index()

    return data


def make_toy_test_flux_data(csv = None):
    # Raw file
    if csv is not None: 
        df = pd.read_csv(csv)
    else:
        df = pd.read_csv(PROJECT_DIR + '/../data/test/toy_test_fluxes.csv')
    # Process the file
    data = df.groupby(['id',]).mean()[['flux', ]]
    data['std'] = df.groupby(['id',]).std()['flux']
    data.columns = ['mean', 'std']

    data = data.reset_index()
    data = data.rename(columns={'id': 'reaction'})  # Rename to match expected column name

    return data
