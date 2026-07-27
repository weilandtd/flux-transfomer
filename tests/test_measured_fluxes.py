"""Tests for the config-driven measured ("bypass") flux features.

These features used to be hard-coded to mouse whole-body respirometry
(EX_o2_e_lung / EX_co2_e_lung -> VO2 / VCO2 / RER) with a default config path
pointing outside the package. They are now described entirely by a caller-
supplied JSON config, so any organism can declare its own.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sigmaflux.ai.data import (
    extract_measured_fluxes,
    get_measured_fluxes_reaction_ids,
    load_measured_fluxes_config,
)


@pytest.fixture
def config_path(tmp_path):
    cfg = {
        "description": "a plain string block must be ignored",
        "respiratory": {
            # Declared under "reactions", so these feed derived features but are
            # not emitted as features in their own right.
            "reactions": {
                "EX_o2_e_lung": "O2 uptake (negative)",
                "EX_co2_e_lung": "CO2 release (positive)",
            },
            "derived": {
                "VO2": {"reaction": "EX_o2_e_lung", "sign": -1},
                "VCO2": {"reaction": "EX_co2_e_lung", "sign": 1},
                "RER": {"ratio": ["VCO2", "VO2"]},
            },
        },
        "protein_synthesis": {
            "reaction_ids": ["protein_synthesis_liver", "protein_synthesis_heart"],
        },
        "summary": {"total_features": 5},
    }
    path = tmp_path / "measured_fluxes.json"
    path.write_text(json.dumps(cfg))
    return str(path)


@pytest.fixture
def flux_df():
    return pd.DataFrame(
        [[-10.0, -20.0],   # EX_o2_e_lung  (uptake, negative)
         [8.0, 18.0],      # EX_co2_e_lung (release, positive)
         [1.0, 2.0],       # protein_synthesis_liver
         [3.0, 4.0]],      # protein_synthesis_heart
        index=["EX_o2_e_lung", "EX_co2_e_lung",
               "protein_synthesis_liver", "protein_synthesis_heart"],
        columns=[0, 1],
    )


def test_no_default_config_path():
    """The config is organism-specific, so there must be no implicit default."""
    with pytest.raises(ValueError, match="explicit config_path"):
        load_measured_fluxes_config(None)


def test_spec_none_yields_nothing():
    assert get_measured_fluxes_reaction_ids(None) == ([], {}, [])
    assert get_measured_fluxes_reaction_ids(False) == ([], {}, [])


def test_explicit_list_needs_no_config():
    rxns, derived, names = get_measured_fluxes_reaction_ids(["R1", "R2"])
    assert rxns == ["R1", "R2"]
    assert derived == {}
    assert names == ["R1", "R2"]


def test_all_selects_every_group(config_path):
    rxns, derived, names = get_measured_fluxes_reaction_ids("all", config_path)
    assert set(rxns) == {"EX_o2_e_lung", "EX_co2_e_lung",
                         "protein_synthesis_liver", "protein_synthesis_heart"}
    assert set(derived) == {"VO2", "VCO2", "RER"}
    # Respiratory source reactions are inputs to derived features, not features.
    assert names == ["VO2", "VCO2", "RER",
                     "protein_synthesis_liver", "protein_synthesis_heart"]


def test_group_selection(config_path):
    _, derived, names = get_measured_fluxes_reaction_ids(
        {"groups": ["protein_synthesis"]}, config_path)
    assert derived == {}
    assert names == ["protein_synthesis_liver", "protein_synthesis_heart"]


def test_derived_can_be_disabled(config_path):
    _, derived, names = get_measured_fluxes_reaction_ids(
        {"groups": ["respiratory"], "derived": False}, config_path)
    assert derived == {}
    assert names == []          # respiratory declares no plain reaction_ids


def test_unsupported_spec_raises(config_path):
    with pytest.raises(ValueError, match="Unsupported measured-flux spec"):
        get_measured_fluxes_reaction_ids("respiratory", config_path)


def test_extract_computes_signed_and_ratio_features(config_path, flux_df):
    data, names = extract_measured_fluxes(flux_df, [0, 1], spec="all",
                                          config_path=config_path)
    assert names == ["VO2", "VCO2", "RER",
                     "protein_synthesis_liver", "protein_synthesis_heart"]
    assert data.shape == (2, 5)
    np.testing.assert_allclose(data[:, 0], [10.0, 20.0])    # VO2 = -EX_o2
    np.testing.assert_allclose(data[:, 1], [8.0, 18.0])     # VCO2 = EX_co2
    np.testing.assert_allclose(data[:, 2], [0.8, 0.9])      # RER = VCO2 / VO2
    np.testing.assert_allclose(data[:, 3], [1.0, 2.0])


def test_extract_with_explicit_reaction_ids(config_path, flux_df):
    data, names = extract_measured_fluxes(
        flux_df, [0], reaction_ids=["protein_synthesis_heart"])
    assert names == ["protein_synthesis_heart"]
    np.testing.assert_allclose(data, [[3.0]])


def test_missing_reactions_are_dropped_not_fatal(config_path, flux_df):
    """A reaction absent from the flux data must not desync data and names."""
    data, names = extract_measured_fluxes(
        flux_df, [0, 1], reaction_ids=["protein_synthesis_liver", "not_in_model"])
    assert names == ["protein_synthesis_liver"]
    assert data.shape == (2, 1)


def test_ratio_guards_against_zero_denominator(config_path, tmp_path):
    zero_o2 = pd.DataFrame(
        [[0.0], [5.0]], index=["EX_o2_e_lung", "EX_co2_e_lung"], columns=[0])
    data, names = extract_measured_fluxes(
        zero_o2, [0], spec={"groups": ["respiratory"]}, config_path=config_path)
    assert names == ["VO2", "VCO2", "RER"]
    assert data[0, 2] == 0.0        # RER falls back to 0 rather than dividing by 0


def test_string_derived_definition_is_rejected(tmp_path, flux_df):
    """Descriptive-string configs cannot be computed and must say so clearly."""
    cfg = {"respiratory": {"reactions": {"EX_o2_e_lung": "x"},
                           "derived": {"VO2": "Oxygen consumption (= -EX_o2_e_lung)"}}}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="described by a string"):
        extract_measured_fluxes(flux_df, [0], spec="all", config_path=str(path))
