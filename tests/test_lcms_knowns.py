"""The LC/MS platform matcher: what it must match, and what it must refuse.

Malate was absent from every organism's measurable space because the GEM writes
``(S)-malate`` and the library ``L-Malic acid``, and the two normalise to ``smal`` and
``mal``.  The gap was invisible -- a metabolite that never becomes a token simply never
appears -- so these tests pin both directions: the names that must resolve, and the
looser matches that would resolve them at the cost of a wrong one.
"""
import os
import warnings

import pandas as pd
import pytest

from sigmaflux.utils.lcms_knowns import (
    _GEM_NAME_ALIASES,
    _alias_key,
    _conflicts,
    match_lcms_knowns,
    normalize_compound,
    read_lcms_knowns,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWNS = os.path.join(REPO_ROOT, "data", "HILIC_LCMS_knowns.csv")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "lcms_mapping_golden.tsv")

#: name, model json, GEM xlsx -- every organism whose mapping this matcher builds.
ORGANISMS = [
    ("I_orientalis", "I_orientalis/imIsor_annotated_hierarchical.json",
     "I_orientalis/imIsor_GEM.xlsx", None),
    ("S_cerevisiae", "S_cerevisiae/sc_annotated_hierarchical.json",
     "S_cerevisiae/scGEM.xlsx", None),
    ("R_toruloides", "R_toruloides/imRt_annotated_hierarchical.json",
     "R_toruloides/imRt_GEM.xlsx", None),
    ("M_musculus_T_Cell", "M_musculus_T_Cell/mmTcell_annotated_hierarchical.json",
     "M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx", None),
    ("E_coli", "E_coli/iAF1260_annotated.json", "E_coli/iAF1260_GEM.xlsx", "E_coli"),
    ("E_coli_core", "E_coli_core/e_coli_core_annotated.json",
     "E_coli_core/example_GEM.xlsx", "E_coli_core"),
]


def _match(gem_xlsx, model_json, fallback):
    """Run the matcher exactly as build_metabolite_mapping_files.py does."""
    with open(gem_xlsx, "rb") as fh:
        if fh.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            pytest.skip(f"{gem_xlsx} is a Git LFS pointer; workbook content unavailable")
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "models", "atomistic"))
    from cobra.io import load_json_model

    metabolites = pd.read_excel(gem_xlsx, sheet_name="Metabolites",
                                usecols=["metid", "metname"])
    if fallback:
        from _bigg_name_fallback import (
            E_COLI_BIGG_NAMES,
            E_COLI_CORE_NAMES,
            fill_missing_names,
        )
        table = E_COLI_BIGG_NAMES if fallback == "E_coli" else E_COLI_CORE_NAMES
        metabolites = fill_missing_names(metabolites, "metid", "metname", table,
                                         fallback == "E_coli")
    model = load_json_model(model_json)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return match_lcms_knowns(
            gem_xlsx=gem_xlsx, knowns_path=KNOWNS,
            model_metabolite_ids=[m.id for m in model.metabolites],
            metabolites_df=metabolites,
        )


@pytest.mark.parametrize("gem_name, compound", [
    ("(S)-malate", "L-Malic acid"),
    ("2-oxoglutarate", "Oxoglutaric acid"),
])
def test_alias_bridges_names_the_normalizer_cannot(gem_name, compound):
    """The two conventions spell these irreconcilably, so the alias table carries them."""
    assert _GEM_NAME_ALIASES[_alias_key(gem_name)] == compound
    assert normalize_compound(gem_name) != normalize_compound(compound)


def test_d_lactate_does_not_match_the_l_enantiomer():
    """The obvious repair -- strip the stereodescriptor -- makes exactly this mistake."""
    assert _conflicts("(R)-lactate", "L-Lactic acid")
    assert _alias_key("(R)-lactate") not in _GEM_NAME_ALIASES


def test_sedoheptulose_phosphate_does_not_match_the_free_sugar():
    """Different compound, different mass; the library lists both separately."""
    assert _conflicts("sedoheptulose 7-phosphate", "Sedoheptulose")
    assert normalize_compound("sedoheptulose 7-phosphate") != normalize_compound("Sedoheptulose")


def test_unqualified_names_do_not_conflict():
    """A name that declares no configuration constrains nothing."""
    assert not _conflicts("L-aspartate", "L-Aspartic acid")
    assert not _conflicts("D-glucose 6-phosphate", "Glucose 6-phosphate")
    assert not _conflicts("pyruvate", "Pyruvic acid")


def test_alias_target_must_exist_in_the_knowns_table():
    """A renamed library entry would otherwise degrade the alias back to no match."""
    compounds = set(read_lcms_knowns(KNOWNS)["compound"])
    assert set(_GEM_NAME_ALIASES.values()) <= compounds


def test_mapping_matches_the_golden_fixture():
    """Lock the resolved mapping for every organism.

    A silently different token set is what this whole file exists to prevent, and the
    only way to see one is to write the mapping down and compare.
    """
    golden = pd.read_csv(GOLDEN, sep="\t", header=None,
                         names=["organism", "experimental_id", "model_id"])

    rows = []
    for name, model_json, gem_xlsx, fallback in ORGANISMS:
        mapping = _match(os.path.join(REPO_ROOT, "models", "atomistic", gem_xlsx),
                         os.path.join(REPO_ROOT, "models", "atomistic", model_json),
                         fallback)
        mapping = mapping.sort_values(["experimental_id", "model_id"])
        rows += [(name, r.experimental_id, r.model_id) for r in mapping.itertuples()]

    actual = pd.DataFrame(rows, columns=["organism", "experimental_id", "model_id"])
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),
                                  golden.reset_index(drop=True))


@pytest.mark.parametrize("organism", ["I_orientalis", "S_cerevisiae", "M_musculus_T_Cell"])
@pytest.mark.parametrize("compound", ["L-Malic acid", "Oxoglutaric acid",
                                      "Sedoheptulose 7-phosphate"])
def test_recovered_metabolites_are_in_the_curated_mapping(organism, compound):
    """These three are measured in every workbook; before the fix none was a token."""
    mapping = pd.read_csv(os.path.join(REPO_ROOT, "models", "atomistic", organism,
                                       f"{organism}_mapping.csv"))
    assert compound in set(mapping["experimental_id"])


@pytest.mark.parametrize("organism", ["I_orientalis", "S_cerevisiae", "M_musculus_T_Cell"])
@pytest.mark.parametrize("short_name", ["Mal", "aKG", "S7P"])
def test_workbook_names_are_readable(organism, short_name):
    """The experiment mapping is what turns a workbook column into a model id."""
    mapping = pd.read_csv(os.path.join(REPO_ROOT, "models", "atomistic", organism,
                                       f"{organism}_experiment_mapping.csv"))
    assert short_name in set(mapping["experimental_id"])


def test_e_coli_core_fallback_table_has_malate_and_s7p():
    """example_GEM.xlsx carries no metabolite names at all, so Mal/S7P need an explicit
    fallback entry -- unlike the yeast/T-cell bug, there is no name here to normalize.
    """
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "models", "atomistic"))
    from _bigg_name_fallback import E_COLI_CORE_NAMES

    assert E_COLI_CORE_NAMES["Mal"] == "L-Malic acid"
    assert E_COLI_CORE_NAMES["S7P"] == "Sedoheptulose 7-phosphate"


def test_e_coli_core_mapping_includes_malate_and_s7p():
    mapping = pd.read_csv(os.path.join(REPO_ROOT, "models", "atomistic", "E_coli_core",
                                       "E_coli_core_mapping.csv"))
    ids = set(mapping["experimental_id"])
    assert "L-Malic acid" in ids
    assert "Sedoheptulose 7-phosphate" in ids
