"""Build the flat ``*_annotated.json`` models from their GEM + AMM workbooks.

Paths resolve against the repository root, not the working directory, since
``convert_mfa_maranas`` passes them straight to ``pd.read_excel``.

Usage:
    python models/atomistic/convert_maranas_models.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / 'src'))

from sigmaflux.core.utils import convert_mfa_maranas  # noqa: E402

_MODELS = pathlib.Path(__file__).parent


def _print_model_summary(label: str, model) -> None:
    """Print reaction/metabolite counts and BIOMASS presence for a converted model.

    Args:
        label: organism name to print as the summary heading.
        model: the converted COBRA model.
    """
    rxn_ids = {r.id for r in model.reactions}
    annotated = [r for r in model.reactions if 'atom_mapping' in r.annotation]
    print(f"\n{label}")
    print(f"  Reactions: {len(model.reactions)} total, {len(annotated)} annotated")
    print(f"  Metabolites: {len(model.metabolites)}")
    # These models carry a single aggregate BIOMASS; build_hierarchical_models.py
    # splits it into the five component reactions afterwards.
    print(f"  BIOMASS present (expected True): {'BIOMASS' in rxn_ids}")


def convert(gem, amm, out, **kwargs):
    """Convert one organism's GEM/AMM workbook pair to the flat annotated JSON model.

    Args:
        gem: GEM workbook path, relative to this file's directory.
        amm: AMM workbook path, relative to this file's directory.
        out: output JSON path, relative to this file's directory.
        kwargs: forwarded to convert_mfa_maranas (e.g. drop_subsystems, biomass_id).
    """
    return convert_mfa_maranas(str(_MODELS / gem), str(_MODELS / amm),
                               str(_MODELS / out), **kwargs)


# I_orientalis
model_lo = convert('I_orientalis/imIsor_GEM.xlsx', 'I_orientalis/imIsor_AMM.xlsx',
                   'I_orientalis/imIsor_annotated.json')
_print_model_summary("I_orientalis", model_lo)

# S_cerevisiae
model_sc = convert('S_cerevisiae/scGEM.xlsx', 'S_cerevisiae/scAMM.xlsx',
                   'S_cerevisiae/sc_annotated.json')
_print_model_summary("S_cerevisiae", model_sc)

# R_toruloides — completeMFA workbook, so the explicit dilution sub-network is dropped
model_rt = convert('R_toruloides/imRt_GEM.xlsx', 'R_toruloides/imRt_AMM.xlsx',
                   'R_toruloides/imRt_annotated.json',
                   drop_subsystems=('dilution',))
_print_model_summary("R_toruloides", model_rt)

# M_musculus_T_Cell — completeMFA workbook; biomass is named BIOMASS_c here
model_mm = convert('M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
                   'M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
                   'M_musculus_T_Cell/mmTcell_annotated.json',
                   drop_subsystems=('dilution',),
                   biomass_id='BIOMASS_c')
_print_model_summary("M_musculus_T_Cell", model_mm)

# E_coli_core / E_coli — the workbooks are normalised in place by
# curate_e_coli_workbooks.py first; run that before this if either xlsx changes.
# Both spell biomass 'biomass_out', which the configs' measured_fluxes and the
# single lumped BIOMASS both refer to as 'BIOMASS'.
model_ec_core = convert('E_coli_core/example_GEM.xlsx', 'E_coli_core/example_AMM.xlsx',
                        'E_coli_core/e_coli_core_annotated.json',
                        biomass_id='biomass_out')
_print_model_summary("E_coli_core", model_ec_core)

model_ec = convert('E_coli/iAF1260_GEM.xlsx', 'E_coli/iAF1260_AMM.xlsx',
                   'E_coli/iAF1260_annotated.json',
                   biomass_id='biomass_out')
_print_model_summary("E_coli", model_ec)
