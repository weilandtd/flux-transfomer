from cobra.io.json import save_json_model

from sigmaflux.core.utils import add_dilution_inputs, convert_mfa_maranas

#: Pools given an explicit unlabeled environmental input, per organism.
#
# A pool that exchanges with the environment is diluted by it, but only if the
# environment is a node in the EMU network. ``co2_e`` is produced internally by
# CO2t_c_e, so it is a core mixing node, while the uptake direction of EX_co2_e
# produces it from nothing -- an empty origin set, no edge, no dilution. With PC_m
# refixing CO2 at ~40 flux units, that error propagates into oxaloacetate and the
# TCA/aspartate branch.
#
# This is a modelling choice, not a fact, which is why it lives here per organism
# rather than being applied unconditionally: whether the diluted pool is co2_e,
# co2_c or co2_m depends on where the exchange is taken to happen, and a closed
# system would want none of them. Note the completeMFA workbooks ship their own
# ``dilution`` subsystem, dropped below -- if that is reinstated for an organism,
# it likely replaces this entry rather than adding to it.
DILUTION_INPUTS = {
    'I_orientalis': ['co2_e'],
    'S_cerevisiae': ['co2_e'],
    'R_toruloides': ['co2_e'],
    'M_musculus_T_Cell': ['co2_e'],
}


def _add_dilution(label: str, model, path: str) -> None:
    """Apply this organism's dilution inputs and re-save."""
    targets = [m for m in DILUTION_INPUTS.get(label, []) if m in model.metabolites]
    added = add_dilution_inputs(model, targets)
    if added:
        save_json_model(model, path)
    print(f"  Dilution inputs: {added or 'none'}")


def _print_model_summary(label: str, model) -> None:
    rxn_ids = {r.id for r in model.reactions}
    annotated = [r for r in model.reactions if 'atom_mapping' in r.annotation]
    print(f"\n{label}")
    print(f"  Reactions: {len(model.reactions)} total, {len(annotated)} annotated")
    print(f"  Metabolites: {len(model.metabolites)}")
    # These models carry a single aggregate BIOMASS; build_hierarchical_models.py
    # splits it into the five component reactions afterwards.
    print(f"  BIOMASS present (expected True): {'BIOMASS' in rxn_ids}")


# I_orientalis
model_lo = convert_mfa_maranas('I_orientalis/imIsor_GEM.xlsx', 'I_orientalis/imIsor_AMM.xlsx', 'I_orientalis/imIsor_annotated.json')
_print_model_summary("I_orientalis", model_lo)
_add_dilution("I_orientalis", model_lo, 'I_orientalis/imIsor_annotated.json')

# S_cerevisiae
model_sc = convert_mfa_maranas('S_cerevisiae/scGEM.xlsx', 'S_cerevisiae/scAMM.xlsx', 'S_cerevisiae/sc_annotated.json')
_print_model_summary("S_cerevisiae", model_sc)
_add_dilution("S_cerevisiae", model_sc, 'S_cerevisiae/sc_annotated.json')

# R_toruloides — completeMFA workbook, so the explicit dilution sub-network is dropped
model_rt = convert_mfa_maranas('R_toruloides/imRt_GEM.xlsx', 'R_toruloides/imRt_AMM.xlsx',
                               'R_toruloides/imRt_annotated.json',
                               drop_subsystems=('dilution',))
_print_model_summary("R_toruloides", model_rt)
_add_dilution("R_toruloides", model_rt, 'R_toruloides/imRt_annotated.json')

# M_musculus_T_Cell — completeMFA workbook; biomass is named BIOMASS_c here
model_mm = convert_mfa_maranas('M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_annotated.json',
                               drop_subsystems=('dilution',),
                               biomass_id='BIOMASS_c')
_print_model_summary("M_musculus_T_Cell", model_mm)
_add_dilution("M_musculus_T_Cell", model_mm, 'M_musculus_T_Cell/mmTcell_annotated.json')
