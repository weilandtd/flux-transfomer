from sigmaflux.core.utils import convert_mfa_maranas


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

# S_cerevisiae
model_sc = convert_mfa_maranas('S_cerevisiae/scGEM.xlsx', 'S_cerevisiae/scAMM.xlsx', 'S_cerevisiae/sc_annotated.json')
_print_model_summary("S_cerevisiae", model_sc)

# R_toruloides — completeMFA workbook, so the explicit dilution sub-network is dropped
model_rt = convert_mfa_maranas('R_toruloides/imRt_GEM.xlsx', 'R_toruloides/imRt_AMM.xlsx',
                               'R_toruloides/imRt_annotated.json',
                               drop_subsystems=('dilution',))
_print_model_summary("R_toruloides", model_rt)

# M_musculus_T_Cell — completeMFA workbook; biomass is named BIOMASS_c here
model_mm = convert_mfa_maranas('M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_annotated.json',
                               drop_subsystems=('dilution',),
                               biomass_id='BIOMASS_c')
_print_model_summary("M_musculus_T_Cell", model_mm)

# E_coli_core / E_coli — the workbooks are normalised in place by
# curate_e_coli_workbooks.py first; run that before this if either xlsx changes.
# Both spell biomass 'biomass_out', which the configs' measured_fluxes and the
# single lumped BIOMASS both refer to as 'BIOMASS'.
model_ec_core = convert_mfa_maranas('E_coli_core/example_GEM.xlsx', 'E_coli_core/example_AMM.xlsx',
                                    'E_coli_core/e_coli_core_annotated.json',
                                    biomass_id='biomass_out')
_print_model_summary("E_coli_core", model_ec_core)

model_ec = convert_mfa_maranas('E_coli/iAF1260_GEM.xlsx', 'E_coli/iAF1260_AMM.xlsx',
                               'E_coli/iAF1260_annotated.json',
                               biomass_id='biomass_out')
_print_model_summary("E_coli", model_ec)
