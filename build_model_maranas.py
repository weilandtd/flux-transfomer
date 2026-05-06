from sigmaflux.core.utils import convert_mfa_maranas


# I_orientalis  
model_lo = convert_mfa_maranas('models/I_orientalis/imIsor_GEM.xlsx', 'models/I_orientalis/imIsor_AMM.xlsx', 'models/I_orientalis/imIsor_annotated.json')

annotated = [r for r in model_lo.reactions if 'atom_mapping' in r.annotation]
print(f"Reactions: {len(model_lo.reactions)} total, {len(annotated)} annotated with atom_mapping")
print(f"Metabolites: {len(model_lo.metabolites)}")


# S_cerevisiae

model_sc = convert_mfa_maranas('models/S_cerevisiae/scGEM.xlsx', 'models/S_cerevisiae/scAMM.xlsx', 'models/S_cerevisiae/sc_annotated.json')
annotated_sc = [r for r in model_sc.reactions if 'atom_mapping' in r.annotation]
print(f"Reactions: {len(model_sc.reactions)} total, {len(annotated_sc)} annotated with atom_mapping")
print(f"Metabolites: {len(model_sc.metabolites)}")
