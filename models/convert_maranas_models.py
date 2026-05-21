from sigmaflux.core.utils import convert_mfa_maranas


def _print_model_summary(label: str, model) -> None:
    rxn_ids = {r.id for r in model.reactions}
    annotated = [r for r in model.reactions if 'atom_mapping' in r.annotation]
    print(f"\n{label}")
    print(f"  Reactions: {len(model.reactions)} total, {len(annotated)} annotated")
    print(f"  Metabolites: {len(model.metabolites)}")
    for cid in ('PROTEIN_c', 'RNA_c', 'DNA_c', 'LIPID_c', 'CARB_c', 'ATPM2_c'):
        print(f"  {cid}: {'OK' if cid in rxn_ids else 'MISSING'}")
    print(f"  BIOMASS present (expected False): {'BIOMASS' in rxn_ids}")


# I_orientalis
model_lo = convert_mfa_maranas('I_orientalis/imIsor_GEM.xlsx', 'I_orientalis/imIsor_AMM.xlsx', 'I_orientalis/imIsor_annotated.json')
_print_model_summary("I_orientalis", model_lo)

# S_cerevisiae
model_sc = convert_mfa_maranas('S_cerevisiae/scGEM.xlsx', 'S_cerevisiae/scAMM.xlsx', 'S_cerevisiae/sc_annotated.json')
_print_model_summary("S_cerevisiae", model_sc)
