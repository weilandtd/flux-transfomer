import cobra
from cobra.io import save_json_model

from sigmaflux.core.utils import convert_mfa_maranas


def add_rt_overflow_routes(model) -> None:
    """Give R. toruloides the two carbon overflow routes the Maranas GEM lacks.

    As converted, the network can dispose of carbon only as CO2 or as coupled
    biomass: every polysaccharide and lipid has exactly one producer and is consumed
    only by a biomass component reaction, at a fixed stoichiometric ratio. Both of
    this yeast's characteristic overflow phenotypes are therefore inexpressible —
    their fluxes are pinned to the growth rate instead of varying independently.

    1. Secreted beta-glucan (EPS). R. toruloides CCT0783 secretes an extracellular
       biopolymer, up to 5.84 g/L under high-glucose / low-nitrogen conditions, once
       the culture acidifies to ~pH 2:

           Hamacek et al. (2025) "Acidification by nitrogen metabolism triggers
           extracellular biopolymer production in an oleaginous yeast."
           Appl Environ Microbiol 91(11). doi:10.1128/aem.00947-25

       Note the reported EPS is glucose-dominant but not characterised as beta-1,3-
       glucan specifically (88.8% glucose, 5.5% galactose, 4.8% mannose), so
       13BDglucan_c stands in as the closest species this network carries rather than
       as a validated match. Added as an export to 13BDglucan_e plus a drain, which
       mirrors how every other secreted species leaves the model.

    2. Stored triacylglycerol. Accumulation is intracellular, so tag_c gets a demand
       reaction with no transport step — nothing crosses the membrane. Named DM_ per
       COBRA convention for an intracellular drain, as against the EX_ that carries
       secreted glucan out of the extracellular pool.

    The naming difference matters to apply_media(): it closes every EX_ reaction
    absent from `sampling.media`, so EX_13BDglucan_e stays shut unless a run opens it
    deliberately. DM_tag_c is never auto-closed, but listing it in that block still
    sets its bounds, since the reopen step looks reactions up by id.

    Effect, with growth held in the sampled window [0.01, 0.5]: max TAG accumulation
    rises 0.0052 -> 0.7722 and the carbon leaving as CO2 at that optimum falls from
    72% to 38%; glucan secretion reaches 11.23, comparable to glucose uptake. Max
    growth is unchanged at 0.98969/h, so the drains add overflow capacity without
    creating a free lunch. The raw hierarchical null space goes 50 -> 52.

    These two are what the GEM is missing that this organism visibly does; they are
    not a complete reconciliation. Carotenoid biosynthesis is also absent and is NOT
    added here — ggdp_c (GGPP) does not exist in the network, so there is no entry
    point to torulene/torularhodin and adding a drain alone would not create one.

    Downstream note: adding these drains lowered the Fisher permutation null floor
    from 8.5098 to 7.4487 nats, and the tokens surviving it from 2 to 6 of 336 — a
    real improvement that nonetheless leaves the selection collapsed. The binding
    constraint is the floor itself, not the network. See models/pretrained/README.md.
    """
    compartment_e = model.metabolites.get_by_id('glc__D_e').compartment

    glucan_c = model.metabolites.get_by_id('13BDglucan_c')
    glucan_pattern = glucan_c.annotation['atom_pattern']
    glucan_e = cobra.Metabolite(id='13BDglucan_e', name=glucan_c.name,
                                compartment=compartment_e, formula=glucan_c.formula)
    glucan_e.annotation['atom_pattern'] = glucan_pattern
    model.add_metabolites([glucan_e])

    export = cobra.Reaction('13BDGLUCANt_c_e', name='(1->3)-beta-D-glucan export',
                            lower_bound=0.0, upper_bound=1000.0)
    model.add_reactions([export])
    export.add_metabolites({glucan_c: -1.0, glucan_e: 1.0})
    export.annotation['atom_mapping'] = (
        f'13BDglucan_c ({glucan_pattern}) -> 13BDglucan_e ({glucan_pattern})')

    eps_drain = cobra.Reaction('EX_13BDglucan_e', name='(1->3)-beta-D-glucan exchange',
                               lower_bound=0.0, upper_bound=1000.0)
    model.add_reactions([eps_drain])
    eps_drain.add_metabolites({glucan_e: -1.0})
    eps_drain.annotation['atom_mapping'] = f'13BDglucan_e ({glucan_pattern}) ->'

    tag_c = model.metabolites.get_by_id('tag_c')
    tag_pattern = tag_c.annotation['atom_pattern']
    tag_drain = cobra.Reaction('DM_tag_c', name='triacylglycerol storage accumulation',
                               lower_bound=0.0, upper_bound=1000.0)
    model.add_reactions([tag_drain])
    tag_drain.add_metabolites({tag_c: -1.0})
    tag_drain.annotation['atom_mapping'] = f'tag_c ({tag_pattern}) ->'


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

# R_toruloides — completeMFA workbook, so the explicit dilution sub-network is dropped.
# Saved after add_rt_overflow_routes rather than by the converter, so the EPS and TAG
# drains are part of the shipped flat model.
model_rt = convert_mfa_maranas('R_toruloides/imRt_GEM.xlsx', 'R_toruloides/imRt_AMM.xlsx',
                               drop_subsystems=('dilution',))
add_rt_overflow_routes(model_rt)
save_json_model(model_rt, 'R_toruloides/imRt_annotated.json')
_print_model_summary("R_toruloides", model_rt)

# M_musculus_T_Cell — completeMFA workbook; biomass is named BIOMASS_c here
model_mm = convert_mfa_maranas('M_musculus_T_Cell/mmTcell_GEM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_AMM_completeMFA.xlsx',
                               'M_musculus_T_Cell/mmTcell_annotated.json',
                               drop_subsystems=('dilution',),
                               biomass_id='BIOMASS_c')
_print_model_summary("M_musculus_T_Cell", model_mm)
