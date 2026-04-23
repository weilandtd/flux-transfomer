
from cobra import Reaction, Model, Metabolite

def add_dummy_reactions(model):
    # Add dummy reactions:
    cyt_atp2adp = Reaction('cyt_atp2adp')
    mit_atp2adp = Reaction('mit_atp2adp')
    cyt_nadh2nad = Reaction('cyt_nadh2nad')
    mit_nadh2nad = Reaction('mit_nadh2nad')
    cyt_nadph2nadp = Reaction('cyt_nadph2nadp')
    cyt_utp2udp = Reaction('cyt_utp2udp')
    mit_nadph2nadp = Reaction('mit_nadph2nadp')
    EX_akg_m = Reaction('EX_akg[m]')
    glycogen_sink = Reaction('EX_glycogen_gluc_eq[c]')
    cyt_udpg2glycogen = Reaction('cyt_udpg2glycogen')
    cyt_glycogen2g1p = Reaction('cyt_glycogen2g1p')
    mit_accoa2coa = Reaction('mit_accoa2coa')
  # Palmitate synthesis
    pa_triglyceride_synthesis = Reaction('triglyceride_synthesis_pa')
    pa_triglyceride_breakdown = Reaction('triglyceride_breakdown_pa')
    pa_triglyceride_exchange = Reaction('triglyceride_excange_pa', )

    # Linoleate synthesis
    li_triglyceride_synthesis = Reaction('triglyceride_synthesis_li')
    li_triglyceride_breakdown = Reaction('triglyceride_breakdown_li')
    li_triglyceride_exchange = Reaction('triglyceride_excange_li')

    # Oleate synthesis
    ol_triglyceride_synthesis = Reaction('triglyceride_synthesis_ol')
    ol_triglyceride_breakdown = Reaction('triglyceride_breakdown_ol')
    ol_triglyceride_exchange = Reaction('triglyceride_excange_ol')

    glycogen_gluc_eq_c = Metabolite(
    'glycogen_gluc_eq_c',
    formula='C6H12O6',
    name='glycogen glucose equivalent',
    compartment='c')

    triglyceride_pa = Metabolite(
    'triglyceride_pa',
    name = "triglyceride", 
    compartment = 'c'
    )

    triglyceride_li = Metabolite(
    'triglyceride_li',
    name = "triglyceride", 
    compartment = 'c'
    )

    triglyceride_ol = Metabolite(
    'triglyceride_ol',
    name = "triglyercide", 
    compartment = 'c'
    )

    fatty_acid = Metabolite(
        'fatty_acid', 
        name = "fatty acid", 
        compartment = 'c'
    )

    model.add_reactions([ pa_triglyceride_breakdown, pa_triglyceride_synthesis, pa_triglyceride_exchange,
                          li_triglyceride_breakdown, li_triglyceride_synthesis, li_triglyceride_exchange,
                          ol_triglyceride_breakdown, ol_triglyceride_synthesis, ol_triglyceride_exchange,
                          cyt_glycogen2g1p, cyt_udpg2glycogen, EX_akg_m, cyt_utp2udp, mit_nadph2nadp, 
                          cyt_atp2adp, mit_atp2adp, cyt_nadh2nad, mit_nadh2nad, cyt_nadph2nadp, 
                          glycogen_sink, mit_accoa2coa])

    cyt_atp2adp.reaction = 'atp_c + h2o_c --> adp_c + pi_c + h_c'
    cyt_utp2udp.reaction = 'utp_c --> udp_c + pi_c'
    mit_atp2adp.reaction = 'atp_m + h2o_m --> adp_m + pi_m + h_m'
    cyt_nadh2nad.reaction = 'nadh_c + h_c --> nad_c'
    cyt_nadph2nadp.reaction = 'nadph_c + h_c --> nadp_c'
    mit_nadh2nad.reaction = 'nadh_m --> nad_m + h_m '
    mit_nadph2nadp.reaction = 'nadph_m + h_c --> nadp_m'
    EX_akg_m.reaction = "akg_m --> "
    mit_accoa2coa.reaction = 'coa_m --> accoa_m'

    mit_atp2adp.bounds = [0,0]
    cyt_nadh2nad.bounds = [0,0]
    cyt_nadph2nadp.bounds = [0,0]
    mit_nadh2nad.bounds = [0,0]
    mit_nadph2nadp.bounds = [0,0]
    EX_akg_m.bounds = [0,0]
    mit_accoa2coa.bounds = [0,0]

    glycogen_sink.add_metabolites({glycogen_gluc_eq_c:-1})
    cyt_udpg2glycogen.add_metabolites({model.metabolites.get_by_id('udpg_c'):-1, glycogen_gluc_eq_c:1, model.metabolites.get_by_id('udp_c'):1 }) 
    cyt_glycogen2g1p.add_metabolites({glycogen_gluc_eq_c:-1, model.metabolites.get_by_id('g1p_c'):1, model.metabolites.get_by_id('pi_c'):-1})

    # This is a dummy reaction to represent the synthesis of triglycerides
    # This reaction should not be atom mapped!!! 
    # This assumes that hte TG pool is large an TG is only unlabled 
    # Synthesis is just a sink 
    pa_triglyceride_synthesis.add_metabolites({model.metabolites.get_by_id('hdca_c'):-3,model.metabolites.get_by_id('glyc_c'):-1, })
    li_triglyceride_synthesis.add_metabolites({model.metabolites.get_by_id('lnlc_c'):-3,model.metabolites.get_by_id('glyc_c'):-1, })
    ol_triglyceride_synthesis.add_metabolites({model.metabolites.get_by_id('ocdcea_c'):-3,model.metabolites.get_by_id('glyc_c'):-1,})

    # Break down is atommpaed
    pa_triglyceride_breakdown.add_metabolites({triglyceride_pa:-1, model.metabolites.get_by_id('hdca_c'):3, model.metabolites.get_by_id('glyc_c'):1})   
    li_triglyceride_breakdown.add_metabolites({triglyceride_li:-1, model.metabolites.get_by_id('lnlc_c'):3, model.metabolites.get_by_id('glyc_c'):1})   
    ol_triglyceride_breakdown.add_metabolites({triglyceride_ol:-1, model.metabolites.get_by_id('ocdcea_c'):3, model.metabolites.get_by_id('glyc_c'):1})

    # Exchange reaction are break down from local storage
    ol_triglyceride_exchange.add_metabolites({triglyceride_ol:1,})
    li_triglyceride_exchange.add_metabolites({triglyceride_li:1,})
    pa_triglyceride_exchange.add_metabolites({triglyceride_pa:1,})

    
    # Generate aromatic atom mapping for triglyceride dymmy reactions
    # Palmitate attom mapping
    TG = ''.join([chr(97+i) for i in range(0,3*16+3)])
    PA1 = ''.join([chr(97+i) for i in range(0,16)])
    PA2 = ''.join([chr(97+i) for i in range(16,32)])
    PA3 = ''.join([chr(97+i) for i in range(32,48)])
    GY = ''.join([chr(97+i) for i in range(48,51)])
    pa_triglyceride_breakdown.annotation["atom_mapping"] = f"triglyceride_pa_c ({TG}) --> hdca_c ({PA1}) + hdca_c ({PA2}) + hdca_c ({PA3}) + glyc_c ({GY})" 
    pa_triglyceride_synthesis.annotation["atom mapping"] = f"hdca_c ({PA1}) + hdca_c ({PA2}) + hdca_c ({PA3}) + glyc_c ({GY}) --> "

    # Linoleate attom mapping
    TG = ''.join([chr(97+i) for i in range(0,3*18+3)])
    LI1 = ''.join([chr(97+i) for i in range(0,18)])
    LI2 = ''.join([chr(97+i) for i in range(18,36)])
    LI3 = ''.join([chr(97+i) for i in range(36,54)])
    GY = ''.join([chr(97+i) for i in range(54,57)])
    li_triglyceride_breakdown.annotation["atom_mapping"] = f"triglyceride_li_c ({TG}) --> lnlc_c ({LI1}) + lnlc_c ({LI2}) + lnlc_c ({LI3}) + glyc_c ({GY})"
    li_triglyceride_synthesis.annotation["atom mapping"] = f"lnlc_c ({LI1}) + lnlc_c ({LI2}) + lnlc_c ({LI3}) + glyc_c ({GY}) --> "

    # Oleate attom mapping
    TG = ''.join([chr(97+i) for i in range(0,3*18+3)])
    OL1 = ''.join([chr(97+i) for i in range(0,18)])
    OL2 = ''.join([chr(97+i) for i in range(18,36)])
    OL3 = ''.join([chr(97+i) for i in range(36,54)])
    GY = ''.join([chr(97+i) for i in range(54,57)])
    ol_triglyceride_breakdown.annotation["atom_mapping"] = f"triglyceride_ol_c ({TG}) --> ocdcea_c ({OL1}) + ocdcea_c ({OL2}) + ocdcea_c ({OL3}) + glyc_c ({GY})"
    ol_triglyceride_synthesis.annotation["atom mapping"] = f"ocdcea_c ({OL1}) + ocdcea_c ({OL2}) + ocdcea_c ({OL3}) + glyc_c ({GY}) --> "

    return model 
