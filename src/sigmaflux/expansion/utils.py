"""

"""

SYMMETRY = ['symmetric', 'Symmetric', 'Symmetry', 'symmetry', '[sym]']


def can_make_met(reaction, metabolite):
    """
    Check if a reaction can produce the metabolite
    :param reaction: cobra.Reaction
    :param metabolite: cobra.Metabolite
    :return: bool
    """
    # If the forward bound is > 0 and the metabolite is in the products
    if reaction.upper_bound > 0:
        if metabolite in reaction.products:
            return True
    # If the reverse bound is < 0 and the metabolite is in the reactants
    if reaction.lower_bound < 0:
        if metabolite in reaction.reactants:
            return True
        
    # Old code based on reversibility flag
    # BUG this is only works if all fluxes are positive  
    # if metabolite in reaction.products:
    #     return True
    # elif reaction.reversibility:
    #     return True
    return False


def check_symetry_kwd(map_str, keywords=SYMMETRY):
    return any([sym in map_str for sym in keywords])

def emu_to_bin(mid, mapping):
    """
    Convert a mapping string to index hash
    :param mapping_string:
    :return:
    """
    # Convert the mapping to binary
    bin_emu = [0 for _ in range(len(mapping))]
    # Mapping -> ABCD...
    # Bin
    for m in mid:
        bin_emu[m-1] = 1
    
    return bin_emu




def intersection(lst1, lst2):
    """
    Intersection between two lists
    :param lst1: list()
    :param lst2:  list()
    :return:
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
