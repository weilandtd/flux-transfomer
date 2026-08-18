"""

"""
import re

SYMMETRY = ['symmetric', 'Symmetric', 'Symmetry', 'symmetry', '[sym]']


def atom_pattern_regex(met_id):
    r"""Regex capturing ``met_id``'s parenthesised atom pattern in a mapping string.

    Excludes hyphens (as well as ``\w``) on the left boundary, since ``\b`` alone can
    match inside a hyphenated BiGG id like ``arbtn-fe3_e`` and mis-attribute the pattern.
    """
    return r"(?<![\w-]){}\s*\(([^)]+)\)".format(re.escape(met_id))


def can_make_met(reaction, metabolite):
    """Whether reaction can produce metabolite, forward or reverse."""
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
    """Convert an EMU's 1-based carbon positions (mid) into a binary indicator vector of length len(mapping)."""
    # Convert the mapping to binary
    bin_emu = [0 for _ in range(len(mapping))]
    # Mapping -> ABCD...
    # Bin
    for m in mid:
        bin_emu[m-1] = 1
    
    return bin_emu




def intersection(lst1, lst2):
    """Intersection between two lists, preserving lst1's order."""
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
