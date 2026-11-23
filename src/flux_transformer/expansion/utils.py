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
    """Whether reaction can produce metabolite, forward or reverse.

    Args:
        reaction: reaction to check.
        metabolite: metabolite to check for production of.
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
    """Whether map_str contains any of the symmetry keywords.

    Args:
        map_str: atom-mapping string to search.
        keywords: symmetry keyword variants to look for.
    """
    return any([sym in map_str for sym in keywords])

def emu_to_bin(mid, mapping):
    """Convert an EMU's 1-based carbon positions (mid) into a binary indicator vector of length len(mapping).

    Args:
        mid: 1-based carbon positions tracked by the EMU.
        mapping: full atom-mapping string, used only for its length.
    """
    # Convert the mapping to binary
    bin_emu = [0 for _ in range(len(mapping))]
    # Mapping -> ABCD...
    # Bin
    for m in mid:
        bin_emu[m-1] = 1
    
    return bin_emu




def intersection(lst1, lst2):
    """Intersection between two lists, preserving lst1's order.

    Args:
        lst1: list whose order is preserved in the result.
        lst2: list to intersect against.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

#: Reaction annotation listing educts the reaction draws label from at coefficient 0.
ZERO_STOICHIOMETRY = 'zero_stoichiometry'


def zero_stoichiometry_educts(rxn):
    """Ids of educts *rxn* draws label from at a stoichiometric coefficient of zero.

    COBRA cannot store a zero coefficient -- ``add_metabolites`` drops the entry
    outright -- so a measurement pool's educt is carried in the annotation instead of the
    stoichiometry. Its absence from S is the point: the pool observes the compartment it
    measures without depleting it.
    """
    return tuple(rxn.annotation.get(ZERO_STOICHIOMETRY, ()) or ())


def labeling_metabolite(rxn, met_id):
    """The ``cobra.Metabolite`` named *met_id*, whether or not *rxn*'s stoichiometry moves it."""
    for met in rxn.metabolites:
        if met.id == met_id:
            return met
    return rxn.model.metabolites.get_by_id(met_id)


def labeling_educts(rxn):
    """Metabolites on *rxn*'s educt side for labeling purposes.

    The stoichiometric reactants plus any zero-stoichiometry educt. Every walk of a
    reaction's participants that exists to follow *carbon* rather than mass must go
    through this, or a measurement pool's mixing leg reads as having no educt at all --
    which builds no EMU edge and silently pins the pool to M+0.
    """
    return list(rxn.reactants) + [labeling_metabolite(rxn, met_id)
                                  for met_id in zero_stoichiometry_educts(rxn)]


def labeling_products(rxn):
    """Metabolites on *rxn*'s product side for labeling purposes."""
    return list(rxn.products)


def labeling_multiplicity(rxn, met_id):
    """Copies of *met_id* the reaction moves, for the atom map's multiplicity check.

    A zero-stoichiometry educt moves no mass but exactly one labeling copy.
    """
    if met_id in zero_stoichiometry_educts(rxn):
        return 1.0
    return abs(rxn.get_coefficient(met_id))
