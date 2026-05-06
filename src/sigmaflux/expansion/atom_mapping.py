import re
from sigmaflux.expansion.utils import emu_to_bin
from sigmaflux.expansion.emu import EMU, OriginSet

from collections import OrderedDict


class MissingAtomMappingError(Exception):
    pass


class AtomMapping(object):
    """
    Class to represent atom mapping
    """

    def __init__(self, rxn):
        """

        :param rxn: cobra.Reaction() with atom_mapping annotation
        """

        if 'atom_mapping' not in rxn.annotation:
            # Todo check if this should be an error or Warning
            raise MissingAtomMappingError('Reaction {} is not annotated with atom_mapping')

        educts_atom_mappings, products_atom_mappings = parse_mapping_string(rxn.annotation['atom_mapping'], rxn)

        # TODO Consistency check
        # for met_id, atom_ids in atom_mappings.items():
        #     stoich = rxn.metabolites.get_by_id(met_id)
        self.product_mappings = OrderedDict(products_atom_mappings)
        self.educt_mappings = OrderedDict(educts_atom_mappings)

        self.rxn = rxn


    @property
    def forward(self) -> bool:
        return self.rxn.upper_bound > 0
    
    @property
    def reverse(self) -> bool:
        return self.rxn.lower_bound < 0

    @property
    def max_emu(self) -> float:
        return max([len(m) for _, m in self.mappings])

    @property
    def mappings(self) -> list:
        return self.product_mappings + self.educt_mappings

    def get_origin_emus(self, target_emu) -> list:
        """
        Here the magic happens
        :param target_emu: mfa.expansion.EMU
        :return: list of mfa.expasion.EMU
        """

        origin_emus = set()
        origin_emus_rev = set()
   
        # Find the origin EMUS that can make the target EMU if the reaction can go forward
        if self.forward &  (target_emu.id in self.product_mappings):

            # Compute the origin set for the target emu
            the_origin_set = compute_origin_emu_set(target_emu, self.product_mappings, self.educt_mappings, self.rxn)
            # We add the origin set to the set of origin sets
            origin_emus.add(OriginSet(the_origin_set))

            # Check for symetries in the target emu
            if len(target_emu.symmetries) > 0:
                # For each symetry we compute the origin emu set
                for sym in range(len(target_emu.symmetries)):
                    the_origin_set_sym = compute_origin_emu_set(target_emu, self.product_mappings, self.educt_mappings, self.rxn, sym=sym)
                    # We add the origin set to the set of origin sets
                    origin_emus.add(OriginSet(the_origin_set_sym))


        # Find the origin EMUS that can make the target EMU if the reaction can go reverse
        if self.reverse & (target_emu.id in self.educt_mappings):
            the_origin_set_rev = compute_origin_emu_set(target_emu, self.educt_mappings, self.product_mappings, self.rxn)
            # We add the origin set to the set of origin sets
            origin_emus_rev.add(OriginSet(the_origin_set_rev))

            # Check for symetries in the target emu
            if len(target_emu.symmetries) > 0:
                # For each symetry we compute the origin emu set
                for sym in range(len(target_emu.symmetries)):
                    the_origin_set_rev_sym = compute_origin_emu_set(target_emu, self.educt_mappings, self.product_mappings, self.rxn, sym=sym)
                    # We add the origin set to the set of origin sets
                    origin_emus_rev.add(OriginSet(the_origin_set_rev_sym))


        # if self.rxn.id == "CSm":
        #     1/0
           
        # Return an origin set
        return origin_emus, origin_emus_rev
    



def parse_mapping_string(map_str, rxn):
    """
    Parse reaction string based on cobra reaction

    :param map_str: " met_a_c (abc) -> met_b_c (ab) + met_c_c (c)"
    :param rxn: cobra.Reaction
    :return: list()
    """

    # TODO make this more robust, needs space before reactant to work 
    if not map_str.startswith(' '):
        map_str = ' ' + map_str
        
    educts_atom_mappings = list()
    products_atom_mappings = list()

    # Reactants
    for met in rxn.reactants:
        # Optimized regex pattern with word boundaries and flexible spacing
        pattern = r"\b{}\s*\(([^)]+)\)".format(re.escape(met.id))
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            educts_atom_mappings.append((met.id, the_mapping))

    # Products
    for met in rxn.products:
        # Optimized regex pattern with word boundaries and flexible spacing
        pattern = r"\b{}\s*\(([^)]+)\)".format(re.escape(met.id))
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            products_atom_mappings.append((met.id, the_mapping))

    return educts_atom_mappings, products_atom_mappings


def compute_origin_emu_set(target_emu, product_mappings, educt_mappings, reaction, sym=-1):
    """AI is creating summary for compute_origin_emu_set

    Args:
        target_emu (EMU): Target EMU to find origins for
        product_mappings (dict): Dictionary for the atom mappings of the products
        educt_mappings (dict): Dictionary for the atom mappings of the educts
        reaction (cobra.Reaction): Reaction object to get metabolite objects
        sym (int, optional): [description]. Defaults to -1.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # Define the target emu
    target_mapping = product_mappings[target_emu.id]
    if sym >= 0:
        # Rearrange the target_mapping according to the symetry
        # e.g. symetry[1] = ('ABC', 'ACB') means ACB is the same as ABC
        # target_mapping = GHJ  then becomes GJH for sym = 1
        # Find mapping of the indices from the symetry tuple (0,1,2) to (0,2,1)
        symetry = target_emu.symmetries[sym]
        # Build index_map by matching letters in the original mapping to the symmetrized mapping
        # symetry is a tuple like ('ABC', 'ACB'), so use the symmetrized string
        sym_mapping = symetry[1]
        orig_mapping = symetry[0]

        # If length of symetry does not match length of target mapping raise error
        if len(sym_mapping) != len(target_mapping):
            raise ValueError(f"Error in atom mapping for reaction {reaction.id} {target_emu}\
                              with target {target_mapping} and symetry {symetry}")
        if sorted(sym_mapping) != sorted(orig_mapping):
            raise ValueError(f"Error in atom mapping for reaction {reaction.id} {target_emu}\
                              with target {target_mapping} and symetry {symetry}")

        # Map each letter in orig_mapping to its index in sym_mapping
        index_map = [sym_mapping.index(c) for c in orig_mapping]
        target_mapping = ''.join([target_mapping[i] for i in index_map])
        
    try:
        bin_target_emu = emu_to_bin(target_emu.mid, target_mapping)
    except IndexError:
        print(f"Error in atom mapping for {target_emu.metabolite} | Check if mapped inconsitently")
        return OriginSet()

    the_orign_emu = set()

    # For each educt find the corresponding emu
    for educt, educt_mapping in educt_mappings.items():
        # Apply find product -> educt mapping
        # p1 (abc) <- e1 (ab) + e2 (c) 
        # product emu 23 -> 011 -> e1 01 + e2 1
        # Hence educt emu e1_2 and e2_1

        # Compute transitions on educts from binary mapping
        bin_educt_emu = [0 for _ in range(len(educt_mapping))]

        # Set the educt mapping to 1 if this atom is 1 in the target emu
        # Mapp by finding the correspoding letter in the target mapping
        for i, letter in enumerate(educt_mapping):
            if letter in target_mapping:
                target_ix = target_mapping.index(letter)
                bin_educt_emu[i] = bin_target_emu[target_ix]

        # Convert binary mapping back to mid 
        mids = [i + 1 for i, v in enumerate(bin_educt_emu) if v == 1]

        if mids:    
            # Check if max mids > len(mapping)
            if max(mids)-1  > len(educt_mapping):
                raise ValueError(f"Error in atom mapping for reaction {educt} {educt_mapping} with target {target_emu} {target_mapping}")

            # Find the metabolite object%debug
            educt_metabolite = next(m for m in reaction.metabolites if m.id == educt)
            new = EMU(educt_metabolite, mids)
            the_orign_emu.add(new)

    # Convert to origin set type 
    return OriginSet(the_orign_emu)

    