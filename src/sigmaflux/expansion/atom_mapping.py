import re
from collections import Counter
from sigmaflux.expansion.utils import atom_pattern_regex, emu_to_bin
from sigmaflux.expansion.emu import EMU, OriginSet


class MissingAtomMappingError(Exception):
    pass


class InconsistentAtomMappingError(ValueError):
    """An atom map names a metabolite a different number of times than it moves."""


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
        self.product_mappings = products_atom_mappings  # list of (met_id, mapping_str)
        self.educt_mappings = educts_atom_mappings       # list of (met_id, mapping_str)

        self.rxn = rxn
        self._check_multiplicity()

    def _check_multiplicity(self):
        """Every molecule the reaction moves must be named in the map.

        Multiplicity is carried by the map, one named copy per molecule with its
        own pattern -- ``atp (P1) + amp (P2) -> adp (P1) + adp (P2)`` -- because
        each copy becomes its own EMU edge. A map that names fewer copies than the
        stoichiometry moves leaves the rest carrying no label, and there is no
        sound way to infer their atoms: guessing them from the coefficient shifted
        the T-cell adenine-nucleotide pool by 0.38 MID units.

        This raises at model load rather than warning, because the consequence is a
        silently wrong forward model and the repair belongs in the atom-mapping
        spreadsheet. Duplicated identical rows are not a violation -- they are
        de-duplicated first, since they describe one molecule written twice.
        """
        for side in (self.product_mappings, self.educt_mappings):
            counts = Counter(met_id for met_id, _ in dict.fromkeys(side))
            for met_id, mapped in counts.items():
                moved = abs(self.rxn.get_coefficient(met_id))
                if abs(moved - mapped) > 1e-9:
                    raise InconsistentAtomMappingError(
                        f"{self.rxn.id}: atom map names {met_id} {mapped} time(s) but the "
                        f"stoichiometry moves {moved:g}. Spell out each molecule with "
                        f"its own pattern, e.g. 'atp (P1) + atp (P2) -> adp (P1) + "
                        f"adp (P2)'.")


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

        def collect(target_side, origin_side, into):
            """One origin set per distinct mapped copy of the target, times symmetries.

            A reaction may name the same product more than once, each copy drawing
            from different educt atoms -- GLUSx_c makes two glutamate, one from
            2-oxoglutarate and one from glutamine. Every copy is a separate way to
            make the EMU and needs its own edge; taking only the first silently
            dropped the others. Copies are deduplicated by mapping string first, so a
            duplicated atom-map row stays one edge rather than becoming two.
            """
            for target_mapping in dict.fromkeys(
                    v for k, v in target_side if k == target_emu.id):
                # sym = -1 is the unpermuted orientation; a symmetric pool is the
                # mean over all of them (weighted in emu_expansion).
                for sym in range(-1, len(target_emu.symmetries)):
                    into.add(OriginSet(compute_origin_emu_set(
                        target_emu, target_side, origin_side, self.rxn, sym=sym,
                        target_mapping=target_mapping)))

        # Find the origin EMUS that can make the target EMU if the reaction can go forward
        if self.forward and any(k == target_emu.id for k, _ in self.product_mappings):
            collect(self.product_mappings, self.educt_mappings, origin_emus)

        # Find the origin EMUS that can make the target EMU if the reaction can go reverse
        if self.reverse and any(k == target_emu.id for k, _ in self.educt_mappings):
            collect(self.educt_mappings, self.product_mappings, origin_emus_rev)

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
        pattern = atom_pattern_regex(met.id)
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            educts_atom_mappings.append((met.id, the_mapping))

    # Products
    for met in rxn.products:
        pattern = atom_pattern_regex(met.id)
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            products_atom_mappings.append((met.id, the_mapping))

    return educts_atom_mappings, products_atom_mappings


def compute_origin_emu_set(target_emu, product_mappings, educt_mappings, reaction, sym=-1,
                           target_mapping=None):
    """AI is creating summary for compute_origin_emu_set

    Args:
        target_emu (EMU): Target EMU to find origins for
        product_mappings (dict): Dictionary for the atom mappings of the products
        educt_mappings (dict): Dictionary for the atom mappings of the educts
        reaction (cobra.Reaction): Reaction object to get metabolite objects
        sym (int, optional): [description]. Defaults to -1.
        target_mapping (str, optional): which mapped copy of the target to use, for
            products named more than once. Defaults to the first.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # Define the target emu
    if target_mapping is None:
        target_mapping = next(v for k, v in product_mappings if k == target_emu.id)
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

    the_orign_emu = []

    # For each educt find the corresponding emu
    for educt, educt_mapping in educt_mappings:
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
            the_orign_emu.append(new)

    # Convert to origin set type 
    return OriginSet(the_orign_emu)

    