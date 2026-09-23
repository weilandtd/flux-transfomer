import re
from collections import Counter
from sigmaflux.expansion.utils import (atom_pattern_regex, emu_to_bin,
                                       labeling_educts, labeling_metabolite,
                                       labeling_multiplicity, labeling_products)
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
        """Parse and validate the atom_mapping annotation of a cobra.Reaction."""

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
        """Every molecule the reaction moves must be named in the map, once per copy (each copy becomes its own EMU edge). Raises at load rather than warning, since an under-named map is a silently wrong forward model."""
        for side in (self.product_mappings, self.educt_mappings):
            counts = Counter(met_id for met_id, _ in dict.fromkeys(side))
            for met_id, mapped in counts.items():
                moved = labeling_multiplicity(self.rxn, met_id)
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

    def get_origin_emus(self, target_emu) -> tuple:
        """Weighted origin sets for the target EMU, forward and reverse, each a ``{OriginSet: weight}`` mapping.

        A weight is a count, not a fraction: one reaction's weights sum to the
        stoichiometric coefficient of the target it moves.
        """

        origin_emus = {}
        origin_emus_rev = {}

        def collect(target_side, origin_side, into):
            """One weighted origin set per distinct mapped copy, times symmetries.

            A reaction may name the same product more than once, each copy drawing
            from different educt atoms; every copy is its own weight and weights
            *accumulate* (not overwrite) since two copies can resolve to the same
            origin EMU. Symmetry orientations, in contrast, are alternative readings
            of one molecule and should merge, each carrying ``1/n`` of a copy.

            Args:
                target_side: (met_id, mapping_str) pairs the target EMU is drawn from.
                origin_side: (met_id, mapping_str) pairs the origin EMUs are drawn from.
                into: dict of {OriginSet: weight} accumulated into, in place.
            """
            n_orientations = 1 + len(target_emu.symmetries)
            for target_mapping in dict.fromkeys(
                    v for k, v in target_side if k == target_emu.id):
                # sym = -1 is the unpermuted orientation; a symmetric pool is the
                # mean over all of them.
                for sym in range(-1, len(target_emu.symmetries)):
                    origin = OriginSet(compute_origin_emu_set(
                        target_emu, target_side, origin_side, self.rxn, sym=sym,
                        target_mapping=target_mapping))
                    into[origin] = into.get(origin, 0.0) + 1.0 / n_orientations

        # Find the origin EMUS that can make the target EMU if the reaction can go forward
        if self.forward and any(k == target_emu.id for k, _ in self.product_mappings):
            collect(self.product_mappings, self.educt_mappings, origin_emus)

        # Find the origin EMUS that can make the target EMU if the reaction can go reverse
        if self.reverse and any(k == target_emu.id for k, _ in self.educt_mappings):
            collect(self.educt_mappings, self.product_mappings, origin_emus_rev)

        return origin_emus, origin_emus_rev
    



def parse_mapping_string(map_str, rxn):
    """Parse an atom_mapping string like ``" met_a_c (abc) -> met_b_c (ab) + met_c_c (c)"`` into (educt_mappings, product_mappings) lists of (met_id, mapping_str).

    Args:
        map_str: atom_mapping annotation string to parse.
        rxn: cobra.Reaction the string belongs to, for its reactant/product ids.
    """

    # TODO make this more robust, needs space before reactant to work 
    if not map_str.startswith(' '):
        map_str = ' ' + map_str
        
    educts_atom_mappings = list()
    products_atom_mappings = list()

    # Reactants -- labeling_educts, not rxn.reactants: a measurement pool's mixing leg
    # moves no mass of the compartment it observes, so the compartment is absent from the
    # stoichiometry. Missing it here yields an empty origin set, which the expansion
    # legitimately skips, leaving the pool with no incoming edge and pinned to M+0.
    for met in labeling_educts(rxn):
        pattern = atom_pattern_regex(met.id)
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            educts_atom_mappings.append((met.id, the_mapping))

    # Products
    for met in labeling_products(rxn):
        pattern = atom_pattern_regex(met.id)
        mappings = re.findall(pattern, map_str)
        for the_mapping in mappings:
            products_atom_mappings.append((met.id, the_mapping))

    return educts_atom_mappings, products_atom_mappings


def compute_origin_emu_set(target_emu, product_mappings, educt_mappings, reaction, sym=-1,
                           target_mapping=None):
    """Trace the target EMU's labeled atoms back through the mapping to find the OriginSet of educt EMUs that produce it.

    Args:
        target_emu: EMU whose origin is being traced.
        product_mappings: (met_id, mapping_str) pairs on the side the target is drawn from.
        educt_mappings: (met_id, mapping_str) pairs on the side the origin EMUs are drawn from.
        reaction: cobra.Reaction the mappings belong to, for error messages and metabolite lookup.
        sym: symmetry orientation to apply to the target mapping before tracing (-1 = unpermuted).
        target_mapping: mapped copy of the target to use, for products named more than once; defaults to the first.
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

            new = EMU(labeling_metabolite(reaction, educt), mids)
            the_orign_emu.append(new)

    # Convert to origin set type 
    return OriginSet(the_orign_emu)

    