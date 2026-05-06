import cobra
import networkx as nx
from sigmaflux.expansion.atom_mapping import AtomMapping, MissingAtomMappingError
from sigmaflux.expansion.emu import OriginSet
from sigmaflux.expansion.utils import can_make_met



class EMUNetwork(object):
    def __init__(self, max_size):
        """
        Class for that describes the collective emu expansion
        :param max_size:
        """

        self.max_size = max_size
        self.subnetworks = dict()
        for i in range(1, max_size + 1):
            self.subnetworks[i] = nx.MultiDiGraph()

    def expand(self, max_size) -> int:

        for i in range(self.max_size + 1, max_size + 1):
            self.subnetworks[i] = nx.MultiDiGraph()

        self.max_size = max_size

    def has_emu(self, item) -> bool:
        if item.size not in self.subnetworks:
            return False
        return self.subnetworks[item.size].has_node(item)


class EMUExpansion(object):
    def __init__(self, model, target_emus):
        """

        :param model: Annotated cobra model
        :param target_emus: set MID [MID, ... ]
        """

        self.model = model

        self.mappings = dict()

        for rxn in model.reactions:
            try:
                mapping = AtomMapping(rxn)
                self.mappings[rxn.id] = mapping

            except MissingAtomMappingError:
                print("missing atommaping", rxn.id)
                pass

        self.target_emus = target_emus

        max_size = max([len(f.mid) for f in self.target_emus])
        self.emu_network = EMUNetwork(max_size)

    def expand_network(self):
        """

        :return: EMU reaction network object
        """
        emus_to_expand = list(self.target_emus)
        # Sort the emus to expand start with largest
        emus_to_expand = sorted(emus_to_expand, key=lambda obj: obj.size, reverse=True)

        counter = 0
        # Make an EMU expansion for each target fragment
        while len(emus_to_expand) > 0:
            # Get the last entry
            the_emu = emus_to_expand.pop()

            # Find the reactions that make the metabolite
            if isinstance(the_emu._metabolite, cobra.Metabolite):
                the_met = self.model.metabolites.get_by_id(the_emu.id)
            else:
                raise TypeError(f'The emu.metabolite is not a cobra.Metabolite but {type(the_emu.metabolite)}')

            the_reactions = [the_rxn for the_rxn in the_met.reactions if can_make_met(the_rxn, the_met)]# key=lambda obj: obj.id)

            # Find all origin EMUs that can result in this specific EMU
            for the_rxn in the_reactions:
                if the_rxn.id not in self.mappings:
                    continue
                the_atom_mapping = self.mappings[the_rxn.id]

                #DWR: one atom maping can have multiple origin sets
                origin_sets, origin_sets_rev = the_atom_mapping.get_origin_emus(the_emu)

                # Symmetries of the target emu
                symmetry_weight = 1/(1+len(the_emu.symmetries))

                # Print some debug info
                # print(f"Origin sets for {the_emu}: {origin_sets}, {origin_sets_rev}")

                # New emus that are not yet in the network
                all_origin_sets = origin_sets.union(origin_sets_rev)
                sum_all_origin_sets = sum(all_origin_sets, OriginSet({}))

                new_emus = [o_emu for o_emu in sum_all_origin_sets
                            if not self.emu_network.has_emu(o_emu)]
                
                for origin_set in origin_sets:
                    # Add the edges to the respective emu network
                    if origin_set.size == the_emu.size:
                        data = self.emu_network.subnetworks[the_emu.size].get_edge_data(origin_set, the_emu)
                        if data is None: 
                            has_edge = False 
                        else: 
                            this_edge_data = {'reaction': the_rxn, 'direction': 'fwd'} 
                            compare = [d == this_edge_data for d in data.values() ]
                            has_edge = any(compare) 

                        if not has_edge: 
                            self.emu_network.subnetworks[the_emu.size].add_edge(origin_set, the_emu,
                                                                                reaction=the_rxn, 
                                                                                direction='fwd',
                                                                                symmetry_weight=symmetry_weight,
                                                                                )

                for origin_set_rev in origin_sets_rev:
                    if origin_set_rev.size == the_emu.size:
                        data = self.emu_network.subnetworks[the_emu.size].get_edge_data(origin_set_rev, the_emu)
                        if data is None: 
                            has_edge = False 
                        else: 
                            this_edge_data = {'reaction': the_rxn, 'direction': 'rvs'} 
                            compare = [d == this_edge_data for d in data.values() ]
                            has_edge = any(compare) 

                        if not has_edge: 
                            self.emu_network.subnetworks[the_emu.size].add_edge(origin_set_rev, the_emu,
                                                                            reaction=the_rxn, 
                                                                            direction='rvs',
                                                                            symmetry_weight=symmetry_weight,
                                                                            )

                #Add a new subnetwork of size new_emu if necessary
                for the_new_emu in new_emus:
                    if the_new_emu.size not in self.emu_network.subnetworks:
                        self.emu_network.expand(the_new_emu.size)

                #print(the_emu,the_atom_mapping.product_mappings , the_atom_mapping.educt_mappings, origin_set)

                # Extract all new emus
                emus_to_expand.extend(new_emus)
                # Sort the new emus to expand
                emus_to_expand = sorted(emus_to_expand, key=lambda obj: obj.size)

            # Count generation
            counter += 1

            if counter % 1000 == 0:
                print(f"EMU expansion step {counter}, {len(emus_to_expand)} emus to expand")
