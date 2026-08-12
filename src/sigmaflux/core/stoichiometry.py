import warnings
from collections import Counter, namedtuple
from networkx import incidence_matrix
import torch
from sigmaflux.expansion.atom_mapping import AtomMapping, MissingAtomMappingError
from sigmaflux.expansion.emu import get_metabolite

AnnotatedStoichiometry = namedtuple('AnnotatedStoichiometry', ['matrix', 'reactions', 'directions',
                                                               'emus', 'boundary', 'core', 'index_hash'])


def _mapped_copies(reaction, cache):
    """``({product met -> count}, {educt met -> count})`` for *reaction*'s atom map.

    Counts *distinct* mapped copies, matching the edges
    :meth:`~sigmaflux.expansion.atom_mapping.AtomMapping.get_origin_emus` actually
    builds: a product named twice with different atoms is two ways to make the EMU,
    while a duplicated atom-map row is one.
    """
    if reaction.id not in cache:
        try:
            mapping = AtomMapping(reaction)
        except MissingAtomMappingError:      # unreachable: unmapped reactions make no edges
            cache[reaction.id] = ({}, {})
        else:
            cache[reaction.id] = tuple(
                Counter(met_id for met_id, _ in dict.fromkeys(side))
                for side in (mapping.product_mappings, mapping.educt_mappings))
    return cache[reaction.id]


def warn_if_underspecified(reaction, direction, target_id, cache):
    """Warn when the atom map names fewer copies of the target than it moves.

    Multiplicity belongs in the *atom map*, as one named copy per molecule with its
    own pattern -- ``atp_c (P1) + amp_c (P2) -> adp_c (P1) + adp_c (P2)``. Each copy
    then becomes its own EMU edge and the edge weight stays 1. Inferring the missing
    copies from the coefficient instead would be guessing which atoms they carry,
    and guessing wrong is invisible: ``CBPS2_m`` moves two ATP and two ADP while
    naming one of each, and weighting its edge by 2 rather than warning shifted the
    whole adenine-nucleotide pool by 0.38 MID units against x3cflux.

    So this reports the map as incomplete and leaves the weight alone. The fix is in
    the atom-mapping spreadsheet, not here.
    """
    product_copies, educt_copies = _mapped_copies(reaction, cache)
    # 'rvs' runs the reaction backwards, so the target is one of its educts.
    copies = (product_copies if direction == 'fwd' else educt_copies).get(target_id, 0)
    coefficient = abs(reaction.get_coefficient(target_id))
    if copies and abs(coefficient - copies) > 1e-9:
        warnings.warn(
            f"{reaction.id}: atom map names {target_id} {copies} time(s) but the "
            f"stoichiometry moves {coefficient:g}. Only the mapped copies carry "
            f"label; spell out each copy with its own pattern to fix this.",
            RuntimeWarning, stacklevel=3)


def get_subnets(model, subnets, device = None, dtype = torch.float64):
    subnet_stoichs = dict()
    subnet_stoichs_plus = dict() 
    emu_hashes = dict() 
    boundary_indices = dict() 
    core_indices = dict()
    reaction_hashes = dict()

    # For each subnet expansion get subnet stoich 
    for size, subnet_graph in subnets.items():
        
        # maps node i index to emu object 
        emu_hash_index_emu = {i: emu for i, emu in enumerate(subnet_graph.nodes())}
        # maps emu object to node index i  
        emu_hash_emu_index = {emu: i for i, emu in enumerate(subnet_graph.nodes())}

        # Initilize en empt stoich matrix
        Sg = torch.zeros([len(subnet_graph.nodes()), len(subnet_graph.edges())], dtype = dtype, device = device) 
        
        # Get boundary nodes 
        boundary_indices[size] = [emu_hash_emu_index[u] for u in subnet_graph.nodes() if len(list(subnet_graph.predecessors(u))) == 0]
        # Get core nodes
        # Note indices need to be hard int64 for indexing tensors
        core_indices[size] = torch.tensor(list(set(emu_hash_index_emu.keys()) - set(boundary_indices[size])), device=device, dtype=torch.int64) 
        boundary_indices[size] = torch.tensor(boundary_indices[size], device=device, dtype=torch.int64)
        
        # Initialize a list to store the reactions / edge mapping
        reaction_indices = torch.zeros(len(subnet_graph.edges()), device = device, dtype = torch.int64,)

        mapped_copies_cache = {}

        for ix, edge in enumerate(subnet_graph.edges):
            data = subnet_graph.get_edge_data(*edge)
            reaction = data['reaction']
            direction = data['direction']
            symmetry_weight = data['symmetry_weight']
            if direction == 'fwd':
                reaction_indices[ix] = model._flux_hash[reaction.id]
                direction_factor = 1
            elif  direction == 'rvs':
                reaction_indices[ix] = model._flux_hash[reaction.id] + model.num_reactions
                direction_factor = -1
            else:
                raise RuntimeError("Expansion contains woring direction annotation!")

            origin_id, target_id = get_metabolite(edge[0])[0], get_metabolite(edge[1])[0]
            warn_if_underspecified(reaction, direction, target_id, mapped_copies_cache)
            weight = symmetry_weight * direction_factor

            # Multiplicity is carried by the edges -- one per mapped copy of the
            # target -- so the weight stays +-1 and only the sign differs between
            # the consumed origin and the produced target.
            for node, met_id in ((edge[0], origin_id), (edge[1], target_id)):
                coefficient = reaction.get_coefficient(met_id)
                Sg[emu_hash_emu_index[node], ix] = weight * coefficient / abs(coefficient)


        # Set up the mapping dicts used later in calculation
        emu_hashes[size] = emu_hash_index_emu
        subnet_stoichs[size] = Sg.cpu().to_sparse_coo().to(device) #COO format for now 
        subnet_stoichs_plus[size] = ((Sg > 0).type(dtype).T).cpu().to_sparse_coo().to(device)
        reaction_hashes[size]  = reaction_indices  

    return subnet_stoichs,subnet_stoichs_plus,emu_hashes, boundary_indices, core_indices,reaction_hashes

def get_subnet_stoichiometries(subnets, device=None, dtype=torch.float32):
    stoichiometries = dict()
    for size, the_subnet in subnets.items():
        matrix = torch.tensor(incidence_matrix(the_subnet, oriented=True).todense(),
                              dtype=dtype, device=device)
        emus = the_subnet.nodes
        reactions = [e[2]['reaction'] for e in the_subnet.edges.data()]
        directions = [e[2]['direction'] for e in the_subnet.edges.data()]
        # Input nodes
        index_hash = {u: i for i, u in enumerate(the_subnet.nodes())}
        boundary = [u for u in the_subnet.nodes() if len(list(the_subnet.predecessors(u))) == 0]
        core = list(set(emus) - set(boundary))

        stoichiometries[size] = AnnotatedStoichiometry(matrix, reactions, directions, emus,
                                                       boundary, core, index_hash)

    return stoichiometries


def get_boundaries(stoichiometries):
    boundaries = list()
    for _, stoich in stoichiometries.items():
        for bd in stoich.boundary:
            boundaries.append(bd)
    return boundaries
