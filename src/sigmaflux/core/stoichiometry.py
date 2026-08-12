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


def edge_multiplier(reaction, direction, target_id, cache):
    """Target molecules this edge makes per unit flux: ``|coefficient| / mapped copies``.

    An EMU edge is one *mapped copy* of the target, but the reaction may move a
    number of copies its atom map does not spell out. ``CBPS2_m`` moves two ATP and
    two ADP while naming each once, so weighting the edge by 1 lets it deliver half
    the label it should. Dividing the coefficient by the number of mapped copies
    fixes that while leaving the common cases untouched: one copy named once gives
    1, and ``accoa_c (ab) + accoa_c (cd) ->`` still gives 1 per edge across its two
    edges, since there the coefficient and the copy count agree.

    Only the *target* coefficient matters. ``forward()`` builds the balance as
    ``Σ w·v·X_origin = Σ w·v·X_target`` with ``S_plus`` a plain ``Sg > 0`` mask, so
    an EMU's labeling is the flux-weighted mean of what produces it and the weights
    only ever act relative to the other edges into that same target. That is also
    why both endpoints of an edge carry the same magnitude: scaling one edge whole
    is a no-op, and it is the ratio between competing edges that carries the
    stoichiometry.
    """
    product_copies, educt_copies = _mapped_copies(reaction, cache)
    # 'rvs' runs the reaction backwards, so the target is one of its educts.
    copies = (product_copies if direction == 'fwd' else educt_copies).get(target_id, 1)
    return abs(reaction.get_coefficient(target_id)) / max(copies, 1)


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
            weight = symmetry_weight * direction_factor * edge_multiplier(
                reaction, direction, target_id, mapped_copies_cache)

            # Sign from each endpoint's own coefficient (origin consumed, target made);
            # magnitude from the target's, so competing edges keep their true ratio.
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
