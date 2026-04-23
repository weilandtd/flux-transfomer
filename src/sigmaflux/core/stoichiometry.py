from collections import namedtuple
from networkx import incidence_matrix
import torch
from sigmaflux.expansion.emu_expansion import EMUExpansion
from sigmaflux.expansion.emu import get_metabolite

AnnotatedStoichiometry = namedtuple('AnnotatedStoichiometry', ['matrix', 'reactions', 'directions',
                                                               'emus', 'boundary', 'core', 'index_hash'])


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

            # Set stoichiometry (-1 or +1) time the direction factor 
            Sg[emu_hash_emu_index[edge[0]], ix] = symmetry_weight * direction_factor * reaction.get_coefficient(get_metabolite(edge[0])[0]) / abs(reaction.get_coefficient(get_metabolite(edge[0])[0])) 
            Sg[emu_hash_emu_index[edge[1]], ix] = symmetry_weight * direction_factor * reaction.get_coefficient(get_metabolite(edge[1])[0]) / abs(reaction.get_coefficient(get_metabolite(edge[1])[0]))
            
            
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
