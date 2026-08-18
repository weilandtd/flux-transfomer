import torch
from sigmaflux.expansion.emu import get_metabolite


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
            edge_weight = data['weight']
            if direction == 'fwd':
                reaction_indices[ix] = model._flux_hash[reaction.id]
                direction_factor = 1
            elif  direction == 'rvs':
                reaction_indices[ix] = model._flux_hash[reaction.id] + model.num_reactions
                direction_factor = -1
            else:
                raise RuntimeError("Expansion contains woring direction annotation!")

            origin_id, target_id = get_metabolite(edge[0])[0], get_metabolite(edge[1])[0]
            weight = edge_weight * direction_factor

            # Multiplicity is carried by the edge weight -- how many copies of the
            # target this origin set makes, see AtomMapping.get_origin_emus -- so
            # only the sign differs between the consumed origin and the produced
            # target.
            for node, met_id in ((edge[0], origin_id), (edge[1], target_id)):
                coefficient = reaction.get_coefficient(met_id)
                Sg[emu_hash_emu_index[node], ix] = weight * coefficient / abs(coefficient)


        # Set up the mapping dicts used later in calculation
        emu_hashes[size] = emu_hash_index_emu
        subnet_stoichs[size] = Sg.cpu().to_sparse_coo().to(device) #COO format for now 
        subnet_stoichs_plus[size] = ((Sg > 0).type(dtype).T).cpu().to_sparse_coo().to(device)
        reaction_hashes[size]  = reaction_indices  

    return subnet_stoichs,subnet_stoichs_plus,emu_hashes, boundary_indices, core_indices,reaction_hashes
