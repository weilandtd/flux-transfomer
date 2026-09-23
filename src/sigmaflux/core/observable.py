# this should be a class where you put in labeling or fluxes, labeling or flux hash, 
# and a mapping between hash to and you get out the observable

import torch 

class Observable(): 
    """Reduces a tensor along one axis into a weighted sum over grouped rows, per observable_mapping."""
    def __init__(self, observable_mapping, observable_weighting,axis=0):
        """Args:
            observable_mapping: per-observable list of row indices to group.
            observable_weighting: per-observable list of weights, aligned with observable_mapping.
            axis: tensor axis the grouping/reduction is applied along.
        """
        self.observable_mapping = observable_mapping
        self.observable_weighting = observable_weighting
        self.axis = axis
        pass

    def get_observable(self, observable ): 
        # Create a new tensor with the same number of rows as the mapping
        new_shape = [len(self.observable_mapping)] + list(observable.shape)[:self.axis] + list(observable.shape)[self.axis+1:]
        new_tensor = torch.zeros(new_shape, dtype=observable.dtype, device=observable.device)   

        # Iterate over the mapping and sum the rows
        for i, indices in enumerate(self.observable_mapping):
            weighting = torch.tensor(self.observable_weighting[i])
            selected_elements = observable.index_select(self.axis, torch.tensor(indices))
            target_shape = [1] * selected_elements.dim()
            target_shape[self.axis] = selected_elements.shape[self.axis]
            weighting = weighting.view(target_shape)
            new_tensor[i] = (selected_elements * weighting).sum(dim=-1)

        
        permute_order = list(range(1, len(new_tensor.shape))) 
        permute_order.insert(self.axis, 0)  

        new_tensor_permuted = new_tensor.permute(permute_order)
        return new_tensor_permuted
        

