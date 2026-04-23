import pandas as pd

import numpy as np

from scipy.sparse import eye as speye
from scipy.sparse import find as sfind
from scipy.sparse import hstack, vstack

from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import milp, linprog
import re 

import time
import logging
import torch 

logger = logging.getLogger(__name__)

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper



def annotate_model(model, annotation_file,index='id',atom_mapping='atom_mapping'):

    atom_mapping_annotation = pd.read_csv(annotation_file)
    atom_mapping_annotation.index = atom_mapping_annotation[index]

    for met, row in atom_mapping_annotation.iterrows():
        try: 
            if type(row[atom_mapping]) == str:
                model.reactions.get_by_id(met).annotation['atom_mapping'] = row[atom_mapping]
            else:
               # print("no atom_mapping", met, row)
                model.reactions.get_by_id(met).annotation['atom_mapping'] = ''
        except: 
            pass

def atommap_boundary(model): 
    all_rxns = set(model.reactions)
    for r in model.boundary: 
        metabolite = list(r.metabolites.keys())[0]
        metabolite_reaction = None 
        for reaction in metabolite.reactions: 
            if reaction in all_rxns and "atom_mapping" in reaction.annotation: 
                metabolite_reaction = reaction
                break 
        if metabolite_reaction is not None: 
            pattern = re.escape(metabolite.id) + r'\s*(\([^)]+\))'
            match = re.search(pattern, metabolite_reaction.annotation['atom_mapping'])
            if match:
                atom_map = match.group(1)
                r.annotation['atom_mapping'] = insert_mapping_into_reaction(r.reaction, metabolite.id, atom_map)
        else: 
            
            raise ValueError("Metabolite is not in the model")



def insert_mapping_into_reaction(reaction, compound_id, mapping):
    # Define a regular expression pattern to find the compound ID
    # and capture the surrounding context
    pattern = re.compile(rf'(\b{re.escape(compound_id)}\b)')

    # Use a replacement function in re.sub()
    def replacement(match):
        # Include the mapping after the compound ID
        return f'{match.group(1)} {mapping}'

    # Replace the pattern in the reaction string with the new string including the mapping
    new_reaction = pattern.sub(replacement, reaction)
    return new_reaction



def solve_for_theta(K,V):
    c = np.ones(K.shape[1]) 
    res = linprog(c=c, A_eq = K, b_eq = V, bounds=(None,None))
    # TODO Chatch stupid erorrs 
    # TODO maybe also make this a tensor? 
    if res.x is None:
        raise ValueError(f"No solution found for theta {res.status}")
    return res.x

### Sparse utils ### 

def select_sparse_column(sparse_matrix, col_idx):
    """
    Select a single column from a sparse matrix, setting all other columns to zero.

    Args:
    sparse_matrix (torch.Tensor): The input sparse matrix.
    col_idx (int): The index of the column to select.

    Returns:
    torch.Tensor: A new sparse matrix with only the selected column.
    """
    rows, cols = sparse_matrix.shape

    # Extract indices and values from the original sparse matrix
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # Filter indices to only keep the desired column
    col_indices = indices[1] == col_idx
    filtered_indices = indices[:, col_indices]
    filtered_values = values[col_indices]

    # Adjust the column indices in filtered_indices to all be col_idx
    filtered_indices[1, :] = col_idx

    # Create a new sparse matrix with the filtered indices and values
    new_sparse_matrix = torch.sparse_coo_tensor(filtered_indices, filtered_values, (rows, cols))

    return new_sparse_matrix

def flatten_sparse_matrix(sparse_matrix):
    """
    Flatten a 2D sparse matrix into a 1D sparse vector.

    Args:
    sparse_matrix (torch.Tensor): A 2D sparse matrix.

    Returns:
    torch.Tensor: A 1D sparse vector.
    """
    rows, cols = sparse_matrix.shape
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # Compute linear indices for the 1D flattened vector
    flat_indices = indices[0] * cols + indices[1]

    # Create the flattened 1D sparse vector
    flat_sparse_matrix = torch.sparse_coo_tensor(flat_indices.unsqueeze(0), values, (rows * cols,))

    return flat_sparse_matrix

def reshape_sparse_tensor_to_3D(sparse_tensor, new_shape):
    """
    Reshape a 2D sparse tensor to a 3D sparse tensor.
    
    Args:
    sparse_tensor (torch.Tensor): A 2D sparse tensor.
    new_shape (tuple): The desired shape (N, X, Y).

    Returns:
    torch.Tensor: A 3D sparse tensor.
    
    """
    if len(new_shape) != 3:
        raise ValueError("New shape must be a tuple of 3 dimensions (N, X, Y)")

    N, M = sparse_tensor.shape
    Y,X = new_shape[1], new_shape[2]

    if M != X * Y:
        raise ValueError("The product of X and Y must be equal to M")

    # Extract original indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # Recalculate indices for the new shape
    new_indices = torch.zeros(3, indices.size(1), dtype=torch.long)
    new_indices[0] = indices[0]  # N dimension remains the same
    new_indices[1] = indices[1] % Y # New X dimension
    new_indices[2] = indices[1] // Y 

    # Create the new sparse tensor
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, values, new_shape)
    return new_sparse_tensor

def multiply_3d_sparse_with_2d(sparse_3d, tensor_2d, transpose_2nd = False, flip_mult = False):
    """
    Multiply each 2D slice of a 3D sparse tensor (along the first dimension) 
    with a 2D dense tensor, leveraging sparse-dense multiplication optimizations.

    Args:
    sparse_3d (torch.Tensor): A 3D sparse tensor of shape [N, C, B].
    tensor_2d (torch.Tensor): A 2D sparse tensor of shape [B, P].
    transpose_2nd (bool): Whether to transpose the second dimension of the sparse matrix.
    flip_mult: whether tensor_2D is multiplied on the right or left of sparse_3d

    Returns:
    torch.Tensor: A 3D tensor of shape [N, C, P] containing the results with type of the tensor 2d.
    """

    # Placeholder for the results
    results = []


    # Loop through the slices of the sparse matrix
    for i in range(sparse_3d.shape[0]):
        # Extract the i-th slice as a 2D sparse tensor
        if transpose_2nd: 
            slice_2d = sparse_3d[i].T.coalesce()
        else: 
            slice_2d = sparse_3d[i].coalesce()
        
        # Perform the sparse-dense multiplication
        if flip_mult: 
            result_slice = torch.mm(tensor_2d, slice_2d)
        else: 
            result_slice = torch.mm(slice_2d, tensor_2d)

        # Store the result slice
        results.append(result_slice)

    # Concatenate the results along the first dimension
    result_tensor = torch.cat([r.unsqueeze(0) for r in results], dim=0)

    return result_tensor

def find_dependent_rows(matrix):
    rank_full = torch.linalg.matrix_rank(matrix)
    dependent_rows = []

    for i in range(matrix.shape[0]):
        matrix_minus_i = torch.cat([matrix[:i], matrix[i+1:]])
        rank_minus_i = torch.linalg.matrix_rank(matrix_minus_i)

        if rank_minus_i == rank_full:
            dependent_rows.append(i)

    return dependent_rows