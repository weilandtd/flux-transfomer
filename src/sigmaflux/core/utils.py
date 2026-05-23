import string

import pandas as pd

import numpy as np


from scipy.optimize import linprog
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
            if isinstance(row[atom_mapping], str):
                model.reactions.get_by_id(met).annotation['atom_mapping'] = row[atom_mapping]
            else:
               # print("no atom_mapping", met, row)
                model.reactions.get_by_id(met).annotation['atom_mapping'] = ''
        except Exception:
            pass

def convert_mfa_maranas(path_to_cobra, path_to_atom_map_xls, path_to_output=None, atom_type='C'):
    """
    Load a Maranas-lab GEM xlsx + AMM xlsx and return a COBRApy model annotated
    with SigmaFlux atom_mapping strings, ready for MFAModel().

    Maranas AMM format (one row per metabolite per reaction, atoms as integers)
    is converted to SigmaFlux string format, e.g.:
        reaction.annotation['atom_mapping'] = "accoa_c (ab) + accoa_c (cd) -> aacoa_c (abcd) + coa_c"

    :param path_to_cobra: path to Maranas GEM xlsx (Reactions + Metabolites sheets)
    :param path_to_atom_map_xls: path to Maranas AMM xlsx (Atom Transition + Symmetry sheets)
    :param path_to_output: if given, save annotated model as JSON
    :param atom_type: element to track, default 'C'
    """
    import cobra

    # Load GEM xlsx into a cobra model
    gem = pd.read_excel(path_to_cobra, sheet_name=None)
    model = cobra.Model()
    for _, row in gem['Metabolites'].iterrows():
        model.add_metabolites([cobra.Metabolite(id=str(row['metid']), name=str(row['metname']))])
    for _, row in gem['Reactions'].iterrows():
        rxn = cobra.Reaction(str(row['Rxn ID']), lower_bound=row['Lower bound'], upper_bound=row['Upper bound'])
        model.add_reactions([rxn])
        rxn.build_reaction_from_string(str(row['Reaction Formula']))

    # Annotate with atom mappings
    amm = pd.read_excel(path_to_atom_map_xls, sheet_name='Atom Transition')
    amm = amm[amm['startNodeSymbol'] == atom_type]

    _chars = string.ascii_lowercase + string.ascii_uppercase + string.digits

    for rxn_id, grp in amm.groupby('rxnAbbreviation', sort=False):
        try:
            rxn = model.reactions.get_by_id(rxn_id)

            # Map sorted unique integer atom IDs → single chars (a-z, A-Z, 0-9)
            all_ids = sorted({int(x) for maps in grp['maps'] for x in str(maps).split(',')})
            lbl = {aid: _chars[i] for i, aid in enumerate(all_ids)}

            reactants, products = [], []
            for _, row in grp.iterrows():
                ids = [int(x) for x in str(row['maps']).split(',')]
                entry = f"{row['metAbbreviation']} ({''.join(lbl[i] for i in ids)})"
                (reactants if row['ReactantProductFlag'] == 'reactant' else products).append(entry)

            amm_mets = set(grp['metAbbreviation'])
            reactants += [m.id for m in rxn.reactants if m.id not in amm_mets]
            products  += [m.id for m in rxn.products  if m.id not in amm_mets]

            rxn.annotation['atom_mapping'] = ' + '.join(reactants) + ' -> ' + ' + '.join(products)
        except Exception:
            pass

    # Symmetries
    try:
        sym = pd.read_excel(path_to_atom_map_xls, sheet_name='Symmetry')
        for _, row in sym.iterrows():
            try:
                met = model.metabolites.get_by_id(str(row['Metab_name']).strip())
                default = [int(x) for x in str(row['default mapping']).split()]
                sym_pos = [int(x) for x in str(row['symmetry mapping']).split()]
                p2l = {p: string.ascii_lowercase[i] for i, p in enumerate(default)}
                entry = ''.join(p2l[p] for p in default) + '.' + ''.join(p2l[p] for p in sym_pos)
                met.annotation.setdefault('symmetries', []).append(entry)
            except Exception:
                pass
    except Exception:
        pass

    _remove_out_metabolites(model)
    _standardize_exchange_directions(model)

    if path_to_output is not None:
        cobra.io.save_json_model(model, str(path_to_output))

    return model


def _remove_out_metabolites(model):
    """Convert Maranas _out boundary metabolites to proper COBRA exchange reactions.

    In Maranas MFA models, EX_ reactions connect an _e metabolite to an _out
    metabolite (the external environment pool).  This converts them to standard
    COBRA boundary reactions that simply deplete the _e metabolite:

        Before:  EX_glc__D_e : glc__D_e → glc__D_out
        After:   EX_glc__D_e : glc__D_e →

    Atom-mapping annotations are updated to remove _out references, and all
    orphaned _out metabolites are removed from the model.
    """
    for reaction in list(model.reactions):
        mets = list(reaction.metabolites.keys())
        out_mets = [m for m in mets if m.id.endswith("_out")]
        e_mets   = [m for m in mets if m.id.endswith("_e")]

        if not out_mets or not e_mets:
            continue

        # Remove _out metabolites from the reaction stoichiometry
        reaction.subtract_metabolites({m: reaction.metabolites[m] for m in out_mets})

        # Strip _out entries from atom_mapping annotation
        if "atom_mapping" in reaction.annotation:
            reaction.annotation["atom_mapping"] = _strip_out_from_atom_mapping(
                reaction.annotation["atom_mapping"]
            )

    # Remove metabolites that are now orphaned (no reactions left)
    orphaned = [m for m in model.metabolites if m.id.endswith("_out") and not m.reactions]
    model.remove_metabolites(orphaned)
    print(f"Removed {len(orphaned)} _out metabolites; EX_ reactions now deplete _e metabolites directly")


def _standardize_exchange_directions(model):
    """Ensure all EX_ reactions are in the depletion direction: met_e → (coeff -1).

    In Maranas models the stoichiometric direction of exchange reactions is
    inconsistent — some produce the _e metabolite (+1, uptake-positive) while
    others consume it (-1, secretion-positive).  This normalises every EX_
    reaction so that:

        positive flux = secretion  (metabolite leaves the system)
        negative flux = uptake     (metabolite enters the system)

    Reactions that already follow the convention are left untouched.
    For reactions where the _e metabolite has coefficient +1, the stoichiometry
    is flipped and the bounds are negated and swapped.
    """
    flipped = []
    for reaction in model.reactions:
        if not reaction.id.startswith("EX_"):
            continue
        e_mets = {m: c for m, c in reaction.metabolites.items() if m.id.endswith("_e")}
        if not e_mets:
            continue
        # If the _e metabolite is a product (+1), flip the reaction direction
        if all(c > 0 for c in e_mets.values()):
            # Negate all stoichiometric coefficients
            reaction.subtract_metabolites({m: 2 * c for m, c in reaction.metabolites.items()})
            lb, ub = reaction.bounds
            reaction.bounds = (-ub, -lb)
            # Flip atom_mapping: swap reactants and products across the '->' arrow
            if "atom_mapping" in reaction.annotation:
                amm = reaction.annotation["atom_mapping"]
                if "->" in amm:
                    lhs, rhs = amm.split("->", 1)
                    reaction.annotation["atom_mapping"] = rhs.strip() + " -> " + lhs.strip()
            flipped.append(reaction.id)

    print(f"Standardized {len(flipped)} EX_ reactions to depletion direction: {flipped}")


def _strip_out_from_atom_mapping(amm_str):
    """Remove all _out metabolite entries from an atom_mapping string.

    Handles entries at any position: first term, after '+', and on either
    side of '->'.  Entries may optionally carry an atom label, e.g.
    ``glc__D_out (abcdef)``.
    """
    if "->" not in amm_str:
        return amm_str

    def _strip_terms(half):
        # Split on '+', strip _out entries, rejoin
        terms = [t.strip() for t in half.split("+")]
        kept = [t for t in terms if not re.match(r'\S+_out\b', t)]
        return " + ".join(kept)

    lhs, rhs = amm_str.split("->", 1)
    return _strip_terms(lhs) + " -> " + _strip_terms(rhs)



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