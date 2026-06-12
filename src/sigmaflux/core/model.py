import torch
import logging 
from torch.nn import Module as TorchModule
from torch.nn import Parameter as TorchParameter

from tqdm import tqdm

from cobra import Model
from cobra.util.array import create_stoichiometric_matrix
from cobra.sampling import sample

from sigmaflux.expansion.emu_expansion import EMUExpansion
from sigmaflux.core.stoichiometry import get_subnets
from sigmaflux.core.experiment import get_tracer_inputs
from sigmaflux.core.mfa import build_emu_met_filter
from sigmaflux.core.calc_labeling import calculate_labeling, get_subnet_fluxes
from sigmaflux.core.utils import time_function, select_sparse_column, flatten_sparse_matrix, reshape_sparse_tensor_to_3D

import pandas as pd

import pickle
import copy 



logger = logging.getLogger(__name__)

class MFAModel(TorchModule, Model):
    """
    Class for an MFA model
    """
    @time_function
    def __init__(self, model, name,
                 device=None,
                 min_flux=1e-4,
                 mu0=1.0,
                 lambda0=1.0,
                 n_processes=1,
                 n_samples=10,
                 initial_flux_distribution=None,
                 dtype=torch.float64,
                 max_reversibility_ratio=10.0,
                 seed=None):
        """

        :param model: cobra.Model annotated with atom transitions
        :param experiments: list of mfa.core.Experiment
        """

        Model.__init__(self, model.copy(), name)
        TorchModule.__init__(self)

        self.expansion = None
        self.experiments = None

        self._S = None

        self.prepared = False
        self._backward_compatible = None

        self.max_emu_size = None
        self.max_mid_size = None
        self.emu_states = None
        
        self._model = model
        self._n_samples = n_samples

        self.min_rev_flux = min_flux
        self.max_reversibility_ratio = max_reversibility_ratio

        self.dtype = dtype

        # Model metabolites
        self._metabolite_hash = {m.id: i for i, m in enumerate(self.metabolites)}
        self.max_emu_mets = len(self._metabolite_hash)

        # Model optimizable parameters
        self.stoichiometry = torch.tensor(create_stoichiometric_matrix(self), 
                                          dtype=dtype, device=device)       
    
        # If no intial value provided sample
        if initial_flux_distribution is None: 
            # Sample a mean flux distribution to initialize the model
            flux_sample = sample(model, n_samples, processes=n_processes, seed=seed)
            initial_fluxes = flux_sample.mean()
        
        else:
            # Make sure the initial flux distribution is a pandas series
            if not isinstance(initial_flux_distribution, pd.Series):
                raise ValueError("initial_flux_distribution must be a pandas Series with reaction ids as index")

            reaction_ids = [r.id for r in self.reactions]
            initial_fluxes = initial_flux_distribution[reaction_ids]

        # Set the initial flux values as a torch tensor
        self.initial_flux_values = torch.tensor(initial_fluxes.values, 
                                           dtype=self.dtype, device=device)
          
        self._flux_hash = {r.id: i for i, r in enumerate(self.reactions)}

        self.num_reactions = len(self.reactions)

        # Reverisbility of each reaction
        # Sample from exponential distribution (mean=1) clamped for numerical stability
        # max_reversibility_ratio controls the maximum r/|v| ratio (default=10 → ~90% reversible)
        random_reversibility = torch.distributions.Exponential(rate=1.0).sample((self.num_reactions,)).to(dtype=self.dtype, device=device)
        random_reversibility = torch.clamp(random_reversibility, max=self.max_reversibility_ratio)
        reversible_reactions = [r for r in self.reactions]
        self._reversibility_hash = {r.id: i for i, r in enumerate(reversible_reactions)}
        self._reversibility_mask = torch.tensor([1.0 if r.reversibility else 0.0 for r in self.reactions]
                                                , dtype=self.dtype, device=device)
        
        initial_reversibility= torch.abs(random_reversibility * self.initial_flux_values) * self._reversibility_mask

        self._reversibility = TorchParameter(initial_reversibility.clone(), requires_grad=True)

        # Fluces 
        self._fluxes = TorchParameter(self.initial_flux_values.clone(), requires_grad=True)

        # Lagrange multipliers for each mass balance
        _lambda0 =  torch.tensor([lambda0 for r in self.metabolites], dtype=self.dtype, device=device)
        self._lambda0 = _lambda0
        #self._lambda = TorchParameter(_lambda0.clone(), requires_grad=True)

        # Lagrange multipliers for each forward and backward flux constrains
        _mu0 = torch.tensor([mu0 for r in range(2*self.num_reactions)], dtype=self.dtype, device=device)
        self._mu0 = _mu0
        #self._mu = TorchParameter(_mu0.clone(), requires_grad=True)

        self.device = device
        logger.info(f"Initalized MFA model with {self.num_reactions} reactions and {self.max_emu_mets} metabolites")
    
    def reset_initial_fluxes(self, initial_flux_distribution=None):
        if initial_flux_distribution is None:
            flux_sample = sample(self._model, self._n_samples)
            mean_flux_solution = flux_sample.mean()
            initial_flux_values = torch.tensor(mean_flux_solution.values, 
                                           dtype=self.dtype, device=self.device)
        else:
            # Make sure the initial flux distribution is a pandas series
            if not isinstance(initial_flux_distribution, pd.Series):
                raise ValueError("initial_flux_distribution must be a pandas Series with reaction ids as index")
            # Get the solution in order of the model reactions
            reaction_ids = [r.id for r in self.reactions]
            initial_fluxes = initial_flux_distribution[reaction_ids]
            initial_flux_values = torch.tensor(initial_fluxes.values,
                                           dtype=self.dtype, device=self.device)
        
        # Over write new initial flux values
        self.initial_flux_values = initial_flux_values
        # Sample from exponential distribution (mean=1) clamped for numerical stability
        # max_reversibility_ratio controls the maximum r/|v| ratio (default=10 → ~90% reversible)
        random_reversibility = torch.distributions.Exponential(rate=1.0).sample((self.num_reactions,)).to(dtype=self.dtype, device=self.device)
        random_reversibility = torch.clamp(random_reversibility, max=self.max_reversibility_ratio)
        initial_reversibility= torch.abs(random_reversibility * initial_flux_values) * self._reversibility_mask

        # Update the model parameters
        sd = self.state_dict()
        sd['_fluxes'] = initial_flux_values.clone()
        sd['_reversibility'] = initial_reversibility.clone()
        self.load_state_dict(sd) 
    

    # count the number of reactions excluding reactions that start with EX and IN 
    def num_reactions(self): 
        return len([r for r in self.reactions if not r.id.startswith("EX") and not r.id.startswith("IN")])
    # count the number of metabolites excluding _a and _v metabolites 
    def num_metabolites(self):
        return len([m for m in self.metabolites if not m.id.endswith("_a") and not m.id.endswith("_v")]) 

    # doesn't have backwards and forward 
    @property
    def fluxes(self) -> torch.Tensor:
        return self._fluxes

    # This is just a read out now 
    @property
    def reversibility(self) -> torch.Tensor:
        return torch.abs(self._reversibility) * self._reversibility_mask
        #return(torch.relu(self._backward_fluxes) + torch.relu(self._forward_fluxes) )/ torch.abs(self.fluxes)

    @property
    def reversible_fluxes(self) -> torch.Tensor:
        # vf = self._forward_fluxes
        # vb = self._backward_fluxes
        reversibility = torch.where(self.reversibility <= self.min_rev_flux, self.min_rev_flux, self.reversibility)
        vf = 0.5 * (torch.abs(self.fluxes) + self.fluxes + reversibility)
        vb = 0.5 * (torch.abs(self.fluxes) - self.fluxes + reversibility)

        # This is needed to minimize the expansion of the fluxes

        # Mapp irreversbile fluxes to vf
        vf = vf - vb * (1-self._reversibility_mask)
        vb = vb * self._reversibility_mask

        return torch.cat((vf, vb)) 

    def get_flux(self,reaction_id):
        return self.fluxes[self._flux_hash[reaction_id]]

    def save_checkpoint(self, file = "checkpoint.pt", meta_data = {}):
    # Load the existing checkpoints if any
        try:
            checkpoints = torch.load(file)
        except Exception:
            checkpoints = []

        check_points_data = self.state_dict()
        # Add metadata to the checkpoint
        for key in meta_data:
            check_points_data[key] = meta_data[key]

        # Save the new checkpoint
        checkpoints.append(check_points_data)

        torch.save(checkpoints, file)

    def load_checkpoint(self, file = "checkpoint.pt", index=-1):
        # Load the checkpoints
        try:
            checkpoints = torch.load(file)
        except Exception:
            print("File not found")
            return

        # Load the state into the model
        if index < len(checkpoints):
            sd = self.state_dict()
            for key in sd:
                sd[key] = checkpoints[index][key]
            self.load_state_dict(sd)
        else:
            print("Invalid index")

    @time_function
    def prepare(self, experiments, dA=None, dB=None, with_backward=True, targets=None):
        """
        Prepares the model expression to be run with MFA
        :param experiments: list of mfa.core.Experiment
        :param dA: optional path to precomputed dA data
        :param dB: optional path to precomputed dB data
        :param with_backward: whether to prepare model for backward propagation
        :param targets: optional set of target EMUs to prepare the model for otherwise
                        the targets will be inferred from the experiments
        :return: None
        """

        self.experiments = experiments

        """
        Integrate data from experiments to build the network
        """
        if targets is None:
            targets = set()
            for the_experiment in self.experiments:
                the_targets = [entry.emu for entry in the_experiment.mid_data]
                targets.update(the_targets)


        # Build network expansions
        self.expansion = EMUExpansion(self, list(targets))
        self.expansion.expand_network()

        # edge_reaction_hash: stores all of the indices in the flux vector of the reactions in each subnet and their ids
        self._S, self._S_plus, self._emu_hashes, self._ix_b, self._ix_c, self.edge_reaction_index \
            = get_subnets(self, self.expansion.emu_network.subnetworks,device=self.device, dtype=self.dtype)

        # Initialize emu_states
        self.max_emu_size = len(self.expansion.emu_network.subnetworks)
        self.max_mid_size = self.max_emu_size + 1


        # TODO CHOOSE THE RIGHT INPUTS
        # Find each metabolite in each subbnet 
        self.emu_met_filter = build_emu_met_filter(self, self.metabolites, self.device)
        self._initial_params = copy.deepcopy(self.state_dict())

        # Build core met indices                    
        core_met_ix = {}
        for subnet in self._ix_c:
            core_met_ix[subnet]= {self._emu_hashes[subnet][emu_index.item()]: i for i,emu_index in enumerate(self._ix_c[subnet])}
        

        if with_backward: 
            # Precompute dA and dB matrices for each subnet
            dBs = {}
            dAs = {} 
            if dA is not None and dB is not None:
                with open(dA, 'rb') as f:
                    dAs = pickle.load(f)
                with open(dB, 'rb') as f:
                    dBs = pickle.load(f)
            else: 
                for subnet in tqdm(self._S): 
                    if self._S[subnet].shape[1] == 0:
                        continue
                    alpha = self._S[subnet].shape[1]
                    T = torch.zeros((self.num_reactions * 2, len(self.edge_reaction_index[subnet])), dtype = self.dtype, device=self.device)
                    T[self.edge_reaction_index[subnet], torch.arange(len(self.edge_reaction_index[subnet]))] = 1
                    T = T.to_sparse()
                    core_S = torch.index_select(self._S[subnet], 0, self._ix_c[subnet])
                    boundary_S = torch.index_select(self._S[subnet], 0, self._ix_b[subnet])
                    core_S_plus = torch.index_select(self._S_plus[subnet], 1, self._ix_c[subnet])
                    M_slices = []
                    N_slices = []
                    for i in range(alpha): 
                        M_slices.append(flatten_sparse_matrix(select_sparse_column(core_S, i) @ core_S_plus).unsqueeze(1))
                        N_slices.append(flatten_sparse_matrix(select_sparse_column(boundary_S, i) @ core_S_plus).unsqueeze(1)) 
                    dM = -torch.cat(M_slices, dim = 1) # (c * c) * (number of edges in subnet)
                    dN = torch.cat(N_slices, dim = 1) # (c * b) * (number of edges in subnet)
                    
                    # Run sparse matrix multiplication on CPU
                    # uses a lot of GPU resources otherwise 
                    tempA = T.cpu() @ (dM.T.cpu())
                    tempB = T.cpu() @ (dN.T.cpu())

                    dA = reshape_sparse_tensor_to_3D(tempA,(self.num_reactions * 2,len(self._ix_c[subnet]),len(self._ix_c[subnet]))) # (number of reactions * 2) x (c * c) 
                    dB = reshape_sparse_tensor_to_3D(tempB, (self.num_reactions*2, len(self._ix_c[subnet]),len(self._ix_b[subnet]))) # (number of reactions * 2) x (c) x (b)

                    # Cast to self.device tensor
                    dBs[subnet] = dB.to(self.device)
                    dAs[subnet] = dA.to(self.device)
    
            self.forward_dict = {
                "S": self._S,
                "S_plus": self._S_plus,
                "emu_hashes": self._emu_hashes,
                "ix_b": self._ix_b,
                "ix_c": self._ix_c,
                "edge_reaction_index": self.edge_reaction_index,
                "emu_met_filter": self.emu_met_filter,
                "dB": dBs, 
                "dA": dAs, 
                "metabolite_hash": self._metabolite_hash,
                "core_met_ix": core_met_ix
            }

            with open('dAs.pkl', 'wb') as f:
                pickle.dump(dAs, f)
            with open('dBs.pkl', 'wb') as f:
                pickle.dump(dBs, f)

            self._backward_compatible = True

        else:          

            self.forward_dict = {
                "S": self._S,
                "S_plus": self._S_plus,
                "emu_hashes": self._emu_hashes,
                "ix_b": self._ix_b,
                "ix_c": self._ix_c,
                "edge_reaction_index": self.edge_reaction_index,
                "emu_met_filter": self.emu_met_filter,
                "metabolite_hash": self._metabolite_hash,
                "core_met_ix": core_met_ix

            }
            self._backward_compatible = False

         # Finally set prepared flag
        self.prepared = True
    

    def forward(self, exp_ix=None):
        if not self.prepared:
            raise RuntimeError('Model is not prepared!')
        labeling = []
        reversible_fluxes = self.reversible_fluxes

        # Recompute A and B matrices for each subnet
        with torch.no_grad(): 
            boundaries = [self._emu_hashes[subnet][b.item()] for subnet in self._ix_b for b in self._ix_b[subnet]]
            v = get_subnet_fluxes(reversible_fluxes, self.forward_dict["edge_reaction_index"])
            self.forward_dict["A_inv"] = {}
            self.forward_dict['A'] = {}
            self.forward_dict["B"] = {}
            for subnet in self.forward_dict["S"]:
                if self.forward_dict["S"][subnet].shape[1] == 0:
                    continue
                core_S = torch.index_select(self.forward_dict["S"][subnet], 0, self.forward_dict["ix_c"][subnet])
                boundary_S = torch.index_select(self.forward_dict["S"][subnet], 0, self.forward_dict["ix_b"][subnet])
                core_S_plus = torch.index_select(self.forward_dict["S_plus"][subnet], 1, self.forward_dict["ix_c"][subnet])

                if not torch.isfinite(v[subnet]).all():
                    bad = torch.where(~torch.isfinite(v[subnet]))[0].tolist()
                    raise RuntimeError(
                        f"Non-finite flux values in subnet {subnet} at edge indices {bad}. "
                        "Check upstream flux parametrisation (e.g. softplus bounds, log-space mapping)."
                    )

                n_fluxes = len(v[subnet])
                offsets = torch.tensor([0])
                V = torch.sparse.spdiags(v[subnet].cpu(), offsets.cpu(), (n_fluxes, n_fluxes),).to(self.device)
                A = -(torch.mm(torch.mm(core_S, V), core_S_plus)).T
                B = (torch.mm(torch.mm(boundary_S, V), core_S_plus)).T

                AD = A.to_dense()
                BD = B.to_dense()

                # We use the inverse for backward calculation
                try:
                    self.forward_dict["A_inv"][subnet] = torch.linalg.pinv(AD)
                except torch.linalg.LinAlgError:
                    row_sums = torch.sum(AD, dim=1)
                    small_indices = torch.where(row_sums < 0.5e-6)[0]
                    small_emus = [self.forward_dict["emu_hashes"][subnet][i.item()] for i in small_indices]
                    raise RuntimeError(
                        f"Degenerate A matrix for subnet {subnet}: "
                        f"{len(small_indices)} EMUs have near-zero row sums "
                        f"{small_emus[:10]}{'...' if len(small_emus) > 10 else ''}"
                    )
                
                self.forward_dict["B"][subnet] = BD

        # Calculate labeling for each experiment in the batch
        if exp_ix is None:
            exp_ix = range(len(self.experiments))
        
        for ix in exp_ix:
            the_experiment = self.experiments[ix]
            tracer_boundary_inputs = get_tracer_inputs(the_experiment, boundaries, device=self.device, dtype = self.dtype)
            the_labeling = calculate_labeling(reversible_fluxes,self.forward_dict , tracer_boundary_inputs, 
                                            self.device, self.dtype)
            labeling.append(the_labeling)

        return torch.stack(labeling).squeeze(0)

