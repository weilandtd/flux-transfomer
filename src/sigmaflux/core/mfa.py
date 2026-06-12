"""
Functions for MFA
"""
from sigmaflux.expansion.emu import OriginSet, EMU
from sigmaflux.core.data import get_model_labeling_data, get_model_flux_data
from tqdm import trange
import numpy as np
import torch
import logging
from sigmaflux.core.scaling import Scaling
from sigmaflux.core.utils import time_function 
# TODO MAKE A CENTRAL DEFINITION
EPSILON = 1e-9

logger = logging.getLogger(__name__)

@time_function
def fit_mfa_model(model, loss_fn,flux_loss_fn, optimizer,
                  lam1=1e-4, lam2=1e-5, epochs=100,
                  simultanous_experiments=3,
                  labeling_scaler= None, flux_scaler = 1, checkpoint = 10, 
                  checkpoint_file='checkpoint.pt', 
                  labeling_checkpoint_file = "labeling_checkpoint.pt", 
                  optimizer_checkpoint = "optimizer_checkpoint.pt", 
                  lr_scheduler = None, with_std = True):
    """AI is creating summary for fit_mfa_model

    Args:
        model ([type]): [description]
        loss_fn ([type]): [description]
        flux_loss_fn ([type]): [description]
        optimizer ([type]): [description]
        lam1 ([type], optional): [description]. Defaults to 1e-4.
        lam2 ([type], optional): [description]. Defaults to 1e-5.
        epochs (int, optional): [description]. Defaults to 100.
        tolerance ([type], optional): [description]. Defaults to 1e-4.
        labeling_scaler ([type], optional): [description]. Defaults to None.
        flux_scaler (int, optional): [description]. Defaults to 1.
        checkpoint (int, optional): [description]. Defaults to 10.
        checkpoint_file (str, optional): [description]. Defaults to 'checkpoint.pt'.
        labeling_checkpoint_file (str, optional): [description]. Defaults to "labeling_checkpoint.pt".
        optimizer_checkpoint (str, optional): [description]. Defaults to "optimizer_checkpoint.pt".
        lr_scheduler ([type], optional): [description]. Defaults to None.
        with_std (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """


    with_std = True
    if labeling_scaler is None: 
        labeling_scaler = torch.ones(len(model.experiments), device=model.device)  

    # check if labeling_scaler is a double or a list
    if isinstance(labeling_scaler, float):
        labeling_scaler = labeling_scaler * torch.ones(len(model.experiments), device=model.device) 

    Scaling(labeling_scaler)

    log = []

    # Generate indices of experiments to run simultanously
    experiment_batches = []
    for i in range(0, len(model.experiments), simultanous_experiments):
        experiment_batches.append(range(i, min(i + simultanous_experiments, len(model.experiments))))

    # Extract index mapped data for each experiment
    batch_data = {}
    for exp_ix in experiment_batches:
        index_labeling, labeling_mean, labeling_scaler, labeling_std = get_model_labeling_data(model, labeling_scaler, with_std, exp_ix=exp_ix)
        index_fluxes, flux_mean, flux_std = get_model_flux_data(model,with_std, exp_ix=exp_ix)                        
        # Create scaling object for this specific batch
        batch_labeling_scaling = Scaling(labeling_scaler)
        
        batch_data[tuple(exp_ix)] = (index_labeling, labeling_mean, labeling_scaler, labeling_std, index_fluxes, 
                                     flux_mean, flux_std, batch_labeling_scaling)

    

    logger.info(f"Optimization parameters: LR:{optimizer.param_groups[0]['lr']}, L1:{lam1}, L2:{lam2}, Epochs:{epochs}," \
                + f"Loss:{loss_fn.__class__.__name__}, Labeling Scaling: {labeling_scaler}")

    model.save_checkpoint(file=checkpoint_file) 
    
    with trange(epochs) as pbar:
        for epoch in pbar:
            epoch_loss = 0.0
            # Only store outputs when we need to save checkpoint (memory efficient)
            if epoch % checkpoint == 0:
                epoch_outputs = []
            
            for exp_ix in experiment_batches:
                # Load correct data for each experiment batch
                index_labeling, labeling_mean, labeling_scaler, labeling_std, index_fluxes, flux_mean, flux_std, batch_labeling_scaling = batch_data[tuple(exp_ix)]
    
                optimizer.zero_grad()
                # Compute labeling prediction and loss
                output = model.forward(exp_ix=exp_ix)
                
                # Only store output for checkpoint epochs (memory efficient)
                if epoch % checkpoint == 0:
                    epoch_outputs.append(output.detach().cpu())

                # Loss function fitting labeling predictions and known fluxes
                simulated_labeling = torch.take(output, index_labeling)
                # Debug line  
                simulated_labeling.retain_grad()

                # Training step 
                labeling_loss = loss_fn(batch_labeling_scaling.scale(simulated_labeling),
                                        batch_labeling_scaling.scale(labeling_mean))
            
                # Lagrange Multipliers to constrain S.v = 0 
                mass_balance_constraints = (model.stoichiometry @ model.fluxes).square()
                # Constrain for vf, vb >= 0 
                #dC/dv = 0  if v >=0  else dC/dv = -v -> positive 
                positive_fluxes_constraints = torch.relu(-1 * model.reversible_fluxes)

                # L1 Regularization 
                l1_reg = lam1 * model.reversible_fluxes.abs().mean()

                # L2 Regularization
                l2_reg = lam2 * model.reversible_fluxes.square().mean() 

                # # Optional flux loss 
                if not index_fluxes.nelement() == 0:
                    measured_fluxes = torch.take(model.fluxes, index_fluxes)
                    
                    #flux_loss = loss_fn(measured_fluxes / flux_mean.squeeze(), flux_mean / flux_mean)
                    flux_loss = flux_loss_fn(measured_fluxes / flux_mean.squeeze(), (flux_mean / flux_mean).squeeze())
                    flux_loss = flux_loss * flux_scaler
                    msrmt_labeling_loss = labeling_loss + flux_loss
                else:
                    msrmt_labeling_loss = labeling_loss
                    flux_loss = labeling_loss * 0.0

                # Total loss function for augmented Lagrangian
                lam = model._lambda0
                mu = model._mu0
                loss = msrmt_labeling_loss  + l1_reg + l2_reg \
                        + lam.T @ mass_balance_constraints \
                        + mu.T @ positive_fluxes_constraints 
        
                # Backpropagation
                loss.backward()
                optimizer.step(closure= lambda: loss)

                epoch_loss += loss.item()

            if lr_scheduler is not None:
                lr_scheduler.step()

            log.append(epoch_loss/len(experiment_batches))

            pbar.set_postfix(loss=epoch_loss/len(experiment_batches), epoch=epoch)
        
            logger.info(f"Optimization step {epoch} :total loss {log[-1]:.5f}, labeling loss {labeling_loss:.5f}," \
                            + f"flux loss {flux_loss:.5f}, l1 {l1_reg:.5f}, l2 {l2_reg:.5f}")
            

            # Save checkpoint
            if epoch % checkpoint == 0: 
                # Add meta data to the checkpoint file
                meta_data = {"loss": loss.item() , "epoch": epoch, 
                             "labeling_loss": labeling_loss.item(), "flux_loss": flux_loss.item(), 
                             "l1": l1_reg.item(), "l2": l2_reg.item()}
                
                model.save_checkpoint(file=checkpoint_file, meta_data=meta_data)
                
                # Aggregate outputs from all experiment batches
                # Concatenate along the experiment dimension (dim=0)
                aggregated_output = torch.cat(epoch_outputs, dim=0)
                
                try:
                    labeling_data = torch.load(labeling_checkpoint_file)
                except Exception:
                    labeling_data = []

                labeling_data.append(aggregated_output)

                torch.save(labeling_data, labeling_checkpoint_file)

                torch.save(optimizer.state_dict(), optimizer_checkpoint)


    logger.info(f"Optimization complete: total loss {log[-1]:.5f}, labeling loss {labeling_loss:.5f}," \
                + f"flux loss {flux_loss:.5f}, l1 {l1_reg:.5f}, l2 {l2_reg:.5f}")
    return log


def filter_conv_inputs(emu_inputs):
    return [b for b in emu_inputs if isinstance(b, OriginSet) and len(b._values) > 1]

def filter_conv_inputs_with_indices(emu_inputs): 
    emus = []
    emu_indices = []
    for i, b in enumerate(emu_inputs): 
        if isinstance(b, OriginSet) and len(b._values) > 1:
            emus.append(b)
            emu_indices.append(i)
    return emus, emu_indices

def calc_emu_input(emu_states, emu_inputs, mid_inputs, ):
    this_emu_inputs = [0.0, ] * ()
    for ix, the_input in enumerate(emu_inputs):
        if isinstance(the_input, OriginSet) and len(OriginSet) > 1:
            this_emu_inputs[ix] = convolution(emu_states, the_input)
        else:
            this_emu_inputs[ix] = extract_input(mid_inputs, the_input)

    return emu_inputs


# Invariant: every component of a multi-element OriginSet must be a core node
# (present in core_met_ix for its subnet size).  This is guaranteed by
# EMUExpansion._prune_orphaned_edges(); a KeyError here means the prune step
# was skipped or a new code path bypasses it.
def convolution(core_met_ix, emu_states, origin_set, device, dtype):
    """
    emu_states matrix is a dictionary of # of subnets torch tensors with # of core emu x # of mids
    """
    emu_size = origin_set.size

    if len(origin_set) > 2:
        emus = origin_set._values 
        ## first convolute every two emus, then convolute the results from that convolution and so on until we have convoluted all 8 elements together 
        all_convolutions = []
        for i in range(0, len(emus), 2): 
            all_convolutions.append(convolution(core_met_ix, emu_states, OriginSet([emus[i], emus[i+1]]), device, dtype))
        
        
        while len(all_convolutions) > 1: 
            new_convolutions = []
            for i in range(0, len(all_convolutions), 2): 
                cur_set_size = all_convolutions[i].shape[0] + all_convolutions[i+1].shape[0] - 2
                t1 = torch.zeros((cur_set_size + 1),device=device, dtype = dtype)
                t2 = torch.zeros((cur_set_size + 1),device=device, dtype = dtype)
                t1[:all_convolutions[i].shape[0]]= all_convolutions[i]
                t2[:all_convolutions[i+1].shape[0]] = all_convolutions[i+1]
                
                conv = torch.tensor([sum([t1[alpha - beta] * t2[beta] for beta in range(alpha + 1)])
                    for alpha in range(cur_set_size + 1)], device=device, dtype=dtype)
                new_convolutions.append(conv)
            all_convolutions = new_convolutions 

        return all_convolutions[0]
    met_emu_size_1 = origin_set._values[0].size
    met_emu_size_2 = origin_set._values[1].size

    metid_1 = core_met_ix[met_emu_size_1][origin_set._values[0]]
    metid_2 = core_met_ix[met_emu_size_2][origin_set._values[1]]
    t1 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)
    t2 = torch.zeros((emu_size + 1,), device=device, dtype = dtype)
    t1[:met_emu_size_1+1] = emu_states[met_emu_size_1][metid_1]
    t2[:met_emu_size_2+1] = emu_states[met_emu_size_2][metid_2]

    conv = [sum([t1[alpha - beta]
                 * t2[beta] for beta in range(alpha + 1)])
            for alpha in range(emu_size + 1)]

    return torch.stack(conv, axis=0)

#def conv(x, y, emu_size, device):
#    return torch.tensor([sum([x[alpha - beta] * y[beta] for beta in range(alpha + 1)]) for alpha in range(emu_size + 1)],dtype = torch.float32, device=device)

def conv(x,y,emu_size, device, dtype): 
    result = torch.zeros(x.shape[0], emu_size+1, device = device, dtype = dtype) 
    for alpha in range(emu_size+1): 
        beta = torch.arange(alpha+1)
        result[:,alpha] = torch.sum(x[:, alpha-beta] @ y[beta], axis=1)
    return result

# can this be done more efficiently? 
def convolution_derivative(core_met_ix, derivative_states, emu_states, origin_set, device, dtype):
    """
    emu_states matrix [M,K]
    """
    emu_size = origin_set.size
    if len(origin_set) > 2:
        emus = origin_set._values
        all_convolutions = []
        all_sets = []
        for i in range(0, len(emus), 2): 
            all_convolutions.append(convolution_derivative(core_met_ix, derivative_states, emu_states, OriginSet([emus[i], emus[i+1]]), device, dtype))
            all_sets.append(OriginSet([emus[i], emus[i+1]]))
    
        while len(all_convolutions) > 1: 
            new_sets = []
            new_convolutions = []
            for i in range(0, len(all_convolutions), 2): 
                origin_set_1 = all_sets[i]
                origin_set_2 = all_sets[i+1]
                cur_set_size = all_convolutions[i].shape[1] + all_convolutions[i+1].shape[1] - 2

                origin1_derivative = torch.zeros((all_convolutions[0].shape[0], cur_set_size + 1,),device=device, dtype = dtype)
                origin2_derivative = torch.zeros((all_convolutions[0].shape[0], cur_set_size + 1,),device=device, dtype = dtype)
                origin1_derivative[:,:all_convolutions[i].shape[1]] = all_convolutions[i]
                origin2_derivative[:,:all_convolutions[i+1].shape[1]] = all_convolutions[i+1]

                origin1_emu = torch.zeros((cur_set_size + 1,),device=device, dtype = dtype)
                origin2_emu = torch.zeros((cur_set_size + 1,),device=device, dtype = dtype)
                origin1_emu[:all_convolutions[i].shape[1]] = convolution(core_met_ix, emu_states, origin_set_1, device, dtype)
                origin2_emu[:all_convolutions[i+1].shape[1]] = convolution(core_met_ix, emu_states, origin_set_2, device, dtype)

                conv_derivative_1 = conv(origin1_derivative , origin2_emu.unsqueeze(-1), cur_set_size, device, dtype)
                conv_derivative_2 = conv(origin2_derivative , origin1_emu.unsqueeze(-1), cur_set_size, device, dtype)   
                
                new_sets.append(OriginSet(origin_set_1._values + origin_set_2._values))
                new_convolutions.append(conv_derivative_1 + conv_derivative_2)
            all_convolutions = new_convolutions 
            all_sets = new_sets

        return all_convolutions[0]

   
    met_emu_size_1 = origin_set._values[0].size
    met_emu_size_2 = origin_set._values[1].size

    metid_1 = core_met_ix[met_emu_size_1][origin_set._values[0]]
    metid_2 = core_met_ix[met_emu_size_2][origin_set._values[1]]
    
    met1_derivative = torch.zeros((derivative_states[met_emu_size_1].shape[1], emu_size + 1,),device=device, dtype = dtype)
    met2_derivative = torch.zeros((derivative_states[met_emu_size_2].shape[1], emu_size + 1,),device=device, dtype = dtype)
    
    met1_derivative[:,:met_emu_size_1+1] = derivative_states[met_emu_size_1][metid_1]
    met2_derivative[:,:met_emu_size_2+1] = derivative_states[met_emu_size_2][metid_2]

    met_emu_1 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)
    met_emu_2 = torch.zeros((emu_size + 1,),device=device, dtype = dtype)

    met_emu_1[:met_emu_size_1+1] = emu_states[met_emu_size_1][metid_1]
    met_emu_2[:met_emu_size_2+1] = emu_states[met_emu_size_2][metid_2]

    
    # for each reaction 
    #for i in range(met1_derivative.shape[0]): 
        # convolve the MIDs 
    conv_derivative_1 = conv(met1_derivative , met_emu_2.unsqueeze(-1), emu_size, device, dtype) 
     
    conv_derivative_2 = conv(met2_derivative , met_emu_1.unsqueeze(-1), emu_size, device, dtype) 

    return conv_derivative_1 + conv_derivative_2


def update_derivative_states(derivative_states, dX, emu_size,core_met_ix): 
    for i, met in enumerate(core_met_ix[emu_size]):
        derivative_states[emu_size][core_met_ix[emu_size][met]] = dX[:,i]


def update_states(emu_states, X, emu_size, core_met_ix):
    for i, met in enumerate(core_met_ix[emu_size]):
        # set for each emu size and each metabolite, the labels for each MID 
        emu_states[emu_size][core_met_ix[emu_size][met]] = X[i]

    
def extract_input(mid_inputs, origin):
    return mid_inputs[origin.ix, :]


def get_labeling(metabolite_hash, emu_states, emu_met_filter, core_met_ix, device, dtype=torch.float64):
    """
    emu_states: list(torch.tensor)
    emu_met_filter: list(torch.tensor)
    """
    labeling = torch.zeros((len(emu_states)+1, len(metabolite_hash),), dtype=dtype, device=device) 
    for subnet in emu_met_filter: 
        for met in emu_met_filter[subnet]: 
            if met in core_met_ix[subnet]: 
                labeling[:(subnet+1), metabolite_hash[met.id]] = emu_states[subnet][core_met_ix[subnet][met]]
    #return sum([es * emf for es, emf in zip(emu_states, emu_met_filter)])
    return labeling


def get_derivative_labeling(metabolite_hash, derivative_states, emu_met_filter, core_met_ix,device, dtype): 
    derivative_labeling = torch.zeros((len(derivative_states)+1, len(metabolite_hash), derivative_states[1].shape[1]),device=device, dtype=dtype)
    for subnet in emu_met_filter:
        for met in emu_met_filter[subnet]:
            if met in core_met_ix[subnet]:
                derivative_labeling[:(subnet+1), metabolite_hash[met.id], :] = derivative_states[subnet][core_met_ix[subnet][met]].T
    return derivative_labeling
    # return sum([torch.einsum('enp, pe -> pen', es, emf) for es, emf in zip(derivative_states, emu_met_filter)])



def build_emu_met_filter(model, metabolites, device=None):
    """

    """
    #emu_met_filter = [0.0 * torch.ones(model.max_mid_size, model.max_emu_mets, dtype=torch.float32, device=device)
    #                  for _ in range(model.max_emu_size)]
    emu_met_filter ={i+1: [] for i in range(model.max_emu_size)}
    for met in metabolites:
        for i in range(model.expansion.emu_network.max_size, 0, -1):
            subnet = model.expansion.emu_network.subnetworks[i]
            # I want to modify this code to check if emu._values[0] has a size equal to subnet size
            # if so, then I want to add the metabolite to the emu_met_filter
            # selects the emus in the subnet 
            emu_mets = []
            for emu in subnet.nodes(): 
                if isinstance(emu, EMU): 
                    emu_mets.append(emu)
                else: 
                    # Walk thtugh all values off the 
                    for emu_values in emu._values:
                        if emu_values.size == i: 
                            emu_mets.append(emu_values)
            found_met = False 
            for j, emu_met in enumerate(emu_mets): 
                if emu_met.id == met.id: 
                    emu_met_filter[i].append(emu_mets[j])
                    found_met = True
                    break
            if found_met: 
                break
            
    return emu_met_filter


def gen_SVD_kernel(S):
    U, D, V = np.linalg.svd(S)
    T = V[np.max(np.where(D > EPSILON)[0]) + 1:, :].T
    return T
