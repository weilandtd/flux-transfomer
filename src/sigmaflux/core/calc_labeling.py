import torch 
from sigmaflux.core.mfa import get_labeling, filter_conv_inputs, \
    convolution, update_states, update_derivative_states, get_derivative_labeling, convolution_derivative, filter_conv_inputs_with_indices
from sigmaflux.expansion.emu import get_metabolite
import logging
from sigmaflux.core.utils import multiply_3d_sparse_with_2d

logger = logging.getLogger(__name__)
import time

def diagonalize_flux(v_f, v_r):
        diag_f = torch.diag(v_f)
        diag_r = torch.diag(v_r)

        # Concatenate the two matrices
        result = torch.cat((diag_f, torch.zeros(diag_f.size(0), diag_r.size(1))), dim=1)
        result = torch.cat((result, torch.cat((torch.zeros(diag_r.size(0), diag_f.size(1)), diag_r), dim=1)), dim=0)

        return result

def get_subnet_fluxes(fluxes, edge_reaction_index): 
    v = dict() 
    for subnet in edge_reaction_index: 
        v[subnet] = fluxes[edge_reaction_index[subnet]]
    return v

class CalcLabeling(torch.autograd.Function): 
    # Y_0 = get_tracer_inputs already
    # emu_hashes converts from index of emu in subnet to emu 
    # all_netabolite_hash converts from id to index of all metabolites
    @staticmethod
    def forward(reversible_fluxes, forward_dict, tracer_boundary_inputs, emu_states, derivative_states, device, dtype): 
        # emu_states = [0.0 * torch.ones(len(S)+1, len(all_metabolite_hash), dtype=dtype, device=device)
        #               for _ in range(len(S))]
        all_X = {}
        all_Y = {}

        for subnet in forward_dict["S"]: 
            if forward_dict["S"][subnet].shape[1] == 0:
                continue
            # calculate A and B
            start_time = time.time()
           
            emu_inputs = [forward_dict["emu_hashes"][subnet][j.item()] for j in forward_dict["ix_b"][subnet]]
                        
            # calculate convolution inputs
            conv_inputs = filter_conv_inputs(emu_inputs)
            conv_input_values = {c: convolution(forward_dict["core_met_ix"], emu_states, c, device, dtype) for c in conv_inputs}
            
            # get all boundary values 
            all_input_values = conv_input_values.copy()
            all_input_values.update(tracer_boundary_inputs)

            # calculate Y 
            Y_list = [all_input_values[inpt] for inpt in emu_inputs]

            if len(Y_list) == 0:
                continue
            
            Y = torch.stack(Y_list, dim=0)

            # Solve the forward EMU problem
            # A X = B Y -> solve for matrix X
            # First compute A-1 B
            A_inv_B = torch.matmul(forward_dict["A_inv"][subnet], forward_dict["B"][subnet])
            # Clip negative fluxes in A_inv_B to zero
            X = torch.clamp(A_inv_B, min=0.0) @ Y

            all_X[subnet] = X
            all_Y[subnet] = Y

            update_states(emu_states, X, subnet, forward_dict["core_met_ix"])
            
            end_time = time.time()
            logger.debug(f"Completed forward of subnet {subnet} in {end_time - start_time:.5f} seconds")
       
        labeling = get_labeling(forward_dict["metabolite_hash"], emu_states, forward_dict["emu_met_filter"], forward_dict["core_met_ix"], device=device, dtype=dtype)
        return labeling, all_X, all_Y, emu_states

    @staticmethod 
    def setup_context(ctx, inputs, output): 
        fluxes, forward_dict, tracer_boundary_inputs, _, derivative_states, device, dtype = inputs 
        _, all_X, all_Y, emu_states = output
        ctx.tracer_boundary_inputs = tracer_boundary_inputs
        ctx.device = device
        ctx.dtype = dtype
        ctx.all_X = all_X
        ctx.all_Y = all_Y
        ctx.forward_dict = forward_dict
        ctx.emu_states = emu_states
        ctx.derivative_states = derivative_states
        ctx.save_for_backward(fluxes)
    
    # I would rather not pass grad_all_X and on 
    @staticmethod 
    def backward(ctx, grad_labeling, grad_all_X, grad_all_Y, grad_emu_states): 
        fluxes = ctx.saved_tensors[0]
        tracer_boundary_inputs = ctx.tracer_boundary_inputs
        device = ctx.device
        dtype = ctx.dtype
        all_X = ctx.all_X
        all_Y = ctx.all_Y
        forward_dict = ctx.forward_dict
        emu_states = ctx.emu_states
        derivative_states = ctx.derivative_states
        grad_v = None
        for derivative_subnet in derivative_states: 
            derivative_states[derivative_subnet].zero_()
        if ctx.needs_input_grad[0]: 
            for subnet in forward_dict["S"]: 
                start_time = time.time()
                if (forward_dict["S"][subnet].shape[1] == 0) or \
                    (subnet not in all_X) or (subnet not in all_Y):
                    continue
                #dBY = torch.einsum('cbn, bp->cpn',  forward_dict["dB"][subnet], all_Y[subnet])
                dBY = multiply_3d_sparse_with_2d(forward_dict["dB"][subnet], all_Y[subnet])  # (number of reactions * 2) x (c) x (p)
                #dAX = torch.einsum('bcn, bp->cpn',  forward_dict["dA"][subnet], all_X[subnet])
                dAX = multiply_3d_sparse_with_2d(forward_dict["dA"][subnet], all_X[subnet]) # (number of reactions * 2) x (c) x (p)
                
                if subnet == 1: 
                    dX = torch.matmul(forward_dict["A_inv"][subnet], dBY - dAX)
                else: 
                    emu_inputs = [forward_dict["emu_hashes"][subnet][j.item()] for j in forward_dict["ix_b"][subnet]]
                    # calculate convolution inputs
                    conv_inputs, conv_indices = filter_conv_inputs_with_indices(emu_inputs)
                    if len(conv_inputs) > 0: 
                        ### this could be made more efficient, should sparsify dY!
                        conv_derivatives = {c: convolution_derivative(forward_dict["core_met_ix"], derivative_states,emu_states, c, device=device, dtype = dtype) for c in conv_inputs}
                        all_Y_derivatives = conv_derivatives.copy()
                        all_Y_derivatives.update({e: torch.zeros(len(fluxes), subnet+1 , dtype=dtype, device=device) for e in tracer_boundary_inputs})
                        dY_list = [all_Y_derivatives[inpt] for inpt in emu_inputs]
                        dY = torch.stack(dY_list, dim=0)
                        ###
                        BdY = torch.einsum('cb, bnp -> ncp', forward_dict["B"][subnet], dY)
                        dX = torch.matmul(forward_dict["A_inv"][subnet], BdY + dBY - dAX)
                    else: 
                        dX = torch.matmul(forward_dict["A_inv"][subnet], dBY - dAX)
                update_derivative_states(derivative_states, dX, subnet, forward_dict["core_met_ix"])
                end_time = time.time()
                logger.debug(f"Completed backward of subnet {subnet} in {end_time - start_time:.5f} seconds")
            
            grad_v = torch.einsum('pcn,pc->n', get_derivative_labeling(forward_dict["metabolite_hash"], derivative_states, forward_dict["emu_met_filter"], forward_dict["core_met_ix"], device = device, dtype = dtype) , grad_labeling)
        return grad_v, None, None, None, None, None, None


def calculate_labeling(reversible_fluxes, forward_dict, tracer_boundary_inputs, device, dtype):
     #emu_states = [0.0 * torch.ones(len(forward_dict["S"])+1, len(forward_dict["metabolite_hash"]), dtype=dtype, device=device)
     #                 for _ in range(len(forward_dict["S"]))]
     emu_states = {subnet: torch.zeros(len(forward_dict["ix_c"][subnet]),subnet+1, device=device, dtype=dtype) for subnet in forward_dict["S"]}
     derivative_states = {subnet: torch.zeros(len(forward_dict["ix_c"][subnet]), len(reversible_fluxes), subnet+1, device=device, dtype=dtype) for subnet in forward_dict["S"]}
     #[0.0 * torch.ones( len(forward_dict["metabolite_hash"]),len(reversible_fluxes),len(forward_dict["S"])+1, dtype=dtype, device=device)
      #                  for _ in range(len(forward_dict["S"]))]
     labeling, _, _, _ = CalcLabeling.apply(reversible_fluxes,forward_dict,tracer_boundary_inputs, emu_states, derivative_states, device, dtype)
     return labeling
