import matplotlib.pyplot as plt 
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.data import get_model_labeling_data, get_model_flux_data
import torch 
import numpy as np

# vary epoch size to find the correct epoch size
def vary_epoch(model, optim, loss_fn, lam1, lam2, labeling_scaler, epochs, plot = False, with_replicates = False, n_reps = 5):
    experimental_scaling = torch.ones(len(model.experiments), device=model.device)
    index_labeling, labeling_mean, _ = get_model_labeling_data(model, experimental_scaling)
    if not with_replicates: 
        n_reps = 1
    labeling_diff = np.zeros((len(epochs), n_reps))
    for i,epoch in enumerate(epochs):
        for j in range(n_reps): 
            model.reset_initial_fluxes()
            fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = labeling_scaler)
            labeling, _ = model()
            simulated_labeling = torch.take(labeling, index_labeling)
            labeling_diff[i,j] = ((simulated_labeling - labeling_mean).mean().item())
    if plot: 
        plt.title("Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MID Difference")
        labeling_diff_mean = np.mean(labeling_diff, axis = 1)
        
        if with_replicates:
            labeling_diff_std = np.std(labeling_diff, axis = 1)
            plt.errorbar(epochs, labeling_diff_mean, yerr = labeling_diff_std, fmt = "bo-")
        else: 
            plt.plot(epochs, labeling_diff_mean, "bo-")
        plt.show()
    return epochs, labeling_diff
    

def vary_l1(model, optim, loss_fn, lam1s, lam2, labeling_scaler, epoch=100,  plot = False,with_replicates = False, n_reps = 5): 
    experimental_scaling = torch.ones(len(model.experiments), device=model.device)
    index_labeling, labeling_mean, _ = get_model_labeling_data(model, experimental_scaling)
    if not with_replicates: 
        n_reps = 1
    labeling_diff = np.zeros((len(lam1s), n_reps))
    for i,lam1 in enumerate(lam1s):
        for j in range(n_reps): 
            model.reset_initial_fluxes()
            fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = labeling_scaler)
            labeling, _ = model()
            simulated_labeling = torch.take(labeling, index_labeling)
            labeling_diff[i,j] = ((simulated_labeling - labeling_mean).mean().item())
    plt.title("L1 Regularization")
    plt.xlabel("L1 Parameter")
    plt.ylabel("MID Difference")
    labeling_diff_mean = np.mean(labeling_diff, axis = 1)
    if plot: 
        if with_replicates:
            labeling_diff_std = np.std(labeling_diff, axis = 1)
            plt.errorbar(lam1s, labeling_diff_mean, yerr = labeling_diff_std, fmt = "bo-")
        else: 
            plt.plot(lam1s, labeling_diff_mean, "bo-")
        plt.show()
    return lam1s, labeling_diff

# write a function that varies l2 regularization parameter
def vary_l2(model, optim, loss_fn, lam1, lam2s,labeling_scaler, epoch=100,  plot = False,with_replicates = False, n_reps = 5): 
    experimental_scaling = torch.ones(len(model.experiments), device=model.device)
    index_labeling, labeling_mean, _ = get_model_labeling_data(model, experimental_scaling)
    if not with_replicates: 
        n_reps = 1
    labeling_diff = np.zeros((len(lam2s), n_reps))
    for i,lam2 in enumerate(lam2s):
        for j in range(n_reps): 
            model.reset_initial_fluxes()
            fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = labeling_scaler)
            labeling, _ = model()
            simulated_labeling = torch.take(labeling, index_labeling)
            labeling_diff[i,j] = ((simulated_labeling - labeling_mean).mean().item())
    plt.title("L2 Regularization")
    plt.xlabel("L2 Parameter")
    plt.ylabel("MID Difference")
    labeling_diff_mean = np.mean(labeling_diff, axis = 1)
    if plot: 
        if with_replicates:
            labeling_diff_std = np.std(labeling_diff, axis = 1)
            plt.errorbar(lam2s, labeling_diff_mean, yerr = labeling_diff_std, fmt = "bo-")
        else: 
            plt.plot(lam2s, labeling_diff_mean, "bo-")
        plt.show()
    return lam2s, labeling_diff

# write a function to vary scaling parameter 
def vary_scaling(model, optim, loss_fn, lam1, lam2, labeling_scalers, epoch=100, plot = False, with_replicates = False, n_reps = 5): 
    experimental_scaling = torch.ones(len(model.experiments), device=model.device)
    index_labeling, labeling_mean, _ = get_model_labeling_data(model, experimental_scaling)
    if not with_replicates: 
        n_reps = 1
    labeling_diff = np.zeros((len(labeling_scalers), n_reps))
    for i,labeling_scaler in enumerate(labeling_scalers):
        for j in range(n_reps): 
            model.reset_initial_fluxes()
            fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = labeling_scaler)
            labeling, _ = model()
            simulated_labeling = torch.take(labeling, index_labeling)
            labeling_diff[i,j] = ((simulated_labeling - labeling_mean).mean().item())
    plt.title("Scaling")
    plt.xlabel("Scaling Parameter")
    plt.ylabel("MID Difference")
    labeling_diff_mean = np.mean(labeling_diff, axis = 1)
    if plot:
        if with_replicates:
            labeling_diff_std = np.std(labeling_diff, axis = 1)
            plt.errorbar(labeling_scalers, labeling_diff_mean, yerr = labeling_diff_std, fmt = "bo-")
        else: 
            plt.plot(labeling_scalers, labeling_diff_mean, "bo-")
        plt.show()
    return labeling_scalers, labeling_diff

def comparing_scaling(model, optim, loss_fn, lam1, lam2, labeling_scale, epoch=100, plot = False, n_reps = 10):
    flux_residuals_no_scale = np.zeros(n_reps)
    flux_residuals_scale = np.zeros(n_reps)
    labeling_residuals_no_scale = np.zeros(n_reps)
    labeling_residuals_scale = np.zeros(n_reps)

    loss_no_scale = np.zeros(n_reps, epoch)
    loss_scale = np.zeros(n_reps,epoch)

    experimental_scaling = torch.ones(len(model.experiments), device=model.device)
    index_labeling, labeling_mean, _ = get_model_labeling_data(model, experimental_scaling)
    index_fluxes, flux_mean = get_model_flux_data(model)

    for i in range(n_reps):
        model.reset_initial_fluxes()
        log = fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = 1)
        simulated_fluxes = torch.take(model.fluxes, index_fluxes)
        labeling, _ = model()
        simulated_labeling = torch.take(labeling, index_labeling)
        flux_residuals_no_scale[i] = (simulated_fluxes - flux_mean).mean()
        labeling_residuals_no_scale[i] = (simulated_labeling - labeling_mean).mean()
        loss_no_scale[i,:] = log

        model.reset_initial_fluxes()
        log = fit_mfa_model(model, loss_fn, optim, epochs=epoch, lam1=lam1 , lam2=lam2, labeling_scaler = labeling_scale)
        simulated_fluxes = torch.take(model.fluxes, index_fluxes)
        labeling, _ = model()
        simulated_labeling = torch.take(labeling, index_labeling)
        flux_residuals_scale[i] = (simulated_fluxes - flux_mean).mean()
        labeling_residuals_scale[i] = (simulated_labeling - labeling_mean).mean()
        loss_scale[i,:] = log
    values = [labeling_residuals_scale.mean(), labeling_residuals_no_scale.mean(), flux_residuals_scale.mean(), flux_residuals_no_scale.mean()]
    errors = [labeling_residuals_scale.std(), labeling_residuals_no_scale.std(), flux_residuals_scale.std(), flux_residuals_no_scale.std()]
    
    if plot: 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        quantities = ['Label Resid. w/ Scale', 'Label Resid. w/o Scale', 'Flux Resid. w/ Scale', 'Flux Resid. w/o Scale']
        ax1.bar(quantities, values, yerr=errors, alpha=0.7, color='blue')
        ax1.set_ylabel('Residual Values')
        ax1.set_title('Residuals')

        ax2.errorbar(np.arange(epoch), loss_scale.mean(axis=1),yerr=loss_scale.std(axis=1), label='Loss Scaled', color = "Red")
        ax2.errorbar(np.arange(epoch), loss_no_scale.mean(axis=1), yerr=loss_no_scale.std(axis=1), label='Loss Unscaled', color = "green")
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch Number')
        ax2.set_title('Loss vs Epoch')
        ax2.legend()
    return values, errors, loss_scale, loss_no_scale
