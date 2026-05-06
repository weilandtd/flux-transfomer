import matplotlib.pyplot as plt


from sigmaflux.core.experiment import Experiment
import torch
import numpy as np 
# Function to generate a MassIsotopomerData plot as a bar plot with error bars

def plot_mass_isotopomer_data(experiment: Experiment, metabolite_id: str, show: bool = True) -> plt.Axes:
    """
    Function to plot all MassIsotopomerData from one experiment in a grid plot 

    parameters
    ----------
    experiment: Experiment
        Experiment object containing the MassIsotopomerData to be plotted

    metabolite_id: str
        The id of the metabolite to be plotted

    returns
    -------
    None
    """
    # Get the MassIsotopomerData for the metabolite
    mid_data = {m.metabolite.id: m for m in experiment.mid_data}
    data = mid_data[metabolite_id]

    # Plot one bar graph for the isotpomers 
    fig, ax = plt.subplots()
    ax.bar(data._mid[1:], data._mean[1:], yerr=data._std[1:], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Fractional enrichment')
    ax.set_xlabel('Mass Isotopomer')
    ax.set_title(f'{metabolite_id}')    

    if show:
        plt.show()
    return ax 
    

# Plot each MassIsotopomerData for each metabolite in a grid plot
def plot_mass_isotopomer_data_grid(experiment: Experiment, show: bool = True) -> plt.Axes:
    """
    Function to plot all MassIsotopomerData from one experiment in a grid plot 

    parameters
    ----------
    experiment: Experiment
        Experiment object containing the MassIsotopomerData to be plotted

    returns
    -------
    None
    """
    # Use tracer enrichments to inform ylim
    tracer_enrichment = max([t.serum_enrichment for t in experiment.tracers])
    

    # Get the MassIsotopomerData for the metabolite
    mid_data = {m.metabolite.id: m for m in experiment.mid_data}

    # Plot one bar graph for the isotpomers 
    # use 4 columns for the plot
    n_cols = 4
    n_rows = len(mid_data) // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    # make a figure title
    tracer_metabolites = [t.metabolite.id for t in experiment.tracers]
    fig.suptitle(f"Tracers :{tracer_metabolites}")

    axs = axs.flatten()

    for i, met_id in enumerate(mid_data):
        data = mid_data[met_id]
        ax = axs[i]
        ax.bar(data._mid[1:], data._mean[1:], yerr=data._std[1:], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Fractional enrichment')
        ax.set_xlabel('Mass Isotopomer')
        ax.set_title(f'{met_id}')    
        ax.set_ylim(0, tracer_enrichment * 1.2)

    if show:
        plt.show()
    return axs

def plot_met_labeling(mfa_model, metabolite, labeling, default_num_mid=7, show = True): 
    # Determine the number of experiments to create subplots
    len(mfa_model.experiments)

    # Setup the figure and subplots
    fig, axs = plt.subplots(4, 2, figsize=(8, 9))
    axs = axs.flatten()
    for i, exp in enumerate(mfa_model.experiments):
        tracer_enrichment = max([t.serum_enrichment for t in exp.tracers])
        found = False 
        for d in exp.mid_data:
            if d.metabolite.id == metabolite:
                found = True 
                exp_mean = d._mean[1:]
                exp_std = d._std[1:]
                color = "orange"
                break 
        if not found: 
            exp_mean = torch.zeros(7)
            exp_std = torch.zeros(7)
            color = "red"

        
        calc_mean = labeling[i, 1:len(exp_mean)+1, mfa_model._metabolite_hash[metabolite]].detach().numpy()
        #calc_std =  all_labeling[:,i, 1:len(exp_mean)+1, mfa_model._metabolite_hash[metabolite]].std(axis=0).detach().numpy()
        n_pairs = len(exp_mean)
        labels = [f'M+{j+1}' for j in range(n_pairs)]
        
        # Positions of the bars
        x = np.arange(n_pairs)
        
        # Width of a bar
        width = 0.35
        
        # Plotting exp_values and calc_values side by side
        axs[i].bar(x - width/2, exp_mean,yerr=exp_std, width=width, label='Exp Value', capsize=5)
        axs[i].bar(x + width/2, calc_mean, width=width, label='Calc Value',capsize=5, color = color)
        
        # Setting the x-axis labels to M+0, M+1, ..., and adding legend and titles
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=45)
        axs[i].set_title(f'Experiment {i+1} {list(exp._tracers.keys())[0]}')
        axs[i].set_ylim(0, tracer_enrichment * 1.2)

        if i == 0:
            axs[i].set_ylabel('Values')
    fig.suptitle(metabolite)

    plt.tight_layout()
    if show:
        plt.show()
    return axs

def plot_exp_labeling(mfa_model, exp_num, labeling, default_num_mid=7, show = True, filename = "exp_labeling.png", figsize = (15,15)):
    experiment = mfa_model.experiments[exp_num]
    # Determine the number of experiments to create subplots
    mid_data = {m.metabolite.id: m for m in experiment.mid_data}
    n_cols =4 
    n_rows = len(mid_data) // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)# Setup the figure and subplots
    axs = axs.flatten()
    tracer_enrichment = max([t.serum_enrichment for t in experiment.tracers])
    tracer_metabolites = [t.metabolite.id for t in experiment.tracers]
    fig.suptitle(f"Tracers :{tracer_metabolites}")

    for i, met_id in enumerate(mid_data):
        data = mid_data[met_id]._mean[1:]
        std_data = mid_data[met_id]._std[1:]
        calc_data = labeling[exp_num,1:len(data)+1,mfa_model._metabolite_hash[met_id]].detach().numpy() 

        n_pairs = len(data)
        labels = [f'M+{j+1}' for j in range(n_pairs)]
        
        # Positions of the bars
        x = np.arange(n_pairs)
        
        # Width of a bar
        width = 0.35
        ax = axs[i]
        # Plotting exp_values and calc_values side by side
        ax.bar(x - width/2, data,yerr=std_data,width= width, label='Exp Value', capsize=5)
        ax.bar(x + width/2, calc_data,width= width, label='Calc Value', capsize=5)
        
        # Setting the x-axis labels to M+0, M+1, ..., and adding legend and titles
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f'{met_id}')    
        
        if i == 0:
            axs[i].set_ylabel('Values')
        ax.set_ylim(0, tracer_enrichment * 1.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    return axs
