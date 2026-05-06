import torch
import pandas as pd
import os

# Checkpoint file keys for MFA model
#KEYS = ["_forward_fluxes", "_backward_fluxes", "_lambda", "_mu"]
KEYS = ["_fluxes", "_reversibility", "_lambda", "_mu"]

def get_dataframe_key(checkpoint, key):
    df = pd.DataFrame()
    for i in range(len(checkpoint)):
        if isinstance(checkpoint[i][key], torch.Tensor):
            df = pd.concat([df, pd.DataFrame(checkpoint[i][key].cpu().numpy())], axis=1)
        else:
            df = pd.concat([df, pd.DataFrame([checkpoint[i][key],])], axis=1)
    
    return df.T

def load_checkpoint(file, keys=KEYS):
    checkpoint = torch.load(file)[1:]
    output = dict()
    for k in keys:
        output[k] = get_dataframe_key(checkpoint, k)
        # Add epoch columns = range(len(chekcpoint))
        output[k]["epoch"] = range(len(checkpoint))
    return output

def load_all_checkpoints(filepattern, keys=KEYS):
    # Load all checkpoint files folloing a string patter i.e "./chkpt_cluster_run_mfa_liver_*.pt"    
    directory = os.path.dirname(filepattern) + "/"
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    output = dict()
    for file in files:
        if file.endswith(".pt"):
            checkpoint = load_checkpoint(directory + file, keys=keys)
            for k in keys:
                if k in output:
                    output[k] = pd.concat([output[k], checkpoint[k]], axis=0, ignore_index=True)
                else:
                    output[k] = checkpoint[k]
    return output
       

    
