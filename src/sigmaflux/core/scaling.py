import torch 
class Scaling(): 
    def __init__(self, scale_by): 
        self.scale_by = scale_by
        pass 
    
    
    def scale(self, quantity, batch_idx=None): 
        if batch_idx is not None:
            # Convert range to list if needed
            if isinstance(batch_idx, range):
                batch_idx = list(batch_idx)
            return quantity / self.scale_by[batch_idx].view(-1, *([1] * (quantity.dim() - 1)))
        return quantity / self.scale_by.view(-1, *([1] * (quantity.dim() - 1)))

    

def get_default_factor(model):
    serum_enrichments = []
    for exp in model.experiments: 
        for mid_entry in exp.mid_data: 
            serum_enrichments.append(mid_entry._mean[1:].sum())
    return sum(serum_enrichments)/len(serum_enrichments)


def get_tracer_enrichments(model): 
    all_serum_enrichments = [] 
    for exp in model.experiments:
        # Get enrichments for all tracers in this experiment
        enrichments = [t.serum_enrichment for t in exp.tracers]
        
        # Check for None or NaN values
        valid_enrichments = [e for e in enrichments if e is not None and not (isinstance(e, float) and e != e)]
        
        if len(valid_enrichments) == 0:
            # If no valid enrichments, default to 1.0
            avg_enrichment = 1.0
            print(f"Warning: No valid serum enrichments found for experiment {exp.tracers}. Using default value of 1.0")
        else:
            avg_enrichment = sum(valid_enrichments) / len(valid_enrichments)
        
        all_serum_enrichments.append(avg_enrichment)
    
    return torch.tensor(all_serum_enrichments, device=model.device, dtype=model.dtype) 
            