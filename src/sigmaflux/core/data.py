import torch
import numpy as np

from sigmaflux.expansion.emu import EMU

DEFAULT_MID_FIELDS = {'id': 'id', 'mid': 'mid', 'mean': 'mean', 'std': 'std'}

DEFAULT_FLUX_FIELDS = {'id': 'reaction', 'mean': 'mean', 'std': 'std'}


class MassIsotopomerData(list):

    def __init__(self, model=None, table=None, annotation=DEFAULT_MID_FIELDS):
        if model is not None and table is not None: 
            metabolites_str_ids = table[annotation['id']].unique()
            values = []

            for met_id in metabolites_str_ids:
                try:
                    the_metabolite = model.metabolites.get_by_id(met_id)
                    met_data = table[table[annotation['id']] == met_id]
                    mid = met_data[annotation['mid']].values
                    # check if mean is nan: 
                    if np.isnan(met_data[annotation['mean']]).any():
                        raise ValueError(f"Mean contains NaN for metabolite {met_id}")
                    mean = met_data[annotation['mean']].values
                    std = met_data[annotation['std']].values
                    entry = MassIsotopomerDataEntry(the_metabolite, mid, mean, std)
                    values.append(entry)
                except KeyError:
                    Warning(f"Metabolite {met_id} not found in model")
                    continue
                except ValueError as ve:
                    Warning(str(ve))
                    continue

            super().__init__(values)
        else: 
            super().__init__()
    
    @classmethod
    def from_list(cls, entries):
        obj = cls()  # Create a new instance without using the standard __init__ parameters
        list.__init__(obj, entries)  # Directly initialize the list part of the object
        return obj

    def __add__(self, other):
        if not isinstance(other, MassIsotopomerData):
            raise ValueError("Operand must be an instance of MassIsotopomerData")
        combined_entries = list(self) + list(other)
        return MassIsotopomerData.from_list(combined_entries)
    
    def __radd__(self, other):
        if other == 0:
            return self
        if not isinstance(other, MassIsotopomerData):
            raise ValueError("Operand must be an instance of MassIsotopomerData")
        combined_entries = list(other) + list(self)
        return MassIsotopomerData.from_list(combined_entries)


class MassIsotopomerDataEntry(object):
    def __init__(self, metabolite, mid, mean, std):
        self.metabolite = metabolite
        self.emu = EMU(metabolite, make_emu_from_mid(mid))
        self._mid = mid
        self._mean = mean
        self._std = std
    # TODO Do we need properties for convinience


def make_emu_from_mid(mid):
    return list(range(1, len(mid)))


class FluxData(list):
    def __init__(self, model, table, annotation=DEFAULT_FLUX_FIELDS):
        flux_str_ids = table[annotation['id']].unique()
        values = []

        for flux_id in flux_str_ids:
            the_reactions = model.reactions.get_by_id(flux_id)
            flux_data = table[table[annotation['id']] == flux_id]
            mean = flux_data[annotation['mean']].values
            std = flux_data[annotation['std']].values
            entry = FluxDataEntry(the_reactions, mean, std)
            values.append(entry)

        super().__init__(values)


class FluxDataEntry(object):
    def __init__(self, reaction, mean, std):
        self.reaction = reaction
        # Ensure mean and std are scalars
        if isinstance(mean, (list, np.ndarray)):
            if len(mean) != 1:
                raise ValueError(f"Mean for reaction {reaction.id} must be a scalar, got array of length {len(mean)}")
            mean = float(mean[0])
        if isinstance(std, (list, np.ndarray)):
            if len(std) != 1:
                raise ValueError(f"Std for reaction {reaction.id} must be a scalar, got array of length {len(std)}")
            std = float(std[0])
        self._mean = mean
        self._std = std

# do not output mid = 0
def get_model_labeling_data(model, experimental_scaling, with_std=False, exp_ix=None):
    indices = []
    mids = []
    mid_std = []
    scalings = []
    
    if exp_ix is None:
        exp_ix = range(len(model.experiments))

    #for e_ix, e in enumerate(model.experiments):
    for e_ix, _e_ix in enumerate(exp_ix):
        e = model.experiments[_e_ix]
        for c in e.mid_data:
            c_ix = model._metabolite_hash[c.metabolite.id]
            for m_ix, m, s in zip(c._mid, c._mean, c._std):
                # only want to fit mid+1 onwards, skipping +0 
                if m_ix == 0: 
                    continue
                ix = create_data_index(model, e_ix, m_ix, c_ix)
                indices.append(ix)
                mids.append(m)
                mid_std.append(s)
                scalings.append(experimental_scaling[e_ix])
    if with_std:
        return torch.tensor(indices, device=model.device), \
               torch.tensor(mids, dtype=model.dtype, device=model.device), \
               torch.tensor(scalings, dtype=model.dtype, device=model.device), \
               torch.tensor(mid_std, dtype=model.dtype, device=model.device)
    else:
        return torch.tensor(indices, device=model.device), \
               torch.tensor(mids, dtype=model.dtype, device=model.device), \
               torch.tensor(scalings, dtype=model.dtype, device=model.device)


def create_data_index(model, e, m, c):
    max_m = model.max_mid_size
    max_c = len(model._metabolite_hash)
    # Convert 1D to 3D index
    # index_1d = index_3d[0] * (size_2 * size_3) + index_3d[1] * size_3 + index_3d[2]
    return e * max_m * max_c + max_c * m + c


def get_model_flux_data(model, with_std=False, exp_ix=None):
    indices = []
    fluxes = []
    fluxes_std = []

    if exp_ix is None:
        exp_ix = range(len(model.experiments))

    # A measured flux is a property of the culture, not of the tracer, so the same
    # FluxData is attached to every per-tracer Experiment. Without this guard each
    # measurement enters once per experiment -- 22 measured fluxes became 66 rows
    # on the 3-tracer T-cell setup. The fit itself is nearly immune (the objective
    # divides the flux block by its own count), but every raw chi-square built from
    # these indices was 3x over-weighted, and so was the n_fluxes that sets the
    # acceptance threshold's effective degrees of freedom.
    seen = set()
    for e_ix, _e_ix in enumerate(exp_ix):
        e = model.experiments[_e_ix]
        for f in e.fluxes:
            if f.reaction.id in seen:
                continue
            seen.add(f.reaction.id)
            f_ix = model._flux_hash[f.reaction.id]
            indices.append(f_ix)
            fluxes.append(f._mean)
            fluxes_std.append(f._std)

    fluxes = np.array(fluxes)
    fluxes_std = np.array(fluxes_std)

    if with_std:
        return torch.tensor(indices, device=model.device), \
               torch.tensor(fluxes, dtype=model.dtype, device=model.device), \
               torch.tensor(fluxes_std, dtype=model.dtype, device=model.device)
    else:
        return torch.tensor(indices, device=model.device), \
               torch.tensor(fluxes, dtype=model.dtype, device=model.device)
