import torch
from sigmaflux.core.data import MassIsotopomerData, FluxData
from sigmaflux.expansion.emu import EMU, OriginSet
from sigmaflux.core.utils import insert_mapping_into_reaction
from cobra import Reaction, Metabolite
import re 


class Experiment(object):
    def __init__(self, mid_data, tracers, fluxes=None, enrichment = None):
        """Bundle MID data, tracers, and optional flux data for one labeling experiment.

        Args:
            mid_data: MassIsotopomerData measured for this experiment.
            tracers: Tracer objects applied in this experiment.
            fluxes: optional measured FluxData for this experiment.
            enrichment: per-tracer enrichment overrides; defaults to each tracer's own enrichment.
        """

        if not isinstance(mid_data, MassIsotopomerData):
            raise ValueError(f'data is of type {type(mid_data)} not {MassIsotopomerData}')
        else:
            # TODO assert if the file-format matches std.
            pass

        if fluxes is None:
            fluxes = []
        elif not isinstance(fluxes, FluxData):
            raise ValueError(f'data is of type {type(fluxes)} not {FluxData}')
        else:
            # TODO
            pass

        if enrichment is None: 
            enrichment = [t.enrichment for t in tracers] 
            
        self.mid_data = mid_data
        self.fluxes = fluxes

        self._tracers = dict()
        for the_tracer in tracers:
            if the_tracer.metabolite.id not in self._tracers:
                self._tracers[the_tracer.metabolite.id] = list()
            self._tracers[the_tracer.metabolite.id].append(the_tracer)

    @property
    def tracers(self):
        return [t for m in self._tracers for t in self._tracers[m]]

    def tracer_metabolites(self):
        return [m for m in self._tracers]

    def get_tracer_by_met(self, met):
        return self._tracers[met]


class Tracer(object):
    def __init__(self, metabolite, enrichment, labeling, serum_enrichment=1.0):
        """A labeled tracer applied to a specific metabolite.

        Args:
            metabolite: metabolite carrying the label.
            enrichment: fraction of the metabolite pool that is labeled.
            labeling: per-atom labeling pattern of the tracer.
            serum_enrichment: enrichment of the tracer in the feeding serum.
        """
        self.metabolite = metabolite
        self.enrichment = enrichment
        self.labeling = labeling
        self.serum_enrichment = serum_enrichment

    def __repr__(self) -> str:
        return f"{self.metabolite}_{self.labeling}_{self.enrichment}"


def get_tracer_inputs(experiment, boundaries, device=None, dtype=torch.float64):
    """Build boundary MID input tensors for an experiment's tracers.

    Args:
        experiment: Experiment supplying the tracer definitions.
        boundaries: EMU/OriginSet boundary objects to build inputs for.
        device: torch device for the returned tensors.
        dtype: torch dtype for the returned tensors.
    """
    boundary_inputs = dict()
    for the_boundary in boundaries:
        # check if the boundary input is a tracer
        if isinstance(the_boundary, EMU) and the_boundary.metabolite in experiment._tracers:

            tracer_list = experiment.get_tracer_by_met(the_boundary.metabolite)
            boundary_inputs[the_boundary] = calculate_tracer_input(tracer_list, the_boundary, device=device, dtype=dtype)

        elif isinstance(the_boundary, OriginSet) \
                and len(the_boundary) == 1 \
                and the_boundary._values[0].metabolite in experiment._tracers:

            tracer_list = experiment.get_tracer_by_met(the_boundary._values[0].metabolite)
            boundary_inputs[the_boundary] = calculate_tracer_input(tracer_list, the_boundary._values[0], device=device, dtype=dtype)
        # if not in the experiment, make all MID 0
        elif isinstance(the_boundary, EMU) and the_boundary.metabolite not in experiment._tracers:

            tensor_input = torch.tensor([0.0, ] * (len(the_boundary.mid) + 1), device=device,dtype=dtype)
            tensor_input[0] = 1.0
            boundary_inputs[the_boundary] = tensor_input

        elif isinstance(the_boundary, OriginSet) \
                and len(the_boundary) == 1 \
                and the_boundary._values[0].metabolite not in experiment._tracers:

            tensor_input = torch.tensor([0.0, ] * (len(the_boundary._values[0].mid) + 1), dtype=dtype,
                                        device=device)
            tensor_input[0] = 1.0
            boundary_inputs[the_boundary] = tensor_input

    return boundary_inputs


def calculate_tracer_input(tracer_list, boundary, device=None, dtype = torch.float64):
    """Combine a metabolite's tracers into one boundary MID tensor.

    Args:
        tracer_list: tracers applied to the boundary's metabolite.
        boundary: EMU boundary the MID is computed for.
        device: torch device for the returned tensor.
        dtype: torch dtype for the returned tensor.
    """
    tracer_mid = [0.0 for i in range(len(boundary.mid) + 1)]
    M0 = 1.0
    for tracer in tracer_list:
        k = sum([tracer.labeling[m - 1] for m in boundary.mid])
        tracer_mid[k] += tracer.enrichment
        M0 -= tracer.enrichment
    tracer_mid[0] += M0
    return torch.tensor(tracer_mid, dtype=dtype, device=device)


def add_tracers_to_model(tracers, model, tracer_influx= 0.01, bound =20):
    """Add input/infusion reactions for each tracer metabolite, carrying its atom mapping onto the new reactions.

    Args:
        tracers: tracer metabolites to add input/infusion reactions for.
        model: cobra model to add the reactions to.
        tracer_influx: fixed flux bound set on each infusion reaction.
        bound: upper bound on each input reaction.
    """

    for tracer in tracers: 
        new_met = Metabolite(
            tracer.id[:-2] + '_t',
            name=tracer.name[:-2] + '_t'
        )
        tracer_input = Reaction(new_met.id + '_input')
        tracer_infusion = Reaction(new_met.id + '_infusion')

        tracer_input.bounds = [0,bound]

        model.add_reactions([tracer_input, tracer_infusion])
        tracer_input.add_metabolites({tracer: 1, new_met: -1})
        tracer_infusion.add_metabolites({new_met: 1})

        # exchange_reaction  = model.reactions.get_by_id(f"EX_{tracer.id[:-2]}[v]")

        for r in tracer.reactions: 
            if "atom_mapping" in r.annotation: 
                annotation = r.annotation["atom_mapping"]
                pattern = r'(?<![\w-])' + re.escape(tracer.id) + r'\s*(\([^)]+\))'
                match = re.search(pattern, annotation)
                if match: 
                    break
        if match:
            atom_map = match.group(1)
            tracer_input_map = tracer_input.reaction 
            tracer_infusion_map = tracer_infusion.reaction
            
            tracer_input_map = insert_mapping_into_reaction(tracer_input_map, tracer.id, atom_map)
            tracer_input_map = insert_mapping_into_reaction(tracer_input_map, new_met.id, atom_map)

            tracer_infusion_map = insert_mapping_into_reaction(tracer_infusion_map, new_met.id, atom_map)
            
            tracer_input.annotation["atom_mapping"] = " " + tracer_input_map
            tracer_infusion.annotation["atom_mapping"] = " " + tracer_infusion_map

            tracer_infusion.bounds = [tracer_influx, tracer_influx]