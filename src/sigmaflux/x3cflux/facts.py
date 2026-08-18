"""Reduce a prepared cobra + MFA model to the JSON the converter reads.

The converter deliberately takes a plain dict rather than live cobra/torch
objects. That keeps :mod:`sigmaflux.x3cflux.convert` importable without torch --
which matters, because x3cflux bundles its own libstdc++/BLAS and can fail to load
in a process that has already imported torch, so the fit runs in a subprocess that
imports numpy and x3cflux only.

The one thing here that cannot be re-derived downstream is the EMU network's
*input set*: it is read off the prepared :class:`~sigmaflux.core.model.MFAModel`
as the boundary nodes of the expansion, not inferred from stoichiometry.
"""

import json


def model_facts(cobra_model, mfa_model, mapped_met_ids, cfg):
    """Reaction/metabolite facts plus the EMU network's input set, as a dict.

    Args:
        cobra_model: the prepared cobra model.
        mfa_model: the built :class:`~sigmaflux.core.model.MFAModel`.
        mapped_met_ids: metabolite ids with a measurement mapping.
        cfg: run config, for labeling tracers and measured fluxes.
    """
    reactions = [{
        "id": r.id,
        "lower_bound": r.lower_bound,
        "upper_bound": r.upper_bound,
        # cobra's Reaction.reversibility is lb < 0 < ub, and that is exactly what
        # MFAModel uses for _reversibility_mask.
        "reversible": bool(r.reversibility),
        "atom_mapping": r.annotation.get("atom_mapping", ""),
        "terminal": bool(r.annotation.get("terminal", False)),
        "stoichiometry": {m.id: c for m, c in r.metabolites.items()},
    } for r in cobra_model.reactions]

    metabolites = [{
        "id": m.id,
        "atom_pattern": m.annotation.get("atom_pattern", ""),
        "symmetries": list(m.annotation.get("symmetries", [])),
        "terminal": bool(m.annotation.get("terminal", False)),
    } for m in cobra_model.metabolites]

    # The EMU network's boundary nodes -- those with no incoming edge, which
    # MFAModel.forward hands to get_tracer_inputs. Only the single-element ones are
    # genuine network inputs with a fixed labeling; the multi-element OriginSets are
    # EMU-cascade convolution nodes, computed from smaller subnets
    # (get_tracer_inputs only ever assigns to len == 1). Collapsing the two would
    # wrongly pin g3p_c, dhap_c, accoa_m, co2_m and friends to pure M+0.
    boundaries = [mfa_model._emu_hashes[subnet][b.item()]
                  for subnet in mfa_model._ix_b for b in mfa_model._ix_b[subnet]]
    inputs, convolutions = [], []
    for b in boundaries:
        values = b._values if hasattr(b, "_values") else [b]
        if len(values) == 1:
            inputs.append({"metabolite": values[0].metabolite,
                           "mid": list(values[0].mid)})
        else:
            convolutions.append({"components": [
                {"metabolite": v.metabolite, "mid": list(v.mid)} for v in values]})

    tracers = [{"metabolite": t["tracer_id"], "labeling": list(t["labeling"]),
                "enrichment": float(t.get("enrichment", 1.0)),
                "key": f"{t['tracer_id']}_{list(t['labeling'])}_"
                       f"{float(t.get('enrichment', 1.0))}"}
               for t in cfg["labeling"]["tracers"]]

    return {
        "reactions": reactions,
        "metabolites": metabolites,
        "input_emus": inputs,
        "input_metabolites": sorted({b["metabolite"] for b in inputs}),
        "convolution_nodes": convolutions,
        "observable_metabolites": sorted(mapped_met_ids),
        "measured_fluxes": list(cfg.get("training", {}).get("measured_fluxes", [])),
        "tracers": tracers,
        "min_rev_flux": float(mfa_model.min_rev_flux),
    }


def dump_model_facts(cobra_model, mfa_model, mapped_met_ids, cfg, path, verbose=True):
    """Write :func:`model_facts` to *path* and return it.

    Args:
        cobra_model: the prepared cobra model.
        mfa_model: the built :class:`~sigmaflux.core.model.MFAModel`.
        mapped_met_ids: metabolite ids with a measurement mapping.
        cfg: run config, for labeling tracers and measured fluxes.
        path: output file path.
        verbose: print a summary of what was written.
    """
    facts = model_facts(cobra_model, mfa_model, mapped_met_ids, cfg)
    with open(path, "w") as fh:
        json.dump(facts, fh, indent=1)
    if verbose:
        print(f"  wrote {path}")
        print(f"    {len(facts['reactions'])} reactions, "
              f"{len(facts['metabolites'])} metabolites")
        print(f"    {len(facts['input_emus'])} input EMUs over "
              f"{len(facts['input_metabolites'])} metabolites, "
              f"{len(facts['convolution_nodes'])} convolution nodes")
    return facts
