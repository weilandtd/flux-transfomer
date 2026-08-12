"""Compare the two forward models: our EMU simulation against x3cflux's.

This is gate G2, and it is the check worth running before any fit. A fit only
tells you an optimiser found a low residual for *some* model; if the two forward
models disagree, the residual is being measured against a network that is not the
one we meant, and every downstream number inherits that. Two independently written
simulators agreeing to ~1e-6 from the same flux is much stronger evidence than
either being self-consistent.

**What agreement does and does not prove.** It rules out a large class of
translation errors -- a lost balance row, a scrambled permutation, a wrong
exchange convention. It does not prove either model is *right*: a bug faithfully
reproduced on both sides shows up as agreement. :mod:`sigmaflux.x3cflux.convert`
deliberately mirrors some quirks of the prepared cobra model rather than repairing
them, so in exactly those places agreement is partly circular and the honest read
is "the conversion is faithful", not "the biology is correct". Disagreement is the
informative direction, and it does not by itself say which side is wrong -- though
x3cflux, being published and independently used, is the better prior.
"""

import numpy as np
import pandas as pd

from .build_config import build_configurations
from .check_network import assert_network
from .convert import convert
from .fluxmap import FluxMap


def measurement_slots(simulator, facts):
    """x3cflux measurement slot -> ``(tracer key, metabolite)``."""
    names, _ = simulator.measurement_names
    multi, _ = simulator.measurement_multi_indices
    keys = [t["key"] for t in facts["tracers"]]
    return [(keys[experiment], name[3:])          # strip the "ms_" prefix
            for name, experiment in zip(names, multi)]


def x3cflux_labeling(simulator, fluxmap, facts, v, r):
    """``{(tracer, metabolite, 'M+k'): value}`` simulated by x3cflux at ``(v, r)``."""
    labeling, _ = simulator.compute_measurements(fluxmap.to_x3c(v, r))
    out = {}
    for (tracer, met), values in zip(measurement_slots(simulator, facts), labeling):
        for k, value in enumerate(np.asarray(values).ravel(), start=1):
            out[(tracer, met, f"M+{k}")] = float(value)
    return out


def compare_mid(facts, labeling_index, our_labeling, v, r, noise_std=0.01,
                verbose=True):
    """Per-isotopomer comparison of the two forward models at one flux state.

    *our_labeling* is the vector our EMU model simulates, aligned to
    *labeling_index* -- the ``(tracer, metabolite, 'M+k')`` triples the benchmark
    scores. Returns a DataFrame with both values and their absolute difference,
    one row per entry the two models have in common.

    The measurement *values* handed to x3cflux are irrelevant here: a simulated MID
    depends on the flux and the network, not on the data it will later be compared
    against. Zeros are used so that no noise realisation has to be invented.
    """
    conv = convert(facts)
    assert_network(conv.network_data, verbose=False)

    import x3cflux as x

    labeling_index = [tuple(k) for k in labeling_index]
    zeros = np.zeros(len(labeling_index))
    flux_ids = list(facts["measured_fluxes"])
    configs = build_configurations(
        facts, conv, labeling_index, flux_ids, zeros,
        np.zeros(len(flux_ids)), np.ones(len(flux_ids)), noise_std)
    simulator = x.create_simulator_from_data(conv.network_data, configs)
    fluxmap = FluxMap(facts, conv, simulator.parameter_space)

    theirs = x3cflux_labeling(simulator, fluxmap, facts, v, r)
    ours = dict(zip(labeling_index, np.asarray(our_labeling, float)))

    inputs = set(facts["input_metabolites"])
    rows = []
    for key in labeling_index:
        if key not in theirs:
            continue
        tracer, met, isotopomer = key
        rows.append({"tracer": tracer, "metabolite": met, "isotopomer": isotopomer,
                     "ours": ours[key], "x3cflux": theirs[key],
                     "abs_diff": abs(theirs[key] - ours[key]),
                     "input_pool": met in inputs})
    frame = pd.DataFrame(rows)

    missing = [k for k in labeling_index if k not in theirs]
    if verbose:
        report(frame, missing)
    return frame


def report(frame, missing=()):
    """Print the comparison, worst offenders first."""
    if frame.empty:
        print("compare_mid: no overlapping entries")
        return
    core = frame[~frame["input_pool"]]
    print(f"compare_mid: {len(frame)} entries "
          f"({len(core)} excluding input pools)")
    if len(missing):
        print(f"  {len(missing)} of our entries absent from x3cflux, "
              f"e.g. {list(missing)[:3]}")
    for label, block in (("all", frame), ("excluding input pools", core)):
        if block.empty:
            continue
        diff = block["abs_diff"]
        print(f"  {label:22s} max={diff.max():.3e}  median={diff.median():.3e}  "
              f"mean={diff.mean():.3e}")
    worst = (core if not core.empty else frame).nlargest(10, "abs_diff")
    print("  worst entries:")
    for _, row in worst.iterrows():
        print(f"    {row.metabolite:14s} {row.isotopomer:5s} "
              f"ours={row.ours: .6f}  x3cflux={row.x3cflux: .6f}  "
              f"|d|={row.abs_diff:.2e}")
