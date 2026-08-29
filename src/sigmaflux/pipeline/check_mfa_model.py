"""Check that a model's atom mappings agree between the in-house EMU model and x3cflux.

An MFA backend can be wrong in ways a fit never reveals: an optimiser happily
converges on a network that has lost a balance row or scrambled an atom mapping and
reports a small residual for a model that is not the one we meant. So the question
this module answers is narrow and comes first, from the config alone, before either
backend is asked to fit anything real: **given its atom mappings, do the in-house EMU
forward model and x3cflux predict the same labeling for the same flux?**

Four checks, in order, each depending on the last:

``build``
    :func:`~sigmaflux.x3cflux.convert.convert` succeeds,
    :func:`~sigmaflux.x3cflux.check_network.check_network` is clean, and a measured
    simulator can be constructed. This is what catches the SIGSEGV classes, because
    x3cflux ships no validator of its own.
``edge_weights``
    Every reaction delivers its whole flux to every EMU it produces, on both sides.
    A structural defect here shows up as a numeric disagreement in ``labeling``
    with no indication of where it came from, so it is worth naming separately.
``labeling``
    Both forward models predict the same MIDs from the same flux, expanding from the
    mapped measurements only. This is the sharpest check of the three: two
    independently written simulators agreeing is far stronger evidence than either
    being self-consistent.
``fit_starts``
    Ipopt takes the converted problem and makes progress from a real starting point,
    capped at a handful of iterations. This does not test convergence -- only that
    the problem is well-posed enough for the optimiser to move.

Usage::

    sigmaflux-check-mfa-model --config paper/configs/e_coli.yaml
    sigmaflux-check-mfa-model --config paper/configs/e_coli.yaml --skip-fit --json

Writes ``model_facts.json`` and ``model_prepared_for_mfa.json`` to ``--out-dir``.
"""

import argparse
import csv
import json
import os
import sys

#: Max absolute MID difference the labeling check accepts.
#:
#: Not the ~1e-6 two exact simulators would reach. The EMU cascade and x3cflux's
#: solver converge their linear systems to different tolerances, and the measured
#: floor on E. coli core is 1.4e-3 on a handful of amino acids. Anything above this
#: is a translation defect worth explaining, not a tolerance to widen.
DEFAULT_LABELING_TOL = 5e-3

#: Iterations to cap the fit-starts check at. Large enough that Ipopt has taken at
#: least one real Newton step, small enough that a fit is seconds, not minutes.
DEFAULT_FIT_ITERS = 3


def measurable_metabolites(cfg, root):
    """Model ids of the metabolites the measurement platform can see.

    Read from the ``model_id`` column of the CSV :func:`mapping_csv_for` picks, so the EMU
    expansion targets, feature selection and inference all describe the same platform.
    Returns ``None`` (meaning "target everything with an atom pattern") when the config
    declares no mapping.

    Args:
        cfg: organism config; supplies ``experiment.mapping_csv`` / ``mapping_csv``.
        root: project root for resolving the mapping CSV path.
    """
    from ._paths import mapping_csv_for, resolve_path

    mapping_csv = mapping_csv_for(cfg)
    if not mapping_csv:
        return None
    path = resolve_path(mapping_csv, root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"mapping_csv does not exist: {path}")
    with open(path) as fh:
        ids = {row["model_id"] for row in csv.DictReader(fh) if row.get("model_id")}
    if not ids:
        raise ValueError(f"mapping_csv has no model_id values: {path}")
    return ids


def prepared_model(cfg, root, out_dir):
    """The MFA-ready cobra model, prepared exactly as ``sample_flux`` prepares it.

    Shares :func:`~sigmaflux.pipeline.sample_flux.prepare_for_mfa` with the sampler
    rather than repeating the sequence, so this validates the network training will
    actually see.

    Args:
        cfg: organism config, supplying sampling/media/growth settings.
        root: project root for resolving the model file path.
        out_dir: directory the prepared model JSON is written to.
    """
    import cobra
    from cobra.io.json import load_json_model

    from ._paths import resolve_path
    from .sample_flux import prepare_for_mfa

    model = load_json_model(resolve_path(cfg["model_file"], root))
    model.solver.configuration.tolerances.feasibility = 1e-9
    prepare_for_mfa(model, cfg)

    path = os.path.join(out_dir, "model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, path)
    print(f"  wrote {path}")
    return model


def interior_flux(model, n_samples=2, seed=None):
    """A feasible interior flux, drawn by cobra's own flux sampler.

    Unlike an FBA vertex, a sampled point lands away from zero on any reaction
    whose feasible range isn't a single point -- avoiding the zeroed branch points
    x3cflux reports as "not numerically solvable" and can segfault on. Uses
    ``achr`` (single-process, suited to a handful of points) rather than the
    default ``auto``/``chrr``.

    Args:
        model: prepared COBRA model to sample.
        n_samples: number of points to draw (only the first is used).
        seed: sampler seed for reproducibility.
    """
    import numpy as np
    import pandas as pd
    from cobra.sampling import sample
    from cobra.util.array import create_stoichiometric_matrix

    samples = sample(model, n=n_samples, method="achr", thinning=100, seed=seed)
    v = samples.iloc[0]

    S = create_stoichiometric_matrix(model)
    resid = float(np.abs(S @ v.values).max())
    if resid >= 1e-6:
        raise RuntimeError(f"Sampled flux is not feasible: max|S.v| = {resid:.3g}")
    print(f"  interior flux: max|S.v| = {resid:.3g}")

    # A pd.Series, not a plain dict: forward_labeling and FluxMap.full_vector both
    # do np.asarray(r) and then index it *positionally*, aligned to model.reactions
    # -- exactly how v (cobra's own solution.fluxes) already behaves. A dict
    # converts to a 0-d object array instead of failing loudly, so the wrong type
    # here would surface many calls downstream rather than here.
    r = pd.Series({rxn.id: (0.5 if rxn.reversibility else 0.0)
                   for rxn in model.reactions})
    return v, r


def check_build(facts, mfa_model, labeling_to, labeling_index, v, r, noise_std=0.01):
    """The model converts to x3cflux, and both simulators build.

    Everything here is construction, not simulation, including x3cflux's own
    simulator build -- the step that scales badly for larger networks -- so a long
    stall here is always attributable to construction, never to the comparison.

    Args:
        facts: model facts dict (from ``dump_model_facts``) driving the x3cflux conversion.
        mfa_model: prepared in-house MFAModel used for the forward labeling prediction.
        labeling_to: index tensor gathering the in-house model's output into labeling_index order.
        labeling_index: ordered (tracer, metabolite, isotopomer) labeling slots.
        v: interior flux vector.
        r: interior reversibility vector.
        noise_std: labeling noise std used to size the x3cflux measurement sigmas.
    """
    import x3cflux as x

    import numpy as np

    from ..x3cflux.build_config import build_configurations
    from ..x3cflux.check_network import check_network
    from ..x3cflux.convert import convert
    from ..x3cflux.fluxmap import FluxMap
    from ._benchmark import forward_labeling

    print("  build x3c")
    conv = convert(facts)
    problems = check_network(conv.network_data, verbose=False)
    result = {"passed": not problems, "n_reactions": len(conv.x3c_reactions),
             "void_pools": conv.void_pools, "problems": problems[:20]}
    if problems:
        print(f"  FAIL {len(problems)} problem(s): {problems[:5]}")
        return result, None, None, None, None
    print(f"    {len(conv.x3c_reactions)} x3cflux reactions, {len(conv.void_pools)} void pools")

    print("  predict in-house")
    ours = forward_labeling(mfa_model, v, r, labeling_to)

    print("  build x3c simulator")
    flux_ids = [rid for rid in facts["measured_fluxes"]
               if rid in {rr["id"] for rr in facts["reactions"]}]
    flux_values = [v[rid] for rid in flux_ids]
    flux_sigma = [max(0.05 * abs(val), 1e-9) for val in flux_values]
    configs = build_configurations(
        facts, conv, labeling_index, flux_ids, np.zeros(len(labeling_index)),
        flux_values, flux_sigma, noise_std)
    simulator = x.create_simulator_from_data(conv.network_data, configs)
    fluxmap = FluxMap(facts, conv, simulator.parameter_space)

    return result, conv, simulator, fluxmap, ours


def check_edge_weights(mfa_model, conv, tol=1e-9):
    """Every reaction delivers its whole flux to every EMU it produces.

    Per ``(EMU, reaction, direction)`` the summed edge weights
    (:meth:`~sigmaflux.expansion.atom_mapping.AtomMapping.get_origin_emus`) must equal
    the reaction's stoichiometric coefficient for that metabolite -- otherwise flux
    carries no label, or label is invented from nothing, and our forward model is
    solving a different network than x3cflux. Structural check only: no flux,
    no simulation.

    Args:
        mfa_model: prepared in-house MFAModel whose EMU network is checked.
        conv: x3cflux conversion result, for the symmetry-share comparison.
        tol: maximum allowed deviation from a full-stoichiometry weight sum.
    """
    from collections import defaultdict

    totals = defaultdict(float)
    coefficients = {}
    for graph in mfa_model.expansion.emu_network.subnetworks.values():
        for _, target, data in graph.edges(data=True):
            key = (repr(target), data["reaction"].id, data["direction"])
            totals[key] += data["weight"]
            coefficients[key] = abs(data["reaction"].get_coefficient(target.id))

    ours = [{"emu": emu, "reaction": rid, "direction": d,
             "weight_sum": total, "coefficient": coefficients[(emu, rid, d)]}
            for (emu, rid, d), total in totals.items()
            if abs(total - coefficients[(emu, rid, d)]) > tol]
    ours.sort(key=lambda item: item["weight_sum"] / item["coefficient"])

    # Only symmetry parents: the other split path (non-integer stoichiometry) uses
    # `parents` for stoichiometric coefficients, which are not a partition of unity.
    # `__sym0` is the same marker fluxmap.FluxMap uses to tell the two apart.
    x3c = [{"reaction": parent, "share_sum": sum(share for _, share in subs)}
           for parent, subs in conv.parents.items()
           if len(subs) > 1 and subs[0][0].endswith("__sym0")]
    x3c = [item for item in x3c if abs(item["share_sum"] - 1.0) > tol]

    passed = not ours and not x3c
    result = {"passed": passed, "n_producers": len(totals),
              "n_our_violations": len(ours), "n_x3c_violations": len(x3c),
              "our_violations": ours[:20], "x3c_violations": x3c[:20]}
    for item in ours[:10]:
        print(f"  FAIL ours: {item['emu']} <- {item['reaction']} "
              f"({item['direction']}) weights sum to {item['weight_sum']:g}, "
              f"stoichiometry moves {item['coefficient']:g}")
    if len(ours) > 10:
        print(f"  ... and {len(ours) - 10} more")
    for item in x3c[:10]:
        print(f"  FAIL x3c: {item['reaction']} symmetry shares sum to "
              f"{item['share_sum']:g}, not 1")
    if passed:
        print(f"  {len(totals)} (EMU, reaction, direction) producers all deliver "
              f"their full stoichiometry")
    return result


def check_labeling(facts, simulator, fluxmap, labeling_index, ours, v, r,
                   tol=DEFAULT_LABELING_TOL):
    """Both forward models predict the same labeling at the same flux.

    Both simulators are already built (:func:`check_build`); this only runs x3cflux's
    forward pass and compares -- fast regardless of network size.

    Args:
        facts: model facts dict driving the comparison.
        simulator: built x3cflux simulator.
        fluxmap: FluxMap translating between in-house and x3cflux parameterizations.
        labeling_index: ordered (tracer, metabolite, isotopomer) labeling slots.
        ours: in-house model's predicted labeling vector.
        v: interior flux vector.
        r: interior reversibility vector.
        tol: max absolute MID difference accepted.
    """
    from ..x3cflux.compare import compare_simulated

    print("  predict x3c")
    frame = compare_simulated(simulator, fluxmap, facts, labeling_index, ours, v, r)

    print("  compare")
    core = frame[~frame["input_pool"]]
    max_diff = float(core["abs_diff"].max()) if not core.empty else 0.0
    n_missing = len(frame.attrs.get("missing", ()))
    passed = max_diff < tol and not n_missing
    result = {"passed": passed, "max_abs_diff": max_diff,
             "median_abs_diff": float(core["abs_diff"].median()) if not core.empty else 0.0,
             "n_compared": len(core), "n_missing": n_missing, "tolerance": tol}
    if n_missing:
        print(f"  FAIL {n_missing} of our labeling slots were never built by x3cflux")
    if max_diff >= tol:
        print(f"  FAIL max |Δ| = {max_diff:.3e} exceeds {tol:.3e}")
    if passed:
        print(f"  max |Δ| = {max_diff:.3e} over {len(core)} slots (tol {tol:.3e})")
    return result


def check_fit_starts(facts, conv, simulator, fluxmap, v, r,
                     max_iter=DEFAULT_FIT_ITERS, isolate=True):
    """Ipopt takes the converted problem and makes progress; convergence is not tested.

    Starts from the same interior flux the labeling check used, so SSR is already
    ~0 before Ipopt steps -- this only checks the problem constructs and Ipopt
    accepts it. ``status == -1`` (``Maximum_Iterations_Exceeded``) is an accepted
    outcome at a small ``max_iter``.

    Args:
        facts: model facts dict driving the fit.
        conv: x3cflux conversion result, for flux bounds.
        simulator: built x3cflux simulator.
        fluxmap: FluxMap translating the interior flux into x3cflux's starting point.
        v: interior flux vector.
        r: interior reversibility vector.
        max_iter: Ipopt iteration cap.
        isolate: run the fit in a forked child so a crash doesn't take the process down.
    """
    from ..x3cflux.fit import flux_bounds, optimize, run_isolated

    bounds = flux_bounds(facts, conv)
    theta_init = fluxmap.to_x3c(v, r)

    def call():
        return optimize(simulator, theta_init, max_iter=max_iter, bounds=bounds)

    outcome = run_isolated(call) if isolate else call()
    if outcome is None:
        print("  FAIL fit crashed or raised")
        return {"passed": False, "crashed": True}

    import numpy as np
    # bool(...): "and" returns whichever operand decided it, and np.isfinite
    # gives a numpy bool_, not a native bool -- json.dump chokes on that.
    passed = bool(outcome["status"] in (0, 1, -1) and np.isfinite(outcome["ssr"]))
    result = {"passed": passed, "status": outcome["status"],
             "status_msg": outcome["status_msg"], "ssr": outcome["ssr"],
             "seconds": outcome["seconds"]}
    if passed:
        print(f"  status {outcome['status']} ({outcome['status_msg']}), "
              f"ssr={outcome['ssr']:.4g}, {outcome['seconds']:.1f}s")
    else:
        print(f"  FAIL status {outcome['status']} ({outcome['status_msg']})")
    return result


def check_model(cfg, root, out_dir, tol=DEFAULT_LABELING_TOL,
                max_iter=DEFAULT_FIT_ITERS, skip_fit=False, noise_std=None,
                seed=None):
    """Run the four checks in order, stopping at the first failure -- each depends on
    the last having passed.

    Args:
        cfg: organism config.
        root: project root for resolving config-relative paths.
        out_dir: directory model_facts.json and the prepared model are written to.
        tol: max absolute MID difference the labeling check accepts.
        max_iter: Ipopt iteration cap for the fit-starts check.
        skip_fit: run only the build/edge_weights/labeling checks.
        noise_std: labeling noise std; defaults to the config's ``features.noise_std``.
        seed: seed for the interior flux sample.

    Returns ``(ok, results)``, ``results`` a ``{"build", "edge_weights", "labeling",
    "fit_starts"}`` dict with one entry per check actually reached.
    """
    from ..x3cflux.facts import dump_model_facts
    from ._benchmark import build_mfa_context

    if noise_std is None:
        noise_std = cfg.get("features", {}).get("noise_std", 0.01)
    results = {}

    print("\nPreparing the model")
    model = prepared_model(cfg, root, out_dir)
    # After preparation, so a compound the platform reports as one whole-cell pool
    # targets that pool. Its compartments are still expanded -- they are its origins.
    keep = measurable_metabolites(cfg, root)

    print("\nComputing an interior flux")
    v, r = interior_flux(model, seed=seed)

    print("\nExpanding the EMU network over the measurable metabolites")
    mfa_model, _, _, labeling_to, labeling_index = build_mfa_context(
        model, cfg["labeling"]["tracers"], initial_flux=v, name=cfg["name"],
        with_backward=False, restrict_targets_to=keep)
    observable = keep or {met for _, met, _ in labeling_index}
    facts = dump_model_facts(model, mfa_model, observable, cfg,
                             os.path.join(out_dir, "model_facts.json"))
    labeling_index = [tuple(k) for k in labeling_index]
    if not labeling_index:
        raise ValueError("No labeling slots. Check that mapping_csv's model_id "
                         "column matches the model.")
    print(f"  {len(labeling_index)} labeling slots")

    print("\n[build]")
    results["build"], conv, simulator, fluxmap, ours = check_build(
        facts, mfa_model, labeling_to, labeling_index, v, r, noise_std)
    if not results["build"]["passed"]:
        return False, results

    print("\n[edge_weights]")
    results["edge_weights"] = check_edge_weights(mfa_model, conv)
    if not results["edge_weights"]["passed"]:
        return False, results

    print("\n[labeling]")
    results["labeling"] = check_labeling(
        facts, simulator, fluxmap, labeling_index, ours, v, r, tol)
    if not results["labeling"]["passed"]:
        return False, results

    if skip_fit:
        results["fit_starts"] = {"passed": True, "skipped": True}
        return True, results

    print("\n[fit_starts]")
    results["fit_starts"] = check_fit_starts(
        facts, conv, simulator, fluxmap, v, r, max_iter=max_iter)
    return results["fit_starts"]["passed"], results


def main(args):
    from ._paths import load_config

    cfg, root = load_config(args.config, args.root)
    organism = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out_dir or os.path.join(
        root, "paper", ".results", "check_mfa_model", organism)
    os.makedirs(out_dir, exist_ok=True)

    print(f"===== check mfa model: {organism} =====")
    print(f"Model   : {cfg['model_file']}")
    print(f"Out dir : {out_dir}")

    ok, results = check_model(cfg, root, out_dir, tol=args.tol,
                              max_iter=args.max_iter, skip_fit=args.skip_fit,
                              seed=args.seed)

    summary = {"organism": organism, "config": args.config,
               "model_file": cfg["model_file"], "passed": ok, "checks": results}
    summary_path = os.path.join(out_dir, "check_mfa_model.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=1)

    print(f"\n===== {organism}: {'PASS' if ok else 'FAIL'} =====")
    for key in ("build", "edge_weights", "labeling", "fit_starts"):
        if key in results:
            mark = "ok  " if results[key]["passed"] else "FAIL"
            print(f"  {mark} {key}")
        else:
            print(f"  --   {key} (not reached)")
    print(f"Summary: {summary_path}")
    if args.json:
        print(json.dumps(summary, indent=1))
    return 0 if ok else 1


def build_parser():
    from ._paths import add_root_argument

    parser = argparse.ArgumentParser(
        description="Check that a model's atom mappings agree between the "
                    "in-house EMU model and x3cflux.")
    parser.add_argument("--config", required=True, help="Organism YAML config")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write model_facts.json and the summary "
                             "(default: paper/.results/check_mfa_model/<organism>)")
    parser.add_argument("--tol", type=float, default=DEFAULT_LABELING_TOL,
                        help=f"Max absolute MID difference for the labeling check "
                             f"(default {DEFAULT_LABELING_TOL})")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_FIT_ITERS,
                        help=f"Ipopt iteration cap for the fit-starts check "
                             f"(default {DEFAULT_FIT_ITERS})")
    parser.add_argument("--skip-fit", action="store_true",
                        help="Build and labeling checks only; do not run Ipopt")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the interior flux sample, for reproducible "
                             "before/after comparisons")
    parser.add_argument("--json", action="store_true",
                        help="Print the summary as JSON on stdout")
    add_root_argument(parser)
    return parser


def cli(argv=None):
    # x3cflux has to load before torch; see sigmaflux.pipeline.mfa_entry.
    from ..x3cflux import preimport, require
    if not preimport():
        require()
    args = build_parser().parse_args(argv)
    return main(args)


if __name__ == "__main__":
    sys.exit(cli())
