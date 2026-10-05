"""
Simulate SigmaFlux's predicted labeling for the experimental panels.

The expensive half of the isotopologue comparison: for every growth condition the paper
plots, this predicts the flux state from the measured data, pushes each draw through the
EMU forward pass, and writes the per-token table paper/scripts/plot_experimental_mid.py
draws. Splitting the two means a plotting choice -- a different floor, a different
annotation -- costs a second, not another pass over the whole bootstrap.

Per model, in ``<run-dir>/experimental_mid/``:
  tokens.csv    one row per (condition, growth condition, tracer, metabolite, isotopomer)
                with the measured mean/std, the predicted value, and the predicted
                interval

Both sides of the comparison carry an uncertainty. The measured side is the replicate
spread. The predicted side is the same bootstrap the flux predictions report, propagated
all the way into labeling space: the measured replicates are resampled across tracers
(``sample_inputs``), every draw is predicted by every ensemble member, and each of those
flux states is simulated through the EMU pass in full. ``summarize_draws`` then reduces
the resulting fractions exactly as it does for fluxes -- the labeling of the ensemble's
medoid_convex flux state at the maximum-likelihood input, bias-corrected bootstrap
percentile interval over the draws.
Propagating in labeling space rather than mapping the flux interval through is the point:
the EMU pass is nonlinear, so the endpoints of a flux interval are not the endpoints of
the labeling interval.

That costs ``(n_bootstrap + 1) x n_members`` EMU passes per growth condition -- minutes
per condition, not seconds, which is why this belongs in a SLURM job
(paper/submit/submit_experimental_mid.sh) rather than at a prompt. ``--n-bootstrap``
trades runtime against how finely the interval's tails resolve.

Conditions and growth conditions are the same ``PREDICTION_SPECS`` selection the flux
panels use, so the two figures describe the identical set of measured states. M+0 is
excluded on both sides -- the training labeling index carries M+1..M+N only, and M+0 is
one minus the rest rather than an independent measurement.

Results are written per model, so a model that fails costs its own panel and nothing
else, and one organism can be recomputed without the others.

Usage:
    python paper/scripts/experimental_mid.py
    python paper/scripts/experimental_mid.py --models m_musculus_t_cell
    python paper/scripts/experimental_mid.py --models e_coli_core --n-bootstrap 20
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.ai.uncertainty import DEFAULT_N_BOOTSTRAP, sample_inputs, summarize_draws
from sigmaflux.pipeline._benchmark import pin_threads
from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions,
    load_condition_inputs)
from sigmaflux.utils import read_reference_fluxes

from _figure_common import (
    PREDICTION_SPECS, UNBOUNDED_FLUX, _REPO_ROOT, add_common_args, growth_rates,
    reference_path, resolve_organism, run_dir_for, selected_models)

sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from validate_inferred_fluxes import agreement, measured_table  # noqa: E402

#: MID token keys -- one predicted fraction is identified by all three.
TOKEN = ["Tracer", "Metabolite", "Isotopomer"]

#: predict(with_mid=True) reports labeling in the schema it was handed, i.e. with the
#: config's display tracer names; measured_table keys on the internal tracer id.
RENAME = {"Metabolite ID": "Metabolite", "Fraction": "Value"}

#: Where a run's simulated tokens live, beside its other per-run artifacts.
TOKENS_SUBDIR = "experimental_mid"


def tokens_path(run):
    """The per-run token table :mod:`plot_experimental_mid` reads."""
    return run / TOKENS_SUBDIR / "tokens.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n-bootstrap", type=int, default=None,
                   help="Replicate resamples per growth condition (default: the config's "
                        "experiment.n_bootstrap). Every draw costs one EMU pass per "
                        "ensemble member.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the replicate bootstrap (default: 42)")
    return p.parse_args()


def tracer_name_map(tracer_cfg, experiments):
    """Config display name -> the internal tracer string the labeling index is keyed on.

    ``build_labeling_index`` names each experiment by its tracers' repr, and
    ``build_experiments`` builds one experiment per config tracer in order, so zipping
    the two is exact -- and needs no checkpoint, unlike ``SigmaFlux.tracer_ids()``.
    """
    return {cfg_entry.get("name", str(exp.tracers[0])):
            "|".join(str(t) for t in exp.tracers)
            for cfg_entry, exp in zip(tracer_cfg, experiments)}


def selected_growth_conditions(org, cond, available, select):
    """The growth conditions of one condition that go into the panel.

    ``max_D`` reads the dilution rate off the reference's own BIOMASS row rather than
    assuming an index, matching how the flux panels pick the same states.
    """
    if select == "wt":
        return [gc for gc in available if gc == 1]
    if select != "max_D":
        return list(available)
    rates = growth_rates(read_reference_fluxes(reference_path(org), cond,
                                               unbounded=UNBOUNDED_FLUX))
    rated = {gc: rates[gc] for gc in available if gc in rates}
    if not rated:
        print(f"  condition {cond}: no BIOMASS row in the reference — keeping every "
              f"growth condition")
        return list(available)
    return [max(rated, key=rated.get)]


def condition_tokens(sf, mid_file, ext_file, mapping_path, cond, gc, flux_names,
                     n_bootstrap, rng, name_to_id):
    """Predicted-vs-measured MID tokens for one growth condition, with intervals.

    Inputs, draws and prediction all go through run_inference_workflow's own helpers,
    so this reports the pipeline's numbers rather than a parallel copy of them, and the
    labeling comes from ``predict(with_mid=True)`` -- the same EMU pass inference itself
    runs, not a second implementation of it.
    """
    df_L, df_Vm, df_err = load_condition_inputs(mid_file, ext_file, cond, gc, mapping_path,
                                                flux_names)
    df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=rng,
                                            n_bootstrap=n_bootstrap,
                                            bounds=sf.flux_bounds())
    n_draws = int(df_L_draws["Sample"].max())      # the MLE row sits at the top index
    _, mids = sf.predict(df_L_draws, df_Vm=df_Vm_draws, aggregate=True, with_mid=True)

    mids = (mids.rename(columns=RENAME)
                .assign(Tracer=lambda d: d["Tracer Name"].map(name_to_id))
                .dropna(subset=["Value", "Tracer"]))
    if not (mids["Sample"] == n_draws).any():
        raise ValueError("the maximum-likelihood flux state has no solvable labeling; "
                         "nothing to report for this growth condition")
    # One simulated state per draw -- the labeling of that draw's aggregate flux state --
    # so the nominal value is the labeling of the fluxes we publish and the interval is
    # that value's own spread across resampled inputs.
    predicted = summarize_draws(mids, n_draws, group_cols=TOKEN)

    measured = measured_table(df_L, name_to_id)
    tokens = agreement(predicted.set_index(TOKEN)["Value"].rename("sim"), measured)
    return tokens.merge(predicted[TOKEN + ["Value_lo", "Value_hi", "sigma_input",
                                           "n_draws"]],
                        on=TOKEN, how="left")


def model_tokens(model, run, org, spec, args):
    """Every measured MID token for one model, paired with its predicted value.

    Returns None when the run has nothing to compare. The predictor builds its EMU
    network once, on first use, and reuses it for every (condition, growth condition).
    """
    cfg = org["cfg"]
    data_dir = resolve_path(org["data"], org["root"])
    mapping_path = resolve_path(mapping_csv_for(cfg), org["root"])
    ext_flux = cfg.get("experiment", {}).get("ext_flux")
    ext_file = resolve_path(ext_flux, org["root"]) if ext_flux else _find_ext_flux_file(data_dir)
    if ext_file is None:
        raise FileNotFoundError(f"no *_ExtFlux.xlsx in {data_dir}")
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None
    n_bootstrap = (args.n_bootstrap if args.n_bootstrap is not None
                   else cfg.get("experiment", {}).get("n_bootstrap", DEFAULT_N_BOOTSTRAP))
    available = _discover_conditions(data_dir)

    sf = load_predictor(cfg["_config_path"], None, data_dir=str(run))
    name_to_id = sf.tracer_ids()

    rng = np.random.default_rng(args.seed)
    parts = []
    for cond in spec["conditions"]:
        if cond not in available:
            print(f"  condition {cond}: no MID workbook in {data_dir} — skipping")
            continue
        for gc in selected_growth_conditions(org, cond,
                                             _inspect_growth_conditions(available[cond]),
                                             spec["select"]):
            started = time.perf_counter()
            try:
                tokens = condition_tokens(sf, available[cond], ext_file, mapping_path,
                                          cond, gc, flux_names, n_bootstrap, rng,
                                          name_to_id)
            except Exception as exc:
                print(f"  {cond}{gc}: skipped — {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]}")
                continue
            print(f"  {cond}{gc}: {len(tokens)} MID tokens, {n_bootstrap} draws "
                  f"[{time.perf_counter() - started:.0f}s]")
            parts.append(tokens.assign(condition=cond, gc=gc))

    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    args = parse_args()
    pin_threads()

    models = [m for m in selected_models(args) if m["key"] in PREDICTION_SPECS]

    print("── Simulating predicted labeling ──")
    for model in models:
        run = run_dir_for(model, args.run_suffix)
        if not list(run.glob("ensemble/model_*.pt")):
            print(f"  {model['label']}: skipped — no ensemble/ in {run.name}; "
                  f"run paper/submit/submit_ensemble.sh {run}")
            continue
        print(f"\n{model['label']}  ({run.name})")
        try:
            org = resolve_organism(run)
            df = model_tokens(model, run, org, PREDICTION_SPECS[model["key"]], args)
        except Exception as exc:
            print(f"  {model['label']}: skipped — {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:160]}")
            continue
        if df is None or df.empty:
            print(f"  {model['label']}: skipped — no token matched between the "
                  f"simulation and the workbook")
            continue
        path = tokens_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.assign(model=model["label"], key=model["key"]).to_csv(path, index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
