"""Does the model return the exchange fluxes it was handed?

The measured fluxes are fed in as tokens and reconstructed through ``Z_full`` like any other
reaction, never hard-copied, so the prediction at those positions can drift from the value
the model was told. Nothing else in the repo checks this: ``show_metrics.py`` scores the same
reactions against *synthetic* ground truth, and ``validate_inferred_fluxes.py`` checks
feasibility and MID agreement but not measured-flux fidelity.

It matters for a projection arm specifically. A projection moves the state along whichever
facets are violated, and a measured flux -- whose value the model was given -- is the one
place an unwanted move is unambiguous rather than a judgement call about accuracy. It is also
exactly where the ratio-test predecessor of ``_project_into_bounds`` did its damage.

Reports, per arm and per measured reaction, the signed error against the handed-in value,
the error in units of the replicate spread, and the pooled relative RMSE.

    python paper/analysis/diagnostics/measured_flux_fidelity.py \\
        --config paper/configs/s_cerevisiae.yaml \\
        paper/.results/s_cerevisiae_20260917
"""

import argparse
import os

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.pipeline.run_uncertainty_benchmark import _resolve_ai_checkpoint
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions,
    _find_ext_flux_file,
    _inspect_growth_conditions,
    load_condition_inputs,
    prepared_network_for,
)

# Matches validate_inferred_fluxes: below this a reported replicate spread is noise, and
# dividing by it would turn a rounding difference into a huge z-score.
STD_FLOOR = 1e-3


def score_arm(run_dir, cfg, config_path, data_dir, mapping_path, conditions, ext_file):
    """Predicted vs handed-in measured flux for every (condition, growth condition)."""
    # Top-level .pt, else the ensemble/ directory -- every current run is the latter.
    checkpoint = _resolve_ai_checkpoint(run_dir)
    if checkpoint is None:
        raise SystemExit(f"no checkpoint in {run_dir} or its ensemble/")
    sf = load_predictor(config_path, checkpoint, data_dir=run_dir)
    flux_names = list(cfg.get("training", {}).get("measured_fluxes", []))
    network = prepared_network_for(checkpoint)

    rows = []
    for condition in conditions:
        mid_file = _discover_conditions(data_dir)[condition]
        for gc in _inspect_growth_conditions(mid_file):
            df_L, df_Vm, df_err = load_condition_inputs(
                mid_file, ext_file, condition, gc, mapping_path, flux_names, model=network)
            # The maximum-likelihood input, not a bootstrap draw: the question is whether
            # the model reproduces the value it was given, so it must be given that value.
            res = sf.predict(df_L, df_Vm=df_Vm)
            pred = (res[res["Type"] == "flux"] if "Type" in res
                    else res).set_index("Flux Name")["Value"]

            given = df_Vm.groupby("Flux Name")["Value"].mean()
            spread = (df_err.groupby("Flux Name")["Value"].mean()
                      if len(df_err) else pd.Series(dtype=float))
            for name, want in given.items():
                if name not in pred.index:
                    continue
                got = float(np.atleast_1d(pred.loc[name])[0])
                sd = max(float(spread.get(name, 0.0)), STD_FLOOR)
                rows.append(dict(
                    arm=os.path.basename(run_dir), condition=condition, gc=gc,
                    reaction=name, given=float(want), predicted=got,
                    error=got - float(want), z=(got - float(want)) / sd))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="experimental data dir; default is experiment.data_dir")
    ap.add_argument("--conditions", nargs="*", default=None)
    add_root_argument(ap)
    args = ap.parse_args()

    cfg, root = load_config(args.config, args.root)
    exp = cfg.get("experiment", {})
    data_dir = args.data_dir or resolve_path(exp["data_dir"], root)
    mapping_path = resolve_path(exp["mapping_csv"], root)
    ext_file = _find_ext_flux_file(data_dir)
    if ext_file is None:
        raise SystemExit(f"no *_ExtFlux.xlsx in {data_dir}")
    conditions = args.conditions or sorted(_discover_conditions(data_dir))

    frames = [score_arm(resolve_path(d, root), cfg, args.config, data_dir,
                        mapping_path, conditions, ext_file)
              for d in args.run_dirs]
    df = pd.concat(frames, ignore_index=True)

    print("\n=== per arm ===")
    for arm, g in df.groupby("arm", sort=False):
        rel = np.sqrt(np.mean((g["error"] / g["given"].abs().clip(1e-9)) ** 2))
        print(f"{arm:42s} n {len(g):4d}  rel RMSE {rel:8.4f}  "
              f"|z| median {g['z'].abs().median():8.3f}  p90 {g['z'].abs().quantile(.9):8.3f}  "
              f"max {g['z'].abs().max():9.3f}")

    print("\n=== per reaction (signed error, mean over conditions) ===")
    piv = df.pivot_table(index="reaction", columns="arm", values="error", aggfunc="mean")
    given = df.groupby("reaction")["given"].mean()
    piv.insert(0, "given", given)
    print(piv.to_string(float_format=lambda v: f"{v:9.4f}"))


if __name__ == "__main__":
    main()
