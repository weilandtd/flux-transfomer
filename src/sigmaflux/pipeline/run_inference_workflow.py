"""
Run SigmaFlux inference for all nutrient conditions and growth rates.

Discovers all condition MID files (<Name>_C.xlsx, <Name>_N.xlsx, …) in the
experimental data directory and runs all growth conditions within each.
Results are saved as per-condition CSVs in the output directory.

Usage:
    sigmaflux-infer \\
        --config     paper/configs/i_orientalis.yaml \\
        --checkpoint paper/.results/i_orientalis_20260603/imisor_transformer.pt \\
        --data-dir   data/I_orientalis \\
        --out-dir    paper/.results/i_orientalis_20260603/predictions

Optional:
    --conditions C N P      run only these conditions (default: all discovered)

Measurement uncertainty is propagated through the ensemble by Monte Carlo -- see
:mod:`sigmaflux.ai.uncertainty` for the model. The number of input draws comes from
experiment.n_input_draws in the config (default 100 when absent) -- not a CLI flag, so
the count used for a run is always the one recorded in its config.
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
from sigmaflux.ai.uncertainty import fit_mid_noise_model, sample_inputs, summarize_draws
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.utils import read_xlsx_mid_data, read_xlsx_extflux


def _discover_conditions(data_dir: str) -> dict[str, str]:
    """Return {condition: mid_file_path} for the condition MID files in data_dir.

    Single-letter nutrient files map by their letter (``<Name>_C.xlsx`` → 'C', …).
    The batch file ``<Name>_batch.xlsx`` (sharing the nutrient files' stem) maps to 'B'.
    """
    found = {}
    stems = set()
    for f in sorted(glob.glob(os.path.join(data_dir, "*_[A-Z].xlsx"))):
        m = re.search(r'^(.*)_([A-Z])\.xlsx$', os.path.basename(f))
        if m:
            stems.add(m.group(1))
            found[m.group(2)] = f
    # Batch condition shares the nutrient files' stem: <stem>_batch.xlsx → 'B'.
    for stem in sorted(stems):
        batch = os.path.join(data_dir, f"{stem}_batch.xlsx")
        if os.path.exists(batch):
            found["B"] = batch
            break
    return found


def _find_ext_flux_file(data_dir: str) -> str | None:
    """Return the *_ExtFlux.xlsx file in data_dir, or None."""
    matches = glob.glob(os.path.join(data_dir, "*_ExtFlux.xlsx"))
    return matches[0] if matches else None


def _inspect_growth_conditions(xlsx_path: str) -> list[int]:
    """Return sorted list of growth condition numbers found in a MID xlsx."""
    pat = re.compile(r'^[A-Z]+(\d+)_Rep\d+$')
    xl  = pd.ExcelFile(xlsx_path)
    df  = xl.parse(xl.sheet_names[0], nrows=0)
    gcs = sorted({int(pat.match(c).group(1)) for c in df.columns if pat.match(c)})
    return gcs if gcs else [1]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SigmaFlux inference for all conditions and growth rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     required=True,
                        help="Organism YAML config (e.g. paper/configs/i_orientalis.yaml)")
    parser.add_argument("--checkpoint", required=True,
                        help="Trained model checkpoint (.pt)")
    parser.add_argument("--data-dir",   required=True,
                        help="Directory containing experimental xlsx files")
    parser.add_argument("--out-dir",    required=True,
                        help="Directory where per-condition prediction CSVs are written")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Conditions to run (default: all discovered from data-dir)")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, root = load_config(args.config, args.root)
    config_path = cfg["_config_path"]

    t           = cfg.get("training", {})
    flux_names  = t.get("measured_fluxes") or None
    organism    = cfg.get("name", "organism")
    # experiment.mapping_csv lets one organism's experimental files use compound ids
    # the training mapping_csv doesn't cover (GC-MS fragments, alternative platform
    # annotations, ...) without touching the file training's restrict_to_mapped filter
    # reads. Defaults to the top-level mapping_csv when the experiment data uses the
    # same ids training does.
    mapping_csv = cfg.get("experiment", {}).get("mapping_csv") or cfg.get("mapping_csv")
    # experiment.ext_flux lets one organism pick a specific *_ExtFlux.xlsx out of a
    # data_dir shared with another organism, when their measured-flux reaction ids
    # differ (e.g. e_coli's iAF1260 ids vs e_coli_core's naming in the same shared
    # file) — same override pattern as experiment.mapping_csv above. Defaults to the
    # single *_ExtFlux.xlsx glob match when absent.
    ext_flux    = cfg.get("experiment", {}).get("ext_flux")
    # experiment.n_input_draws is the sole source for the Monte Carlo draw count -- no
    # CLI override, so a run's count is always the one its config records.
    n_input_draws = cfg.get("experiment", {}).get("n_input_draws", 100)

    def _p(rel):
        return resolve_path(rel, root)

    checkpoint   = os.path.abspath(args.checkpoint)
    data_dir     = os.path.abspath(args.data_dir)
    out_dir      = os.path.abspath(args.out_dir)
    mapping_path = _p(mapping_csv) if mapping_csv else None

    ext_file = _p(ext_flux) if ext_flux else _find_ext_flux_file(data_dir)
    if ext_file is None:
        print(f"ERROR: no *_ExtFlux.xlsx found in {data_dir}")
        sys.exit(1)

    all_condition_files = _discover_conditions(data_dir)
    if not all_condition_files:
        print(f"ERROR: no condition xlsx files (*_C.xlsx etc.) found in {data_dir}")
        sys.exit(1)

    conditions = args.conditions or sorted(all_condition_files.keys())
    missing = [c for c in conditions if c not in all_condition_files]
    if missing:
        print(f"ERROR: conditions {missing} not found in {data_dir}. "
              f"Available: {sorted(all_condition_files.keys())}")
        sys.exit(1)

    # Ensemble when a sibling <model-dir>/ensemble exists (mean over members), else single.
    sf = load_predictor(config_path, checkpoint)

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Data dir    : {data_dir}")
    print(f"ExtFlux file: {ext_file}")
    print(f"Conditions  : {conditions}")
    print(f"Output dir  : {out_dir}")
    print()

    rng = np.random.default_rng(seed=42)

    # Every condition's labeling data is read up front because the MID noise model pools
    # replicate spread across all of them: one point's 3 replicates give a 2-dof variance
    # estimate far too noisy to use on its own.
    labeling = {}
    for condition in conditions:
        mid_file = all_condition_files[condition]
        for gc in _inspect_growth_conditions(mid_file):
            df_L = read_xlsx_mid_data(
                xlsx_path        = mid_file,
                mapping_csv      = mapping_path,
                growth_condition = gc,
            )
            df_L['Sample'] = df_L['Sample'].rank(method='dense').astype(int)
            labeling[(condition, gc)] = df_L

    noise = fit_mid_noise_model(labeling.values())
    print(f"MID noise model : sigma^2 = {noise.a:.3e} + {noise.b:.3e}*p(1-p)")
    print(f"                  sigma = {noise.sigma(0.0):.4f} at p=0, "
          f"{noise.sigma(0.5):.4f} at p=0.5  ({len(labeling)} growth conditions pooled)")
    print()

    n_flux = len(sf.flux_names)
    is_ensemble = hasattr(sf, 'members')

    for condition in conditions:
        mid_file = all_condition_files[condition]
        gcs      = _inspect_growth_conditions(mid_file)

        print(f"══ Condition {condition} ({os.path.basename(mid_file)}) ══")
        print(f"   Growth conditions: {gcs}")

        cond_frames = []
        for gc in gcs:
            print(f"  ── Growth condition {gc} ──")

            df_L = labeling[(condition, gc)]
            df_Vm, df_err = read_xlsx_extflux(
                xlsx_path        = ext_file,
                condition        = condition,
                flux_names       = flux_names,
                return_errors    = True,
                growth_condition = gc,
            )
            for _df in (df_Vm, df_err):
                _df['Sample'] = _df['Sample'].rank(method='dense').astype(int)

            n_reps = df_L['Sample'].nunique()
            print(f"     Replicates: {n_reps}  |  MID rows: {len(df_L)}  |  Vm rows: {len(df_Vm)}")

            df_L_draws, df_Vm_draws = sample_inputs(
                df_L, df_Vm, df_err, noise, n_draws=n_input_draws, rng=rng)

            print(f"     Running inference ({n_input_draws} input draws "
                  f"+ 1 maximum-likelihood input)...")

            if is_ensemble:
                raw = sf.predict(df_L_draws, df_Vm=df_Vm_draws, aggregate=False)
            else:
                raw = sf.predict(df_L_draws, df_Vm=df_Vm_draws)

            # Outputs come back per sample in flux-then-reversibility order, and a reaction
            # can appear in both, so Type is part of what identifies a predicted quantity.
            member_key = ['Sample', 'Member'] if 'Member' in raw.columns else ['Sample']
            raw['Type'] = (
                raw.groupby(member_key, sort=False).cumcount()
                   .lt(n_flux).map({True: 'flux', False: 'reversibility'})
            )

            agg = summarize_draws(raw, n_draws=n_input_draws,
                                  group_cols=['Type', 'Flux Name'])
            agg.insert(0, 'GrowthCondition', gc)
            agg.insert(0, 'Condition', condition)

            cond_frames.append(agg)
            print(f"     → {len(agg)} rows after aggregation")

        combined = pd.concat(cond_frames, ignore_index=True)
        out_path = os.path.join(out_dir, f"predictions_{organism}_{condition}_{timestamp}.csv")
        combined.to_csv(out_path, index=False)
        print(f"   Saved → {out_path}  ({len(combined)} rows)\n")

    print("Inference complete.")


if __name__ == "__main__":
    cli()
