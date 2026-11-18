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
    --fbapro                project every member draw with FBApro onto S.v = 0, pinning
                            the reactions ``experiment.knockouts`` lists for that growth
                            condition (e.g. ``B4: [R1]``) to zero

Measurement uncertainty is propagated through the ensemble by resampling cross-tracer
combinations of the measured MID replicates, one flux draw each; ``experiment.n_bootstrap``
sets how many. See :mod:`sigmaflux.ai.uncertainty` for the composition.
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
from sigmaflux.ai.uncertainty import (
    DEFAULT_N_BOOTSTRAP, sample_inputs, summarize_draws,
)
from sigmaflux.pipeline._paths import (
    add_root_argument, load_config, mapping_csv_for, resolve_path,
)
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


def prepared_network_for(checkpoint):
    """The run's ``model_prepared_for_mfa.json``, or None if it is not beside *checkpoint*.

    Needed only for its measurement-pool annotations, which decide whether a measured
    compound resolves to one whole-cell token or to one per compartment.
    """
    for directory in (os.path.dirname(os.path.abspath(checkpoint)),
                      os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))):
        path = os.path.join(directory, "model_prepared_for_mfa.json")
        if os.path.isfile(path):
            from cobra.io.json import load_json_model
            return load_json_model(path)
    return None


def load_condition_inputs(mid_file, ext_file, condition, gc, mapping_path, flux_names,
                          model=None):
    """Measured MID and exchange-flux frames for one growth condition.

    ``Sample`` is dense-ranked on both sides so the replicate ids line up regardless of
    how the workbook numbered them.

    Args:
        mid_file: per-condition MID xlsx.
        ext_file: the *_ExtFlux.xlsx for this organism.
        condition: condition prefix, e.g. 'B'.
        gc: growth-condition number within that file.
        mapping_path: experimental-id -> model-id CSV.
        flux_names: measured flux columns to keep, or None for all.
        model: prepared cobra model, so compounds the platform reports as one pool
            resolve to that pool's token rather than to one token per compartment.

    Returns ``(df_L, df_Vm, df_err)``.
    """
    df_L = read_xlsx_mid_data(xlsx_path=mid_file, mapping_csv=mapping_path,
                              growth_condition=gc)
    df_L['Sample'] = df_L['Sample'].rank(method='dense').astype(int)
    df_Vm, df_err = read_xlsx_extflux(xlsx_path=ext_file, condition=condition,
                                      flux_names=flux_names, return_errors=True,
                                      growth_condition=gc)
    for _df in (df_Vm, df_err):
        _df['Sample'] = _df['Sample'].rank(method='dense').astype(int)
    return df_L, df_Vm, df_err


def predict_draws(sf, df_L_draws, df_Vm_draws, n_flux):
    """Raw per-(sample, member) predictions, tagged flux vs reversibility.

    Outputs come back per sample in flux-then-reversibility order and a reaction can
    appear in both, so 'Type' is assigned by position -- it is part of what identifies a
    predicted quantity, not a property of the name.

    Args:
        sf: predictor, single or ensemble.
        df_L_draws: labeling inputs, one Sample per draw.
        df_Vm_draws: matching measured-flux inputs.
        n_flux: number of flux outputs, i.e. where the reversibility block starts.
    """
    raw = (sf.predict(df_L_draws, df_Vm=df_Vm_draws, aggregate=False)
           if hasattr(sf, 'members') else sf.predict(df_L_draws, df_Vm=df_Vm_draws))
    member_key = ['Sample', 'Member'] if 'Member' in raw.columns else ['Sample']
    raw['Type'] = (raw.groupby(member_key, sort=False).cumcount()
                      .lt(n_flux).map({True: 'flux', False: 'reversibility'}))
    return raw


def fbapro_draws(sf, raw, n_flux, knockout_rxn_ids):
    """*raw* with every member's flux states projected by FBApro, knockouts pinned to 0.

    Args:
        sf: ensemble predictor.
        raw: :func:`predict_draws` output.
        n_flux: number of flux outputs.
        knockout_rxn_ids: reactions to pin to zero; empty projects onto S.v = 0 alone.
    """
    if 'Member' not in raw.columns:
        raise ValueError("FBApro weights by member variance, so it needs an ensemble")
    flux = raw['Type'] == 'flux'
    names = raw.loc[flux, 'Flux Name'].to_numpy().reshape(-1, n_flux)
    if not (names == np.asarray(sf.flux_names)).all():
        raise ValueError("flux rows are not in flux_names order per member draw")
    blocks = raw.loc[flux, 'Value'].to_numpy().reshape(raw['Member'].nunique(), -1, n_flux)
    raw = raw.copy()
    raw.loc[flux, 'Value'] = sf.project_knockouts(blocks, knockout_rxn_ids).ravel()
    return raw


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
    parser.add_argument("--fbapro", action="store_true",
                        help="Project predictions with FBApro, pinning experiment.knockouts")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, root = load_config(args.config, args.root)
    config_path = cfg["_config_path"]

    t           = cfg.get("training", {})
    flux_names  = t.get("measured_fluxes") or None
    organism    = cfg.get("name", "organism")
    mapping_csv = mapping_csv_for(cfg)
    # experiment.ext_flux lets one organism pick a specific *_ExtFlux.xlsx out of a
    # data_dir shared with another organism, when their measured-flux reaction ids
    # differ (e.g. e_coli's iAF1260 ids vs e_coli_core's naming in the same shared
    # file) — same override pattern as experiment.mapping_csv above. Defaults to the
    # single *_ExtFlux.xlsx glob match when absent.
    ext_flux    = cfg.get("experiment", {}).get("ext_flux")
    # Each tracer is its own culture, so replicate i of one says nothing about replicate i
    # of another; draws are always bootstrapped combinations, experiment.n_bootstrap only
    # sets how many.
    n_bootstrap = cfg.get("experiment", {}).get("n_bootstrap", DEFAULT_N_BOOTSTRAP)
    knockouts   = cfg.get("experiment", {}).get("knockouts") or {}

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

    # Only for its measurement-pool annotations: without them a compound the platform
    # reports as one pool would be written into each compartment separately, and every
    # such token would then be dropped as absent from the trained token index.
    network = prepared_network_for(checkpoint)

    n_flux = len(sf.flux_names)

    for condition in conditions:
        mid_file = all_condition_files[condition]
        gcs      = _inspect_growth_conditions(mid_file)

        print(f"══ Condition {condition} ({os.path.basename(mid_file)}) ══")
        print(f"   Growth conditions: {gcs}")

        cond_frames = []
        for gc in gcs:
            print(f"  ── Growth condition {gc} ──")

            df_L, df_Vm, df_err = load_condition_inputs(
                mid_file, ext_file, condition, gc, mapping_path, flux_names,
                model=network)

            n_draws = int(n_bootstrap)
            reps_by_tracer = df_L.groupby('Tracer Name')['Sample'].nunique()
            print(f"     Replicates per tracer: {reps_by_tracer.to_dict()}"
                  f"  |  MID rows: {len(df_L)}  |  Vm rows: {len(df_Vm)}")

            df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=rng,
                                                    n_bootstrap=n_bootstrap,
                                                    bounds=sf.flux_bounds())

            print(f"     Running inference ({n_draws} bootstrapped cross-tracer draws "
                  f"+ 1 maximum-likelihood input, bias-corrected percentile interval)...")

            raw = predict_draws(sf, df_L_draws, df_Vm_draws, n_flux)
            if args.fbapro:
                pinned = knockouts.get(f"{condition}{gc}", [])
                print(f"     FBApro: pinning {pinned or 'nothing'} to zero")
                raw = fbapro_draws(sf, raw, n_flux, pinned)

            agg = summarize_draws(raw, n_draws=n_draws,
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
