"""
Run SigmaFlux inference for all nutrient conditions and growth rates.

Discovers all condition MID files (<Name>_C.xlsx, <Name>_N.xlsx, …) in the
experimental data directory and runs all growth conditions within each.
Results are saved as per-condition CSVs in the output directory.

Usage:
    python paper/pipeline/inference/run_inference_workflow.py \\
        --config     paper/configs/i_orientalis.yaml \\
        --checkpoint paper/.results/i_orientalis_20260603/imisor_transformer.pt \\
        --data-dir   data/I_orientalis \\
        --out-dir    paper/.results/i_orientalis_20260603/predictions

Optional:
    --conditions C N P      run only these conditions (default: all discovered)
    --n-flux-samples 50     flux draws per bio-replicate (default: 50)
    --flux-to-sample EX_glc__D_e
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sigmaflux.ai.inference import load_predictor
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


def _expand_samples(
    df_L:           pd.DataFrame,
    df_Vm:          pd.DataFrame,
    flux_to_sample: str,
    df_err:         pd.DataFrame | None,
    n_samples:      int,
    rng:            np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replicate df_L and df_Vm n_samples times per bio-rep, perturbing flux_to_sample.

    For bio-rep R and flux sample S the synthetic sample ID is:
        (R - 1) * n_samples + S          (S in 0 .. n_samples-1)

    The perturbed flux is drawn uniformly from [value - error, value + error].
    """
    bio_reps = sorted(df_L['Sample'].unique())
    L_parts, V_parts = [], []

    err_idx = (
        df_err.set_index(['Sample', 'Flux Name'])['Value'].to_dict()
        if df_err is not None else {}
    )

    for rep in bio_reps:
        df_L_rep = df_L[df_L['Sample'] == rep].copy()
        vm_sample = rep if rep in df_Vm['Sample'].values else int(df_Vm['Sample'].iloc[0])
        df_Vm_rep = df_Vm[df_Vm['Sample'] == vm_sample].copy()

        base_rows = df_Vm_rep.loc[df_Vm_rep['Flux Name'] == flux_to_sample, 'Value']
        base_val  = float(base_rows.iloc[0]) if len(base_rows) else None
        err_val   = err_idx.get((vm_sample, flux_to_sample))

        for s in range(n_samples):
            syn_id = (rep - 1) * n_samples + s

            df_L_syn = df_L_rep.copy()
            df_L_syn['Sample'] = syn_id
            L_parts.append(df_L_syn)

            df_Vm_syn = df_Vm_rep.copy()
            df_Vm_syn['Sample'] = syn_id
            if base_val is not None and err_val is not None and err_val > 0:
                df_Vm_syn.loc[df_Vm_syn['Flux Name'] == flux_to_sample, 'Value'] = \
                    float(rng.uniform(base_val - err_val, base_val + err_val))
            V_parts.append(df_Vm_syn)

    return pd.concat(L_parts, ignore_index=True), pd.concat(V_parts, ignore_index=True)


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
    parser.add_argument("--n-flux-samples", type=int, default=50,
                        help="Independent flux draws per bio-replicate")
    parser.add_argument("--flux-to-sample", default="EX_glc__D_e",
                        help="Reaction whose flux is perturbed within ±error per bio-rep")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, root = load_config(args.config, args.root)
    config_path = cfg["_config_path"]

    t           = cfg.get("training", {})
    flux_names  = t.get("measured_fluxes") or None
    organism    = cfg.get("name", "organism")
    mapping_csv = cfg.get("mapping_csv")

    def _p(rel):
        return resolve_path(rel, root)

    checkpoint   = os.path.abspath(args.checkpoint)
    data_dir     = os.path.abspath(args.data_dir)
    out_dir      = os.path.abspath(args.out_dir)
    mapping_path = _p(mapping_csv) if mapping_csv else None

    ext_file = _find_ext_flux_file(data_dir)
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

    for condition in conditions:
        mid_file = all_condition_files[condition]
        gcs      = _inspect_growth_conditions(mid_file)

        print(f"══ Condition {condition} ({os.path.basename(mid_file)}) ══")
        print(f"   Growth conditions: {gcs}")

        cond_frames = []
        for gc in gcs:
            print(f"  ── Growth condition {gc} ──")

            df_L = read_xlsx_mid_data(
                xlsx_path        = mid_file,
                mapping_csv      = mapping_path,
                growth_condition = gc,
            )
            df_Vm, df_err = read_xlsx_extflux(
                xlsx_path        = ext_file,
                condition        = condition,
                flux_names       = flux_names,
                return_errors    = True,
                growth_condition = gc,
            )

            for _df in (df_L, df_Vm, df_err):
                _df['Sample'] = _df['Sample'].rank(method='dense').astype(int)

            bio_reps = sorted(df_L['Sample'].unique())
            print(f"     Bio-reps: {bio_reps}  |  MID rows: {len(df_L)}  |  Vm rows: {len(df_Vm)}")

            df_L_exp, df_Vm_exp = _expand_samples(
                df_L, df_Vm,
                flux_to_sample = args.flux_to_sample,
                df_err         = df_err,
                n_samples      = args.n_flux_samples,
                rng            = rng,
            )

            n_syn = df_L_exp['Sample'].nunique()
            print(f"     Running inference ({len(bio_reps)} bio-reps × "
                  f"{args.n_flux_samples} flux samples = {n_syn} synthetic samples)...")
            result = sf.predict(df_L_exp, df_Vm=df_Vm_exp)

            result['BioRep']          = result['Sample'] // args.n_flux_samples + 1
            result['GrowthCondition'] = gc
            result['Condition']       = condition

            n_flux = len(sf.flux_names)
            result['Type'] = (
                result.groupby('Sample', sort=False).cumcount()
                      .lt(n_flux).map({True: 'flux', False: 'reversibility'})
            )

            agg = (
                result
                .groupby(['Condition', 'GrowthCondition', 'BioRep', 'Type', 'Flux Name'],
                         sort=False)['Value']
                .mean()
                .reset_index()
            )
            cond_frames.append(agg)
            print(f"     → {len(agg)} rows after aggregation")

        combined = pd.concat(cond_frames, ignore_index=True)
        out_path = os.path.join(out_dir, f"predictions_{organism}_{condition}_{timestamp}.csv")
        combined.to_csv(out_path, index=False)
        print(f"   Saved → {out_path}  ({len(combined)} rows)\n")

    print("Inference complete.")


if __name__ == "__main__":
    cli()
