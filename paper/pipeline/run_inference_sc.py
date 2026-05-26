"""
Run flux inference for all growth conditions and bio-replicates — S. cerevisiae.

For each growth condition (dilution rate), input fluxes are sampled uniformly within
[flux - error, flux + error] (using GLC_IN ± GLC_IN_e from the ExtFlux file).  Each
bio-replicate is paired with N_FLUX_SAMPLES independently drawn flux vectors;
predictions are averaged across flux samples to yield one estimate per bio-replicate.

Edit the USER CONFIGURATION section below with your paths, then run:

    .venv/bin/python paper/pipeline/run_inference_sc.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.inference import SigmaFlux
from sigmaflux.utils import read_xlsx_mid_data, read_xlsx_extflux


# ============================================================
# USER CONFIGURATION — edit these paths for your run
# ============================================================

CONFIG      = "paper/configs/s_cerevisiae.yaml"
CHECKPOINT  = "/scratch/gpfs/MONA/Toki/MFA/sigmaflux/S_cerevisiae_20260523/sc_transformer.pt"

MID_FILE    = "data/S_cervesiae/S_cerevisiae_C.xlsx"
EXT_FILE    = "data/S_cervesiae/S_cerevisiae_ExtFlux.xlsx"
CONDITION   = "C"           # C, N, or P

MAPPING_CSV = "models/S_cerevisiae/S_cerevisiae_mapping.csv"

ORGANISM    = "S_cerevisiae"
OUT_DIR     = "paper/output"    # directory where predictions CSV is saved

# Flux sampling — perturb GLC_IN within its measurement error bounds
FLUX_TO_SAMPLE    = "EX_glc__D_e"  # reaction whose flux is perturbed
N_FLUX_SAMPLES    = 50             # independent uniform samples per bio-rep
GROWTH_CONDITIONS = None           # None = all; or a list e.g. [1, 2, 3]

# ============================================================
# END USER CONFIGURATION
# ============================================================


def _inspect_mid_file(xlsx_path: str) -> list[int]:
    """Return sorted list of GrowthCondition numbers present in the MID file."""
    import re
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
    Other fluxes keep their point values.
    """
    bio_reps = sorted(df_L['Sample'].unique())
    L_parts  = []
    V_parts  = []

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
                sampled = float(rng.uniform(base_val - err_val, base_val + err_val))
                df_Vm_syn.loc[df_Vm_syn['Flux Name'] == flux_to_sample, 'Value'] = sampled
            V_parts.append(df_Vm_syn)

    return pd.concat(L_parts, ignore_index=True), pd.concat(V_parts, ignore_index=True)


if __name__ == '__main__':
    os.chdir(_REPO_ROOT)

    def _p(rel: str) -> str:
        return os.path.join(_REPO_ROOT, rel)

    with open(_p(CONFIG)) as _f:
        _cfg = yaml.safe_load(_f)
    _flux_names    = _cfg.get('training', {}).get('measured_fluxes', None)
    _negate_fluxes = set(_cfg.get('training', {}).get('negate_fluxes', []))

    _ckpt_path = _p(CHECKPOINT)
    _data_dir  = os.path.join(os.path.dirname(os.path.dirname(_ckpt_path)), 'data')
    sf = SigmaFlux(_p(CONFIG), _ckpt_path, data_dir=_data_dir)

    all_gcs    = _inspect_mid_file(_p(MID_FILE))
    gcs_to_run = GROWTH_CONDITIONS if GROWTH_CONDITIONS is not None else all_gcs
    print(f"MID file:     {MID_FILE}")
    print(f"Condition:    {CONDITION}")
    print(f"Growth conds: {gcs_to_run}")
    print(f"Flux samples: {N_FLUX_SAMPLES} (perturbing {FLUX_TO_SAMPLE})")
    print()

    rng        = np.random.default_rng(seed=42)
    all_frames = []

    for gc in gcs_to_run:
        print(f"── Growth condition {gc} ──")

        df_L = read_xlsx_mid_data(
            xlsx_path        = _p(MID_FILE),
            mapping_csv      = _p(MAPPING_CSV),
            growth_condition = gc,
        )

        df_Vm, df_err = read_xlsx_extflux(
            xlsx_path        = _p(EXT_FILE),
            condition        = CONDITION,
            flux_names       = _flux_names,
            return_errors    = True,
            growth_condition = gc,
        )

        # Flip sign for fluxes where the xlsx uses opposite convention to the model
        if _negate_fluxes:
            mask = df_Vm['Flux Name'].isin(_negate_fluxes)
            df_Vm.loc[mask, 'Value'] *= -1

        # Normalise sample IDs to 1-based within this GC
        for _df in (df_L, df_Vm, df_err):
            _df['Sample'] = _df['Sample'].rank(method='dense').astype(int)

        bio_reps = sorted(df_L['Sample'].unique())
        print(f"  Bio-reps: {bio_reps}  |  MID rows: {len(df_L)}  |  "
              f"Vm rows: {len(df_Vm)}")

        df_L_exp, df_Vm_exp = _expand_samples(
            df_L, df_Vm,
            flux_to_sample = FLUX_TO_SAMPLE,
            df_err         = df_err,
            n_samples      = N_FLUX_SAMPLES,
            rng            = rng,
        )

        n_syn = df_L_exp['Sample'].nunique()
        print(f"  Running inference ({len(bio_reps)} bio-reps × "
              f"{N_FLUX_SAMPLES} flux samples = {n_syn} synthetic samples)...")
        result = sf.predict(df_L_exp, df_Vm=df_Vm_exp)

        result['BioRep']          = result['Sample'] // N_FLUX_SAMPLES + 1
        result['GrowthCondition'] = gc
        result['Condition']       = CONDITION

        # Tag flux vs reversibility rows
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
        all_frames.append(agg)
        print(f"  → {len(agg)} rows after aggregation")

    combined  = pd.concat(all_frames, ignore_index=True)
    out_dir   = _p(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(
        out_dir,
        f"predictions_{ORGANISM}_{CONDITION}_{timestamp}.csv",
    )
    combined.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(combined)} rows)")
