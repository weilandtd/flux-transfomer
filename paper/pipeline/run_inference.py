"""
Run flux inference for all bio-replicates in a single condition xlsx.

Edit the USER CONFIGURATION section below with your paths, then run:

    .venv/bin/python paper/pipeline/run_inference.py
"""

import os
import sys
from datetime import datetime

import pandas as pd
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.inference import SigmaFlux
from sigmaflux.utils import read_xlsx_mid_data, read_xlsx_extflux


# ============================================================
# USER CONFIGURATION — edit these paths for your run
# ============================================================

CONFIG      = "paper/configs/i_orientalis.yaml"
CHECKPOINT  = "paper/I_orientalis_20260508/model/imisor_transformer.pt"

MID_FILE    = "data/I_orientalis/I_orientalis_C.xlsx"
EXT_FILE    = "data/I_orientalis/I_orientalis_ExtFlux.xlsx"
CONDITION   = "C"           # C, N, P, or B

MAPPING_CSV = "models/I_orientalis/I_orientalis_mapping.csv"

ORGANISM    = "I_orientalis"
OUT_DIR     = "paper/output"    # directory where predictions CSV is saved

# ============================================================
# END USER CONFIGURATION
# ============================================================


def _inspect_mid_file(xlsx_path: str) -> tuple[list[str], list[int]]:
    xl = pd.ExcelFile(xlsx_path)
    tracers  = xl.sheet_names
    df_first = xl.parse(tracers[0], nrows=1)
    rep_cols = [c for c in df_first.columns if str(c).startswith('Rep_')]
    bio_reps = sorted(int(str(c).split('_')[1]) for c in rep_cols)
    return tracers, bio_reps


if __name__ == '__main__':
    os.chdir(_REPO_ROOT)

    def _p(rel: str) -> str:
        return os.path.join(_REPO_ROOT, rel)

    with open(_p(CONFIG)) as _f:
        _cfg = yaml.safe_load(_f)
    _flux_names = _cfg.get('training', {}).get('measured_fluxes', None)

    _ckpt_path = _p(CHECKPOINT)
    _data_dir  = os.path.join(os.path.dirname(os.path.dirname(_ckpt_path)), 'data')
    sf = SigmaFlux(_p(CONFIG), _ckpt_path, data_dir=_data_dir)

    tracers, bio_reps = _inspect_mid_file(_p(MID_FILE))
    print(f"MID file:  {MID_FILE}")
    print(f"Condition: {CONDITION}")
    print(f"Tracers:   {tracers}")
    print(f"Bio-reps:  {bio_reps}")
    print()

    df_L = read_xlsx_mid_data(
        xlsx_path   = _p(MID_FILE),
        mapping_csv = _p(MAPPING_CSV),
    )
    print()

    df_Vm = read_xlsx_extflux(
        xlsx_path  = _p(EXT_FILE),
        condition  = CONDITION,
        flux_names = _flux_names,
    )
    print()

    print(f"Running inference ({df_L['Sample'].nunique()} bio-replicates)...")
    result = sf.predict(df_L, df_Vm=df_Vm)

    out_dir = _p(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(out_dir, f"predictions_{ORGANISM}_{CONDITION}_{timestamp}.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(result)} rows)")
