"""
OBSOLETE, will now raise TypeError if run -- kept only as the record of the Stage-3
experiment that led to the sentinel-value fix (see the plan). It compares
labeling_mask=None against the mask predict() used to build, but the labeling_mask
argument has since been removed from model.forward() entirely: both train.py and
inference.py fill unmeasured/dropped tokens with MISSING_LABEL_VALUE instead of masking
them from attention (see model.py, train.py, inference.py), and the parameter itself is
gone, not just unused. Rerunning this script will fail at `sf.predict(...)` with
"forward() got an unexpected keyword argument 'labeling_mask'" -- not a bug to fix, since
the mechanism it probes no longer exists anywhere in the codebase.

Original purpose, historical: did the train/inference labeling-mask mismatch actually
matter on real SC data, using the then-existing checkpoint, no retrain? It did (median
relative diff 1-3%, max up to 4.6x on specific reactions) -- that result motivated fixing
the mismatch, but the attention-masking fix it was built around was itself replaced by the
sentinel-value approach after masking cost real synthetic accuracy with no real-data
benefit (see verify_sentinel_fix_sc.py-style checks in the plan's Stage 3 section instead).

Usage (historical only):
    python paper/analysis/diagnostics/verify_mask_parity_sc.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "paper" / "scripts"))

from sigmaflux.ai.inference import load_predictor
from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path, load_config
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, load_condition_inputs)

RUN_DIR = _REPO_ROOT / "paper/.results/s_cerevisiae_20260831"
CONFIG = _REPO_ROOT / "paper/configs/s_cerevisiae.yaml"
DATA_DIR = _REPO_ROOT / "data/S_cervesiae"


def main():
    cfg, root = load_config(str(CONFIG), None)
    mapping_csv = resolve_path(mapping_csv_for(cfg), root)
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None

    sf = load_predictor(str(CONFIG), str(RUN_DIR / "ensemble" / "model_0.pt"), device="cpu")

    ext_file = _find_ext_flux_file(str(DATA_DIR))
    conditions = _discover_conditions(str(DATA_DIR))

    from cobra.io.json import load_json_model
    cobra_model = load_json_model(str(RUN_DIR / "model_prepared_for_mfa.json"))

    orig_forward = sf._model.forward

    def unmasked_forward(*args, **kwargs):
        kwargs["labeling_mask"] = None
        return orig_forward(*args, **kwargs)

    rows = []
    for cond in ["C", "N", "P"]:
        if cond not in conditions:
            continue
        mid_file = conditions[cond]
        for gc in [1, 2, 3, 4]:
            try:
                df_L, df_Vm, df_err = load_condition_inputs(
                    mid_file, ext_file, cond, gc, mapping_csv, flux_names, model=cobra_model)
            except Exception:
                continue
            if df_L.empty:
                continue

            sf._model.forward = orig_forward
            pred_masked = sf.predict(df_L, df_Vm=df_Vm)

            sf._model.forward = unmasked_forward
            pred_unmasked = sf.predict(df_L, df_Vm=df_Vm)
            sf._model.forward = orig_forward

            m = pred_masked.set_index("Flux Name")["Value"]
            u = pred_unmasked.set_index("Flux Name")["Value"]
            diff = (u - m)
            scale = m.abs().clip(lower=1.0)
            rel = (diff.abs() / scale)
            rows.append(dict(Condition=cond, GC=gc, n_reactions=len(m),
                             max_abs_diff=diff.abs().max(), median_rel_diff=rel.median(),
                             max_rel_diff=rel.max(),
                             worst_reaction=rel.idxmax(), worst_rel=rel.max()))
            print(f"{cond}{gc}: max|diff|={diff.abs().max():.4g}  median rel diff={rel.median():.4g}"
                  f"  max rel diff={rel.max():.4g} (on {rel.idxmax()})")

    out = pd.DataFrame(rows)
    out_path = RUN_DIR / "mask_parity_check_s_cerevisiae.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
