"""
Wall time of SigmaFlux's reported flux prediction on benchmark samples.

Predicts the validation split's noisy benchmark inputs (the same draw the MFA benchmark
fits) through ``EnsembleSigmaFlux.predict(aggregate=True)`` -- every member, projected and
aggregated per sample -- and records the time per repeat. Model loading is timed
separately: it is paid once, not per sample.

Output: <run-dir>/timing/sigmaflux_predict.json

Usage:
    python paper/scripts/time_sigmaflux.py --run-dir paper/.results/i_orientalis_20260921 \\
        --config paper/configs/i_orientalis.yaml
"""

import argparse
import json
import pathlib
import time

import torch

from sigmaflux.ai.inference import load_predictor
from sigmaflux.pipeline._benchmark import load_benchmark_inputs, pin_threads, select_samples
from sigmaflux.pipeline._paths import add_root_argument, load_config, resolve_path
from sigmaflux.pipeline.run_ai_benchmark import build_df_L, build_df_Vm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Training run directory")
    p.add_argument("--config", required=True, help="Organism YAML config")
    add_root_argument(p)
    p.add_argument("--n-samples", type=int, default=1000, help="Samples per repeat")
    p.add_argument("--repeats", type=int, default=3, help="Timed repeats")
    p.add_argument("--seed", type=int, default=42, help="Benchmark noise seed")
    return p.parse_args()


def main():
    args = parse_args()
    pin_threads()
    cfg, root = load_config(args.config, args.root)
    data_dir = pathlib.Path(resolve_path(args.run_dir, root))

    started = time.perf_counter()
    sf = load_predictor(args.config, str(data_dir / "ensemble"), data_dir=str(data_dir))
    load_seconds = time.perf_counter() - started

    _, mapped_met_ids, labeling_df, flux_df, _ = load_benchmark_inputs(str(data_dir), cfg, root)
    sel, _ = select_samples("val", sf.split_source, flux_df.shape[1], cfg, 0, args.n_samples)
    inv_tracer = {v: k for k, v in sf._tracer_name_to_id.items()}
    measured = cfg.get("training", {}).get("measured_fluxes", [])
    df_L = build_df_L(labeling_df, sel, 0.01, args.seed, inv_tracer, keep_met_ids=mapped_met_ids)
    df_Vm = build_df_Vm(flux_df, measured, sel, 0.01, 0.05, labeling_df, args.seed,
                        keep_met_ids=mapped_met_ids)

    seconds = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        sf.predict(df_L, df_Vm=df_Vm, aggregate=True)
        seconds.append(time.perf_counter() - started)
        print(f"  {len(sel)} samples in {seconds[-1]:.1f}s")

    out = data_dir / "timing"
    out.mkdir(exist_ok=True)
    record = dict(n_samples=len(sel), n_members=len(sf.members), seconds=seconds,
                  load_seconds=load_seconds, threads=torch.get_num_threads())
    (out / "sigmaflux_predict.json").write_text(json.dumps(record, indent=2))
    print(f"Saved: {out / 'sigmaflux_predict.json'}  (model load {load_seconds:.1f}s)")


if __name__ == "__main__":
    main()
