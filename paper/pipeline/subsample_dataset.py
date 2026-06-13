"""
Randomly subsample N samples from a generated training dataset.

Reads the contiguous labeling/flux/reversibility chunks + flux_samples.csv from
SRC, draws N column indices uniformly without replacement (seeded), and writes a
single-chunk dataset to DST so that downstream token selection and training both
operate on the *same* random subset.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def load_contiguous(src, stem):
    files = glob.glob(os.path.join(src, f"{stem}_*.h5"))
    spans = []
    for f in files:
        m = re.search(r"_(\d+)_(\d+)\.h5$", f)
        spans.append((int(m.group(1)), int(m.group(2)), f))
    spans.sort(key=lambda x: (x[0], x[1]))
    chosen, covered = [], 0
    for start, end, f in spans:
        if start == covered:
            df = pd.read_hdf(f)
            if df.shape[1] != end - start:
                print(f"  skip partial {os.path.basename(f)} ({df.shape[1]} != {end-start})")
                continue
            chosen.append(df)
            covered = end
    out = pd.concat(chosen, axis=1)
    print(f"  {stem}: {out.shape} (contiguous 0:{covered})")
    return out, covered


def main(a):
    os.makedirs(a.dst, exist_ok=True)
    lab, nlab = load_contiguous(a.src, "wb_model_labeling_data")
    flx, nflx = load_contiguous(a.src, "wb_model_flux_data")
    rev, nrev = load_contiguous(a.src, "wb_model_reversibility_data")
    n_avail = min(nlab, nflx, nrev)
    flux_csv = pd.read_csv(os.path.join(a.src, "flux_samples.csv"), index_col=0)

    n = min(a.n, n_avail)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n_avail, size=n, replace=False))
    print(f"Random subsample: {n} of {n_avail} (seed={a.seed})")

    suffix = f"_0_{n}"
    lab.iloc[:, idx].to_hdf(os.path.join(a.dst, f"wb_model_labeling_data{suffix}.h5"), key="labeling", mode="w")
    flx.iloc[:, idx].to_hdf(os.path.join(a.dst, f"wb_model_flux_data{suffix}.h5"), key="fluxes", mode="w")
    rev.iloc[:, idx].to_hdf(os.path.join(a.dst, f"wb_model_reversibility_data{suffix}.h5"), key="reversibility", mode="w")
    flux_csv.iloc[idx].to_csv(os.path.join(a.dst, "flux_samples.csv"))

    src_model = os.path.join(a.src, "model_prepared_for_mfa.json")
    if os.path.isfile(src_model):
        import shutil
        shutil.copy(src_model, os.path.join(a.dst, "model_prepared_for_mfa.json"))
    print(f"Wrote {n}-sample subset to {a.dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
