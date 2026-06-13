"""
Token-level informative-measurement selection for flux estimation.

This is the **token-granular** counterpart of ``informative_metabolites.py``.
Instead of scoring/selecting whole metabolites (all of a metabolite's mass
isotopologues treated as one block), every individual **token** -- a single
``(metabolite, M+k)`` mass-isotopologue fraction -- is a separate candidate
measurement.  This matters because a metabolite is often informative only at a
specific isotopologue (e.g. only ``glu__L_c M+5`` carries flux information while
its other M+k carry almost none); selecting at the metabolite level would either
drag in uninformative tokens or drop informative ones.

It replaces the pipeline's correlation score (``corr_yl`` max|r| token
information in ``train_ai_model.py``) with the Fisher / D-optimal experimental
design machinery in ``sigmaflux.design.observability``:

* each token's prior-averaged Fisher information about the free fluxes is
  estimated from generated samples (cross-fitted empirical sensitivity Gram),
* a permutation null calibrates a significance floor (``significant`` flag),
* greedy D-optimal selection picks a non-redundant subset of tokens (a token
  whose sensitivity lies in already-covered flux directions adds ~nothing).

A token candidate keyed ``met|M+k`` carries one column per tracer experiment
(the same fragment measured under each tracer), so its information is the joint
information across the tracers, matching how the data is actually collected.

Outputs a per-token CSV and prints (i) the external-flux distribution summary
(glucose uptake + biomass component fluxes) and (ii) the headline counts:
total candidate tokens vs significant vs greedily-selected tokens.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from cobra.io.json import load_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.ai.utils import compute_sparse_nullspace
from sigmaflux.design import observability as obs


def _load_labeling(labeling_path):
    """Read the labeling matrix from a single ``.h5`` or a directory of
    ``wb_model_labeling_data_<start>_<end>.h5`` generation chunks, concatenated
    along the sample axis in a contiguous non-overlapping cover (so labeling
    columns line up with flux-sample rows)."""
    if not os.path.isdir(labeling_path):
        return pd.read_hdf(labeling_path)
    files = glob.glob(os.path.join(labeling_path, "wb_model_labeling_data_*.h5"))
    if not files:
        raise FileNotFoundError(f"No wb_model_labeling_data_*.h5 in {labeling_path}")
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
                print(f"WARNING: {os.path.basename(f)} has {df.shape[1]} "
                      f"samples but declares {end - start}; skipping.")
                continue
            chosen.append(df)
            covered = end
    return pd.concat(chosen, axis=1)


def load_generated_tokens(model_json, flux_csv, labeling_h5, n_samples):
    """Load Z, theta samples, and per-**token** MID samples.

    Returns ``theta`` ``(n, d)`` and ``token_states`` keyed by
    ``"<met_id>|M+k"`` -> ``(n, n_tracers)`` (one column per tracer experiment
    in which that fragment is measured)."""
    model = load_json_model(model_json)
    reaction_ids = [r.id for r in model.reactions]
    S = create_stoichiometric_matrix(model)
    Z = np.asarray(compute_sparse_nullspace(S), dtype=np.float64)  # (n_rxn, d)

    flux = pd.read_csv(flux_csv, index_col=0)
    lab = _load_labeling(labeling_h5)                    # (n_tokens, n_lab_samples)

    n = min(n_samples, flux.shape[0], lab.shape[1])
    V = flux.iloc[:n][reaction_ids].to_numpy(np.float64)  # (n, n_rxn)
    theta = V @ Z                                         # (n, d)
    lab = lab.iloc[:, :n]

    # One candidate per (Metabolite, Isotopomer); its columns are the tracers.
    token_states = {}
    for (met_id, iso), sub in lab.groupby(level=["Metabolite", "Isotopomer"]):
        token_states[f"{met_id}|{iso}"] = sub.to_numpy(np.float64).T  # (n, n_tracers)
    return theta, token_states, Z.shape[1]


def report_external_fluxes(flux_csv, n_samples):
    """Print the distribution (range/quantiles) of the external/boundary fluxes
    that anchor the design problem: glucose uptake and the biomass-component
    fluxes."""
    flux = pd.read_csv(flux_csv, index_col=0).iloc[:n_samples]
    keys = ["EX_glc__D_e", "BIOMASS", "PROTEIN_c", "RNA_c", "DNA_c",
            "LIPID_c", "CARB_c", "EX_o2_e", "EX_co2_e", "EX_etoh_e",
            "EX_ac_e", "EX_glyc_e", "EX_succ_e"]
    present = [k for k in keys if k in flux.columns]
    sub = flux[present]
    desc = sub.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc = desc[["min", "5%", "25%", "50%", "75%", "95%", "max", "mean", "std"]]
    print(f"\n=========  EXTERNAL / BIOMASS FLUX DISTRIBUTION  (n={len(sub)})  =========")
    with pd.option_context("display.width", 160, "display.float_format", lambda x: f"{x:8.3f}"):
        print(desc.to_string())
    missing = [k for k in keys if k not in flux.columns]
    if missing:
        print(f"(not in model / flux table: {', '.join(missing)})")
    return desc


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    report_external_fluxes(args.flux_samples, args.n_samples)

    print(f"\nLoading generated data ({args.n_samples} samples) ...")
    theta, token_states, d = load_generated_tokens(
        args.model, args.flux_samples, args.labeling, args.n_samples)

    # Random subsample that MATCHES train_ai_model.py exactly: it does
    #   rng = np.random.default_rng(seed); idx = rng.permutation(n_total)[:N]
    # on the same start-index-ordered sample columns, so running selection on
    # the same (seed, N) picks the identical random subset the model trains on.
    if args.subsample_n is not None and args.subsample_n < theta.shape[0]:
        rng = np.random.default_rng(args.subsample_seed)
        idx = rng.permutation(theta.shape[0])[:args.subsample_n]
        theta = theta[idx]
        token_states = {k: v[idx] for k, v in token_states.items()}
        print(f"Random subsample (matches training): {args.subsample_n} of "
              f"{idx.max()+1 if len(idx) else 0}-pool (seed={args.subsample_seed})")

    n_all_tokens = len(token_states)
    print(f"  free-flux directions d = {d}; "
          f"candidate TOKENS = {n_all_tokens}; samples = {theta.shape[0]}")

    df, summary = obs.rank_from_samples(
        theta, token_states,
        noise_std=args.noise_std,
        gain_threshold=args.gain_threshold,
        budget=args.budget,
        run_mi_audit=not args.no_mi,
    )

    # split the "met|M+k" key back into columns for readability
    df = df.rename(columns={"metabolite": "token"})
    df[["metabolite_id", "isotopomer"]] = df["token"].str.split("|", expand=True)
    front = ["token", "metabolite_id", "isotopomer", "selected", "significant",
             "greedy_rank", "marginal_gain_nats", "standalone_gain_nats"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(args.output, index=False)

    n_sig = int(df["significant"].sum())
    n_sel = int(df["selected"].sum())

    print("\n========================  TOKEN SELECTION SUMMARY  ========================")
    print(f"Free-flux directions (null space)        : {summary['n_free_flux_directions']}")
    print(f"Effective rank (info concentration)      : {summary['effective_rank']:.2f}")
    print(f"Directions resolved >=50% / 90% / 99% var: "
          f"{summary['n_resolved_directions']} / {summary['n_resolved_90pct']} / {summary['n_resolved_99pct']}")
    print(f"Noise std assumed                        : {summary['noise_std']}")
    print(f"Null significance floor (nats)           : {summary['null_floor_nats']:.4f}")
    print("-" * 74)
    print(f"TOTAL candidate tokens                   : {n_all_tokens}")
    print(f"  ... significant (info > null floor)    : {n_sig}")
    print(f"  ... selected (greedy D-optimal subset) : {n_sel}")
    print(f"  unique metabolites among selected      : {df.loc[df.selected, 'metabolite_id'].nunique()}")
    print("-" * 74)

    ev = summary["eigenvalues"]
    if len(ev):
        print("Top observable eigenvalues (info gain/direction): "
              + ", ".join(f"{e:.1f}" for e in ev[:10]))

    print("\n----------  Greedy minimal informative token subset  ----------")
    sel = df[df["selected"]].sort_values("greedy_rank")
    cols = ["token", "marginal_gain_nats", "standalone_gain_nats"]
    if "nonlinearity_gap_nats" in df.columns:
        cols.append("nonlinearity_gap_nats")
    with pd.option_context("display.max_rows", 80, "display.width", 120):
        print(sel[cols].to_string(index=False))

    # how many isotopologue tokens per selected metabolite (the headline point:
    # selection keeps specific M+k, not whole metabolites)
    print("\n----  Selected tokens per metabolite (top 25 metabolites)  ----")
    per_met = (df[df.selected].groupby("metabolite_id")["isotopomer"]
               .agg(lambda s: ", ".join(sorted(s, key=lambda x: int(x.split("+")[1])))))
    per_met_n = df[df.selected].groupby("metabolite_id").size().sort_values(ascending=False)
    for met in per_met_n.head(25).index:
        print(f"  {met:<22} [{per_met_n[met]}]  {per_met[met]}")

    print(f"\nResults written to: {args.output}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="prepared COBRA model json (model_prepared_for_mfa.json)")
    p.add_argument("--flux-samples", required=True, help="flux_samples.csv")
    p.add_argument("--labeling", required=True, help="generated labeling .h5 file OR directory of chunks")
    p.add_argument("--n-samples", type=int, default=10000,
                   help="how many aligned samples to load (use the full pool, e.g. 20000)")
    p.add_argument("--subsample-n", type=int, default=None,
                   help="random subsample size AFTER loading; matches train_ai_model permutation")
    p.add_argument("--subsample-seed", type=int, default=42,
                   help="seed for the subsample permutation (= training config seed)")
    p.add_argument("--noise-std", type=float, default=0.02,
                   help="labeling measurement noise std (as in apply_labeling_noise)")
    p.add_argument("--gain-threshold", type=float, default=1e-2,
                   help="stop greedy selection when marginal info gain (nats) < this")
    p.add_argument("--budget", type=int, default=None, help="optional hard cap on subset size")
    p.add_argument("--no-mi", action="store_true", help="skip the MI / nonlinearity audit")
    p.add_argument("--output", default="paper/analysis/output/informative_tokens.csv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
