"""
KO flux-sampling for the knockout-detection experiment.

Reuses an existing WT run (its flux_samples.csv + EMU h5 chunks) and only samples
the KO block fresh, so we don't redo the expensive WT generation. For each of the
top-K non-essential reactions by WT flux variance, the reaction is constrained to 0
and a plain flux-sampling pass is run. A held-out subset of KO reactions is marked
"valonly" (their samples must go to validation only).

Outputs in --out-dir:
  flux_samples.csv   WT rows (copied verbatim) followed by KO rows  [rows = samples]
  ko_labels.csv      one row per sample: ko_reaction ("WT" or rxn_id), split
  + copies of the WT EMU h5 chunks (so only the KO block needs generating, at idx>=N_WT)

Usage:
  sigmaflux-sample-ko-flux --config paper/configs/i_orientalis.yaml \
      --wt-run paper/.results/i_orientalis_20260620 \
      --out-dir paper/.results/i_orientalis_ko_20260628 \
      --top-k 30 --holdout 5 --samples-per-ko 400
"""
import argparse, os, glob, shutil
import numpy as np, pandas as pd, yaml
from cobra.io import load_json_model

from sigmaflux.pipeline.reusable_sampler import ReusableSampler
from sigmaflux.pipeline.sample_flux import _do_sample


def cli(argv=None):
    """Console-script entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--wt-run', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--top-k', type=int, default=30)
    ap.add_argument('--holdout', type=int, default=5)
    ap.add_argument('--samples-per-ko', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-h5-copy', action='store_true', help='smoke test: skip copying WT h5 chunks')
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    s = dict(cfg['sampling'])
    s['stratified_sampling'] = {'enabled': False}    # plain sampling for KO pools
    s['n_samples'] = args.samples_per_ko
    measured = set(cfg['training'].get('measured_fluxes') or [])
    os.makedirs(args.out_dir, exist_ok=True)

    # ── WT block (reuse) ─────────────────────────────────────────────────────
    wt_csv = os.path.join(args.wt_run, 'flux_samples.csv')
    wt = pd.read_csv(wt_csv, index_col=0)                  # rows=samples, cols=reactions
    n_wt = len(wt)
    print(f"WT: {wt.shape} (samples x reactions) from {wt_csv}")
    rxn_cols = list(wt.columns)

    # ── pick top-K KO reactions by WT flux variance ──────────────────────────
    std = wt.std(axis=0)
    cand = [r for r in rxn_cols if r not in measured and not r.startswith('EX_')]
    ranked = std[cand].sort_values(ascending=False)
    model = load_json_model(os.path.join(args.wt_run, 'model_prepared_for_mfa.json'))
    chosen = []
    for r in ranked.index:
        if len(chosen) >= args.top_k:
            break
        try:
            rxn = model.reactions.get_by_id(r)
        except KeyError:
            continue
        with model:
            rxn.bounds = (0, 0)
            if model.optimize().status == 'optimal':
                chosen.append(r)
    print(f"chosen {len(chosen)} non-essential top-variance KO reactions")

    rng = np.random.default_rng(args.seed)
    holdout = set(rng.choice(chosen, size=min(args.holdout, len(chosen)), replace=False).tolist())
    print(f"held-out (val-only) KO reactions: {sorted(holdout)}")

    # ── sample each KO ───────────────────────────────────────────────────────
    # One sampler, built once; each KO below only mutates that reaction's bounds in the
    # already-built sampler instead of rebuilding it (see reusable_sampler.py). hopsy is
    # devel-only/unavailable in a released cobra (see _do_sample's fallback), so it never
    # actually reaches sampler.sample(); "optgp" just gives it working bookkeeping.
    method = s.get("method", "optgp")
    sampler = ReusableSampler(model, method=("optgp" if method == "hopsy" else method),
                              thinning=s["thinning"], processes=s["processes"], seed=args.seed)
    ko_blocks, labels = [wt[rxn_cols].reset_index(drop=True)], \
                        [pd.DataFrame({'ko_reaction': ['WT'] * n_wt, 'split': ['mixed'] * n_wt})]
    for i, r in enumerate(chosen):
        with sampler.temporary_bounds({r: (0, 0)}):
            if sampler.model.optimize().status != 'optimal':
                print(f"  skip {r} (infeasible)"); continue
            samp = _do_sample(sampler, s, args.seed + (i + 1) * 1000)
        samp = samp.reindex(columns=rxn_cols)             # align reaction columns
        # enforce exact zero for the knocked-out reaction
        samp[r] = 0.0
        ko_blocks.append(samp.reset_index(drop=True))
        labels.append(pd.DataFrame({'ko_reaction': [r] * len(samp),
                                    'split': ['valonly' if r in holdout else 'mixed'] * len(samp)}))
        print(f"  KO [{i+1}/{len(chosen)}] {r}: {len(samp)} samples ({'val-only' if r in holdout else 'mixed'})")

    flux_samples = pd.concat(ko_blocks, ignore_index=True)
    ko_labels = pd.concat(labels, ignore_index=True)
    assert len(flux_samples) == len(ko_labels)
    flux_samples.to_csv(os.path.join(args.out_dir, 'flux_samples.csv'))
    ko_labels.to_csv(os.path.join(args.out_dir, 'ko_labels.csv'))
    print(f"\nflux_samples: {flux_samples.shape}  (WT={n_wt}, KO={len(flux_samples)-n_wt})")
    print(f"saved -> {args.out_dir}/flux_samples.csv  + ko_labels.csv")

    # ── reuse WT artifacts: prepared model + WT h5 chunks (idx 0..n_wt) ───────
    shutil.copy(os.path.join(args.wt_run, 'model_prepared_for_mfa.json'), args.out_dir)
    if not args.skip_h5_copy:
        for f in glob.glob(os.path.join(args.wt_run, 'wb_model_*_data_*.h5')):
            shutil.copy(f, args.out_dir)
    print(f"copied prepared model + WT h5 chunks (KO block to generate: idx {n_wt}..{len(flux_samples)})")
    print(f"NEXT: generate EMU labeling for idx {n_wt}..{len(flux_samples)} only")


if __name__ == '__main__':
    cli()
