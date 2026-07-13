"""
believe.py — one-stop reliability tool for a real experiment.

Input : an experiment's MID measurements + measured exchange fluxes (the standard
        xlsx files) and the trained 20-model ensemble.
Does  : runs all 20 models on every replicate, and for each reaction reports
          - predicted flux  mu  +/-  calibrated 95% error bar (2.6*sigma)
          - BELIEVE HIGH / LOW  (bar < TOL * glucose uptake)
          - if LOW, the bottleneck: MEASUREMENT (fix the experiment) vs MODEL
            (better tracer/model), from a 2-way ANOVA over model x replicate.
No ground truth needed: the calibration factor c and threshold TOL were fixed once
on simulated data and are shipped constants here.

Usage:
  python paper/analysis/ensemble/believe.py \
    --config paper/configs/i_orientalis.yaml \
    --ensemble-dir <run>/ensemble --data-dir data/I_orientalis --out-dir <run>/believe
"""
import argparse, glob, os, sys, numpy as np, pandas as pd, yaml
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
sys.path.insert(0, os.path.join(_REPO, 'paper', 'pipeline', 'inference'))
from sigmaflux.ai.inference import SigmaFlux
from sigmaflux.utils import read_xlsx_mid_data, read_xlsx_extflux
from run_inference_workflow import _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions, _expand_samples

C_CALIB = 1.34          # shipped calibration factor (fit once on sim val -> 95% coverage)
Z = 1.96                # 95% normal quantile


def anova_1d(v):        # v: (M, R) model x replicate -> (var_model, var_rep)
    M, R = v.shape
    g = v.mean(); mm = v.mean(1); rm = v.mean(0)
    MS_m = (R * ((mm - g) ** 2).sum()) / max(M - 1, 1)
    MS_r = (M * ((rm - g) ** 2).sum()) / max(R - 1, 1)
    MS_e = ((v - mm[:, None] - rm[None, :] + g) ** 2).sum() / max((M - 1) * (R - 1), 1)
    return max((MS_m - MS_e) / R, 0.0), max((MS_r - MS_e) / M, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ensemble-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tol', type=float, default=0.20)      # believe if 95% bar < TOL * glucose uptake
    ap.add_argument('--n-flux-samples', type=int, default=50)
    ap.add_argument('--max-models', type=int, default=None)  # smoke test
    ap.add_argument('--conditions', nargs='+', default=None) # smoke test
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    mapping = cfg.get('mapping_csv'); mapping = mapping if os.path.isabs(mapping) else os.path.join(_REPO, mapping)
    conds = _discover_conditions(args.data_dir); ext = _find_ext_flux_file(args.data_dir)
    if args.conditions: conds = {k: v for k, v in conds.items() if k in args.conditions}
    members = sorted(glob.glob(os.path.join(args.ensemble_dir, 'model_*.pt')))
    if args.max_models: members = members[:args.max_models]
    print(f"ensemble: {len(members)} models | conditions: {sorted(conds)} | calibration c={C_CALIB}, tol={args.tol}")

    # cube[condition][reaction] accumulates (model, replicate) predictions
    cube = {c: {} for c in conds}          # condition -> reaction -> list per model of per-replicate arrays
    glucose = {}                            # condition -> |glucose uptake|
    flux_names = None
    for mi, member in enumerate(members):
        sf = SigmaFlux(args.config, member)
        flux_names = list(sf.flux_names); n_flux = len(flux_names)
        rng = np.random.default_rng(42)
        for cond in conds:
            mid_file = conds[cond]
            rep_frames = []
            for gc in _inspect_growth_conditions(mid_file):
                df_L = read_xlsx_mid_data(xlsx_path=mid_file, mapping_csv=mapping, growth_condition=gc)
                df_Vm, df_err = read_xlsx_extflux(xlsx_path=ext, condition=cond, flux_names=cfg['training']['measured_fluxes'],
                                                  return_errors=True, growth_condition=gc)
                for _d in (df_L, df_Vm, df_err): _d['Sample'] = _d['Sample'].rank(method='dense').astype(int)
                if mi == 0 and cond not in glucose:
                    g = df_Vm[df_Vm['Flux Name'] == 'EX_glc__D_e']['Value']
                    glucose[cond] = float(abs(g.mean())) if len(g) else np.nan
                dL, dV = _expand_samples(df_L, df_Vm, flux_to_sample='EX_glc__D_e', df_err=df_err,
                                         n_samples=args.n_flux_samples, rng=rng)
                res = sf.predict(dL, df_Vm=dV)
                res['BioRep'] = res['Sample'] // args.n_flux_samples + 1
                res['Type'] = res.groupby('Sample', sort=False).cumcount().lt(n_flux).map({True: 'flux', False: 'rev'})
                fl = res[res.Type == 'flux']
                # per (growth, biorep) replicate = mean over synthetic flux draws
                p = fl.groupby(['BioRep', 'Flux Name'])['Value'].mean().unstack('Flux Name').reindex(columns=flux_names)
                p.index = [f"{gc}_{b}" for b in p.index]
                rep_frames.append(p)
            allrep = pd.concat(rep_frames)                 # (replicates, reactions) for this model+condition
            for r in flux_names:
                cube[cond].setdefault(r, []).append(allrep[r].values)   # per model: array over replicates
        del sf

    # ── decide per condition, per reaction ──
    for cond in sorted(conds):
        rows = []
        glc = glucose.get(cond, np.nan)
        for r in flux_names:
            V = np.array(cube[cond][r])                    # (M, R)
            V = V[:, np.all(np.isfinite(V), axis=0)] if V.ndim == 2 else V
            if V.ndim != 2 or V.shape[1] < 2: continue
            mu_med, mu_mean = float(np.median(V)), float(V.mean())      # robust median (reported) + mean (compare)
            mad = float(np.median(np.abs(V - np.median(V))))
            sig = 1.4826 * mad if mad > 1e-12 else float(V.std())       # MAD-based robust spread (std fallback)
            bar = Z * C_CALIB * sig                         # calibrated 95% half-width
            rel = bar / glc if glc and glc > 0 else np.nan
            vm, vr = anova_1d(V)
            bottleneck = 'measurement' if vr > vm else 'model'
            rows.append({'reaction': r, 'flux': mu_med, 'flux_mean': mu_mean, 'err95': bar, 'rel_bar': rel,
                         'believe': 'HIGH' if (rel is not np.nan and rel < args.tol) else 'LOW',
                         'bottleneck': bottleneck,
                         'meas_frac': vr / (vm + vr) if (vm + vr) > 0 else np.nan})
        df = pd.DataFrame(rows)
        out = os.path.join(args.out_dir, f'believe_{cond}.csv')
        df.to_csv(out, index=False)
        hi = (df.believe == 'HIGH').sum()
        low = df[df.believe == 'LOW']
        print(f"\n══ Condition {cond} (glucose uptake={glc:.2f}) ══  {len(df)} reactions")
        print(f"  BELIEVE (HIGH): {hi}/{len(df)}  ({100*hi/len(df):.0f}%)")
        print(f"  LOW confidence: {len(low)}  -> bottleneck: model={ (low.bottleneck=='model').sum()}  measurement={ (low.bottleneck=='measurement').sum()}")
        print(f"  saved -> {out}")
    print("\ndone.")


if __name__ == '__main__':
    main()
