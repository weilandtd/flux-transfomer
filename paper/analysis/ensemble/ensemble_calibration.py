"""
Calibrate the ensemble uncertainty into a validated per-reaction reliability rule.

On the held-out val set (truth known) the 20 members give mean mu and std sigma per
(sample, reaction). We:
  1. measure COVERAGE of mu +/- 2 sigma  (does the truth land inside ~95% of the time?)
  2. fit ONE calibration factor c so mu +/- 1.96 c sigma has 95% coverage
     (deep ensembles are usually overconfident -> c > 1)
  3. turn c*sigma into a per-reaction calibrated relative uncertainty
        u_r = 1.96 c median(sigma_r) / (flux dynamic range on val)
  4. find the u_r threshold that earns an R2 guarantee -> the BELIEVE set.

Output: reliability diagram + decision map (R2 vs u_r) + a per-reaction table
(reaction, predicted flux, calibrated 95% half-width, believe HIGH/LOW).

Usage: python paper/analysis/ensemble/ensemble_calibration.py --ensemble-dir <run>/ensemble
"""
import argparse, glob, os, sys, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import norm
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src'))
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'believe': '#4C72B0', 'nope': '#DD8452', 'ref': '#000000'}


def per_reaction_r2(pred, gt):
    sr = ((pred - gt) ** 2).sum(0); st = ((gt - gt.mean(0)) ** 2).sum(0)
    return np.where(st > 1e-9, 1 - sr / st, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble-dir', required=True)
    ap.add_argument('--n-val', type=int, default=2000)
    ap.add_argument('--r2-target', type=float, default=0.9)
    args = ap.parse_args()

    members = sorted(glob.glob(os.path.join(args.ensemble_dir, 'model_*.pt')))
    ck0 = torch.load(members[0], map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck0['model_config'], ck0['null_space_data']
    flux_names = list(ck0['flux_names']); val_idx = np.asarray(ck0['val_indices'], int)[:args.n_val]
    lab, flux, rev = load_data(ck0['data_dir']); ti = ck0.get('token_index')
    if ti is not None: lab = lab.loc[ti]
    _, base = create_tracer_mapping(lab); S, cobra = load_stoichiometric_matrix(ck0['model_json'])
    vocabs = build_radial_vocabs(cobra, lab, base, h_compound=cfg_dict['d_comp']//2,
                                 h_label=cfg_dict['d_label']//2, h_tracer=cfg_dict['d_tracer']//2)
    val_ds, _, _, _, temb = prepare_id_dataset(lab, flux, rev, vocabs=vocabs, train_indices=val_idx,
                                               measured_fluxes=ck0.get('measured_fluxes'), norm_stats_input=ck0.get('norm_stats'))
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__ if k in cfg_dict})
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    (c, l, t, fr), tg = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))
    c, l, t, fr, tg = [x.to(dev) for x in (c, l, t, fr, tg)]
    gt = tg[:, :len(flux_names)].cpu().numpy(); K = len(flux_names)
    preds = []
    for f in members:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                        compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'], tracer_emb_table=temb,
                        reaction_emb_table=radial_embedding_table(K, ck['h_reaction']),
                        type_emb_table=radial_embedding_table(2, ck['h_type']), null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
        m.load_state_dict(ck['model_state_dict']); m.eval(); m.to(dev)
        fid = torch.as_tensor(np.asarray(m.fixed_idx), dtype=torch.long, device=dev)
        with torch.no_grad():
            preds.append(m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid), measured_flux_ids=fid, return_attn=False)[0].cpu().numpy())
    P = np.stack(preds)                                    # (M, S, K)
    mu, sig = P.mean(0), P.std(0)                           # (S,K)
    err = np.abs(mu - gt)
    # standardized residual (floor sigma by a small fraction of each reaction's flux range)
    frange = gt.std(0, keepdims=True)                      # (1,K) flux dynamic range on val
    sig_eff = np.maximum(sig, 1e-3 * np.clip(frange, 1e-9, None))
    z = err / sig_eff
    fin = np.isfinite(z)

    # ── 1) coverage + calibration ──
    cov2 = float(np.mean(err[fin] < 2 * sig_eff[fin]))
    cov196 = float(np.mean(err[fin] < 1.96 * sig_eff[fin]))
    cfac = float(np.percentile(z[fin], 95) / 1.96)         # calibration factor
    cov_cal = float(np.mean(err[fin] < 1.96 * cfac * sig_eff[fin]))
    print(f"ensemble: M={len(members)} | val={len(val_idx)} | reactions={K}")
    print(f"RAW coverage of  mu +/- 2 sigma      : {cov2:.3f}   (target 0.95)")
    print(f"RAW coverage of  mu +/- 1.96 sigma   : {cov196:.3f}")
    print(f"calibration factor c (for 95% cover) : {cfac:.2f}   ({'overconfident' if cfac>1 else 'ok/under'})")
    print(f"coverage of  mu +/- 1.96*c*sigma     : {cov_cal:.3f}")

    # ── 2) per-reaction calibrated relative uncertainty + R2 ──
    r2 = per_reaction_r2(mu, gt)
    med_sig = np.median(sig, 0)                            # per-reaction typical spread
    u = 1.96 * cfac * med_sig / np.clip(frange[0], 1e-9, None)   # calibrated 95% half-width / flux range
    df = pd.DataFrame({'reaction': flux_names, 'r2': r2, 'u_rel': u,
                       'pred_scale': frange[0], 'half_width_95': 1.96 * cfac * med_sig})
    df = df[np.isfinite(df.r2) & np.isfinite(df.u_rel)]

    # ── 3) find u threshold that earns the R2 guarantee ──
    order = df.sort_values('u_rel')
    thr = None
    for q in np.linspace(0.05, 1.0, 40):
        cut = order['u_rel'].quantile(q)
        if order[order.u_rel <= cut]['r2'].median() >= args.r2_target:
            thr = cut
        else:
            break
    thr = thr if thr is not None else order['u_rel'].quantile(0.1)
    df['believe'] = df.u_rel <= thr
    bel = df[df.believe]
    print(f"\nBELIEVE rule: calibrated relative uncertainty u_rel <= {thr:.3f}")
    print(f"  believable reactions: {len(bel)}/{len(df)}  (median R2={bel.r2.median():.3f})")
    print(f"  not believable      : {len(df)-len(bel)}/{len(df)}  (median R2={df[~df.believe].r2.median():.3f})")
    df.sort_values('u_rel').to_csv(os.path.join(args.ensemble_dir, 'reliability_table.csv'), index=False)

    # ── plot A: reliability diagram (nominal vs empirical coverage) ──
    levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    emp_raw = [np.mean(err[fin] < norm.ppf(0.5 + p/2) * sig_eff[fin]) for p in levels]
    emp_cal = [np.mean(err[fin] < norm.ppf(0.5 + p/2) * cfac * sig_eff[fin]) for p in levels]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], '--', color=COLORS['ref'], lw=0.8)
    ax.plot(levels, emp_raw, 'o-', color=COLORS['nope'], label='raw')
    ax.plot(levels, emp_cal, 's-', color=COLORS['believe'], label=f'calibrated (c={cfac:.1f})')
    ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Empirical coverage'); ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    fig.savefig(os.path.join(args.ensemble_dir, 'reliability_diagram.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.ensemble_dir, 'reliability_diagram.png'), dpi=200, bbox_inches='tight')

    # ── plot B: decision map — R2 vs calibrated uncertainty, believe threshold ──
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    d = df.copy(); d['u_c'] = d.u_rel.clip(upper=2.0); d['r2c'] = d.r2.clip(lower=-1)
    sns.scatterplot(data=d, x='u_c', y='r2c', hue='believe', palette={True: COLORS['believe'], False: COLORS['nope']},
                    s=14, alpha=0.6, edgecolor='none', ax=ax, legend=False)
    ax.axvline(thr, color='k', ls='--', lw=0.8); ax.axhline(args.r2_target, color='gray', ls=':', lw=0.7)
    ax.set_xlabel('Calibrated relative uncertainty  (95% width / flux range)'); ax.set_ylabel('Actual R² (val)')
    ax.set_ylim(-1.05, 1.05); sns.despine(ax=ax)
    fig.savefig(os.path.join(args.ensemble_dir, 'decision_map.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.ensemble_dir, 'decision_map.png'), dpi=200, bbox_inches='tight')
    print(f"\nsaved -> reliability_diagram.svg, decision_map.svg, reliability_table.csv")


if __name__ == '__main__':
    main()
