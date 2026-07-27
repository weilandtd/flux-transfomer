"""
Ensemble per-reaction confidence.

--mode val: forward the shared held-out val set through all ensemble members;
  per (sample, reaction) get mean/std across members. Reports per-reaction
  confidence (mean std) and CALIBRATION (does ensemble std track actual error?).

--mode experimental: aggregate the per-member inference CSVs (run the inference
  workflow once per member first) into per-condition per-reaction mean +/- std.

Usage:
  python paper/analysis/ensemble_predict.py --mode val --ensemble-dir <run>/ensemble
  python paper/analysis/ensemble_predict.py --mode experimental --infer-dir <run>/ensemble_inference
"""
import argparse, glob, os, re, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import spearmanr
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'main': '#4C72B0', 'ref': '#000000'}


def per_reaction_r2(pred, gt):
    sr = ((pred - gt) ** 2).sum(0); st = ((gt - gt.mean(0)) ** 2).sum(0)
    return np.where(st > 1e-9, 1 - sr / st, np.nan)


def _val_predictions(ensemble_dir):
    """Forward the shared val set through every member -> (n_members, n_val, n_rxn), gt, names."""
    import torch
    from torch.utils.data import DataLoader
    from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                                   prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
    from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
    members = sorted(glob.glob(os.path.join(ensemble_dir, 'model_*.pt')))
    ck0 = torch.load(members[0], map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck0['model_config'], ck0['null_space_data']
    flux_names = list(ck0['flux_names']); val_idx = np.asarray(ck0['val_indices'], int)
    lab, flux, rev = load_data(ck0['data_dir']); ti = ck0.get('token_index')
    if ti is not None: lab = lab.loc[ti]
    _, base = create_tracer_mapping(lab); S, cobra = load_stoichiometric_matrix(ck0['model_json'])
    vocabs = build_radial_vocabs(cobra, lab, base, h_compound=cfg_dict['d_comp']//2,
                                 h_label=cfg_dict['d_label']//2, h_tracer=cfg_dict['d_tracer']//2)
    val_ds, _, _, _, temb = prepare_id_dataset(lab, flux, rev, vocabs=vocabs, train_indices=val_idx,
                                               measured_fluxes=ck0.get('measured_fluxes'), norm_stats_input=ck0.get('norm_stats'))
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__ if k in cfg_dict})
    (c, l, t, fr), tg = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    c, l, t, fr, tg = [x.to(dev) for x in (c, l, t, fr, tg)]
    gt = tg[:, :len(flux_names)].cpu().numpy()
    preds = []
    for f in members:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                        compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'], tracer_emb_table=temb,
                        reaction_emb_table=radial_embedding_table(len(flux_names), ck['h_reaction']),
                        type_emb_table=radial_embedding_table(2, ck['h_type']), null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
        m.load_state_dict(ck['model_state_dict']); m.eval(); m.to(dev)
        fid = torch.as_tensor(np.asarray(m.fixed_idx), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid), measured_flux_ids=fid, return_attn=False)
        preds.append(out[0].cpu().numpy())
    return np.stack(preds), gt, flux_names, len(members)


def mode_val(args):
    P, gt, names, n = _val_predictions(args.ensemble_dir)          # (n_members, n_val, n_rxn)
    mean, std = P.mean(0), P.std(0)                                # (n_val, n_rxn)
    r2 = per_reaction_r2(mean, gt)
    abs_err = np.abs(mean - gt)
    # WT-scale normalisation so reactions are comparable
    scale = gt.std(0); scale[scale < 1e-9] = 1.0
    std_rel, err_rel = std / scale, abs_err / scale
    ok = np.isfinite(std_rel) & np.isfinite(err_rel)
    rho = spearmanr(std_rel[ok].ravel(), err_rel[ok].ravel()).correlation
    print(f"ensemble: {n} members | val samples={gt.shape[0]} | reactions={len(names)}")
    print(f"ensemble mean per-reaction R2 median: {np.nanmedian(r2):.3f}")
    print(f"per-reaction confidence (median std / WT-SD): {np.nanmedian(std_rel.mean(0)):.3f}")
    print(f"CALIBRATION Spearman(ensemble std, |error|): {rho:.3f}  (higher = std tracks error)")

    # Model-specific outputs live in the run's output/ dir (sibling of ensemble/).
    output_dir = os.path.join(os.path.dirname(os.path.normpath(args.ensemble_dir)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({'reaction': names, 'r2': r2, 'conf_std_rel': std_rel.mean(0), 'err_rel': err_rel.mean(0)})
    df.to_csv(os.path.join(output_dir, 'ensemble_val_confidence.csv'), index=False)

    # reliability curve: bin by ensemble std, plot mean |error| (log-log axes)
    x = std_rel[ok].ravel(); y = err_rel[ok].ravel()
    q = np.quantile(x, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(x, q[1:-1]), 0, 9)
    bx = [x[bins == b].mean() for b in range(10)]; by = [y[bins == b].mean() for b in range(10)]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.lineplot(x=bx, y=by, marker='o', color=COLORS['main'], ax=ax)
    pos = [v for v in bx + by if v > 0]
    lo, hi = min(pos), max(pos)
    ax.plot([lo, hi], [lo, hi], '--', lw=0.8, color=COLORS['ref'])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Ensemble std (/ WT SD)'); ax.set_ylabel('Actual |error| (/ WT SD)')
    sns.despine(ax=ax)
    out = os.path.join(output_dir, 'ensemble_val_calibration.svg')
    fig.savefig(out, bbox_inches='tight')
    print(f"saved -> {out} + {os.path.join(output_dir, 'ensemble_val_confidence.csv')}")


def mode_experimental(args):
    # aggregate per-member inference CSVs: infer-dir/member_*/predictions_*_<cond>_*.csv
    members = sorted(glob.glob(os.path.join(args.infer_dir, 'member_*')))
    print(f"aggregating {len(members)} members from {args.infer_dir}")
    rows = []
    for mi, md in enumerate(members):
        for csv in glob.glob(os.path.join(md, 'predictions_*_*.csv')):
            d = pd.read_csv(csv)
            d = d[d['Type'] == 'flux']                       # fluxes only (drop reversibility rows)
            # this member's prediction = mean over bio-reps / growth / flux draws
            g = d.groupby(['Condition', 'Flux Name'])['Value'].mean().reset_index()
            g = g.rename(columns={'Condition': 'cond', 'Flux Name': 'reaction', 'Value': 'flux'})
            g['member'] = mi
            rows.append(g)
    if not rows:
        print("no member prediction CSVs found — run the per-member inference first"); return
    allp = pd.concat(rows, ignore_index=True)
    conds = sorted(allp['cond'].unique())
    agg = allp.groupby(['cond', 'reaction'])['flux'].agg(['mean', 'std', 'count']).reset_index()
    agg['cv'] = agg['std'] / agg['mean'].abs().clip(1e-9)
    agg.to_csv(os.path.join(args.infer_dir, 'ensemble_experimental_confidence.csv'), index=False)
    for c in sorted(conds):
        s = agg[agg.cond == c]
        print(f"cond {c}: {len(s)} reactions | median CV across members = {s['cv'].median():.3f} | "
              f"reactions with CV<0.2 (confident): {(s['cv']<0.2).sum()}/{len(s)}")
    print(f"saved -> {os.path.join(args.infer_dir, 'ensemble_experimental_confidence.csv')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['val', 'experimental'], required=True)
    ap.add_argument('--ensemble-dir', default=None)
    ap.add_argument('--infer-dir', default=None)
    args = ap.parse_args()
    (mode_val if args.mode == 'val' else mode_experimental)(args)
