"""
KO vs non-KO prediction-shift significance.

For each KO reaction R, compare the model's PREDICTED flux for R in:
  - KO-of-R validation samples (R was knocked out)   vs
  - non-KO (WT) validation samples (R active, normal condition)
Mann-Whitney U test on the raw predictions -> per-reaction p-value.
Bars: mean |predicted|/WT_std[R] per group (dimensionless, comparable across reactions).
Two panels split by the reaction's normal (WT) per-reaction R²: >0.5 vs <=0.5.
Expectation: well-predicted (R²>0.5) reactions show a larger, more significant KO shift.

Usage: python paper/analysis/ko_significance.py --checkpoint <ko.pt> --run-dir <ko run>
"""
import argparse, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig

mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'non-KO': '#4C72B0', 'KO': '#DD8452'}


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'


def r2(pred, true):
    ss = ((true - pred) ** 2).sum(); st = ((true - true.mean()) ** 2).sum()
    return 1 - ss / st if st > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--r2-cut', type=float, default=0.5)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck['model_config'], ck['null_space_data']
    flux_names = list(ck['flux_names']); name2idx = {n: i for i, n in enumerate(flux_names)}
    val_idx = np.asarray(ck['val_indices'], int)
    labels = pd.read_csv(os.path.join(args.run_dir, 'ko_labels.csv'), index_col=0).iloc[val_idx].reset_index(drop=True)

    lab, flux, rev = load_data(ck['data_dir']); ti = ck.get('token_index')
    if ti is not None: lab = lab.loc[ti]
    _, base = create_tracer_mapping(lab); S, cobra = load_stoichiometric_matrix(ck['model_json'])
    vocabs = build_radial_vocabs(cobra, lab, base, h_compound=cfg_dict['d_comp']//2,
                                 h_label=cfg_dict['d_label']//2, h_tracer=cfg_dict['d_tracer']//2)
    val_ds, _, _, _, temb = prepare_id_dataset(lab, flux, rev, vocabs=vocabs, train_indices=val_idx,
                                               measured_fluxes=ck.get('measured_fluxes'), norm_stats_input=ck.get('norm_stats'))
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__ if k in cfg_dict})
    m = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                    compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'], tracer_emb_table=temb,
                    reaction_emb_table=radial_embedding_table(len(flux_names), ck['h_reaction']),
                    type_emb_table=radial_embedding_table(2, ck['h_type']), null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
    m.load_state_dict(ck['model_state_dict']); m.eval(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m.to(dev)
    (c, l, t, fr), tg = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))
    c, l, t, fr, tg = [x.to(dev) for x in (c, l, t, fr, tg)]
    fid = torch.as_tensor(np.asarray(m.fixed_idx), dtype=torch.long, device=dev)
    with torch.no_grad():
        vp = m(c, l, t, fr, measured_fluxes=tg.index_select(1, fid), measured_flux_ids=fid, return_attn=False)[0].cpu().numpy()
    gt = tg[:, :len(flux_names)].cpu().numpy()

    ko_rxns = [r for r in labels['ko_reaction'].unique() if r != 'WT']
    wt = labels['ko_reaction'].values == 'WT'
    rows, long = [], []
    for r in ko_rxns:
        j = name2idx[r]; koR = labels['ko_reaction'].values == r
        pk, pn = vp[koR, j], vp[wt, j]                       # predicted flux: KO vs non-KO(WT)
        wt_std = gt[wt, j].std() or 1.0
        p = mannwhitneyu(np.abs(pk), np.abs(pn), alternative='two-sided').pvalue
        R2 = r2(vp[wt, j], gt[wt, j])                        # normal (WT) predictability of R
        rows.append({'reaction': r, 'n_KO': koR.sum(), 'n_nonKO': wt.sum(), 'R2_wt': R2,
                     'mean_nonKO_rel': np.abs(pn).mean()/wt_std, 'mean_KO_rel': np.abs(pk).mean()/wt_std,
                     'p': p, 'sig': stars(p)})
        for v in pn: long.append({'reaction': r, 'cond': 'non-KO', 'rel': abs(v)/wt_std, 'R2': R2})
        for v in pk: long.append({'reaction': r, 'cond': 'KO', 'rel': abs(v)/wt_std, 'R2': R2})
    res = pd.DataFrame(rows).sort_values('R2_wt', ascending=False)
    L = pd.DataFrame(long)
    res.to_csv(os.path.join(args.run_dir, 'ko_significance.csv'), index=False)
    print(res.to_string(index=False))
    hi = res[res.R2_wt > args.r2_cut]; lo = res[res.R2_wt <= args.r2_cut]
    print(f"\nsignificant (p<0.05): R2>{args.r2_cut}: {(hi.p<0.05).mean():.2f} ({len(hi)} rxns) | "
          f"R2<={args.r2_cut}: {(lo.p<0.05).mean():.2f} ({len(lo)} rxns)")

    # ── two-panel grouped bar plot ───────────────────────────────────────────
    for tag, sub in [('r2hi', res[res.R2_wt > args.r2_cut]), ('r2lo', res[res.R2_wt <= args.r2_cut])]:
        order = list(sub['reaction'])
        if not order: continue
        d = L[L.reaction.isin(order)]
        fig, ax = plt.subplots(figsize=(max(3.5, 0.5 * len(order) + 1.5), 3.5))
        sns.barplot(data=d, x='reaction', y='rel', hue='cond', order=order,
                    hue_order=['non-KO', 'KO'], palette=COLORS, errorbar='se', ax=ax)
        ymax = d.groupby(['reaction', 'cond'])['rel'].mean().max()
        for k, rxn in enumerate(order):
            s = sub[sub.reaction == rxn]['sig'].iloc[0]
            ax.text(k, ymax * 1.05, s, ha='center', va='bottom', fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel('|predicted flux| / WT std')
        ax.set_xticklabels(order, rotation=60, ha='right', fontsize=9)
        ax.legend(title='', frameon=False, fontsize=10)
        sns.despine(ax=ax)
        out = os.path.join(args.run_dir, f'ko_significance_{tag}.svg')
        fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
        print(f"saved -> {out}")


if __name__ == '__main__':
    main()
