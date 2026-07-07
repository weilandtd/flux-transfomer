"""
KO-detection evaluation. For every KO validation sample, does the model drive the
knocked-out reaction's predicted flux to ~0?

A "hit" = |v_pred[R]| < tau_rel * WT_std[R] (predicted flux is small relative to that
reaction's normal WT variation). Reported split by seen vs held-out KO reactions and
by free vs dependent reaction (null-space structure).

Usage:
  python paper/analysis/ko_eval.py --checkpoint <ko.pt> --run-dir <ko run> --tau-rel 0.1
"""
import argparse, json, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tau-rel', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck['model_config'], ck['null_space_data']
    flux_names = list(ck['flux_names'])
    name2idx = {n: i for i, n in enumerate(flux_names)}
    val_idx = np.asarray(ck['val_indices'], int)

    labels = pd.read_csv(os.path.join(args.run_dir, 'ko_labels.csv'), index_col=0)
    fs = pd.read_csv(os.path.join(args.run_dir, 'flux_samples.csv'), index_col=0)
    wt_std = fs[labels['ko_reaction'].values == 'WT'].std(axis=0)        # per-reaction WT scale

    # free (identity) vs dependent reactions from the null-space basis
    Z = np.asarray(nsd['Z_free'], float)
    is_free = np.array([(np.sum(np.abs(Z[i]) > 1e-6) == 1) for i in range(Z.shape[0])])

    # build val dataset (same machinery as the trainer/benchmark)
    lab, flux, rev = load_data(ck['data_dir'])
    ti = ck.get('token_index')
    if ti is not None: lab = lab.loc[ti]
    _, base_tr = create_tracer_mapping(lab)
    _, cobra = load_stoichiometric_matrix(ck['model_json'])
    vocabs = build_radial_vocabs(cobra, lab, base_tr, h_compound=cfg_dict.get('d_comp', 32)//2,
                                 h_label=cfg_dict.get('d_label', 10)//2, h_tracer=cfg_dict.get('d_tracer', 2)//2)
    val_ds, _, _, _, tracer_emb = prepare_id_dataset(lab, flux, rev, vocabs=vocabs, train_indices=val_idx,
                                                     measured_fluxes=ck.get('measured_fluxes'),
                                                     norm_stats_input=ck.get('norm_stats'))
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__ if k in cfg_dict})
    model = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                        compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'],
                        tracer_emb_table=tracer_emb,
                        reaction_emb_table=radial_embedding_table(len(flux_names) or cfg.n_fluxes, ck['h_reaction']),
                        type_emb_table=radial_embedding_table(2, ck['h_type']),
                        null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
    model.load_state_dict(ck['model_state_dict']); model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)

    (cids, lids, tids, fracs), targets = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))
    cids, lids, tids, fracs, targets = [t.to(device) for t in (cids, lids, tids, fracs, targets)]
    fixed_idx = torch.as_tensor(np.asarray(model.fixed_idx), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(cids, lids, tids, fracs, measured_fluxes=targets.index_select(1, fixed_idx),
                    measured_flux_ids=fixed_idx, return_attn=False)
    v_pred = out[0].cpu().numpy()                                         # (n_val, 387)

    val_labels = labels.iloc[val_idx].reset_index(drop=True)
    rows = []
    for i in range(len(val_labels)):
        r = val_labels['ko_reaction'].iloc[i]
        if r == 'WT' or r not in name2idx:
            continue
        ridx = name2idx[r]
        pred = abs(float(v_pred[i, ridx]))
        scale = float(wt_std[r]) if wt_std[r] > 1e-9 else 1.0
        rows.append({'reaction': r, 'split': val_labels['split'].iloc[i],
                     'pred_abs': pred, 'wt_std': scale, 'rel': pred / scale,
                     'hit': pred < args.tau_rel * scale, 'is_free': bool(is_free[ridx])})
    df = pd.DataFrame(rows)

    def rate(sub, lab):
        if len(sub) == 0: print(f"  {lab}: (none)"); return
        print(f"  {lab}: n={len(sub)}  hit-rate={sub['hit'].mean():.2f}  "
              f"median rel={sub['rel'].median():.3f}  (reactions={sub['reaction'].nunique()})")
    print(f"KO-detection (hit = |v_pred[R]| < {args.tau_rel}·WT_std[R]) on {len(df)} KO val samples")
    rate(df, "ALL KO")
    rate(df[df.split == 'mixed'], "seen KO reactions")
    rate(df[df.split == 'valonly'], "held-out KO reactions")
    rate(df[df.is_free], "free (identity) reactions")
    rate(df[~df.is_free], "dependent reactions")
    print("\nper-reaction hit-rate:")
    per = df.groupby(['reaction', 'split', 'is_free']).agg(n=('hit', 'size'), hit=('hit', 'mean'),
                                                           med_rel=('rel', 'median')).reset_index()
    print(per.to_string(index=False))
    out = args.out or os.path.join(args.run_dir, 'ko_eval.csv')
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
