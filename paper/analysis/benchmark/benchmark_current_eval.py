"""
Version benchmark — CURRENT model (FluxDecoder) on IO held-out val.
Computes per-reaction R2 two ways on the SAME split:
  (a) as-designed: with the 10 measured-flux bypass tokens
  (b) labeling-only: measured_fluxes removed (apples-to-apples vs the old model)
Saves paper/.results/version_benchmark/current_io.json
"""
import argparse, json, os, numpy as np, torch
from torch.utils.data import DataLoader
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


def per_reaction_r2(pred, gt):
    sr = ((pred - gt) ** 2).sum(0); st = ((gt - gt.mean(0)) ** 2).sum(0)
    return np.where(st > 1e-9, 1 - sr / st, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', default='paper/.results/version_benchmark')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck['model_config'], ck['null_space_data']
    flux_names = list(ck['flux_names'])
    val_idx = np.asarray(ck['val_indices'], int)

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
    gt = targets[:, :len(flux_names)].cpu().numpy()
    fixed_idx = torch.as_tensor(np.asarray(model.fixed_idx), dtype=torch.long, device=device)

    def evalr2(use_flux):
        with torch.no_grad():
            if use_flux:
                out = model(cids, lids, tids, fracs, measured_fluxes=targets.index_select(1, fixed_idx),
                            measured_flux_ids=fixed_idx, return_attn=False)
            else:
                out = model(cids, lids, tids, fracs, measured_fluxes=None, return_attn=False)
        return per_reaction_r2(out[0].cpu().numpy(), gt)

    n_params = sum(v.numel() for v in ck['model_state_dict'].values())
    res = {'model': 'FluxDecoder (current)', 'n_params': int(n_params),
           'd_model': cfg_dict.get('d_model'), 'n_layers': cfg_dict.get('n_transformer_layers') or cfg_dict.get('n_layers'),
           'n_free': cfg_dict.get('n_free'), 'n_fluxes': cfg_dict.get('n_fluxes'),
           'n_tokens': len(ti) if ti is not None else None, 'n_measured_fluxes': len(ck.get('measured_fluxes', [])),
           'n_val': int(len(val_idx))}
    for tag, uf in [('as_designed', True), ('labeling_only', False)]:
        r2 = evalr2(uf)
        res[tag] = {'r2_median': float(np.nanmedian(r2)),
                    'n_r2_gt_0.9': int(np.nansum(r2 > 0.9)),
                    'n_r2_gt_0.95': int(np.nansum(r2 > 0.95)),
                    'r2_per_reaction': [None if np.isnan(v) else float(v) for v in r2]}
        print(f"{tag}: median={res[tag]['r2_median']:.3f} #>0.9={res[tag]['n_r2_gt_0.9']} #>0.95={res[tag]['n_r2_gt_0.95']}")
    out = os.path.join(args.out_dir, 'current_io.json')
    with open(out, 'w') as f: json.dump(res, f)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
