"""
Version benchmark — CURRENT model (FluxDecoder) on IO held-out val.
Computes per-reaction R2 two ways on the SAME split:
  (a) as-designed: with the 10 measured-flux bypass tokens
  (b) labeling-only: measured_fluxes removed (apples-to-apples vs the old model)
Saves paper/.results/version_benchmark/current_io.json
"""
import argparse, json, os, numpy as np, torch
from sigmaflux.ai.checkpoint import build_flux_decoder, load_eval_context, resolve_device
from sigmaflux.ai.train import per_reaction_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', default='paper/.results/version_benchmark')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device()
    ctx = load_eval_context(args.checkpoint)
    model = build_flux_decoder(ctx, device=device)

    ck, cfg_dict = ctx.ckpt, ctx.ckpt['model_config']
    flux_names, val_idx = ctx.flux_names, ctx.sample_indices
    ti = ck.get('token_index')

    (cids, lids, tids, fracs), targets = ctx.full_batch(device=device)
    gt = targets[:, :len(flux_names)].cpu().numpy()
    fixed_idx = model.fixed_idx_t.to(device)

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
