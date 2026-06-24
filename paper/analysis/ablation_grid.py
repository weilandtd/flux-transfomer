"""
Ablation grid: validation R2 as a function of how much input is removed.

Independently removes a fraction of the LABELING metabolites (MIDs) and a fraction
of the KNOWN measured fluxes, for all combinations of {0,10,20,50,70}% x {0,10,20,50,70}%,
averaged over random removals. Reports a matrix of median per-reaction R2 (over the
reactions that are identifiable in the full model, full R2 > 0.2) and saves a heatmap.

MID removal   = zero the fraction tokens of the removed metabolites (treat as unmeasured).
Flux removal  = drop those bypass tokens (pass a subset to measured_flux_ids).

Usage:
    python paper/analysis/ablation_grid.py \
        --checkpoint paper/.results/i_orientalis_20260620/imisor_transformer.pt \
        --n-samples 400 --seeds 5
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table,
                               load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig

strip_c = lambda m: m[:-2] if len(m) > 2 and m[-2] == '_' else m
LEVELS = [0, 10, 20, 50, 70]                                # % removed


def per_reaction_r2(pred, gt):
    ss_res = ((pred - gt) ** 2).sum(0)
    ss_tot = ((gt - gt.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 1e-9, 1 - ss_res / ss_tot, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-samples', type=int, default=400)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck['model_config'], ck['null_space_data']
    flux_names = ck.get('flux_names', [])

    lab, flux, rev = load_data(ck['data_dir'], max_samples=args.n_samples)
    ti = ck.get('token_index')
    if ti is not None:
        lab = lab.loc[ti]
    _, base_tr = create_tracer_mapping(lab)
    _, cobra = load_stoichiometric_matrix(ck['model_json'])
    vocabs = build_radial_vocabs(cobra, lab, base_tr,
                                 h_compound=cfg_dict.get('d_comp', 32) // 2,
                                 h_label=cfg_dict.get('d_label', 10) // 2,
                                 h_tracer=cfg_dict.get('d_tracer', 2) // 2)
    vi = np.arange(min(args.n_samples, lab.shape[1]))
    val_ds, _, _, _, tracer_emb = prepare_id_dataset(lab, flux, rev, vocabs=vocabs, train_indices=vi,
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"device: {device}")

    (cids, lids, tids, fracs), targets = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))
    cids, lids, tids, fracs = cids.to(device), lids.to(device), tids.to(device), fracs.to(device)
    targets = targets.to(device)
    gt = targets[:, :len(flux_names)].cpu().numpy()
    fixed_idx = np.array(model.fixed_idx)                    # reaction indices of the known fluxes
    cvocab = vocabs['compound_vocab']
    tok_met = np.array([strip_c(cvocab[int(c)]) if int(c) < len(cvocab) else str(int(c)) for c in cids[0].cpu().numpy()])
    mets = sorted(set(tok_met))
    print(f"val samples={gt.shape[0]} | reactions={len(flux_names)} | metabolites={len(mets)} | known fluxes={len(fixed_idx)}")

    def predict(keep_mid_mask, keep_flux_ids):
        f = fracs.clone()
        f[:, torch.as_tensor(~keep_mid_mask, device=device)] = 0.0
        if len(keep_flux_ids) > 0:
            ids = torch.as_tensor(np.asarray(keep_flux_ids), dtype=torch.long, device=device)
            mf = targets.index_select(1, ids)
        else:
            mf, ids = None, None
        with torch.no_grad():
            out = model(cids, lids, tids, f, measured_fluxes=mf, measured_flux_ids=ids, return_attn=False)
        return per_reaction_r2(out[0].cpu().numpy(), gt)

    # identifiable reactions defined by the FULL model
    full_r2 = predict(np.ones(len(tok_met), bool), fixed_idx)
    ident = full_r2 > 0.2
    print(f"identifiable reactions (full R2>0.2): {int(ident.sum())}/{len(flux_names)}\n")

    grid = np.zeros((len(LEVELS), len(LEVELS)))
    for i, mid_rm in enumerate(LEVELS):
        for j, flx_rm in enumerate(LEVELS):
            meds = []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 * i + 7 * j + s)
                n_mid_drop = int(round(mid_rm / 100 * len(mets)))
                drop_mets = set(rng.choice(mets, size=n_mid_drop, replace=False)) if n_mid_drop else set()
                keep_mid = ~np.isin(tok_met, list(drop_mets))
                n_flx_keep = len(fixed_idx) - int(round(flx_rm / 100 * len(fixed_idx)))
                keep_flux = np.sort(rng.choice(fixed_idx, size=n_flx_keep, replace=False)) if n_flx_keep else np.array([], int)
                r2 = predict(keep_mid, keep_flux)
                meds.append(np.nanmedian(r2[ident]))
            grid[i, j] = np.mean(meds)

    # ── report ──────────────────────────────────────────────────────────────────
    print("median R2 over identifiable reactions  (rows = % MIDs removed, cols = % known fluxes removed)")
    print("            flux:  " + "  ".join(f"{c:>5d}%" for c in LEVELS))
    for i, mid_rm in enumerate(LEVELS):
        print(f"  MID {mid_rm:3d}% removed:  " + "  ".join(f"{grid[i, j]:6.3f}" for j in range(len(LEVELS))))

    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(grid, cmap='viridis', vmin=max(0, grid.min()), vmax=1, aspect='equal')
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"{c}%" for c in LEVELS])
    ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels([f"{c}%" for c in LEVELS])
    ax.set_xlabel("known fluxes removed"); ax.set_ylabel("MID metabolites removed")
    for i in range(len(LEVELS)):
        for j in range(len(LEVELS)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center',
                    color='white' if grid[i, j] < 0.6 else 'black', fontsize=8)
    ax.set_title(f"Validation R² vs input removal (median over {int(ident.sum())} identifiable reactions)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median R²")
    fig.tight_layout()
    out = args.out or args.checkpoint.replace('.pt', '_ablation_grid.svg')
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nsaved -> {out} (+ .png)")


if __name__ == '__main__':
    main()
