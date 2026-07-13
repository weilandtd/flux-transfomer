"""
Three-way variance decomposition: MODEL vs MID-measurement vs FLUX-measurement.

Real flux uncertainty has three independent sources:
  - sigma^2_model : ensemble disagreement (epistemic / prediction reliability)
  - sigma^2_MID   : how ISOTOPE-LABELING measurement noise propagates to the flux
  - sigma^2_flux  : how EXTERNAL-FLUX measurement noise propagates to the flux
We build a factorial perturbation design per flux state: I MID-noise instances x
J flux-noise instances x M ensemble members, then a 3-way random-effects ANOVA
(main effects; interactions pooled into residual) attributes the variance.

Validation of channel ISOLATION: run three configs —
  full (both noisy), MID-only (flux clean -> sigma^2_flux~0), flux-only (MID clean
  -> sigma^2_MID~0) — proving each channel is attributed to the right source.

Actionable output per reaction: fractions from model / MID / flux -> tells a
biologist whether to improve the model, the labeling, or the flux measurement.

Usage: python paper/analysis/ensemble/variance_decomposition_3way.py --ensemble-dir <run>/ensemble
"""
import argparse, glob, os, sys, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src'))
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.train import apply_labeling_noise
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'model': '#4C72B0', 'MID': '#DD8452', 'flux': '#55A868'}


def anova_3way(v):
    """v: (M, I, J, S, K) -> variance components (S,K): model, MID, flux, resid (main-effects model)."""
    M, I, J = v.shape[:3]
    g   = v.mean(axis=(0, 1, 2))                       # (S,K)
    vm  = v.mean(axis=(1, 2))                           # (M,S,K)
    vi  = v.mean(axis=(0, 2))                           # (I,S,K)
    vj  = v.mean(axis=(0, 1))                           # (J,S,K)
    SS_m = I * J * ((vm - g) ** 2).sum(0)
    SS_i = M * J * ((vi - g) ** 2).sum(0)
    SS_j = M * I * ((vj - g) ** 2).sum(0)
    SS_t = ((v - g) ** 2).sum(axis=(0, 1, 2))
    SS_r = np.clip(SS_t - SS_m - SS_i - SS_j, 0, None)
    df_r = M * I * J - (M - 1) - (I - 1) - (J - 1) - 1
    MS_m, MS_i, MS_j = SS_m / (M - 1), SS_i / (I - 1), SS_j / (J - 1)
    MS_r = SS_r / max(df_r, 1)
    var_model = np.clip((MS_m - MS_r) / (I * J), 0, None)
    var_MID   = np.clip((MS_i - MS_r) / (M * J), 0, None)
    var_flux  = np.clip((MS_j - MS_r) / (M * I), 0, None)
    return var_model, var_MID, var_flux, MS_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble-dir', required=True)
    ap.add_argument('--n-states', type=int, default=50)
    ap.add_argument('--n-mid', type=int, default=5)
    ap.add_argument('--n-flux', type=int, default=5)
    ap.add_argument('--sigma-mid', type=float, default=0.01)
    ap.add_argument('--sigma-flux', type=float, default=0.05)   # relative to each flux's scale
    args = ap.parse_args()

    members = sorted(glob.glob(os.path.join(args.ensemble_dir, 'model_*.pt')))
    ck0 = torch.load(members[0], map_location='cpu', weights_only=False)
    cfg_dict, nsd = ck0['model_config'], ck0['null_space_data']
    flux_names = list(ck0['flux_names']); val_idx = np.asarray(ck0['val_indices'], int)[:args.n_states]
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
    fr = fr if fr.dim() == 3 else fr.unsqueeze(-1)
    Sn, K = len(val_idx), len(flux_names)
    models = []
    for f in members:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                        compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'], tracer_emb_table=temb,
                        reaction_emb_table=radial_embedding_table(K, ck['h_reaction']),
                        type_emb_table=radial_embedding_table(2, ck['h_type']), null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
        m.load_state_dict(ck['model_state_dict']); m.eval(); m.to(dev); models.append(m)
    fid = torch.as_tensor(np.asarray(models[0].fixed_idx), dtype=torch.long, device=dev)
    mf_clean = tg.index_select(1, fid)                             # (S, n_measured)
    flux_scale = mf_clean.std(0, keepdim=True).clamp_min(1e-6)     # per measured-flux scale
    print(f"M={len(models)} | S={Sn} | I(MID)={args.n_mid} | J(flux)={args.n_flux} | sig_MID={args.sigma_mid} sig_flux={args.sigma_flux}")

    def run(sig_mid, sig_flux):
        mids = []
        for k in range(args.n_mid):
            torch.manual_seed(100 + k)
            mids.append(apply_labeling_noise(fr, c, t, sig_mid) if sig_mid > 0 else fr)
        fluxes = []
        for j in range(args.n_flux):
            torch.manual_seed(500 + j)
            fluxes.append(mf_clean + sig_flux * flux_scale * torch.randn_like(mf_clean) if sig_flux > 0 else mf_clean)
        v = np.zeros((len(models), args.n_mid, args.n_flux, Sn, K), dtype=np.float32)
        with torch.no_grad():
            for mi_, m in enumerate(models):
                for i, midv in enumerate(mids):
                    for j, fv in enumerate(fluxes):
                        v[mi_, i, j] = m(c, l, t, midv, measured_fluxes=fv, measured_flux_ids=fid, return_attn=False)[0].cpu().numpy()
        return anova_3way(v)

    configs = {'full': (args.sigma_mid, args.sigma_flux),
               'MID_only': (args.sigma_mid, 0.0),
               'flux_only': (0.0, args.sigma_flux)}
    res = {}
    for name, (sm, sf) in configs.items():
        vmod, vmid, vflx, _ = run(sm, sf)
        tot = vmod + vmid + vflx
        ok = tot > 1e-9
        res[name] = dict(model=np.nanmedian((vmod/np.clip(tot,1e-12,None))[ok]),
                         MID=np.nanmedian((vmid/np.clip(tot,1e-12,None))[ok]),
                         flux=np.nanmedian((vflx/np.clip(tot,1e-12,None))[ok]),
                         vmod=vmod, vmid=vmid, vflx=vflx, tot=tot, ok=ok)
        print(f"  [{name:9s}] median fractions -> model={res[name]['model']:.3f}  MID={res[name]['MID']:.3f}  flux={res[name]['flux']:.3f}")

    # validation isolation check
    print("\nvalidation (isolation):")
    print(f"  MID_only  -> flux fraction should be ~0:  {res['MID_only']['flux']:.3f}")
    print(f"  flux_only -> MID  fraction should be ~0:  {res['flux_only']['MID']:.3f}")

    # ── per-reaction stacked composition at the full config ──
    full = res['full']
    tot_rxn = np.nanmedian(full['tot'], axis=0)                 # (K,) per-reaction total variance
    keep = tot_rxn > np.nanpercentile(tot_rxn, 25)             # (K,) drop the lowest-variance quartile
    comp = pd.DataFrame({
        'reaction': [flux_names[i] for i in np.where(keep)[0]],
        'model': np.nanmedian(full['vmod'], 0)[keep], 'MID': np.nanmedian(full['vmid'], 0)[keep],
        'flux': np.nanmedian(full['vflx'], 0)[keep]})
    tot = comp[['model', 'MID', 'flux']].sum(1).clip(lower=1e-12)
    for ch in ['model', 'MID', 'flux']: comp[ch + '_frac'] = comp[ch] / tot
    comp.to_csv(os.path.join(args.ensemble_dir, 'variance_decomp_3way.csv'), index=False)
    n = len(comp)
    print(f"\nper-reaction (n={n} variable reactions), dominant source:")
    dom = comp[['model_frac', 'MID_frac', 'flux_frac']].idxmax(1).str.replace('_frac', '')
    print("  ", dom.value_counts().to_dict())

    # plot 1: overall composition bars per config
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    labels = list(configs); bottoms = np.zeros(len(labels))
    for ch in ['model', 'MID', 'flux']:
        vals = [res[c][ch] for c in labels]
        ax.bar(labels, vals, bottom=bottoms, color=COLORS[ch], label=ch, width=0.6)
        bottoms += vals
    ax.set_ylabel('Median variance fraction'); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=9)
    ax.set_xticklabels(labels, rotation=20, ha='right'); sns.despine(ax=ax)
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_3way_configs.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_3way_configs.png'), dpi=200, bbox_inches='tight')

    # plot 2: per-reaction stacked composition (sorted by flux fraction), full config
    comp_s = comp.sort_values('flux_frac', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(comp_s)); bottoms = np.zeros(len(comp_s))
    for ch in ['model', 'MID', 'flux']:
        ax.bar(x, comp_s[ch + '_frac'], bottom=bottoms, color=COLORS[ch], width=1.0, label=ch)
        bottoms += comp_s[ch + '_frac'].values
    ax.set_xlim(-0.5, len(comp_s) - 0.5); ax.set_ylim(0, 1); ax.set_xlabel(f'Reactions (n={n}, sorted)')
    ax.set_ylabel('Variance fraction'); ax.legend(frameon=False, fontsize=9, ncol=3, loc='lower center'); sns.despine(ax=ax)
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_3way_perrxn.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_3way_perrxn.png'), dpi=200, bbox_inches='tight')
    print(f"\nsaved -> {args.ensemble_dir}/variance_decomp_3way*.svg/csv")


if __name__ == '__main__':
    main()
