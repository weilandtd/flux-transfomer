"""
Deconvolute prediction variance into MEASUREMENT vs MODEL sources (validation on sim).

For a set of simulated flux states we synthesize R "replicates" by injecting known
measurement noise (sigma) into the clean MIDs (apply_labeling_noise), then run all M
ensemble members. Per (state, reaction) this gives an M x R prediction matrix whose
spread mixes two sources:
  - sigma^2_model     : disagreement across ensemble members (epistemic / prediction)
  - sigma^2_replicate : how measurement noise propagates to the flux (experimental)
A balanced two-way random-effects ANOVA separates them.

VALIDATION: sweep the injected sigma. If the method works, sigma^2_replicate must
RISE with sigma (and be ~0 at sigma=0), while sigma^2_model stays ~flat (it is the
ensemble spread, independent of measurement-noise level). That proves the split.

Usage: python paper/analysis/ensemble/variance_decomposition.py \
         --ensemble-dir <run>/ensemble --n-states 60 --n-reps 10
"""
import argparse, glob, os, sys, numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src'))
from sigmaflux.ai.data import (load_data, create_tracer_mapping, build_radial_vocabs,
                               prepare_id_dataset, radial_embedding_table, load_stoichiometric_matrix)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.train import apply_labeling_noise
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 12})
COLORS = {'model': '#4C72B0', 'meas': '#DD8452', 'resid': '#B0B0B0'}


def two_way_anova(v):
    """v: (M, R, S, K) -> variance components (S, K) via balanced 2-way random effects."""
    M, R = v.shape[0], v.shape[1]
    grand = v.mean(axis=(0, 1))                       # (S,K)
    model_means = v.mean(axis=1)                       # (M,S,K)
    rep_means = v.mean(axis=0)                         # (R,S,K)
    SS_model = R * ((model_means - grand) ** 2).sum(0)
    SS_rep   = M * ((rep_means   - grand) ** 2).sum(0)
    resid = v - model_means[:, None] - rep_means[None, :] + grand
    SS_resid = (resid ** 2).sum(axis=(0, 1))
    MS_model = SS_model / (M - 1)
    MS_rep   = SS_rep / (R - 1)
    MS_resid = SS_resid / ((M - 1) * (R - 1))
    var_model = np.clip((MS_model - MS_resid) / R, 0, None)
    var_rep   = np.clip((MS_rep   - MS_resid) / M, 0, None)
    var_resid = MS_resid
    return var_model, var_rep, var_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble-dir', required=True)
    ap.add_argument('--n-states', type=int, default=60)
    ap.add_argument('--n-reps', type=int, default=10)
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument('--report-sigma', type=float, default=0.02)
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
    fr = fr if fr.dim() == 3 else fr.unsqueeze(-1)                 # (S, Ntok, 1)
    Sn, K = len(val_idx), len(flux_names)

    # load all M models once
    models = []
    for f in members:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        m = FluxDecoder(cfg, token_experiment_ids=ck['model_state_dict']['token_experiment_ids'],
                        compound_emb_table=vocabs['compound_emb_table'], label_emb_table=vocabs['label_emb_table'], tracer_emb_table=temb,
                        reaction_emb_table=radial_embedding_table(K, ck['h_reaction']),
                        type_emb_table=radial_embedding_table(2, ck['h_type']), null_space_data=nsd, flux_norm_stats=ck.get('flux_norm_stats'))
        m.load_state_dict(ck['model_state_dict']); m.eval(); m.to(dev); models.append(m)
    fid = torch.as_tensor(np.asarray(models[0].fixed_idx), dtype=torch.long, device=dev)
    mf = tg.index_select(1, fid)                                  # clean measured fluxes (MID-noise-only test)
    print(f"M={len(models)} models | S={Sn} states | R={args.n_reps} reps | sigmas={args.sigmas}")

    rows = []          # sweep summary
    per_rxn = {}       # per-reaction fractions at report sigma
    for sig in args.sigmas:
        # precompute R replicate inputs (SAME replicate fed to all models -> valid ANOVA)
        reps = []
        for r in range(args.n_reps):
            torch.manual_seed(1000 + int(sig * 1e5) + r)
            noisy = apply_labeling_noise(fr, c, t, sig) if sig > 0 else fr
            reps.append(noisy)
        preds = np.zeros((len(models), args.n_reps, Sn, K), dtype=np.float32)
        with torch.no_grad():
            for mi, m in enumerate(models):
                for ri, noisy in enumerate(reps):
                    out = m(c, l, t, noisy, measured_fluxes=mf, measured_flux_ids=fid, return_attn=False)
                    preds[mi, ri] = out[0].cpu().numpy()
        var_model, var_rep, var_resid = two_way_anova(preds)
        tot = var_model + var_rep + var_resid
        ok = tot > 1e-9
        meas_frac = np.where(ok, var_rep / np.clip(tot, 1e-12, None), np.nan)
        rows.append({'sigma': sig,
                     'var_model': float(np.nanmedian(var_model[ok])),
                     'var_rep':   float(np.nanmedian(var_rep[ok])),
                     'var_resid': float(np.nanmedian(var_resid[ok])),
                     'meas_frac': float(np.nanmedian(meas_frac))})
        print(f"  sigma={sig:<6}  median var_model={rows[-1]['var_model']:.3g}  "
              f"var_rep={rows[-1]['var_rep']:.3g}  measurement_fraction={rows[-1]['meas_frac']:.3f}")
        if abs(sig - args.report_sigma) < 1e-9:
            # per-reaction: median over states of measurement fraction (reactions with real variance)
            rxn_tot = np.nanmedian(tot, axis=0); keep = rxn_tot > np.nanpercentile(rxn_tot, 25)
            per_rxn = {'meas_frac': np.nanmedian(meas_frac, axis=0), 'keep': keep}

    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.ensemble_dir, 'variance_decomp_sweep.csv'), index=False)

    # ── plot 1: validation sweep — var_rep tracks sigma, var_model flat ──
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.plot(df.sigma, df.var_rep, 'o-', color=COLORS['meas'], label='measurement (σ²_rep)')
    ax.plot(df.sigma, df.var_model, 's-', color=COLORS['model'], label='model (σ²_ens)')
    ax.set_xlabel('Injected measurement noise σ'); ax.set_ylabel('Median flux variance'); sns.despine(ax=ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_sweep.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_sweep.png'), dpi=200, bbox_inches='tight')

    # ── plot 2: per-reaction measurement fraction at report sigma ──
    if per_rxn:
        mfrac = per_rxn['meas_frac'][per_rxn['keep']]
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sns.histplot(mfrac, bins=20, color=COLORS['meas'], ax=ax)
        ax.axvline(0.5, color='k', ls='--', lw=0.8)
        ax.set_xlabel(f'Measurement fraction of variance (σ={args.report_sigma})'); ax.set_ylabel('# reactions')
        sns.despine(ax=ax)
        fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_perrxn.svg'), bbox_inches='tight')
        fig.savefig(os.path.join(args.ensemble_dir, 'variance_decomp_perrxn.png'), dpi=200, bbox_inches='tight')
        n = int(per_rxn['keep'].sum())
        print(f"\nper-reaction @ σ={args.report_sigma} (n={n} variable reactions):")
        print(f"  measurement-limited (meas_frac>0.5): {int((mfrac>0.5).sum())}/{n}")
        print(f"  model-limited        (meas_frac<0.5): {int((mfrac<0.5).sum())}/{n}")
    print(f"\nsaved -> {args.ensemble_dir}/variance_decomp_*.svg/csv")


if __name__ == '__main__':
    main()
