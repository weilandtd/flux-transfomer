"""
Print cross-attention report for a trained FluxDecoder.

For each of the 38 null-space y-dimensions, prints the top-10% most-attended
labeling tokens (MID measurements), sorted by attention weight.

  gate→0  : model relies on MID signal → expect sharp, concentrated attention
  gate→1  : model relies on v_known    → expect diffuse/uniform attention

Usage:
    python paper/show_attention.py I_orientalis_20260508/imisor_transformer.pt
    python paper/show_attention.py ...pt --n-samples 256 --top-pct 0.05
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig


def strip_compartment(met_id):
    for suffix in ('_c', '_m', '_e', '_x', '_n', '_r', '_g', '_l'):
        if met_id.endswith(suffix):
            return met_id[:-2]
    return met_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--top-pct',   type=float, default=0.10,
                        help='Fraction of tokens to show per y-dim (default 0.10 = top 10%%)')
    parser.add_argument('--out', default=None,
                        help='Write output to this file instead of stdout')
    parser.add_argument('--save-matrix', default=None,
                        help='Save flux-token attention matrix as .npz (e.g. flux_token.npz)')
    parser.add_argument('--save-self-attn', default=None,
                        help='Save metabolite-token self-attention matrix (.npz) for the '
                             'attention-vs-distance analysis')
    parser.add_argument('--drop-flux-frac', type=float, default=0.0,
                        help='randomly drop this fraction of known measured fluxes before '
                             'extracting attention (tests if MID attention sharpens)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_dict   = ckpt['model_config']
    nsd        = ckpt['null_space_data']
    data_dir   = ckpt['data_dir']
    model_json = ckpt.get('model_json')
    flux_names = ckpt.get('flux_names', [])

    # ── Rebuild data (for vocabs + val samples) ───────────────────────────────
    labeling_df, flux_df, reversibility_df = load_data(data_dir, max_samples=args.n_samples * 2)

    # Restrict to the exact tokens the model was trained on, in the saved order,
    # so the labeling sequence and the radial label vocab match the checkpoint
    # (training may have filtered to a Fisher-selected token subset).
    token_index = ckpt.get('token_index')
    if token_index is not None:
        labeling_df = labeling_df.loc[token_index]
        print(f"Restricted to {labeling_df.shape[0]} trained tokens (from checkpoint token_index)")

    _, base_tracer_names = create_tracer_mapping(labeling_df)
    _, cobra_model       = load_stoichiometric_matrix(model_json)

    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound = cfg_dict.get('d_comp', 32) // 2,
        h_label    = cfg_dict.get('d_label', 10) // 2,
        h_tracer   = cfg_dict.get('d_tracer', 2)  // 2,
    )

    val_indices = np.arange(min(args.n_samples, labeling_df.shape[1]))
    (val_dataset, _, _, tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df, flux_df, reversibility_df,
        vocabs=vocabs,
        train_indices=val_indices,
        measured_fluxes=ckpt.get('measured_fluxes'),
        norm_stats_input=ckpt.get('norm_stats'),
    )
    val_loader = DataLoader(val_dataset, batch_size=min(args.n_samples, 64), shuffle=False)

    # ── Reconstruct model ─────────────────────────────────────────────────────
    cfg = FluxDecoderConfig(**{k: cfg_dict[k] for k in FluxDecoderConfig.__dataclass_fields__
                               if k in cfg_dict})
    token_experiment_ids_tensor = ckpt['model_state_dict']['token_experiment_ids']

    # Rebuild the frozen reaction/type radial tables exactly as training did
    # (deterministic given (N, H)); load_state_dict restores the learned weights.
    flux_dim_ck        = len(flux_names) or cfg.n_fluxes
    reaction_emb_table = radial_embedding_table(flux_dim_ck, ckpt['h_reaction'])
    type_emb_table     = radial_embedding_table(2, ckpt['h_type'])

    model = FluxDecoder(
        cfg,
        token_experiment_ids = token_experiment_ids_tensor,
        compound_emb_table   = vocabs['compound_emb_table'],
        label_emb_table      = vocabs['label_emb_table'],
        tracer_emb_table     = tracer_combo_emb_table,
        reaction_emb_table   = reaction_emb_table,
        type_emb_table       = type_emb_table,
        null_space_data      = nsd,
        flux_norm_stats      = ckpt.get('flux_norm_stats'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Running on {device}")

    out_file = open(args.out, 'w') if args.out else None
    def out(msg=''):
        print(msg)
        if out_file:
            out_file.write(msg + '\n')

    # ── Optionally drop a fraction of the known measured fluxes ────────────────
    keep_flux_ids = np.array(model.fixed_idx)
    if getattr(args, 'drop_flux_frac', 0.0) > 0 and len(keep_flux_ids) > 0:
        rng = np.random.default_rng(getattr(args, 'seed', 0))
        n_keep = len(keep_flux_ids) - int(round(args.drop_flux_frac * len(keep_flux_ids)))
        keep_flux_ids = np.sort(rng.choice(keep_flux_ids, size=max(0, n_keep), replace=False))
        print(f"Dropping {args.drop_flux_frac:.0%} of known fluxes: keeping "
              f"{len(keep_flux_ids)}/{len(model.fixed_idx)} for attention extraction")

    # ── Collect attention weights ─────────────────────────────────────────────
    attn_sum    = None
    n_collected = 0

    with torch.no_grad():
        for batch in val_loader:
            (cids, lids, tids, fracs), targets = batch
            cids  = cids.to(device)
            lids  = lids.to(device)
            tids  = tids.to(device)
            fracs = fracs.to(device)
            measured_fluxes, mfi = None, None
            if len(keep_flux_ids) > 0:
                measured_fluxes = targets[:, keep_flux_ids].to(device)
                mfi = torch.as_tensor(keep_flux_ids, dtype=torch.long, device=device)

            _, _, attn = model(cids, lids, tids, fracs,
                               measured_fluxes=measured_fluxes, measured_flux_ids=mfi,
                               return_attn=True)
            batch_sum = attn.cpu().sum(dim=0).numpy()   # (38, N)
            attn_sum  = batch_sum if attn_sum is None else attn_sum + batch_sum
            n_collected += attn.shape[0]

    attn_mean = attn_sum / n_collected   # (38, N)

    # ── Build token labels from the first batch ───────────────────────────────
    first_batch = next(iter(DataLoader(val_dataset, batch_size=1)))
    (cids0, lids0, tids0, _), _ = first_batch
    cids0 = cids0[0].numpy()
    lids0 = lids0[0].numpy()
    tids0 = tids0[0].numpy()

    compound_vocab = vocabs['compound_vocab']
    label_vocab    = vocabs['label_vocab']

    token_labels = []
    for c, l, t in zip(cids0, lids0, tids0):
        met  = strip_compartment(compound_vocab[c]) if c < len(compound_vocab) else str(c)
        iso  = label_vocab[l]                        if l < len(label_vocab)    else str(l)
        trac = tracer_combo_vocab[t]                 if t < len(tracer_combo_vocab) else str(t)
        trac_short = trac.replace('glc__D_e_', '').replace('_0.5', '')
        token_labels.append(f"{met} {iso} [{trac_short}]")

    # Bypass measured-flux tokens are appended after the labeling tokens.
    for fname in (ckpt.get('measured_fluxes') or []):
        token_labels.append(f"{fname} [flux]")

    # ── Encoder self-attention over metabolite tokens (for attention-vs-distance) ─
    if args.save_self_attn:
        import torch.backends.mha as _mha
        try:
            _mha.set_fastpath_enabled(False)        # force the python MHA path so weights are returned
        except Exception:
            pass
        _store = []                                  # per-layer (H, S, S) batch-summed weights
        def _wrap(layer):
            _orig = layer.self_attn.forward
            def _fwd(q, k, v, **kw):
                kw['need_weights'] = True
                kw['average_attn_weights'] = False        # keep per-head
                o, w = _orig(q, k, v, **kw)               # w: (B, H, S, S)
                _store.append(w.detach().sum(0))          # reduce batch -> (H, S, S) (memory-safe)
                return o, w
            layer.self_attn.forward = _fwd
        for _L in model.encoder.layers:
            _wrap(_L)

        n_flux_tok = len(ckpt.get('measured_fluxes') or [])
        accum, sa_n = None, 0
        with torch.no_grad():
            for batch in val_loader:
                (cids, lids, tids, fracs), targets = batch
                cids, lids = cids.to(device), lids.to(device)
                tids, fracs = tids.to(device), fracs.to(device)
                mf = targets[:, model.fixed_idx].to(device) if len(model.fixed_idx) > 0 else None
                _store.clear()
                model(cids, lids, tids, fracs, measured_fluxes=mf, return_attn=False)
                batch_lh = np.stack([s.cpu().numpy() for s in _store], 0)   # (L, H, S, S)
                accum = batch_lh if accum is None else accum + batch_lh
                sa_n += cids.shape[0]
        self_attn_lh = accum / sa_n                            # (L, H, S, S)
        n_lab = self_attn_lh.shape[-1] - n_flux_tok
        lh_block = self_attn_lh[:, :, :n_lab, :n_lab]          # metabolite-token block (L, H, n_lab, n_lab)
        np.savez(args.save_self_attn,
                 self_attn_lh=lh_block,                        # per layer & head
                 self_attn=lh_block.mean((0, 1)),              # layer/head-averaged (backward compat)
                 token_labels=np.array(token_labels[:n_lab], dtype=object))
        print(f"Saved self-attention → {args.save_self_attn}  per-head {lh_block.shape}")

    # ── Gate values + top-flux labels per y-dim ───────────────────────────────
    Z        = np.array(nsd.get('Z_free', []), dtype=float)   # (386, 38)
    gate_raw = ckpt['model_state_dict'].get('fusion_gate')
    gate     = torch.sigmoid(gate_raw).numpy() if gate_raw is not None else np.full(38, 0.5)

    # Flux-token attention matrix: Z @ attn_mean[:n_free] → (n_fluxes, N)
    # attn_mean has shape (n_free + n_rev, N); only the first n_free rows
    # correspond to the null-space y-dims that Z projects from.
    n_free     = Z.shape[1]                  # 38
    flux_token = Z @ attn_mean[:n_free]      # (386, 38) @ (38, N) → (386, N)

    n_ydims = n_free          # 38 flux basis dims (ignore reversibility query rows)
    N       = attn_mean.shape[1]
    top_k   = max(1, math.ceil(N * args.top_pct))

    # Sort y-dims by gate descending (fusion-heavy first)
    order = np.argsort(-gate[:n_ydims])

    out(f"Cross-attention top-{args.top_pct*100:.0f}% tokens per y-dimension")
    out(f"  {n_collected} samples averaged  |  {N} tokens total  |  showing top {top_k}")
    out(f"  Sorted by gate value (↓ = more fusion / v_known reliance)\n")

    for j in order:
        g = gate[j] if j < len(gate) else 0.5

        top_flux = '—'
        if Z.ndim == 2 and Z.shape[1] > j and flux_names:
            top_i    = int(np.argmax(np.abs(Z[:, j])))
            top_flux = flux_names[top_i] if top_i < len(flux_names) else str(top_i)

        row     = attn_mean[j]
        top_idx = np.argsort(-row)[:top_k]

        p            = row / (row.sum() + 1e-12)
        entropy      = float(-np.sum(p * np.log(p + 1e-12)))
        entropy_norm = entropy / math.log(N)

        out(f"y{j:02d}  gate={g:.3f}  entropy={entropy_norm:.3f}  top flux: {top_flux}")
        out(f"  {'Token':<45}  {'Attn':>7}  {'% total':>8}")
        out(f"  " + "─" * 65)
        for idx in top_idx:
            lbl = token_labels[idx] if idx < len(token_labels) else f"tok{idx}"
            w   = row[idx]
            pct = 100.0 * w / (row.sum() + 1e-12)
            out(f"  {lbl:<45}  {w:.5f}  {pct:>7.2f}%")
        out()

    if args.save_matrix:
        save_path = args.save_matrix
        fixed_idx = np.array(nsd.get('fixed_idx', []), dtype=int)
        free_idx  = np.array(nsd.get('free_idx',  []), dtype=int)
        np.savez(
            save_path,
            flux_token   = flux_token,          # (n_fluxes, N_tokens)  main matrix
            attn_mean    = attn_mean[:n_free],   # (38, N_tokens)        flux-query attention only
            Z            = Z,                   # (n_fluxes, 38)        null-space basis
            gate         = gate,                # (38,)                 gate values
            flux_names   = np.array(flux_names, dtype=object),
            token_labels = np.array(token_labels, dtype=object),
            fixed_idx    = fixed_idx,           # indices of measured (fixed) fluxes
            free_idx     = free_idx,            # indices of free fluxes
        )
        print(f"Saved flux-token matrix → {save_path}")
        print(f"  flux_token shape : {flux_token.shape}  (fluxes × tokens)")
        print(f"  attn_mean  shape : {attn_mean.shape}   (y-dims × tokens)")

    if out_file:
        out_file.close()
        print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
