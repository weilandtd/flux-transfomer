"""
Version benchmark — OLD model (LabelingToFluxTransformer) on the IO dataset.

Trains the initial-version model on the SAME IO data and the SAME train/val split
saved in the current FluxDecoder checkpoint, with matched architecture width and
training budget, so accuracy / compute / convergence are head-to-head comparable.

The old model is labeling-only by design (no measured-flux bypass) — that's the
"as-designed" arm of the comparison. R2 is per-reaction on the held-out val split,
computed in normalized target space (R2 is affine-invariant, so identical to raw).

Usage:
    python paper/analysis/benchmark_old_model.py \
        --checkpoint paper/.results/i_orientalis_20260620/imisor_transformer.pt \
        --old-repo   <path-to>/mammalian_flux_atlas \
        --out-dir    paper/.results/version_benchmark
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader, Subset


def per_reaction_r2(pred, gt):
    ss_res = ((pred - gt) ** 2).sum(0)
    ss_tot = ((gt - gt.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 1e-9, 1 - ss_res / ss_tot, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--old-repo', required=True)
    ap.add_argument('--out-dir', default='paper/.results/version_benchmark')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--var-floor', type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, args.old_repo)
    from mfa.ai import (LabelingToFluxTransformer, load_data, create_tracer_mapping,
                        load_stoichiometric_matrix, prepare_dataset, var_norm_mse_loss)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # ── same split + data as the current checkpoint ──────────────────────────
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_idx = np.asarray(ck['train_indices'], int)
    val_idx   = np.asarray(ck['val_indices'], int)
    flux_names = list(ck['flux_names'])
    data_dir = ck['data_dir']
    model_json = ck['model_json']
    n_rxn = len(flux_names)
    print(f"split: train={len(train_idx)} val={len(val_idx)} | reactions={n_rxn}")

    lab, flux, rev = load_data(data_dir)            # all samples, original order
    thm, base = create_tracer_mapping(lab)
    n_exp = len(lab.index.get_level_values('Tracer').unique())
    input_dim = lab.shape[0] // n_exp
    S, cobra = load_stoichiometric_matrix(model_json)

    # old prepare_dataset SELECTS the given indices (and z-scores labeling inputs;
    # targets stay raw — the var-norm loss handles their scale). Build train/val
    # directly from their index sets, matching the current checkpoint's split.
    train_ds, norm_stats = prepare_dataset(lab, flux, rev, thm,
                                           train_indices=train_idx, normalize=True)
    val_ds, _ = prepare_dataset(lab, flux, rev, thm,
                                train_indices=val_idx, normalize=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds), shuffle=False)

    # ── model (matched width to current) ─────────────────────────────────────
    model = LabelingToFluxTransformer(
        stoichiometric_matrix=S, input_dim=input_dim,
        flux_dim=n_rxn, reversibility_dim=n_rxn, num_base_tracers=len(base),
        combined_dim=args.d_model, dropout=args.dropout,
        num_attn_heads=args.n_heads, num_transformer_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OLD params: {n_params:,}  (d_model={args.d_model}, layers={args.n_layers})")

    # var-norm loss, variance from train targets. Apply a variance FLOOR (the old
    # recipe had none -> near-constant reactions blew up the loss; the current
    # version fixed this via variance_min_cutoff). Floor here for a fair comparison.
    train_tgts = torch.cat([t for _, t in train_loader], 0)
    var = train_tgts.var(0).clamp(min=args.var_floor)
    flux_var, rev_var = var[:n_rxn].to(device), var[n_rxn:].to(device)
    flux_loss = partial(var_norm_mse_loss, precomputed_variance=flux_var)
    rev_loss  = partial(var_norm_mse_loss, precomputed_variance=rev_var)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # held-out val tensors for R2
    (xb_val, yb_val) = next(iter(val_loader))
    xb_val, yb_val = xb_val.to(device), yb_val.to(device)
    gt_val = yb_val[:, :n_rxn].cpu().numpy()

    best_val, best_state, best_epoch, no_improve = float('inf'), None, -1, 0
    tr_curve, va_curve = [], []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            fp, rp = model(xb)
            loss = flux_loss(fp, yb[:, :n_rxn]) + rev_loss(rp, yb[:, n_rxn:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.shape[0]
        tr = tot / len(train_ds)

        model.eval()
        with torch.no_grad():
            fp, rp = model(xb_val)
            vloss = (flux_loss(fp, yb_val[:, :n_rxn]) + rev_loss(rp, yb_val[:, n_rxn:])).item()
        sched.step(vloss)
        tr_curve.append(tr); va_curve.append(vloss)

        if vloss < best_val:
            best_val, best_epoch, no_improve = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if epoch % 25 == 0 or no_improve >= args.patience:
            print(f"epoch {epoch:4d}  train {tr:.4f}  val {vloss:.4f}  best@{best_epoch}", flush=True)
        if no_improve >= args.patience:
            print(f"early stop @ {epoch} (no val improve {args.patience})")
            break
    train_time = time.time() - t0

    # ── eval best model: per-reaction R2 on val ──────────────────────────────
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        fp, _ = model(xb_val)
    r2 = per_reaction_r2(fp.cpu().numpy(), gt_val)

    res = {
        'model': 'LabelingToFluxTransformer (old, labeling-only)',
        'n_params': int(n_params),
        'd_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': args.n_heads,
        'input_dim_per_exp': int(input_dim), 'n_experiments': int(n_exp),
        'seq_len_tokens': int(n_exp),
        'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
        'epochs_run': len(tr_curve), 'best_epoch': int(best_epoch),
        'train_time_sec': round(train_time, 1),
        'best_val_loss': round(best_val, 6),
        'r2_median': float(np.nanmedian(r2)),
        'r2_mean': float(np.nanmean(r2)),
        'n_r2_gt_0.9': int(np.nansum(r2 > 0.9)),
        'n_r2_gt_0.95': int(np.nansum(r2 > 0.95)),
        'r2_per_reaction': [None if np.isnan(v) else float(v) for v in r2],
        'flux_names': flux_names,
        'train_curve': tr_curve, 'val_curve': va_curve,
    }
    out = os.path.join(args.out_dir, 'old_model_io.json')
    with open(out, 'w') as f:
        json.dump(res, f)
    torch.save(best_state, os.path.join(args.out_dir, 'old_model_io.pt'))
    print(f"\n=== OLD model (labeling-only) on IO ===")
    print(f"params={n_params:,}  train_time={train_time:.1f}s  epochs={len(tr_curve)} best@{best_epoch}")
    print(f"R2 median={res['r2_median']:.3f}  mean={res['r2_mean']:.3f}  "
          f"#R2>0.9={res['n_r2_gt_0.9']}  #R2>0.95={res['n_r2_gt_0.95']}")
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
