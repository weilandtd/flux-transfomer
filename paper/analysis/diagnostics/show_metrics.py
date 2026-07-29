"""
Print evaluation metrics stored inside a saved checkpoint — no HDF5 needed.

Usage:
    python paper/analysis/show_metrics.py I_orientalis_20260508/data/imisor_transformer.pt
"""
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--random', action='store_true',
                        help='Replace R² scores with random values (null-hypothesis baseline)')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'log' in ckpt and not args.random:
        log = ckpt['log']
        print(f"Training: {len(log['train_loss'])} epochs")
        print(f"  Final train loss : {log['train_loss'][-1]:.6g}")
        print(f"  Final val   loss : {log['val_loss'][-1]:.6g}")

    vc   = ckpt['val_correlations']
    cfg  = ckpt.get('model_config', {})
    nsd  = ckpt.get('null_space_data', {})

    n_total   = cfg.get('n_fluxes', '?')
    fixed_idx = nsd.get('fixed_idx', [])
    free_idx  = nsd.get('free_idx',  [])
    n_free_basis = cfg.get('n_free', '?')

    print(f"\n=== Reaction space ===")
    print(f"  Total reactions          : {n_total}")
    print(f"  Measured (soft input)    : {len(fixed_idx)}  ← v_known fed through fusion_layer (soft feature, not hard-copied)")
    print(f"  Free reactions           : {len(free_idx)}  ← predicted via null-space")
    print(f"  Null-space basis dim     : {n_free_basis}  ← what the model ACTUALLY predicts (y ∈ R^{n_free_basis})")
    print(f"  (All {n_total} fluxes are algebraically determined by these {n_free_basis} values)")

    if 'flux_r2_per_reaction' not in vc:
        print("\n(No per-reaction R² in this checkpoint)")
        return

    # Load reaction equations from cobra model for human-readable descriptions
    rxn_eq = {}
    model_json = ckpt.get('model_json')
    if model_json and os.path.exists(model_json):
        from cobra.io.json import load_json_model
        cobra_model = load_json_model(model_json)
        rxn_eq = {r.id: r.reaction for r in cobra_model.reactions}

    flux_names = ckpt.get('flux_names', [])
    fixed_names = {flux_names[i] for i in fixed_idx} if flux_names else set()

    all_names = vc['flux_reaction_names']
    all_r2    = vc['flux_r2_per_reaction']

    if args.random:
        import numpy as np
        rng = np.random.default_rng(0)
        all_r2 = list(rng.uniform(-1, 1, size=len(all_r2)))
        print(f"[--random] R² scores replaced with uniform random values in [-1, 1]  (null-hypothesis baseline)\n")

    pairs         = list(zip(all_r2, all_names))
    pairs_fixed   = [(r2, n) for r2, n in pairs if n in fixed_names]
    pairs_free    = [(r2, n) for r2, n in pairs if n not in fixed_names]

    # Categorize free reactions only (measured reactions shown separately below)
    trivial   = [(r2, n) for r2, n in pairs_free if r2 > 0.9999]
    learned   = [(r2, n) for r2, n in pairs_free if 0 <= r2 <= 0.9999]
    bad       = [(r2, n) for r2, n in pairs_free if r2 < 0]

    import statistics

    # Measured-reaction R² (now genuinely predicted through Z_full, not hard-copied)
    if pairs_fixed:
        print(f"\n=== Measured-flux R² (soft input → predicted via Z_full, n={len(pairs_fixed)}) ===")
        header = f"  {'Reaction':<30}  {'R²':>8}"
        print(header)
        print('  ' + '─' * 42)
        for r2, name in sorted(pairs_fixed, key=lambda x: -x[0]):
            flag = '  ✓' if r2 > 0.5 else ('  ?' if r2 > 0 else '  ✗')
            print(f"  {name:<30}  {r2:>+8.3f}{flag}")

    print(f"\n=== Free-reaction R² breakdown (n={len(pairs_free)} variable reactions) ===")
    print(f"  R²≈1.0  (trivially pinned by measured fluxes) : {len(trivial):3d}")
    print(f"  R²∈[0, 1)  (genuinely learned from labeling)  : {len(learned):3d}", end="")
    if learned:
        lr = [r for r, _ in learned]
        print(f"  mean={sum(lr)/len(lr):.3f}  median={statistics.median(lr):.3f}")
    else:
        print()
    print(f"  R²<0   (model does worse than mean predictor) : {len(bad):3d}", end="")
    if bad:
        print(f"  (worst: {min(r for r, _ in bad):.1f})")
    else:
        print()

    # Overall R² across all samples and fluxes
    if 'val_predictions' in vc and 'val_actuals' in vc:
        import numpy as np
        preds = np.array(vc['val_predictions'], dtype=float).flatten()
        actuals = np.array(vc['val_actuals'], dtype=float).flatten()
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        overall_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        print(f"\n=== Overall R² (all samples × all fluxes) ===")
        print(f"  R² = {overall_r2:+.4f}")

    EQ_WIDTH = 72  # truncate long equations to keep lines readable

    def fmt_eq(name):
        eq = rxn_eq.get(name, '')
        if len(eq) > EQ_WIDTH:
            eq = eq[:EQ_WIDTH - 1] + '…'
        return eq

    # Central carbon metabolism R²
    try:
        import json
        central_carbon_file = 'models/atomistic/I_orientalis/central_carbon_metabolism.json'
        if os.path.exists(central_carbon_file):
            with open(central_carbon_file, 'r') as f:
                cc_data = json.load(f)
            cc_reactions = set(cc_data.get('reaction_ids', []))

            # Filter reactions by central carbon metabolism IDs
            cc_pairs = [(r2, n) for r2, n in pairs if n in cc_reactions]

            if cc_pairs:
                cc_r2 = [r2 for r2, _ in cc_pairs]
                cc_mean = sum(cc_r2) / len(cc_r2)
                cc_median = sorted(cc_r2)[len(cc_r2) // 2]

                print(f"\n=== Central Carbon Metabolism R² (n={len(cc_pairs)}) ===")
                print(f"  Mean R²   = {cc_mean:+.4f}")
                print(f"  Median R² = {cc_median:+.4f}")
                print(f"  Subsystems: {', '.join(cc_data.get('central_carbon_subsystems', []))}")

                # Show per-reaction breakdown for central carbon
                cc_learned = [(r2, n) for r2, n in cc_pairs if 0 <= r2 <= 0.9999]
                cc_bad = [(r2, n) for r2, n in cc_pairs if r2 < 0]
                cc_trivial = [(r2, n) for r2, n in cc_pairs if r2 > 0.9999]

                print(f"  R²≈1.0  (trivially pinned): {len(cc_trivial)}")
                print(f"  R²∈[0,1) (learned)       : {len(cc_learned)}")
                print(f"  R²<0    (poorly predicted): {len(cc_bad)}")

                if cc_learned:
                    cc_learned_sorted = sorted(cc_learned, key=lambda x: -x[0])
                    print(f"\n  Central carbon reactions (best → worst):")
                    for r2, name in cc_learned_sorted:
                        eq = fmt_eq(name)
                        print(f"    {name:<30}  {r2:+.4f}  │  {eq}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        pass  # Central carbon file not found or invalid JSON

    # Show the genuinely learned reactions sorted best → worst
    if learned:
        learned_sorted = sorted(learned, key=lambda x: -x[0])
        col1 = 44  # name + R² + bar
        header = f"  {'Reaction':<30}  {'R²':>7}  {'':20}  │  Equation"
        rule   = '  ' + '─' * (col1 + 4 + EQ_WIDTH)
        print(f"\n--- Genuinely learned reactions (R²∈[0,1), best → worst) ---")
        print(header)
        print(rule)
        for r2, name in learned_sorted:
            bar = '█' * int(r2 * 20)
            eq  = fmt_eq(name)
            print(f"  {name:<30}  {r2:+.4f}  {bar:<20}  │  {eq}")

    if bad:
        bad_sorted = sorted(bad, key=lambda x: x[0])
        header = f"  {'Reaction':<30}  {'R²':>7}  │  Equation"
        rule   = '  ' + '─' * (42 + EQ_WIDTH)
        print(f"\n--- Poorly predicted reactions (R²<0, worst → best) ---")
        print(header)
        print(rule)
        for r2, name in bad_sorted:
            eq = fmt_eq(name)
            print(f"  {name:<30}  {r2:>+7.2f}  │  {eq}")

    # ── Observability → R² correlation ───────────────────────────────────────
    import numpy as np
    from scipy import stats

    Z   = np.array(nsd.get('Z_free',      []), dtype=float)   # (n_rxn, n_basis)
    obs = np.array(nsd.get('obs_weights', []), dtype=float)   # (n_basis,)

    if Z.ndim == 2 and obs.size == Z.shape[1] and flux_names:
        name_to_idx = {n: i for i, n in enumerate(flux_names)}

        obs_pred, actual = [], []
        for r2, name in zip(all_r2, all_names):
            if name not in name_to_idx:
                continue
            row_sq = Z[name_to_idx[name], :] ** 2
            denom  = row_sq.sum()
            obs_pred.append((row_sq @ obs) / denom if denom > 1e-10 else 0.0)
            actual.append(r2)

        obs_pred = np.array(obs_pred)
        actual   = np.array(actual)

        # Spearman is rank-based → robust to the R²=-22k outliers
        rho, p_s = stats.spearmanr(obs_pred, actual)
        # Pearson on clipped R² to reduce explosion influence
        r_p, p_p = stats.pearsonr(obs_pred, np.clip(actual, -5, 1))

        print(f"\n=== Observability → R² correlation (n={len(actual)} reactions) ===")
        print(f"  Spearman ρ = {rho:+.3f}  (p={p_s:.1e})  ← rank-based, robust to explosion outliers")
        print(f"  Pearson  r = {r_p:+.3f}  (p={p_p:.1e})  ← R² clipped to [-5, 1]")

        # Bin reactions by predicted observability; show actual R² per bin
        edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]
        labels = ['[0.00,0.05)', '[0.05,0.10)', '[0.10,0.20)',
                  '[0.20,0.30)', '[0.30,0.50)', '[0.50,1.00]']
        print(f"\n  {'Pred-obs bin':<14}  {'n':>4}  {'mean R²':>8}  {'median R²':>10}  {'%R²>0':>7}")
        print(f"  " + "─" * 52)
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            mask = (obs_pred >= lo) & (obs_pred < hi)
            n = mask.sum()
            if n == 0:
                continue
            r2_bin = actual[mask]
            pct_pos = 100.0 * (r2_bin > 0).mean()
            print(f"  {lab:<14}  {n:>4}  {r2_bin.mean():>+8.3f}  {np.median(r2_bin):>+10.3f}  {pct_pos:>6.1f}%")

    print(f"\nReversibility R²  mean={vc['reversibility_mean']:.4f}  median={vc['reversibility_median']:.4f}  n={vc['reversibility_n']}")

    # ── Gated fusion: per-y-dimension gate analysis ───────────────────────────
    import numpy as np

    state = ckpt.get('model_state_dict', {})
    gate_raw = state.get('fusion_gate')
    if gate_raw is None:
        return

    gate_raw = gate_raw.float().numpy()           # (n_basis,) raw logits
    gate     = 1.0 / (1.0 + np.exp(-gate_raw))   # sigmoid → (0,1)

    Z   = np.array(nsd.get('Z_free',      []), dtype=float)   # (n_rxn, n_basis)
    obs = np.array(nsd.get('obs_weights', []), dtype=float)   # (n_basis,)

    measured = ckpt.get('measured_fluxes') or []
    measured_label = ', '.join(measured) if measured else 'v_known'
    print(f"\n=== Fusion gate  (gate→0: trust transformer/MID  |  gate→1: trust [{measured_label}]) ===")
    print(f"  mean gate = {gate.mean():.3f}   min = {gate.min():.3f}   max = {gate.max():.3f}")

    # Correlation: obs_weight vs gate (hypothesis: high obs → low gate)
    if obs.size == gate.size:
        from scipy import stats as _stats
        rho_g, p_g = _stats.spearmanr(obs, gate)
        print(f"  Spearman ρ(obs_weight, gate) = {rho_g:+.3f}  (p={p_g:.1e})")
        print(f"  {'> 0: fusion learns from MID-observable dims' if rho_g > 0 else '< 0: confirms hypothesis — obs dims rely on transformer'}")

    # Sort dims by gate value descending (most fusion-reliant first)
    order = np.argsort(-gate)

    print(f"\n  {'y-dim':>5}  {'gate':>6}  {'bar':<20}  {'obs_w':>6}  │  top fluxes (|Z| loadings)")
    print(f"  " + "─" * 90)

    name_list = flux_names if flux_names else []

    for j in order:
        g   = gate[j]
        bar = '█' * int(g * 20)
        ow  = f"{obs[j]:.3f}" if obs.size == gate.size else "  ─  "

        # Top 3 fluxes driven by this y-dimension (largest |Z[:,j]|)
        if Z.ndim == 2 and Z.shape[1] > j:
            col     = np.abs(Z[:, j])
            top_idx = np.argsort(-col)[:3]
            top_str = "  ".join(
                f"{name_list[k] if k < len(name_list) else k}({col[k]:.2f})"
                for k in top_idx
            )
        else:
            top_str = ""

        print(f"  {j:>5}  {g:.4f}  {bar:<20}  {ow:>6}  │  {top_str}")

    # Summary: high-gate vs low-gate dims
    high_gate = order[:10]
    low_gate  = order[-10:]
    if obs.size == gate.size:
        print(f"\n  Top-10 gate (fusion-dominated) obs_weight mean : {obs[high_gate].mean():.3f}")
        print(f"  Bot-10 gate (transformer-dom.) obs_weight mean : {obs[low_gate].mean():.3f}")


if __name__ == '__main__':
    main()
