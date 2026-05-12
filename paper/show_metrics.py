"""
Print evaluation metrics stored inside a saved checkpoint — no HDF5 needed.

Usage:
    python paper/show_metrics.py I_orientalis_20260508/imisor_transformer.pt
"""
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'log' in ckpt:
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

    EQ_WIDTH = 72  # truncate long equations to keep lines readable

    def fmt_eq(name):
        eq = rxn_eq.get(name, '')
        if len(eq) > EQ_WIDTH:
            eq = eq[:EQ_WIDTH - 1] + '…'
        return eq

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

    print(f"\nReversibility R²  mean={vc['reversibility_mean']:.4f}  median={vc['reversibility_median']:.4f}  n={vc['reversibility_n']}")


if __name__ == '__main__':
    main()
