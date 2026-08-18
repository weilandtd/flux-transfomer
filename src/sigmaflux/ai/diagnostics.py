"""
Diagnostics for understanding model learning bottlenecks:
1. Labeling signal coverage per reaction
2. Attention weight analysis (which tokens does each query attend to?)
"""

import numpy as np
import torch
import pandas as pd
from collections import defaultdict


def analyze_labeling_coverage(labeling_df, flux_names, compound_to_reaction):
    """For each reaction, count distinct labeling patterns for its metabolites.
    Placeholder: currently returns the same total training-metabolite count for
    every reaction (TODO: extract per-reaction metabolites)."""
    labeling_patterns = []

    # Get metabolites involved in training data
    training_metabolites = set()
    for tracer_id, met, iso in labeling_df.index:
        training_metabolites.add(met)

    # For each reaction, count how many labeling patterns involve its metabolites
    reaction_coverage = {}
    for rxn_name in flux_names:
        # Find metabolites in this reaction (need model to extract this)
        # For now, approximate: count how many patterns mention metabolites in training set
        # This is a placeholder — full version needs stoichiometric matrix
        reaction_coverage[rxn_name] = len(training_metabolites)  # TODO: extract per-reaction metabolites

    return reaction_coverage


def extract_attention_weights(model, val_loader, device, return_per_query=True):
    """Run validation with return_attn=True and collect attention weights.

    Returns:
        dict with 'attn_weights' ((n_val_samples, n_queries, N_total) array, batches
        concatenated) and 'query_metadata' (list of one {'flux_targets': array} dict
        per sample).
    """
    model.eval()
    all_attn_weights = []
    query_metadata = []

    with torch.no_grad():
        for batch_data in val_loader:
            (cids, lids, tids, fracs), targets = batch_data
            cids = cids.to(device)
            lids = lids.to(device)
            tids = tids.to(device)
            fracs = fracs.to(device)
            targets = targets.to(device)

            measured_fluxes = None
            if len(model.fixed_idx) > 0:
                measured_fluxes = targets[:, model.fixed_idx]

            # Forward with attention weights
            _, _, attn_weights = model(
                cids, lids, tids, fracs,
                measured_fluxes=measured_fluxes,
                return_attn=True
            )  # (B, n_queries, N_total)

            all_attn_weights.append(attn_weights.detach().cpu().numpy())
            # Store metadata about each batch (flux magnitudes)
            for i in range(targets.shape[0]):
                query_metadata.append({
                    'flux_targets': targets[i, :model.flux_dim].detach().cpu().numpy(),
                })

    # Aggregate across batches
    attn_all = np.concatenate(all_attn_weights, axis=0)  # (B*n_batches, n_queries, N_total)

    return {
        'attn_weights': attn_all,  # (n_samples*n_queries, n_queries, N_total)
        'query_metadata': query_metadata,
    }


def analyze_attention_patterns(attn_weights, flux_names, flux_r2, n_labeling_tokens):
    """Compute per-reaction attention statistics (attention to labeling vs. flux
    tokens, entropy, dead zones) from batch-averaged attention weights. Returns a
    DataFrame with one row per reaction query."""
    # Average attention across batch dimension
    mean_attn = attn_weights.mean(axis=0)  # (n_queries, N_total)

    n_queries = mean_attn.shape[0]
    n_total_tokens = mean_attn.shape[1]
    n_flux_tokens = n_total_tokens - n_labeling_tokens

    results = []
    for q_idx in range(n_queries):
        attn_q = mean_attn[q_idx, :]  # (N_total,)

        lab_attn = attn_q[:n_labeling_tokens].sum()
        flux_attn = attn_q[n_labeling_tokens:].sum()

        results.append({
            'reaction': flux_names[q_idx] if q_idx < len(flux_names) else f'query_{q_idx}',
            'r2': flux_r2[q_idx] if q_idx < len(flux_r2) else np.nan,
            'attn_to_labeling': lab_attn,
            'attn_to_flux': flux_attn,
            'max_attn_token': attn_q.argmax(),
            'entropy': -np.sum(attn_q[attn_q > 1e-10] * np.log(attn_q[attn_q > 1e-10])),
            'is_dead': (attn_q.max() < 0.01),  # all attention evenly spread = dead zone
        })

    return pd.DataFrame(results)


def diagnose_signal_bottleneck(model, val_loader, device, flux_names, flux_r2, labeling_df):
    """
    Main diagnostic: correlate attention patterns with learning success.
    """
    print("\n=== Labeling Signal & Attention Diagnostics ===")

    # Extract attention weights
    attn_dict = extract_attention_weights(model, val_loader, device)
    attn_weights = attn_dict['attn_weights']

    # Analyze attention patterns
    n_lab_tokens = len(labeling_df)
    attn_df = analyze_attention_patterns(
        attn_weights, flux_names, flux_r2, n_lab_tokens
    )

    # Correlate attention to learning
    high_r2_mask = attn_df['r2'] > 0.5
    low_r2_mask = attn_df['r2'] < 0.2

    print(f"\nHigh-R² reactions (>0.5):")
    print(f"  Mean attention to labeling: {attn_df.loc[high_r2_mask, 'attn_to_labeling'].mean():.3f}")
    print(f"  Mean entropy: {attn_df.loc[high_r2_mask, 'entropy'].mean():.3f}")
    print(f"  Dead zones (%): {100 * attn_df.loc[high_r2_mask, 'is_dead'].sum() / high_r2_mask.sum():.1f}%")

    print(f"\nLow-R² reactions (<0.2):")
    print(f"  Mean attention to labeling: {attn_df.loc[low_r2_mask, 'attn_to_labeling'].mean():.3f}")
    print(f"  Mean entropy: {attn_df.loc[low_r2_mask, 'entropy'].mean():.3f}")
    print(f"  Dead zones (%): {100 * attn_df.loc[low_r2_mask, 'is_dead'].sum() / low_r2_mask.sum():.1f}%")

    # Identify reactions with dead attention
    dead_attn = attn_df[attn_df['is_dead']]
    if len(dead_attn) > 0:
        print(f"\n⚠ Dead attention zones ({len(dead_attn)} reactions):")
        print(f"  Mean R² of dead-attention reactions: {dead_attn['r2'].mean():.4f}")
        print(f"  Examples: {', '.join(dead_attn.head(5)['reaction'].tolist())}")

    return attn_df
