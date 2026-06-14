"""Shared token feature selection core for SigmaFlux.

Building blocks used by both the pipeline step (extract_informative_features.py)
and the offline analysis script (paper/analysis/informative_tokens.py).
"""
import numpy as np
import pandas as pd
from cobra.io.json import load_json_model
from cobra.util.array import create_stoichiometric_matrix

from sigmaflux.ai import fisher_selection
from sigmaflux.ai.utils import compute_sparse_nullspace


def build_theta_and_token_states(
    labeling_df: pd.DataFrame,
    flux_df: pd.DataFrame,
    model_json: str,
    n_samples: int,
    reversibility_df: pd.DataFrame | None = None,
    subsample_n: int | None = None,
    seed: int = 42,
    skip_constraints: list | None = None,
):
    """Project fluxes onto the null-space basis and build per-token MID arrays.

    Args:
        labeling_df:      (n_tokens, n_samples) with MultiIndex (Tracer, Metabolite, Isotopomer).
        flux_df:          (n_reactions, n_samples) with reaction IDs as index.
        model_json:       Path to COBRA model JSON.
        n_samples:        Maximum number of samples to use.
        reversibility_df: Optional (n_reactions, n_samples) reversibility data. When
                          provided, variable reversibilities are log1p-z-scored and
                          appended to theta so Fisher selection also captures tokens
                          informative about reaction reversibility.
        subsample_n:      If set, randomly subsample this many from the first n_samples.
        seed:             RNG seed for the subsample permutation.

    Returns:
        theta:        (n, d) or (n, d+n_rev) free-flux (+ reversibility) coordinates.
        token_states: dict keyed "met_id|M+k" → (n, n_tracers) array.
        d:            Number of free-flux directions (null-space dimension, excludes rev).
    """
    model = load_json_model(model_json)
    reaction_ids = [r.id for r in model.reactions]
    S = create_stoichiometric_matrix(model)
    if skip_constraints:
        from sigmaflux.ai.utils import drop_constraint_rows
        metabolite_ids = [m.id for m in model.metabolites]
        S = drop_constraint_rows(S, metabolite_ids, reaction_ids, skip_constraints)
    Z = np.asarray(compute_sparse_nullspace(S), dtype=np.float64)  # (n_rxn, d)
    d = Z.shape[1]

    n = min(n_samples, flux_df.shape[1], labeling_df.shape[1])
    V = flux_df.loc[reaction_ids].iloc[:, :n].to_numpy(np.float64).T  # (n, n_rxn)
    theta = V @ Z  # (n, d)

    if reversibility_df is not None:
        rev = reversibility_df.iloc[:, :n].to_numpy(np.float64).T  # (n, n_rxn)
        rev_std = rev.std(axis=0)
        rev = rev[:, rev_std > 1e-6]          # drop constant reversibilities
        rev = np.log1p(np.abs(rev))
        rev = (rev - rev.mean(axis=0, keepdims=True)) / (rev.std(axis=0, keepdims=True) + 1e-8)
        theta = np.concatenate([theta, rev], axis=1)  # (n, d + n_rev_variable)
        print(f"Reversibility augmentation: {rev.shape[1]} variable reactions appended to theta")

    lab = labeling_df.iloc[:, :n]
    token_states = {}
    for (met_id, iso), sub in lab.groupby(level=["Metabolite", "Isotopomer"]):
        token_states[f"{met_id}|{iso}"] = sub.to_numpy(np.float64).T  # (n, n_tracers)

    if subsample_n is not None and subsample_n < n:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)[:subsample_n]
        theta = theta[idx]
        token_states = {k: v[idx] for k, v in token_states.items()}
        print(f"Random subsample: {subsample_n} of {n} samples (seed={seed})")

    return theta, token_states, d


def apply_mapping_filter(token_states: dict, mapping_csv_path: str) -> dict:
    """Restrict token candidates to experimentally measurable metabolites.

    Args:
        token_states:      dict keyed "met_id|M+k" → (n, n_tracers) array.
        mapping_csv_path:  Path to mapping CSV with columns experimental_id, model_id.

    Returns:
        Filtered token_states with only keys whose met_id is in the mapping CSV.
    """
    mapping = pd.read_csv(mapping_csv_path)
    allowed = set(mapping["model_id"].unique())
    before = len(token_states)
    filtered = {k: v for k, v in token_states.items() if k.split("|")[0] in allowed}
    print(f"Mapping filter: {before} → {len(filtered)} tokens "
          f"({before - len(filtered)} dropped, {len(allowed)} allowed model IDs)")
    return filtered


def select_informative_tokens(
    theta: np.ndarray,
    token_states: dict,
    noise_std: float = 0.02,
    gain_threshold: float = 0.01,
    budget: int | None = None,
    run_mi_audit: bool = False,
):
    """Rank and select tokens by Fisher/D-optimal information about free fluxes.

    Args:
        theta:           (n, d) free-flux coordinates.
        token_states:    dict keyed "met_id|M+k" → (n, n_tracers).
        noise_std:       Labeling measurement noise std for Fisher estimation.
        gain_threshold:  Stop greedy selection when marginal gain < this (nats).
        budget:          Hard cap on selected token count (None = unlimited).
        run_mi_audit:    Run MI / nonlinearity audit (slow).

    Returns:
        df:      per-token DataFrame with columns token, metabolite_id, isotopomer,
                 selected, significant, greedy_rank, marginal_gain_nats,
                 standalone_gain_nats, fisher_trace [+ MI columns if audit].
        summary: dict from rank_from_samples.
    """
    df, summary = fisher_selection.rank_from_samples(
        theta, token_states,
        noise_std=noise_std,
        gain_threshold=gain_threshold,
        budget=budget,
        run_mi_audit=run_mi_audit,
    )
    df = df.rename(columns={"metabolite": "token"})
    df[["metabolite_id", "isotopomer"]] = df["token"].str.split("|", expand=True)
    front = ["token", "metabolite_id", "isotopomer", "selected", "significant",
             "greedy_rank", "marginal_gain_nats", "standalone_gain_nats"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df, summary
