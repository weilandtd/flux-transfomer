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
    rel_gain_threshold: float = 0.05,
    n_permutations: int = 5,
    budget: int | None = None,
    token_filter: str = "significant",
    run_mi_audit: bool = False,
):
    """Rank and select tokens by Fisher/D-optimal information about free fluxes.

    Args:
        theta:               (n, d) free-flux coordinates.
        token_states:        dict keyed "met_id|M+k" → (n, n_tracers).
        noise_std:           Labeling measurement noise std for Fisher estimation.
        gain_threshold:      Absolute stop threshold for greedy selection (nats).
        rel_gain_threshold:  Stop when next gain < this fraction of the first gain.
        n_permutations:      Null-calibration permutations to set significance floor.
        budget:              Hard cap on token count (None = unlimited).
        token_filter:        'significant' | 'selected' — which tokens are marked
                             included (the training gate).
        run_mi_audit:        Run MI / nonlinearity audit (slow).

    Returns:
        df:      per-token DataFrame with columns metabolite_id, isotopomer,
                 included, significant, rank, standalone_nats, marginal_nats
                 [+ MI columns if audit].
        summary: dict from rank_from_samples.
    """
    df, summary = fisher_selection.rank_from_samples(
        theta, token_states,
        noise_std=noise_std,
        gain_threshold=gain_threshold,
        rel_gain_threshold=rel_gain_threshold,
        n_permutations=n_permutations,
        budget=budget,
        run_mi_audit=run_mi_audit,
    )
    df[["metabolite_id", "isotopomer"]] = df["metabolite"].str.split("|", expand=True)
    df = df.drop(columns=["metabolite", "fisher_trace"])
    df = df.rename(columns={
        "greedy_rank":        "rank",
        "marginal_gain_nats": "marginal_nats",
        "standalone_gain_nats": "standalone_nats",
    })
    if token_filter == "significant":
        df["included"] = df["significant"]
    elif token_filter == "selected":
        df["included"] = df["selected"]
    else:
        raise ValueError(f"token_filter must be 'significant' or 'selected', got {token_filter!r}")
    df = df.drop(columns=["selected"])
    front = ["metabolite_id", "isotopomer", "included", "significant", "rank",
             "standalone_nats", "marginal_nats"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df, summary


def select_significant_tokens_fast(theta, token_states, noise_std=0.01,
                                   n_permutations=10, seed=42, ridge=1e-6):
    """Fast significance-only token selection via the matrix-determinant lemma.

    Scales to genome / whole-body models where :func:`select_informative_tokens` is
    intractable (dense d×d Fisher grams + a per-candidate d³ log-determinant + greedy).
    Each token's standalone information is computed without ever forming a d×d gram:

        standalone(m) = logdet(F0 + G_m) - logdet(F0)
                      = logdet(I_k + (1/sigma^2) * B_m Sigma B_m^T)     # a k×k determinant

    with ``B_m = (pinv(theta) @ L_m).T`` (k×d) and ``Sigma = cov(theta) + ridge = F0^{-1}``
    (never inverted). Significance is a permutation test whose floor is the family-wise
    max standalone under a broken theta<->L pairing (``pinv``/``Sigma`` are reused, so the
    permutations are cheap). There is NO greedy redundancy step, so ``included ==
    significant``. Complexity O(C·d²) time, O(C·d) memory (vs O(K·C·d³) / O(C·d²) for the
    greedy Fisher path). Reproduces that path's significant set to within near-noise
    borderline tokens.

    Returns ``(df, summary)`` matching :func:`select_informative_tokens`' schema.
    """
    keys = list(token_states)
    tc = np.asarray(theta, np.float64)
    tc = tc - tc.mean(0, keepdims=True)
    D = tc.shape[1]
    pinv = np.linalg.pinv(tc)                                   # (D, n) — once
    cov = np.atleast_2d(np.cov(tc.T))
    Sigma = cov + ridge * np.trace(cov) / D * np.eye(D)        # F0^{-1} — once
    inv_sig2 = 1.0 / (noise_std ** 2)
    Ik = np.eye(np.asarray(token_states[keys[0]]).shape[1])

    def standalone(L):
        B = (pinv @ (L - L.mean(0, keepdims=True))).T          # (k, D)
        s, v = np.linalg.slogdet(Ik + inv_sig2 * (B @ Sigma @ B.T))
        return float(v) if s > 0 else 0.0

    real = {t: standalone(np.asarray(token_states[t], np.float64)) for t in keys}
    rng = np.random.default_rng(seed)
    floor = 0.0
    for _ in range(max(1, n_permutations)):
        perm = rng.permutation(tc.shape[0])
        for t in keys:
            floor = max(floor, standalone(np.asarray(token_states[t], np.float64)[perm]))

    rows = [{"metabolite_id": t.split("|", 1)[0], "isotopomer": t.split("|", 1)[1],
             "standalone_nats": real[t], "marginal_nats": real[t],
             "significant": real[t] > floor} for t in keys]
    df = pd.DataFrame(rows).sort_values("standalone_nats", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(len(df))
    df["included"] = df["significant"]                         # no greedy trim
    df = df[["metabolite_id", "isotopomer", "included", "significant", "rank",
             "standalone_nats", "marginal_nats"]]
    summary = {"n_free_flux_directions": D, "null_floor_nats": floor,
               "n_selected": int(df["significant"].sum()),
               "effective_rank": None, "eigenvalues": []}
    return df, summary
