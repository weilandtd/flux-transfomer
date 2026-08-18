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

    When *reversibility_df* is given, variable reversibilities are log1p-z-scored and appended
    to theta so Fisher selection also captures tokens informative about reversibility.
    Returns ``(theta, token_states, d)`` — free-flux (+ reversibility) coordinates, a dict
    keyed ``"met_id|M+k"`` of ``(n, n_tracers)`` arrays, and the null-space dimension.
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
    """Restrict token candidates to metabolites listed in *mapping_csv_path* (columns
    experimental_id, model_id); returns the filtered ``token_states``.
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

    *token_filter* ('significant' or 'selected') picks which tokens the returned DataFrame
    marks ``included`` — the training gate. Returns ``(df, summary)``; *df* has columns
    metabolite_id, isotopomer, included, significant, rank, standalone_nats, marginal_nats
    (plus MI columns if *run_mi_audit*).
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


def _token_standalone_nats(L, pinv, Sigma, inv_sig2, Ik):
    """Det-lemma standalone information (nats) of one token's labeling matrix ``L``.

    ``logdet(I_k + inv_sig2 * B Sigma B^T)`` with ``B = (pinv @ centered(L)).T`` (k×D);
    returns 0.0 when the determinant is non-positive.
    """
    B = (pinv @ (L - L.mean(0, keepdims=True))).T              # (k, D)
    sign, val = np.linalg.slogdet(Ik + inv_sig2 * (B @ Sigma @ B.T))
    return float(val) if sign > 0 else 0.0


def select_significant_tokens_fast(theta, token_states, noise_std=0.01,
                                   n_permutations=10, seed=42, ridge=1e-6):
    """Fast significance-only token selection via the matrix-determinant lemma, for genome /
    whole-body models where :func:`select_informative_tokens`'s dense d×d Fisher grams and
    per-candidate d³ log-determinant are intractable. Each token's standalone information is a
    k×k determinant (never forms a d×d gram); significance is a permutation test, and there is
    no greedy redundancy step, so ``included == significant``. Returns ``(df, summary)``
    matching :func:`select_informative_tokens`'s schema.
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

    real = {t: _token_standalone_nats(np.asarray(token_states[t], np.float64),
                                      pinv, Sigma, inv_sig2, Ik) for t in keys}
    rng = np.random.default_rng(seed)
    floor = 0.0
    for _ in range(max(1, n_permutations)):
        perm = rng.permutation(tc.shape[0])
        for t in keys:
            floor = max(floor, _token_standalone_nats(
                np.asarray(token_states[t], np.float64)[perm], pinv, Sigma, inv_sig2, Ik))

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
