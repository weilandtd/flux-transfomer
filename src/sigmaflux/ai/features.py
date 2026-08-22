"""Shared token feature selection core for SigmaFlux.

Building blocks used by the pipeline step (extract_informative_features.py).
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
        labeling_df: labeling DataFrame (rows are tokens, columns are samples).
        flux_df: flux DataFrame (rows are reactions, columns are samples).
        model_json: path to the COBRA model JSON, for the null-space basis.
        n_samples: number of samples to use (capped by the DataFrame sizes).
        reversibility_df: optional reversibility DataFrame; appended to theta if given.
        subsample_n: further random subsample size, if smaller than n_samples.
        seed: subsampling random seed.

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

    Args:
        token_states: dict of token id -> MID array, to filter.
        mapping_csv_path: CSV path with experimental_id/model_id columns.
    """
    mapping = pd.read_csv(mapping_csv_path)
    allowed = set(mapping["model_id"].unique())
    before = len(token_states)
    filtered = {k: v for k, v in token_states.items() if k.split("|")[0] in allowed}
    print(f"Mapping filter: {before} → {len(filtered)} tokens "
          f"({before - len(filtered)} dropped, {len(allowed)} allowed model IDs)")
    return filtered


def drop_extracellular_tokens(labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Drop extracellular (``_e``-suffixed metabolite) tokens; returns the filtered
    ``labeling_df``.

    Args:
        labeling_df: labeling DataFrame indexed by (Metabolite, Isotopomer, ...).
    """
    n_before = labeling_df.shape[0]
    mask = ~labeling_df.index.get_level_values("Metabolite").str.endswith("_e")
    filtered = labeling_df.loc[mask]
    print(f"Excluded extracellular tokens: {n_before} → {filtered.shape[0]}")
    return filtered


def select_significant_tokens(
    theta: np.ndarray,
    token_states: dict,
    noise_std: float = 0.02,
    n_permutations: int = 5,
    seed: int = 42,
    exact: bool = True,
    run_mi_audit: bool = False,
):
    """Select tokens by Fisher-information significance against free fluxes.

    A token is included iff its standalone information about theta exceeds a
    permutation-calibrated null floor -- no redundancy trimming on top,
    ``included == significant`` always.

    Args:
        theta: free-flux (+ reversibility) coordinates, (n_samples, d).
        token_states: dict of token id -> per-sample MID array.
        noise_std: assumed labeling measurement noise std.
        n_permutations: permutations used for the significance-floor calibration.
        seed: permutation and sensitivity-Gram cross-fit random seed.
        exact: cross-fit, bias-corrected sensitivity Gram estimate (costlier,
            ``token_selection_method: fisher``) vs a single-shot estimate
            (cheaper, biased, ``fisher_fast``); everything else is identical.
        run_mi_audit: also compute the mutual-information nonlinearity audit.

    Returns ``(df, summary)``; *df* has columns metabolite_id, isotopomer, included,
    significant, rank, standalone_nats (plus MI columns if *run_mi_audit*).
    """
    df, summary = fisher_selection.rank_from_samples(
        theta, token_states,
        noise_std=noise_std,
        n_permutations=n_permutations,
        seed=seed,
        exact=exact,
        run_mi_audit=run_mi_audit,
    )
    df[["metabolite_id", "isotopomer"]] = df["metabolite"].str.split("|", expand=True)
    df = df.drop(columns=["metabolite"])
    df = df.rename(columns={"standalone_gain_nats": "standalone_nats"})
    df["included"] = df["significant"]
    front = ["metabolite_id", "isotopomer", "included", "significant", "rank", "standalone_nats"]
    df = df[front + [c for c in df.columns if c not in front]]
    return df, summary
