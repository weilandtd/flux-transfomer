"""
Fisher information and D-optimal greedy selection of informative labeling tokens.
==================================================================================

Answers, from the *forward simulator or generated samples alone* (no estimator
training), the experimental-design question: which metabolite isotopologue tokens
most reduce uncertainty in the estimated free fluxes, and what is the minimal
non-redundant informative subset?

Framework
---------
Fluxes at steady state live in the null space of the stoichiometric matrix S.
We work in the free-flux coordinates ``theta = Z.T @ v`` where ``Z`` is an
orthonormal null-space basis (the same basis the training pipeline uses,
``compute_sparse_nullspace``).  The measurement is the stacked MID vector
``L = g(v)`` produced by the differentiable MFA model, corrupted by labeling
noise with std ``sigma`` (matching ``apply_labeling_noise``).

**Layer 1 -- Fisher information / optimal experimental design (primary).**
Linearising the simulator about a flux state, ``L approx g(v0) + J (theta -
theta0)`` with Jacobian ``J = dL/dtheta``, the Fisher information that
measuring metabolite ``m`` contributes about ``theta`` is

    F_m = (1/sigma^2) * J_m^T J_m            (d x d, with sigma^2 I noise)

averaged over flux states drawn from the prior (Bayesian / expected Fisher).
In the linear-Gaussian regime this *is* the mutual information
``I(theta; L_S) = 0.5 * logdet(I + F_S Sigma_prior)``, so the Fisher object and
an "MI-based" selection coincide -- we just get it in closed form.  The
informative subset is built by greedy **D-optimal** selection (maximising
``logdet`` of the posterior precision), which automatically penalises
redundancy: a metabolite whose sensitivity lies in already-covered flux
directions adds almost nothing.  The eigenspectrum of the total information
reports how many independent flux directions the candidate set can resolve --
the quantitative version of the "one tracer resolves only a few directions"
observation.

**Layer 2 -- mutual-information audit (nonlinearity check).**
Layer 1 is local/linear.  Because labeling is a nonlinear, sometimes
non-monotone function of fluxes, we cross-check with mutual information
estimated on the simulated samples: a fast Gaussian-copula closed form
``I(theta; L_m)`` (a strict upgrade over Pearson, rank based, shrinkage
regularised), and a k-NN (KSG, via scikit-learn) estimate on 1-D projections
to flag metabolites whose nonlinear dependence the linear/Fisher metric
under-counts.

Everything here uses only the simulator and its generated outcomes.
"""

from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Null-space (free-flux) basis
# ----------------------------------------------------------------------------
def free_flux_basis(model) -> np.ndarray:
    """
    Orthonormal null-space basis ``Z`` of the model's stoichiometric matrix.

    Columns of ``Z`` are the free-flux directions ``theta = Z.T @ v`` with
    ``v = Z @ theta`` for any steady-state flux ``v`` (``S @ v = 0``).  Uses the
    same varimax-rotated basis as the training pipeline so results are
    comparable.

    :param model: a prepared :class:`sigmaflux.core.model.MFAModel`
    :return: ``Z`` of shape ``(n_reactions, d)`` ordered as ``model.reactions``
    """
    from cobra.util.array import create_stoichiometric_matrix
    from sigmaflux.ai.utils import compute_sparse_nullspace

    S = create_stoichiometric_matrix(model._model)  # (n_mets, n_reactions)
    Z = np.asarray(compute_sparse_nullspace(S), dtype=np.float64)
    if Z.shape[0] != len(model.reactions):
        raise ValueError(
            f"Null-space basis has {Z.shape[0]} rows but model has "
            f"{len(model.reactions)} reactions; check reaction ordering."
        )
    return Z


# ----------------------------------------------------------------------------
# Helpers to drive the simulator
# ----------------------------------------------------------------------------
def _set_net_fluxes(model, v: np.ndarray) -> None:
    """Overwrite the model's net fluxes (keeping reversibility) for forward()."""
    sd = model.state_dict()
    sd["_fluxes"] = torch.as_tensor(v, dtype=model.dtype, device=model.device).clone()
    model.load_state_dict(sd)


def _set_reversibility(model, r: np.ndarray) -> None:
    sd = model.state_dict()
    sd["_reversibility"] = torch.as_tensor(r, dtype=model.dtype, device=model.device).clone()
    model.load_state_dict(sd)


def _labeling(model, exp_ixs) -> np.ndarray:
    """Run the forward simulator -> ``(n_exp, max_mid, n_met)`` numpy array."""
    with torch.no_grad():
        L = model(exp_ix=list(exp_ixs))
    L = L.detach().to("cpu").to(torch.float64).numpy()
    if L.ndim == 2:  # single experiment, forward() squeezed the batch axis
        L = L[None, ...]
    return L  # (n_exp, max_mid, n_met)


def candidate_measurements(model):
    """
    Default candidate measurement set derived from the prepared EMU network:
    every metabolite that appears in any subnet, with ``n_atoms`` = the largest
    fragment size at which it occurs (its measurable MID length).  Works for any
    model regardless of ``atom_pattern`` annotations.

    :return: sorted list of ``(metabolite_id, n_atoms)``
    """
    sizes = {}
    for _, graph in model.expansion.emu_network.subnetworks.items():
        for node in graph.nodes():
            entries = []
            if hasattr(node, "_values"):  # OriginSet
                if len(node._values) == 1:
                    v = node._values[0]
                    entries = [(v.id, v.size)]
            elif hasattr(node, "id"):  # EMU
                entries = [(node.id, node.size)]
            for met_id, s in entries:
                sizes[met_id] = max(sizes.get(met_id, 0), int(s))
    return sorted(sizes.items())


def _measurement_columns(model, measurements):
    """Resolve ``[(met_id, n_atoms)]`` to ``[(met_id, col, n_atoms)]``."""
    out = []
    for met_id, n_atoms in measurements:
        if met_id not in model._metabolite_hash:
            continue
        col = model._metabolite_hash[met_id]
        n_atoms = int(n_atoms)
        if n_atoms + 1 > model.max_mid_size:
            n_atoms = model.max_mid_size - 1
        out.append((met_id, col, n_atoms))
    return out


def _extract_tokens(L, columns):
    """
    From labeling ``(n_exp, max_mid, n_met)`` pull each metabolite's MID across
    all experiments.  Returns dict ``met_id -> (n_exp, n_atoms+1)``.
    """
    tok = {}
    for met_id, col, n_atoms in columns:
        tok[met_id] = L[:, : n_atoms + 1, col]  # (n_exp, n_atoms+1)
    return tok


# ----------------------------------------------------------------------------
# Layer 1a -- Fisher Gram matrices via finite-difference Jacobians (rigorous)
# ----------------------------------------------------------------------------
def compute_fisher_grams(
    model,
    Z: np.ndarray,
    flux_df,
    measurements,
    exp_ixs=None,
    reversibility_df=None,
    n_states: int = 25,
    eps: float = 1e-2,
    seed: int = 0,
    jacobian: bool = True,
):
    """
    Per-metabolite Fisher Gram matrices ``G_m = sum_exp mean_states J_m^T J_m``
    on the free-flux coordinates, from central finite-difference Jacobians of
    the simulator along the null-space directions.

    The Fisher information is then ``F_m = G_m / sigma^2``.  We return the
    noise-free Gram so the noise level can be applied (and varied) downstream.

    :param model: prepared MFAModel (forward() runnable)
    :param Z: null-space basis ``(n_reactions, d)`` from :func:`free_flux_basis`
    :param flux_df: DataFrame of net-flux samples, columns = reaction ids,
        one row per sample (e.g. the pipeline's ``flux_samples.csv``)
    :param measurements: iterable of ``(metabolite_id, n_atoms)``
    :param exp_ixs: experiment indices to include (default: all experiments).
        Multiple experiments = multiple tracers; their information adds.
    :param reversibility_df: optional DataFrame of per-sample reversibilities
        (columns = reaction ids).  If absent the model's current/sampled
        reversibility is used and held fixed during differentiation.
    :param n_states: number of flux states to average the Fisher over
    :param eps: relative finite-difference step (fraction of RMS flux)
    :param seed: RNG seed for state subsampling
    :return: dict with keys ``grams`` (met_id -> (d,d)), ``theta_states``
        ((n_used, d)), ``token_states`` (met_id -> (n_used, n_exp*(n_atoms+1))),
        ``n_states`` and ``d``.
    """
    if exp_ixs is None:
        exp_ixs = range(len(model.experiments))
    exp_ixs = list(exp_ixs)
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    reaction_ids = [r.id for r in model.reactions]
    columns = _measurement_columns(model, measurements)

    rng = np.random.default_rng(seed)
    n_avail = len(flux_df)
    pick = rng.choice(n_avail, size=min(n_states, n_avail), replace=False)

    grams = {met_id: np.zeros((d, d)) for met_id, _, _ in columns}
    counts = {met_id: 0 for met_id, _, _ in columns}
    theta_states = []
    token_states = {met_id: [] for met_id, _, _ in columns}

    for si in pick:
        v = flux_df.iloc[si][reaction_ids].to_numpy(dtype=np.float64)
        if reversibility_df is not None:
            r = reversibility_df.iloc[si].reindex(reaction_ids).fillna(0.0).to_numpy(np.float64)
            _set_reversibility(model, np.abs(r))

        h = eps * (np.sqrt(np.mean(v ** 2)) + 1e-9)  # step in theta units

        # base labeling at this state (for MI / token samples)
        _set_net_fluxes(model, v)
        try:
            L0 = _labeling(model, exp_ixs)
        except RuntimeError:
            continue  # degenerate flux state, skip

        # central finite differences along each null-space direction
        if jacobian:
            dL = np.empty((d,) + L0.shape, dtype=np.float64)  # (d, n_exp, max_mid, n_met)
            ok = True
            for j in range(d):
                step = h * Z[:, j]
                _set_net_fluxes(model, v + step)
                try:
                    Lp = _labeling(model, exp_ixs)
                except RuntimeError:
                    ok = False
                    break
                _set_net_fluxes(model, v - step)
                try:
                    Lm = _labeling(model, exp_ixs)
                except RuntimeError:
                    ok = False
                    break
                dL[j] = (Lp - Lm) / (2.0 * h)
            if not ok:
                continue

        theta_states.append(Z.T @ v)
        for met_id, col, n_atoms in columns:
            if jacobian:
                # token Jacobian: (n_exp, n_atoms+1, d)
                J = np.transpose(dL[:, :, : n_atoms + 1, col], (1, 2, 0))
                # sum Fisher over experiments (independent measurements)
                for e in range(J.shape[0]):
                    Je = J[e]  # (n_atoms+1, d)
                    grams[met_id] += Je.T @ Je
                counts[met_id] += 1
            token_states[met_id].append(L0[:, : n_atoms + 1, col].reshape(-1))

    for met_id in grams:
        if counts[met_id] > 0:
            grams[met_id] /= counts[met_id]  # mean over states
        token_states[met_id] = np.asarray(token_states[met_id])

    return {
        "grams": grams,
        "theta_states": np.asarray(theta_states),
        "token_states": token_states,
        "n_states": len(theta_states),
        "d": d,
    }


# ----------------------------------------------------------------------------
# Layer 1b -- Empirical (global-linear) sensitivity from generated data only
# ----------------------------------------------------------------------------
def empirical_sensitivity_grams(theta: np.ndarray, token_states: dict,
                                cross_fit: bool = True, n_splits: int = 5, seed: int = 0):
    """
    Fast alternative to :func:`compute_fisher_grams` that never re-runs the
    simulator: estimate the prior-averaged *linear* sensitivity of each
    metabolite's MID to the free fluxes by least squares on already-generated
    ``(theta, L)`` samples, and form the Fisher Gram ``G_m``.

    This is the principled, noise-weightable, joint generalisation of the
    pipeline's ``corr_yl`` (which is the normalised, max-reduced version).  It
    is global-linear -- use the Jacobian version when local structure matters,
    and the MI audit to catch non-Gaussian dependence.

    **Cross-fitting (default).**  A least-squares sensitivity is positively
    biased: over finite samples a metabolite that is *independent* of the
    fluxes still shows a spurious gradient, inflating its apparent information
    (the multiple-testing bias that plagues MI estimators).  We remove it by
    estimating the sensitivity on two disjoint folds (``B_A``, ``B_B``) and
    using the cross-product ``G = sym(B_A^T B_B)``: spurious gradients differ
    between folds and cancel in expectation, while real signal survives.  The
    signed cross-products are averaged over ``n_splits`` random folds (variance
    reduction) and only then projected to the nearest PSD matrix (negative
    eigenvalues -> 0), so a pure-noise metabolite yields ``G ~ 0``.

    :param theta: ``(n_samples, d)`` free-flux coordinates
    :param token_states: dict ``met_id -> (n_samples, p_m)`` MID samples
    :param cross_fit: use the unbiased cross-fitted estimator (recommended)
    :param n_splits: number of random 2-fold splits to average over
    :return: dict ``met_id -> G_m (d, d)``
    """
    theta = np.asarray(theta, dtype=np.float64)
    n, d = theta.shape

    def _project_psd(G):
        G = 0.5 * (G + G.T)
        w, V = np.linalg.eigh(G)
        w = np.clip(w, 0.0, None)
        return (V * w) @ V.T

    if not cross_fit or n < 20:
        tc = theta - theta.mean(0, keepdims=True)
        pinv = np.linalg.pinv(tc)
        grams = {}
        for met_id, L in token_states.items():
            L = np.asarray(L, dtype=np.float64)
            if L.ndim != 2 or L.shape[0] != n:
                grams[met_id] = np.zeros((d, d))
                continue
            B = (pinv @ (L - L.mean(0, keepdims=True))).T
            grams[met_id] = B.T @ B
        return grams

    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(max(1, n_splits)):
        perm = rng.permutation(n)
        iA, iB = perm[: n // 2], perm[n // 2:]
        thA = theta[iA] - theta[iA].mean(0, keepdims=True)
        thB = theta[iB] - theta[iB].mean(0, keepdims=True)
        splits.append((iA, iB, np.linalg.pinv(thA), np.linalg.pinv(thB)))

    grams = {}
    for met_id, L in token_states.items():
        L = np.asarray(L, dtype=np.float64)
        if L.ndim != 2 or L.shape[0] != n:
            grams[met_id] = np.zeros((d, d))
            continue
        Graw = np.zeros((d, d))
        for iA, iB, pinvA, pinvB in splits:
            BA = (pinvA @ (L[iA] - L[iA].mean(0, keepdims=True))).T  # (p, d)
            BB = (pinvB @ (L[iB] - L[iB].mean(0, keepdims=True))).T
            Graw += 0.5 * (BA.T @ BB + BB.T @ BA)
        grams[met_id] = _project_psd(Graw / len(splits))
    return grams


# ----------------------------------------------------------------------------
# Prior precision and greedy D-optimal subset selection
# ----------------------------------------------------------------------------
def _prior_precision(theta_prior: np.ndarray, d: int, ridge: float = 1e-6):
    """Precision matrix of the flux prior over theta (with ridge)."""
    if theta_prior is None or len(theta_prior) <= 1:
        return np.eye(d)
    cov = np.cov(np.asarray(theta_prior, dtype=np.float64).T)
    cov = np.atleast_2d(cov)
    cov = cov + ridge * np.trace(cov) / d * np.eye(d)
    return np.linalg.inv(cov)


def _slogdet(M):
    sign, val = np.linalg.slogdet(M)
    return val if sign > 0 else -np.inf


def greedy_d_optimal(
    grams: dict,
    theta_prior: np.ndarray,
    noise_std: float = 0.02,
    budget: int = None,
    gain_threshold: float = 1e-3,
    rel_gain_threshold: float = 0.01,
    ridge: float = 1e-6,
):
    """
    Greedy D-optimal metabolite selection on the Fisher Gram matrices.

    Posterior precision after measuring subset ``S`` is
    ``P_S = F0 + (1/sigma^2) sum_{m in S} G_m`` where ``F0`` is the flux-prior
    precision.  We greedily add the metabolite with the largest
    ``logdet(P_{S+m}) - logdet(P_S)`` (marginal information gain, in nats),
    stopping when a metabolite resolves a new flux direction no longer (the
    "elbow") or ``budget`` is reached.  ``logdet`` naturally discounts
    metabolites redundant with those already chosen.

    Stopping uses ``max(gain_threshold, rel_gain_threshold * first_gain)``: a
    *relative* elbow is the robust default, because once the resolvable flux
    directions are covered, extra metabolites only keep shrinking variance
    along already-covered directions (noise averaging), so an absolute nats
    threshold would over-select.  Set ``rel_gain_threshold=0`` to disable.

    :return: dict with ``order`` (selected met_ids), ``marginal_gain`` (nats,
        aligned with ``order``), ``cumulative_information`` (logdet P_S after
        each pick, relative to logdet F0), ``standalone_gain`` (met_id ->
        gain if it were the only measurement), and ``F0``.
    """
    met_ids = [m for m, G in grams.items() if np.isfinite(G).all() and np.any(G)]
    if not met_ids:
        return {"order": [], "marginal_gain": [], "cumulative_information": [],
                "standalone_gain": {}, "F0": None}
    d = next(iter(grams.values())).shape[0]
    inv_sig2 = 1.0 / (noise_std ** 2)
    F0 = _prior_precision(theta_prior, d, ridge)
    base = _slogdet(F0)

    standalone = {m: _slogdet(F0 + inv_sig2 * grams[m]) - base for m in met_ids}

    P = F0.copy()
    cur = base
    remaining = set(met_ids)
    order, gains, cum = [], [], []
    stop = gain_threshold
    while remaining and (budget is None or len(order) < budget):
        best_m, best_g = None, -np.inf
        for m in remaining:
            g = _slogdet(P + inv_sig2 * grams[m]) - cur
            if g > best_g:
                best_g, best_m = g, m
        if not order:  # first pick sets the relative elbow threshold
            stop = max(gain_threshold, rel_gain_threshold * best_g)
        if best_g < stop:
            break
        order.append(best_m)
        gains.append(best_g)
        P = P + inv_sig2 * grams[best_m]
        cur = _slogdet(P)
        cum.append(cur - base)
        remaining.discard(best_m)

    return {
        "order": order,
        "marginal_gain": gains,
        "cumulative_information": cum,
        "standalone_gain": standalone,
        "F0": F0,
    }


def observable_spectrum(grams: dict, theta_prior, noise_std: float = 0.02,
                        ridge: float = 1e-6, resolve_ratio: float = 2.0):
    """
    Eigenspectrum of the total Fisher information relative to the flux prior.

    Returns the generalised eigenvalues of ``(F0 + total_information, F0)`` --
    each is the factor by which measuring *all* candidate metabolites sharpens
    the posterior along one flux direction.  ``n_resolved`` counts directions
    whose precision improves by at least ``resolve_ratio`` (the quantitative
    "how many flux directions can labeling resolve" number).
    """
    if not grams:
        return {"eigenvalues": np.array([]), "n_resolved": 0,
                "variance_reduction": np.array([]), "effective_rank": 0.0}
    d = next(iter(grams.values())).shape[0]
    inv_sig2 = 1.0 / (noise_std ** 2)
    total = sum((G for G in grams.values()), np.zeros((d, d)))
    F0 = _prior_precision(theta_prior, d, ridge)
    # generalised eigenvalues of (F0 + inv_sig2*total) x = lam F0 x
    L = np.linalg.cholesky(F0)
    Linv = np.linalg.inv(L)
    M = Linv @ (F0 + inv_sig2 * total) @ Linv.T
    eig = np.linalg.eigvalsh(M)[::-1]                 # posterior/prior precision ratio
    eig = np.clip(eig, 1.0, None)
    var_red = 1.0 - 1.0 / eig                         # fraction of prior variance removed, in [0,1)
    info = eig - 1.0                                  # information eigenvalues (>=0)
    # participation ratio: effective number of strongly-informed directions
    eff_rank = float((info.sum() ** 2) / (np.sum(info ** 2) + 1e-30)) if info.sum() > 0 else 0.0
    return {
        "eigenvalues": eig,
        "variance_reduction": var_red,
        "n_resolved": int(np.sum(eig >= resolve_ratio)),
        "n_resolved_90": int(np.sum(var_red >= 0.90)),
        "n_resolved_99": int(np.sum(var_red >= 0.99)),
        "effective_rank": eff_rank,
        "resolve_ratio": resolve_ratio,
    }


# ----------------------------------------------------------------------------
# Layer 2 -- mutual-information audit (nonlinearity check)
# ----------------------------------------------------------------------------
def _normal_scores(X: np.ndarray) -> np.ndarray:
    """Rank-transform each column to standard-normal scores (Gaussian copula)."""
    from scipy.stats import norm

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    ranks = X.argsort(axis=0).argsort(axis=0) + 1.0
    return norm.ppf(ranks / (n + 1.0))


def copula_mutual_information(theta: np.ndarray, L: np.ndarray) -> float:
    """
    Gaussian-copula estimate of ``I(theta; L)`` in nats (multivariate, both
    sides), with Ledoit-Wolf shrinkage of the joint covariance.

        I = 0.5 * ( logdet C_tt + logdet C_LL - logdet C )

    Captures the full rank-correlation (Gaussian-copula) dependence structure --
    a strict upgrade over Pearson |r| -- but, like any second-order method,
    misses genuinely non-monotone dependence (that is what the KSG gap flags).
    """
    from sklearn.covariance import LedoitWolf

    theta = np.asarray(theta, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if L.ndim == 1:
        L = L[:, None]
    if theta.shape[0] < 5 or L.shape[0] < 5:
        return 0.0
    Zt = _normal_scores(theta)
    Zl = _normal_scores(L)
    J = np.hstack([Zt, Zl])
    C = LedoitWolf().fit(J).covariance_
    dt = theta.shape[1]
    Ctt, Cll = C[:dt, :dt], C[dt:, dt:]
    mi = 0.5 * (_slogdet(Ctt) + _slogdet(Cll) - _slogdet(C))
    return float(max(mi, 0.0))


def ksg_nonlinearity_gap(theta: np.ndarray, L: np.ndarray, n_neighbors: int = 5) -> dict:
    """
    Flag metabolite-flux dependence that the Gaussian/linear score under-counts.

    Both sides are reduced to a 1-D summary (top PCA component of ``L``; the
    flux direction most correlated with it), keeping the k-NN (KSG) estimator
    in its low-dimensional comfort zone.  We then compare a KSG estimate of
    ``I(L_pc; theta_dir)`` to the Gaussian-copula MI on the *same* projections.

    A positive ``gap = ksg - copula`` flags **non-Gaussian** dependence the
    second-order score misses -- note this includes both genuinely
    *non-monotone* coupling **and** near-deterministic relationships (where the
    Gaussian-copula MI saturates while KSG keeps climbing).  Treat a large gap
    as "inspect this metabolite", not as proof of nonlinearity.

    :return: dict with ``ksg``, ``copula_1d`` and ``gap`` (all nats), or zeros
        if scikit-learn is unavailable / sample too small.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception:
        return {"ksg": 0.0, "copula_1d": 0.0, "gap": 0.0}

    theta = np.asarray(theta, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if L.ndim == 1:
        L = L[:, None]
    n = theta.shape[0]
    if n < max(10, n_neighbors + 1):
        return {"ksg": 0.0, "copula_1d": 0.0, "gap": 0.0}

    # top PC of L
    Lc = L - L.mean(0, keepdims=True)
    if not np.any(Lc):
        return {"ksg": 0.0, "copula_1d": 0.0, "gap": 0.0}
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False)
    l_pc = Lc @ Vt[0]
    # flux direction most correlated with l_pc
    tc = theta - theta.mean(0, keepdims=True)
    corr = tc.T @ l_pc
    if not np.any(corr):
        return {"ksg": 0.0, "copula_1d": 0.0, "gap": 0.0}
    t_dir = tc @ (corr / (np.linalg.norm(corr) + 1e-12))

    ksg = float(mutual_info_regression(
        l_pc.reshape(-1, 1), t_dir, n_neighbors=n_neighbors, random_state=0)[0])
    cop = copula_mutual_information(t_dir.reshape(-1, 1), l_pc.reshape(-1, 1))
    return {"ksg": ksg, "copula_1d": cop, "gap": ksg - cop}


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def rank_informative_metabolites(
    model,
    flux_df,
    measurements,
    Z=None,
    exp_ixs=None,
    reversibility_df=None,
    noise_std: float = 0.02,
    n_states: int = 25,
    eps: float = 1e-2,
    sensitivity: str = "jacobian",
    budget: int = None,
    gain_threshold: float = 1e-3,
    run_mi_audit: bool = True,
    seed: int = 0,
):
    """
    End-to-end ranking + minimal-subset selection of informative metabolites.

    :param model: prepared MFAModel
    :param flux_df: net-flux samples (columns = reaction ids)
    :param measurements: iterable of ``(metabolite_id, n_atoms)`` candidates
    :param Z: optional precomputed null-space basis (else computed)
    :param sensitivity: ``"jacobian"`` (finite-difference simulator Jacobian,
        rigorous) or ``"empirical"`` (global-linear from generated samples,
        fast, no re-run)
    :param noise_std: labeling measurement noise std (as in apply_labeling_noise)
    :param budget / gain_threshold: stop greedy selection at a size or when the
        marginal information gain (nats) falls below the threshold
    :param run_mi_audit: also compute the Layer-2 MI / nonlinearity audit
    :return: ``(results_df, summary)`` where ``results_df`` is a pandas
        DataFrame, one row per candidate metabolite, sorted by selection, and
        ``summary`` is a dict (observable spectrum, resolved directions, etc.)
    """
    if Z is None:
        Z = free_flux_basis(model)
    d = Z.shape[1]
    if measurements is None:
        measurements = candidate_measurements(model)

    fg = compute_fisher_grams(
        model, Z, flux_df, measurements, exp_ixs=exp_ixs,
        reversibility_df=reversibility_df, n_states=n_states, eps=eps, seed=seed,
        jacobian=(sensitivity == "jacobian"))
    theta_states = fg["theta_states"]
    token_states = fg["token_states"]
    if sensitivity == "jacobian":
        grams = fg["grams"]
    elif sensitivity == "empirical":
        grams = empirical_sensitivity_grams(theta_states, token_states)
    else:
        raise ValueError(f"sensitivity must be 'jacobian' or 'empirical', got {sensitivity}")

    # prior over theta from the full flux sample set
    reaction_ids = [r.id for r in model.reactions]
    theta_prior = np.asarray(flux_df[reaction_ids].to_numpy(np.float64)) @ Z

    return _summarize(
        grams, theta_states, token_states, theta_prior, d, noise_std,
        budget, gain_threshold, run_mi_audit, sensitivity, fg.get("n_states", 0))


def rank_from_samples(
    theta: np.ndarray,
    token_states: dict,
    noise_std: float = 0.02,
    budget: int = None,
    gain_threshold: float = 1e-3,
    rel_gain_threshold: float = 0.05,
    n_permutations: int = 5,
    run_mi_audit: bool = True,
    calibrate_null: bool = True,
    null_seed: int = 12345,
):
    """
    Rank/select informative metabolites directly from generated samples, with
    **no simulator re-run** -- the natural entry point when you already have a
    batch of generated ``(flux, labeling)`` data (e.g. the training-data h5).

    Uses the empirical (global-linear) cross-fitted sensitivity Fisher.  This
    is the rigorous, noise-weighted, joint, redundancy-aware generalisation of
    the pipeline's ``corr_yl`` max\\|r\\| token-information score.

    **Null calibration.**  The empirical Fisher has a residual finite-sample
    floor (a metabolite independent of the fluxes still shows tiny apparent
    information).  With ``calibrate_null`` we shuffle ``theta`` against the
    labels, recompute the information, and take the *maximum* over all
    ``n_permutations x n_tokens`` null standalone gains as a significance floor:
    only metabolites whose information exceeds it are selected, and a
    ``significant`` column flags them.  This is the permutation-null
    significance test for the design score.

    That maximum is a family-wise floor, so it is deliberately conservative --
    and being an extreme-value statistic it is also unstable.  Observed floors
    on three yeast runs with near-identical standalone distributions: 2.98
    (I. orientalis), ~4.53 (S. cerevisiae), 8.51 (R. toruloides), the last
    admitting 2 of 336 tokens.  Widening the candidate pool does not raise it
    (R. toruloides gave 8.5098 at both 336 and 2260 tokens), so it is set by a
    single extreme draw under the fixed ``null_seed``.  Prefer a high quantile
    of ``null_vals`` if a less conservative floor is wanted; note this changes
    which tokens every organism selects.

    :param theta: ``(n_samples, d)`` free-flux coordinates ``Z.T @ v``
    :param token_states: dict ``metabolite_id -> (n_samples, p_m)`` MID samples
    :return: ``(results_df, summary)`` as in :func:`rank_informative_metabolites`
    """
    theta = np.asarray(theta, dtype=np.float64)
    grams = empirical_sensitivity_grams(theta, token_states)

    null_floor = 0.0
    if calibrate_null and len(theta) >= 20:
        rng = np.random.default_rng(null_seed)
        null_vals = []
        for _ in range(n_permutations):
            perm = rng.permutation(len(theta))
            null_grams = empirical_sensitivity_grams(theta[perm], token_states, seed=int(rng.integers(1 << 30)))
            null_sel = greedy_d_optimal(null_grams, theta, noise_std=noise_std,
                                        budget=1, gain_threshold=0.0, rel_gain_threshold=0.0)
            null_vals.extend(null_sel["standalone_gain"].values())
        if null_vals:
            # family-wise floor: the largest information any independent
            # metabolite produced by chance across the permutations
            null_floor = float(np.max(null_vals))

    # null floor -> the per-metabolite significance flag; a relative elbow on
    # the marginal gains -> the non-redundant selection (stop when a new
    # metabolite adds little beyond the directions already covered).
    return _summarize(
        grams, theta, token_states, theta, theta.shape[1], noise_std,
        budget, gain_threshold, run_mi_audit,
        "empirical(samples)", len(theta), null_floor=null_floor,
        rel_gain_threshold=rel_gain_threshold)


def _summarize(grams, theta_states, token_states, theta_prior, d, noise_std,
               budget, gain_threshold, run_mi_audit, sensitivity, n_states,
               null_floor=0.0, rel_gain_threshold=0.01):
    """Assemble the results DataFrame + summary from per-metabolite Grams."""
    import pandas as pd

    # per-metabolite standalone information, and the significance flag
    F0 = _prior_precision(theta_prior, d)
    base = _slogdet(F0)
    inv_sig2 = 1.0 / (noise_std ** 2)
    standalone_all = {m: _slogdet(F0 + inv_sig2 * G) - base for m, G in grams.items()}
    significant = {m for m, v in standalone_all.items() if v > null_floor}

    # non-redundant selection runs only over significant metabolites
    grams_sig = {m: grams[m] for m in significant}
    sel = greedy_d_optimal(
        grams_sig, theta_prior, noise_std=noise_std, budget=budget,
        gain_threshold=gain_threshold, rel_gain_threshold=rel_gain_threshold)
    spec = observable_spectrum(grams, theta_prior, noise_std=noise_std)

    order_rank = {m: i for i, m in enumerate(sel["order"])}
    marginal = dict(zip(sel["order"], sel["marginal_gain"]))

    rows = []
    for met_id, G in grams.items():
        standalone = standalone_all.get(met_id, 0.0)
        row = {
            "metabolite": met_id,
            "selected": met_id in order_rank,
            "significant": standalone > null_floor,
            "greedy_rank": order_rank.get(met_id, np.nan),
            "marginal_gain_nats": marginal.get(met_id, np.nan),
            "standalone_gain_nats": standalone,
            "fisher_trace": float(np.trace(G)) / (noise_std ** 2),
        }
        if run_mi_audit and met_id in token_states and len(token_states[met_id]):
            Lm = token_states[met_id]
            row["copula_mi_nats"] = copula_mutual_information(theta_states, Lm)
            gap = ksg_nonlinearity_gap(theta_states, Lm)
            row["ksg_mi_1d_nats"] = gap["ksg"]
            row["nonlinearity_gap_nats"] = gap["gap"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"n_free_flux_directions": int(d), "n_resolved_directions": 0,
                    "n_selected": 0, "selected_order": [], "total_information_nats": 0.0,
                    "n_states_used": int(n_states), "noise_std": noise_std,
                    "sensitivity": sensitivity, "eigenvalues": np.array([])}
    df = df.sort_values(
        by=["selected", "greedy_rank", "standalone_gain_nats"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    summary = {
        "n_free_flux_directions": int(d),
        "n_resolved_directions": spec["n_resolved"],
        "n_resolved_90pct": spec.get("n_resolved_90"),
        "n_resolved_99pct": spec.get("n_resolved_99"),
        "effective_rank": spec.get("effective_rank"),
        "resolve_ratio": spec.get("resolve_ratio"),
        "eigenvalues": spec["eigenvalues"],
        "variance_reduction": spec.get("variance_reduction"),
        "n_selected": len(sel["order"]),
        "selected_order": sel["order"],
        "total_information_nats": float(sel["cumulative_information"][-1])
        if sel["cumulative_information"] else 0.0,
        "n_states_used": int(n_states),
        "noise_std": noise_std,
        "sensitivity": sensitivity,
        "null_floor_nats": float(null_floor),
        "n_significant": int(df["significant"].sum()),
    }
    return df, summary
