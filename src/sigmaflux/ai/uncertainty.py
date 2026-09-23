"""
Uncertainty quantification for SigmaFlux flux predictions.

The experiment's inputs are measurements, so they are draws from a distribution rather
than known constants. This module propagates the measured replicates through the
ensemble and reports the plug-in maximum-likelihood estimate together with the resulting
interval.

The replicates are used directly rather than resampled from a fitted noise model: they
already are the realizations of the measurement distribution, and drawing synthetic
samples from a distribution fitted to them would manufacture precision the data does not
have. (The replicate noise is genuinely heteroscedastic -- a constant sigma has R^2 = 0.000
against it, versus 0.851 for sigma^2 = a + b*p(1-p) -- but that is a fact about the data,
not a step this propagation needs; a fitted noise model would only be needed again for a
likelihood/chi-squared construction, which this module does not attempt.) Measured exchange
fluxes carry no replicates -- one value and one reported error per growth condition -- so
they are instead resampled from that error, one draw per MID draw, via the sign-truncated
Beta in :func:`resample_flux`.

Replicates of *different tracers* are recombined, however. Each tracer is its own culture
measured on its own sheet, so pairing replicate 1 of one with replicate 1 of another
asserts a correspondence that does not exist and reaches only min_t R_t of the prod_t R_t
legitimate combinations -- and leaves an under-replicated tracer contributing nothing past
its own replicate count. Draws are therefore always bootstrapped pairings, ``n_bootstrap``
of them. This resamples measured values only, nothing from a fitted distribution, and the
interval's degrees of freedom stay tied to the real replicate count -- pooled as
``sum_t(R_t - 1)``, which is just ``R - 1`` for a single tracer -- so extra draws sharpen
the estimate of the spread without narrowing the interval on resampling alone.

**The interval is the uncertainty of the reported value.** Each draw is reduced to one
aggregate by the same estimator that produces ``Value``, and the interval is the percentile
of those aggregates -- the same quantity an MFA confidence interval describes, so the two
can be compared directly. It is deliberately *not* the spread of individual member
predictions: that answers "where would one member land", which is a wider question and
made our intervals incomparable with the MFA fits they sit beside.

Two spreads are reported, and only the first enters the interval:

* ``sigma_input``  -- the bootstrap standard error of ``Value``: how far the aggregate
  moves when the measured replicates it is built from are resampled.
* ``sigma_model``  -- diagnostic only: how far the members disagree at the
  maximum-likelihood input. It says where an interval's width comes from; it is not part
  of one, because the members are not resampled.

No factor is fitted to make the interval cover any known answer -- coverage against
held-out ground truth is a diagnostic (:func:`coverage_by_reaction`), and tuning to it
would make it true by construction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import truncnorm

#: Bootstrap draws when the config's ``experiment`` block does not say. Large enough for a
#: 2.5% tail to be resolvable in the percentile interval; past that the limit is how many
#: distinct replicate combinations exist, not how many are drawn.
DEFAULT_N_BOOTSTRAP = 200

# The isotopomers of one metabolite under one tracer are a composition: they sum to 1.
_MID_GROUP = ['Tracer Name', 'Metabolite ID']
_MID_POINT = _MID_GROUP + ['Isotopomer']


def resample_flux(value: float, err: float, n_samples: int,
                  rng: np.random.Generator, lb: float = None,
                  ub: float = None) -> np.ndarray:
    """n_samples draws of one measured flux, bounded to the side of zero it was measured on.

    An exchange flux is either uptake or secretion in a given condition, never both, so the
    measurement's Gaussian error is truncated at zero on the physical side. Draws come from
    a Beta on [0, M] scaled by the measured sign, with M the truncated law's 97.5th
    percentile and (alpha, beta) matched to its mean and variance:
    kappa = x(1-x)/v - 1, alpha = kappa*x, beta = kappa*(1-x).

    kappa is bounded below by ~2.17 (the value == 0 limit) and grows with |value|/err, so
    alpha and beta are always positive -- there is no infeasible case to fall back on.

    ``lb``/``ub`` truncate that law to the reaction's own bounds. A draw is a claim about
    what the flux might really have been, so it has to be a value the model admits: the
    measurement's error bar does not know about the network, and without this a quarter of
    the draws for a well-measured uptake can fall outside the polytope the prediction is
    then projected into. Truncation is exact, by inverse CDF rather than by rejection or
    clipping -- clipping would pile the excluded mass onto the bound itself, which is the
    one place a flux is most likely to make the labeling system singular.

    Args:
        value: measured flux value.
        err: reported uncertainty of that value.
        n_samples: number of draws.
        rng: random generator used for the draw.
        lb, ub: the reaction's bounds; None leaves that side untruncated.
    """
    a = -abs(value) / err   # zero, standardized -- the truncation point
    tn = truncnorm(a, np.inf, loc=abs(value), scale=err)
    m, s, M = tn.mean(), tn.std(), tn.ppf(0.975)
    x, v = m / M, (s / M) ** 2
    kappa = x * (1 - x) / v - 1
    sign = 1.0 if value >= 0 else -1.0
    a_beta, b_beta = kappa * x, kappa * (1 - x)
    if lb is None and ub is None:
        return sign * M * rng.beta(a_beta, b_beta, size=n_samples)

    # The draw is sign*M*u with u ~ Beta on [0, 1]; map [lb, ub] onto u and truncate there.
    lo = -np.inf if lb is None else float(lb)
    hi = np.inf if ub is None else float(ub)
    u_lo, u_hi = (lo / M, hi / M) if sign > 0 else (-hi / M, -lo / M)
    u_lo, u_hi = max(0.0, u_lo), min(1.0, u_hi)
    q_lo = stats.beta.cdf(u_lo, a_beta, b_beta) if u_hi > u_lo else 0.0
    q_hi = stats.beta.cdf(u_hi, a_beta, b_beta) if u_hi > u_lo else 0.0
    if q_hi - q_lo < 1e-12:
        # The bounds exclude essentially the whole measured law -- the measurement and the
        # model disagree about this reaction. Report that rather than resampling noise.
        import warnings
        warnings.warn(f"measured flux {value:g} +/- {err:g} has almost no mass inside its "
                      f"bounds [{lo:g}, {hi:g}]; draws are held at the nearest bound")
        return np.full(n_samples, float(np.clip(value, lo, hi)))
    q = rng.uniform(q_lo, q_hi, size=n_samples)
    return sign * M * stats.beta.ppf(q, a_beta, b_beta)


def _bootstrap_replicates(df_L, n_bootstrap: int, rng: np.random.Generator):
    """*n_bootstrap* synthetic replicates, each drawing one replicate per tracer independently.

    A synthetic replicate is a combination such as (T1-R2, T2-R4): the tracers were grown
    and measured separately, so any pairing of their replicates is as legitimate as the
    index-matched one, and there are ``prod_t R_t`` of them rather than ``min_t R_t``.
    Drawing with replacement (rather than enumerating the product) keeps the count
    independent of the tracer count and handles unequal replicate counts per tracer.

    Args:
        df_L: long-format labeling data for one growth condition.
        n_bootstrap: number of synthetic replicates to build.
        rng: random generator used for the replicate draws.
    """
    by_tracer = {tracer: [g[_MID_POINT + ['Fraction']]
                          for _, g in sub.groupby('Sample', sort=True)]
                 for tracer, sub in df_L.groupby('Tracer Name', sort=True)}
    return [reps[rng.integers(len(reps))].assign(Sample=s)
            for s in range(n_bootstrap)
            for reps in by_tracer.values()]


def sample_inputs(df_L, df_Vm, df_err, rng: np.random.Generator,
                  n_bootstrap: int = DEFAULT_N_BOOTSTRAP, bounds=None):
    """The measured inputs to evaluate the model at, plus the replicate mean as the
    maximum-likelihood row.

    Returns ``(df_L_draws, df_Vm_draws)`` carrying samples ``0 .. n_bootstrap-1`` plus
    sample ``n_bootstrap`` holding the per-(tracer, metabolite, isotopomer) replicate mean
    and the measured flux values unperturbed. That row is the plug-in MLE input: by
    invariance of the maximum-likelihood estimate under transformation, ``f(mean)`` is the
    estimate, whereas averaging ``f`` over the individual replicates is a different,
    Jensen-biased quantity. Each draw is paired with an independent flux draw.

    Draws are always bootstrapped combinations (:func:`_bootstrap_replicates`). Taking
    replicate index *i* of every tracer together instead -- which this used to do -- is not
    a defensible alternative but a mistake: it asserts a correspondence between separately
    grown cultures that does not exist, reaches ``min_t R_t`` of the ``prod_t R_t``
    legitimate combinations, and leaves an under-replicated tracer contributing nothing at
    all on the draws past its own replicate count.

    Args:
        df_L: long-format labeling data, one row per (bio-rep, tracer, metabolite, isotopomer).
        df_Vm: measured-flux data for this growth condition.
        df_err: reported uncertainty per measured flux, or None.
        rng: random generator used for the replicate and flux draws.
        n_bootstrap: number of bootstrapped cross-tracer replicate combinations.
        bounds: ``{flux name: (lb, ub)}`` to truncate the flux draws to, or None to draw
            from the reported error alone (see :func:`resample_flux`).
    """
    n_draws = int(n_bootstrap)
    if n_draws < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    mu = df_L.groupby(_MID_POINT, sort=False)['Fraction'].mean().reset_index()
    parts = _bootstrap_replicates(df_L, n_draws, rng)
    parts.append(mu.assign(Sample=n_draws))

    return (pd.concat(parts, ignore_index=True),
            _flux_draws(df_Vm, df_err, n_draws, rng, bounds))


def _flux_draws(df_Vm, df_err, n_draws: int, rng: np.random.Generator,
                bounds=None) -> pd.DataFrame:
    """Measured fluxes resampled from their reported error, plus the unperturbed row.

    Fluxes without a usable error are held at their measured value; ``np.isfinite`` guards
    the unmeasured case, since ``float(nan)`` passes an ``is not None`` check but makes the
    resampler overflow.
    """
    base = df_Vm.drop_duplicates('Flux Name').reset_index(drop=True)
    err_idx = (df_err.drop_duplicates('Flux Name').set_index('Flux Name')['Value'].to_dict()
               if df_err is not None else {})

    draws = {}
    for name, value in zip(base['Flux Name'], base['Value']):
        err = err_idx.get(name)
        if (value is not None and np.isfinite(value) and err is not None and err > 0):
            lb, ub = (bounds or {}).get(name, (None, None))
            draws[name] = resample_flux(float(value), float(err), n_draws, rng, lb, ub)

    parts = []
    for s in range(n_draws + 1):
        part = base.copy()
        part['Sample'] = s
        if s < n_draws:
            for name, values in draws.items():
                part.loc[part['Flux Name'] == name, 'Value'] = float(values[s])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _bc_percentile_interval(drawn, keys, point, alpha):
    """Bias-corrected bootstrap percentile interval, one row per group.

    The draws are already a nonparametric bootstrap -- measured replicates resampled with
    replacement across tracers (:func:`sample_inputs`) -- so the interval is the matching
    percentile of the bootstrap distribution rather than a scale fitted to it. That is
    what lets it come out asymmetric: a flux whose predictions pile up against zero on one
    side and spread on the other now reports that shape instead of averaging it away.

    ``z0`` is Efron's bias correction: the bootstrap distribution generally does not centre
    on the point estimate (which is evaluated at the maximum-likelihood input, not at any
    resample), and a plain percentile interval would inherit that offset. It is read off
    the fraction of bootstrap values below the point estimate and clamped to +/- one
    half-observation, so an estimate outside the bootstrap range shifts the quantiles by a
    finite amount rather than an infinite one.

    The acceleration term of a full BCa is not estimated: it needs a jackknife over the
    measured replicates, and with as few as two per tracer that estimate would be noisier
    than the skewness it corrects.

    Args:
        drawn: the bootstrap rows (Sample < n_draws), long-format with 'Value'.
        keys: the group columns identifying one predicted quantity.
        point: the point estimate per group, indexed by those keys.
        alpha: two-sided miss rate.
    """
    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    bounds = {}
    for key, values in drawn.groupby(keys, sort=False)['Value']:
        v = values.to_numpy()
        theta = float(point.loc[key])
        n = len(v)
        if n < 2 or np.ptp(v) == 0.0:
            bounds[key] = (theta, theta)      # nothing resampled, or nothing moved
            continue
        p0 = np.clip((v < theta).mean(), 1.0 / (2 * n), 1.0 - 1.0 / (2 * n))
        z0 = stats.norm.ppf(p0)
        q = stats.norm.cdf(2.0 * z0 + np.array([z_lo, z_hi]))
        bounds[key] = tuple(np.quantile(v, q))

    idx = point.index
    return (pd.Series([bounds[k][0] for k in idx], index=idx),
            pd.Series([bounds[k][1] for k in idx], index=idx))


def summarize_draws(raw: pd.DataFrame, n_draws: int, group_cols=('Flux Name',),
                    alpha: float = 0.05) -> pd.DataFrame:
    """Reduce raw per-(sample, member) predictions to the MLE point and its interval.

    The point estimate is the ensemble *medoid_convex* aggregate at the unperturbed
    input (see :func:`_medoid_weights`) -- the same aggregation
    :meth:`EnsembleSigmaFlux.predict` uses, evaluated at the MLE input rather than
    averaged over the replicates. One weight vector is solved from the *full* joint
    (Member x key) matrix -- simplex weights minimizing distance to the coordinatewise
    median -- and applied identically across every key, not recomputed key by key,
    since the weights describe which members' overall flux state to blend, not an
    independent choice per reaction.

    medoid_convex rather than the plain mean or a per-reaction median because it is the
    only one of the three that is both a flux state and measurably closer to real data.
    Each member predicts ``v = y @ Z_full.T``, so it lies in the stoichiometric null
    space; both the mean and medoid_convex are convex combinations of those and stay
    there, and stay inside the (convex) bounding box whenever the members do. A
    per-reaction median is neither, and measurably so -- on E. coli core LC-MS it
    reported ``max|S.v| = 5.74`` against a glucose uptake of 8.6, where the mean gave
    2e-6. Between the two feasible options, medoid_convex measured better against real
    SC flux data than the plain mean (pooled R² 0.242 vs 0.228,
    `paper/analysis/diagnostics/compare_aggregation_sc.py`), because it concentrates
    weight near the coordinatewise median instead of spreading uniformly across every
    member. What is given up relative to the mean is nothing measured so far; watch
    ``sigma_model``, which reports member spread regardless of which aggregate is used.

    The interval is the bias-corrected bootstrap percentile (:func:`_bc_percentile_interval`)
    of the *aggregated* draws -- each draw reduced by the same estimator that produces
    ``Value`` (:func:`_aggregate_draws`) -- so it reports the uncertainty of the value we
    publish, which is what an MFA confidence interval reports too. Taking the percentile of
    the raw per-member draws instead would describe where a single member lands, a wider
    and different question.

    Being a percentile rather than a fitted scale, it is **asymmetric** wherever the draws
    are. Its resolution is bounded by the replicate structure, not the draw count: with
    4/2/2 replicates there are only 16 distinct MID configurations to resample, and the
    tails are as fine as that allows.

    ``sigma_model`` is carried alongside as a diagnostic -- member disagreement at the
    maximum-likelihood input -- but nothing is computed from it.

    The sigma columns are kept as diagnostics -- they say where the width comes from, which
    an interval alone does not -- but nothing is computed from them any more.

    None of this is fitted to coverage; coverage (:func:`coverage_by_reaction`) is the check
    on it, not a target.

    Args:
        raw: long predictions with Sample, Member, Value and the group columns.
        n_draws: number of replicate draws; sample ``n_draws`` is the unperturbed MLE row.
        group_cols: columns identifying one predicted quantity. A reaction can appear both
            as a flux and as a reversibility, so 'Type' belongs here alongside the name.
        alpha: two-sided miss rate of the reported interval.
    """
    if 'Member' not in raw.columns:      # single model -- an ensemble of one
        raw = raw.assign(Member=0)

    keys = list(group_cols)
    mle = raw[raw['Sample'] == n_draws]
    drawn = raw[raw['Sample'] < n_draws]

    at_mle = mle.groupby(keys, sort=False)['Value']
    # One medoid_convex weight vector from the full (Member x key) matrix -- see the
    # docstring above for why this must be joint, not per-key.
    wide = mle.pivot_table(index='Member', columns=keys, values='Value', sort=False)
    weights = _medoid_weights(wide.to_numpy(float), how='medoid_convex')
    agg_value = pd.Series(weights @ wide.to_numpy(float), index=wide.columns, name='Value')
    out = pd.DataFrame({
        'Value':       agg_value,
        'sigma_model': at_mle.std(ddof=1).fillna(0.0),
    })
    # The estimator re-applied to every draw. This, not the raw per-member values, is the
    # bootstrap distribution of what gets reported.
    agg_drawn = _aggregate_draws(drawn, keys)
    # How many draws actually reached the percentile. A draw can be lost upstream -- a
    # flux state whose labeling did not solve is dropped rather than carried as NaN -- and
    # an interval from a handful of survivors is not the same statement as one from 200.
    out['n_draws'] = agg_drawn.groupby(keys, sort=False)['Value'].size()
    out['sigma_input'] = (agg_drawn.groupby(keys, sort=False)['Value']
                                   .std(ddof=1).fillna(0.0))

    out['Value_lo'], out['Value_hi'] = _bc_percentile_interval(
        agg_drawn, keys, out['Value'], alpha)
    return out.reset_index()


def _aggregate_draws(drawn, keys, how='medoid_convex'):
    """The point estimator applied to each bootstrap draw on its own.

    Bootstrapping a statistic means recomputing *that statistic* on every resample, so the
    weights are solved again per draw rather than reused from the maximum-likelihood row:
    which members sit nearest the coordinatewise median is itself a function of the input,
    and holding the weights fixed would report the spread of a different estimator than
    the one ``Value`` uses.
    """
    frames = []
    for sample, g in drawn.groupby('Sample', sort=False):
        wide = g.pivot_table(index='Member', columns=keys, values='Value', sort=False)
        w = _medoid_weights(wide.to_numpy(float), how=how)
        agg = pd.Series(w @ wide.to_numpy(float), index=wide.columns, name='Value')
        frames.append(agg.reset_index().assign(Sample=sample))
    return pd.concat(frames, ignore_index=True)


def _medoid_weights(members: np.ndarray, how: str = "medoid") -> np.ndarray:
    """Aggregation weights picking (or blending) the member(s) closest to the
    coordinatewise median, as an alternative to the plain per-reaction mean.

    The mean is a convex combination of null-space vectors, so it stays a valid flux
    state (S.v=0, and inside any convex box every member satisfies) -- see the comment
    on ``EnsembleSigmaFlux.predict``. But it is a *blend*, and may not correspond to any
    single coherent flux state if the ensemble disagrees at a branch point. This offers
    two alternatives that stay closer to the "typical" member:

    Args:
        members: (n_members, n_reactions) raw per-member values, one row each.
        how: ``"medoid"`` -- one-hot at the single member closest (z-scored L2) to the
            coordinatewise median. Exactly one real member's own output, so it is on the
            null-space manifold by construction, but -- unlike the mean -- is not
            guaranteed to lie inside the FVA box (individual members frequently do not).
            ``"medoid_convex"`` -- simplex weights minimizing distance to the median.
            Still a convex combination, so it keeps the mean's box-feasibility guarantee,
            while concentrating weight near the median rather than spreading uniformly.

    Distances are scaled per reaction by the median absolute deviation across members,
    not raw L2, so the largest-magnitude reactions (glucose uptake, biomass) don't dominate
    the distance and "closest member" doesn't collapse to "closest on one big flux".
    """
    median = np.median(members, axis=0)
    scale = np.median(np.abs(members - median), axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (members - median) / scale

    n = len(members)
    if how == "medoid":
        i = int(np.argmin(np.linalg.norm(z, axis=1)))
        w = np.zeros(n)
        w[i] = 1.0
        return w
    if how == "medoid_convex":
        a = z.T  # (n_reactions, n_members)
        w0 = np.full(n, 1.0 / n)
        res = minimize(
            lambda w: float(np.sum((a @ w) ** 2)), w0,
            jac=lambda w: 2 * a.T @ (a @ w),
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            method="SLSQP",
        )
        return res.x
    raise ValueError(f"Unknown how={how!r}")


def coverage_by_reaction(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """Fraction of samples whose true flux falls inside its interval, per reaction.

    Scored per reaction in signed, linear flux units. Fluxes span many orders of magnitude
    across reactions, so a figure pooled over all of them describes the largest reactions
    and nothing else; and taking magnitudes would let a sign error score as a hit.

    This is a diagnostic only. Nothing in this module is fitted to make it come out at the
    nominal level -- if it were, the number would carry no evidence.

    Args:
        pred: columns 'reaction', 'sample', 'Value_lo', 'Value_hi'.
        truth: columns 'reaction', 'sample', 'gt'.
    """
    joined = pred.merge(truth, on=['reaction', 'sample'])
    inside = (joined['Value_lo'] <= joined['gt']) & (joined['gt'] <= joined['Value_hi'])
    return inside.groupby(joined['reaction']).mean()
