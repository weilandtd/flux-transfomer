"""
Uncertainty quantification for FluxTransformer flux predictions.

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


def _percentile_interval(drawn, keys, alpha):
    """Bootstrap percentile interval of each group's draws, as ``(lo, hi)`` Series.

    The draws are already a nonparametric bootstrap -- measured replicates resampled with
    replacement across tracers (:func:`sample_inputs`) -- so the interval is the matching
    percentile of the bootstrap distribution rather than a scale fitted to it, and comes
    out asymmetric wherever the draws are.

    Args:
        drawn: the bootstrap rows (Sample < n_draws), long-format with 'Value'.
        keys: the group columns identifying one predicted quantity.
        alpha: two-sided miss rate.
    """
    values = drawn.groupby(keys, sort=False)['Value']
    return values.quantile(alpha / 2.0), values.quantile(1.0 - alpha / 2.0)


def summarize_draws(raw: pd.DataFrame, n_draws: int, group_cols=('Flux Name',),
                    alpha: float = 0.05) -> pd.DataFrame:
    """Reduce raw per-(sample, member) predictions to the MLE point and its interval.

    The point estimate is the ensemble medoid at the unperturbed input (see
    :func:`_medoid_weights`): the one member closest to the coordinatewise median over
    the full (Member x key) matrix, reported whole -- so it is a flux state the model
    actually produced -- and the same aggregate :meth:`EnsembleFluxTransformer.predict` uses.
    The interval is the bootstrap percentile (:func:`_percentile_interval`) of the medoid
    re-chosen on every draw (:func:`_aggregate_draws`), so it is the uncertainty of the
    reported value, as an MFA confidence interval is.

    ``sigma_model`` (member disagreement at the MLE input) and ``sigma_input`` (bootstrap
    standard error of ``Value``) are diagnostics; nothing is computed from them.

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
    out = _aggregate_draws(mle, keys).set_index(keys)[['Value']]
    out['sigma_model'] = mle.groupby(keys, sort=False)['Value'].std(ddof=1).fillna(0.0)
    agg_drawn = _aggregate_draws(raw[raw['Sample'] < n_draws], keys)
    by_key = agg_drawn.groupby(keys, sort=False)['Value']
    # Draws can be lost upstream (a labeling that did not solve), so count the survivors.
    out['n_draws'] = by_key.size()
    out['sigma_input'] = by_key.std(ddof=1).fillna(0.0)
    out['Value_lo'], out['Value_hi'] = _percentile_interval(agg_drawn, keys, alpha)
    return out.reset_index()


def _aggregate_draws(drawn, keys, how='medoid'):
    """The point estimator applied to each sample on its own.

    The medoid is chosen again per draw rather than reused from the maximum-likelihood
    row: which member sits nearest the median is itself a function of the input.
    """
    frames = []
    for sample, g in drawn.groupby('Sample', sort=False):
        wide = g.pivot_table(index='Member', columns=keys, values='Value', sort=False)
        w = _medoid_weights(wide.to_numpy(float), how=how)
        agg = pd.Series(w @ wide.to_numpy(float), index=wide.columns, name='Value')
        frames.append(agg.reset_index().assign(Sample=sample))
    return pd.concat(frames, ignore_index=True)


def _medoid_weights(members: np.ndarray, how: str = "medoid") -> np.ndarray:
    """One-hot weights on the member closest to the coordinatewise median.

    Distances are scaled per reaction by the median absolute deviation across members,
    so the largest-magnitude reactions don't decide "closest" alone.

    Args:
        members: (n_members, n_reactions) raw per-member values, one row each.
        how: ``"medoid"``, the only aggregate.
    """
    if how != "medoid":
        raise ValueError(f"Unknown how={how!r}")
    median = np.median(members, axis=0)
    scale = np.median(np.abs(members - median), axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    dist = np.linalg.norm((members - median) / scale, axis=1)
    if not np.isfinite(dist).all():
        raise ValueError("non-finite distance between ensemble members")
    w = np.zeros(len(members))
    w[int(np.argmin(dist))] = 1.0
    return w


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
