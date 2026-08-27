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
``sum_t(R_t - 1)``, which is just ``R - 1`` for a single tracer -- so the extra draws
sharpen the estimate of the spread without narrowing the interval on resampling alone, and the reported interval is the bootstrap percentile itself rather than
a symmetric band fitted to its width.

Two variance sources compose into the reported interval, measured together rather than
assumed independent (they are the pooled spread of every (replicate, member) prediction,
capturing their interaction rather than assuming ``sigma_total^2 = sigma_model^2 +
sigma_input^2``):

* ``sigma_model``  -- epistemic: ensemble members disagree, at the maximum-likelihood input.
* ``sigma_input``  -- measurement: the ensemble consensus moves with which replicate it sees.

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
                  rng: np.random.Generator) -> np.ndarray:
    """n_samples draws of one measured flux, bounded to the side of zero it was measured on.

    An exchange flux is either uptake or secretion in a given condition, never both, so the
    measurement's Gaussian error is truncated at zero on the physical side. Draws come from
    a Beta on [0, M] scaled by the measured sign, with M the truncated law's 97.5th
    percentile and (alpha, beta) matched to its mean and variance:
    kappa = x(1-x)/v - 1, alpha = kappa*x, beta = kappa*(1-x).

    kappa is bounded below by ~2.17 (the value == 0 limit) and grows with |value|/err, so
    alpha and beta are always positive -- there is no infeasible case to fall back on.

    Args:
        value: measured flux value.
        err: reported uncertainty of that value.
        n_samples: number of draws.
        rng: random generator used for the draw.
    """
    a = -abs(value) / err   # zero, standardized -- the truncation point
    tn = truncnorm(a, np.inf, loc=abs(value), scale=err)
    m, s, M = tn.mean(), tn.std(), tn.ppf(0.975)
    x, v = m / M, (s / M) ** 2
    kappa = x * (1 - x) / v - 1
    sign = 1.0 if value >= 0 else -1.0
    return sign * M * rng.beta(kappa * x, kappa * (1 - x), size=n_samples)


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
                  n_bootstrap: int = DEFAULT_N_BOOTSTRAP):
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
    """
    n_draws = int(n_bootstrap)
    if n_draws < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    mu = df_L.groupby(_MID_POINT, sort=False)['Fraction'].mean().reset_index()
    parts = _bootstrap_replicates(df_L, n_draws, rng)
    parts.append(mu.assign(Sample=n_draws))

    return pd.concat(parts, ignore_index=True), _flux_draws(df_Vm, df_err, n_draws, rng)


def _flux_draws(df_Vm, df_err, n_draws: int, rng: np.random.Generator) -> pd.DataFrame:
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
            draws[name] = resample_flux(float(value), float(err), n_draws, rng)

    parts = []
    for s in range(n_draws + 1):
        part = base.copy()
        part['Sample'] = s
        if s < n_draws:
            for name, values in draws.items():
                part.loc[part['Flux Name'] == name, 'Value'] = float(values[s])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize_draws(raw: pd.DataFrame, n_draws: int, group_cols=('Flux Name',),
                    alpha: float = 0.05) -> pd.DataFrame:
    """Reduce raw per-(sample, member) predictions to the MLE point and its interval.

    The point estimate is the ensemble median at the unperturbed input -- the same
    aggregation :meth:`EnsembleSigmaFlux.predict` uses, evaluated at the MLE input rather
    than averaged over the replicates.

    Two variance sources compose into the interval:

    * ``sigma_model``  -- member disagreement at the maximum-likelihood input (epistemic).
    * ``sigma_input``  -- how far the ensemble consensus moves across replicates
      (measurement).

    ``sigma_total`` is measured directly over every (replicate, member) draw pooled
    together, not assumed to be ``sqrt(sigma_model^2 + sigma_input^2)`` -- pooling captures
    their interaction, which that identity would not.

    The reported interval is ``point +/- t * sigma_total``, not the empirical percentile of
    the draws: with as few as 3 replicates the 2.5th/97.5th percentiles sit at the sample
    extrema and cannot resolve a 2.5% tail. Because sigma is *estimated* from a small sample
    the quantile is Student-t rather than normal; the degrees of freedom follow the
    replicate axis, since with 3 replicates against ~20 members it is now the thinner one.
    None of this is fitted to coverage -- it is the standard treatment of an estimated
    scale, and coverage (:func:`coverage_by_reaction`) is the check on it, not a target.

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
    out = pd.DataFrame({
        'Value':       at_mle.median(),
        'sigma_model': at_mle.std(ddof=1).fillna(0.0),
    })
    out['sigma_input'] = (drawn.groupby(keys + ['Sample'], sort=False)['Value']
                                .median().reset_index()
                                .groupby(keys, sort=False)['Value']
                                .std(ddof=1).fillna(0.0))
    out['sigma_total'] = drawn.groupby(keys, sort=False)['Value'].std(ddof=1).fillna(0.0)

    dof = max(n_draws - 1, 1)
    half = stats.t.ppf(1.0 - alpha / 2.0, dof) * out['sigma_total']
    out['Value_lo'] = out['Value'] - half
    out['Value_hi'] = out['Value'] + half
    return out.reset_index()


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
