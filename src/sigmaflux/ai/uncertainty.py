"""
Uncertainty quantification for SigmaFlux flux predictions.

The experiment's inputs are measurements, so they are draws from a distribution rather
than known constants. This module estimates that distribution from the measurement
replicates, propagates it through the ensemble by Monte Carlo, and reports the plug-in
maximum-likelihood estimate together with the resulting spread.

Two measured quantities enter, and they carry their uncertainty differently:

* **MID fractions** are supplied as individual replicates, so the noise model is fitted
  from them and the model is fed the replicate *mean*. The relevant input uncertainty is
  therefore the standard error of that mean, ``sigma / sqrt(n_reps)`` -- not the raw
  replicate spread.
* **Measured exchange fluxes** arrive already aggregated: one value and one ``_err`` per
  growth condition, with no replicates to average. ``_err`` is taken at face value as the
  uncertainty of the reported number, and is *not* divided again.

What comes out is a percentile interval over the propagated draws plus the variance
components behind it, so a caller can also form a symmetric interval. No factor is fitted
to make the interval cover any known answer -- coverage against held-out ground truth is a
diagnostic (:func:`coverage_by_reaction`), and tuning to it would make it true by
construction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import truncnorm

# The isotopomers of one metabolite under one tracer are a composition: they sum to 1.
_MID_GROUP = ['Tracer Name', 'Metabolite ID']
_MID_POINT = _MID_GROUP + ['Isotopomer']


@dataclass(frozen=True)
class MidNoiseModel:
    """Replicate noise of a measured MID fraction as a function of its magnitude.

    ``sigma^2(p) = a + b * p * (1 - p)``. The ``p(1-p)`` term is the multinomial shape a
    composition of fractions carries by construction; ``a`` is the magnitude-independent
    floor the instrument still has as ``p -> 0``. A single scalar sigma applied to every
    measurement is the special case ``b = 0``, which the replicates reject outright.
    """

    a: float
    b: float

    def sigma(self, fraction):
        """Standard deviation of one replicate measurement at these fractions."""
        p = np.clip(np.asarray(fraction, dtype=float), 0.0, 1.0)
        return np.sqrt(self.a + self.b * p * (1.0 - p))


def _replicate_stats(frames):
    """Per-MID-point replicate mean and variance, pooled over many measurement frames.

    Each frame must hold one (condition, growth condition): the same
    (tracer, metabolite, isotopomer) has a different true value in each, so they are kept
    apart by frame and only the replicate axis within a frame is treated as repeat
    measurement.

    Args:
        frames: iterable of long-format df_L frames, one per (condition, growth condition).
    """
    parts = []
    for i, df in enumerate(frames):
        part = df[_MID_POINT + ['Fraction']].copy()
        part['_frame'] = i
        parts.append(part)

    grouped = pd.concat(parts, ignore_index=True).groupby(['_frame'] + _MID_POINT)['Fraction']
    stats = pd.DataFrame({
        'mean': grouped.mean(),
        'var':  grouped.var(ddof=1),
        'n':    grouped.size(),
    })
    return stats[(stats['n'] >= 2) & stats['var'].notna()]


def fit_mid_noise_model(frames, n_bins: int = 12) -> MidNoiseModel:
    """Fit :class:`MidNoiseModel` to the replicate spread pooled across every MID point.

    A point measured in 3 replicates yields a variance estimate with 2 degrees of freedom,
    whose own relative error is ~50% -- far too noisy to use per point, which is why this
    returns a pooled curve rather than a per-point sigma. Binning by magnitude and
    averaging the *squared* replicate deviation gives an unbiased variance per bin; the
    two-parameter curve is then fitted to those bins, weighted by how many points each
    holds. Coefficients are constrained non-negative because a variance component cannot
    be negative.

    Args:
        frames: iterable of long-format df_L frames, one per (condition, growth condition).
        n_bins: magnitude bins the fit is computed over.
    """
    stats = _replicate_stats(frames)
    if stats.empty:
        raise ValueError("no MID point has replicates to estimate a noise model from")

    binned = stats.assign(_bin=pd.qcut(stats['mean'], n_bins, duplicates='drop'))
    agg = binned.groupby('_bin', observed=True).agg(
        p=('mean', 'mean'), var=('var', 'mean'), n=('var', 'size'))

    p, w = agg['p'].to_numpy(), np.sqrt(agg['n'].to_numpy())
    design = np.column_stack([np.ones_like(p), p * (1.0 - p)])
    coef, _ = nnls(design * w[:, None], agg['var'].to_numpy() * w)
    return MidNoiseModel(a=float(coef[0]), b=float(coef[1]))


def _renormalize(draws: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Rescale each compositional group of every draw back to sum 1.

    A diagonal covariance ignores that the isotopomers of one metabolite must sum to 1, so
    an independent draw per isotopomer leaves the composition off-constraint. Rescaling the
    group projects that violation out and recovers most of the negative off-diagonal
    covariance the constraint implies. Fractions are clipped at zero first, since a
    negative abundance is unphysical; that clip biases near-zero points marginally upward,
    which is the accepted cost of a diagonal approximation.
    """
    out = np.clip(draws, 0.0, None)
    n_groups = int(codes.max()) + 1
    for i in range(out.shape[0]):
        totals = np.bincount(codes, weights=out[i], minlength=n_groups)
        scale = np.where(totals > 0, totals, 1.0)[codes]
        out[i] /= scale
    return out


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


def sample_inputs(df_L, df_Vm, df_err, noise: MidNoiseModel, n_draws: int,
                  rng: np.random.Generator):
    """Monte Carlo draws of the measured inputs, plus the unperturbed maximum-likelihood row.

    Returns ``(df_L_draws, df_Vm_draws)`` carrying samples ``0 .. n_draws``, where sample
    ``n_draws`` holds the replicate mean and the measured flux values unperturbed. That row
    is the plug-in MLE input: by invariance of the maximum-likelihood estimate under
    transformation, ``f(mean)`` is the estimate, whereas averaging ``f`` over the individual
    replicates is a different, Jensen-biased quantity.

    Args:
        df_L: long-format labeling data, one row per (bio-rep, tracer, metabolite, isotopomer).
        df_Vm: measured-flux data for this growth condition.
        df_err: reported uncertainty per measured flux, or None.
        noise: fitted MID noise model.
        n_draws: number of perturbed draws.
        rng: random generator used for the perturbation.
    """
    n_reps = df_L['Sample'].nunique()
    mu = (df_L.groupby(_MID_POINT, sort=False)['Fraction'].mean().reset_index())

    # The model is fed the replicate mean, so its uncertainty is the standard error.
    sigma = noise.sigma(mu['Fraction'].to_numpy()) / np.sqrt(n_reps)
    centre = mu['Fraction'].to_numpy()
    perturbed = centre + sigma * rng.standard_normal((n_draws, centre.size))
    codes = pd.factorize(pd.MultiIndex.from_frame(mu[_MID_GROUP]))[0]
    fractions = np.vstack([_renormalize(perturbed, codes), centre[None, :]])

    template = mu.drop(columns='Fraction')
    df_L_draws = pd.concat(
        [template.assign(Sample=s, Fraction=fractions[s]) for s in range(n_draws + 1)],
        ignore_index=True)

    df_Vm_draws = _flux_draws(df_Vm, df_err, n_draws, rng)
    return df_L_draws, df_Vm_draws


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
    than averaged over perturbed ones. The interval is the percentile interval of the
    propagated draws, which respects the skew a flux near zero has and which a symmetric
    band would misreport.

    The two variance components are reported alongside, each isolated by holding the other
    axis fixed, so they can be compared without assuming they add -- the map is nonlinear,
    so ``sigma_total^2 = sigma_input^2 + sigma_model^2`` need not hold and is not imposed.

    Args:
        raw: long predictions with Sample, Member, Value and the group columns.
        n_draws: number of perturbed draws; sample ``n_draws`` is the unperturbed row.
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
        # Epistemic: member disagreement at the maximum-likelihood input.
        'sigma_model': at_mle.std(ddof=1).fillna(0.0),
    })

    consensus = (drawn.groupby(keys + ['Sample'], sort=False)['Value']
                      .median().reset_index())
    # Measurement propagation: how far the ensemble consensus moves with the input.
    out['sigma_input'] = (consensus.groupby(keys, sort=False)['Value']
                                   .std(ddof=1).fillna(0.0))

    spread = drawn.groupby(keys, sort=False)['Value']
    out['sigma_total'] = spread.std(ddof=1).fillna(0.0)
    out['Value_lo'] = spread.quantile(alpha / 2.0)
    out['Value_hi'] = spread.quantile(1.0 - alpha / 2.0)
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
