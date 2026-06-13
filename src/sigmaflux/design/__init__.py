"""
sigmaflux.design -- experimental-design / observability tools.

Identify which metabolite labelings are informative for *flux* estimation,
using only the forward simulator (the differentiable MFA model and its
generated samples) -- no estimator training required.

The entry point is :func:`sigmaflux.design.observability.rank_informative_metabolites`.
"""

from sigmaflux.design.observability import (
    free_flux_basis,
    candidate_measurements,
    compute_fisher_grams,
    empirical_sensitivity_grams,
    greedy_d_optimal,
    observable_spectrum,
    copula_mutual_information,
    ksg_nonlinearity_gap,
    rank_informative_metabolites,
    rank_from_samples,
)

__all__ = [
    "free_flux_basis",
    "candidate_measurements",
    "compute_fisher_grams",
    "empirical_sensitivity_grams",
    "greedy_d_optimal",
    "observable_spectrum",
    "copula_mutual_information",
    "ksg_nonlinearity_gap",
    "rank_informative_metabolites",
    "rank_from_samples",
]
