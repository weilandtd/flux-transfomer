"""
Prepare and sample an MFA flux problem for SigmaFlux AI data generation.

Prepares a COBRA sampling problem from an annotated model, writes sampled
fluxes, and saves the exact model that those fluxes belong to.

Example:
    sigmaflux-sample-flux --config paper/configs/i_orientalis.yaml
"""

import argparse
import datetime
import math
import os
import warnings
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import yaml
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.json import load_json_model
from cobra.sampling import sample

from ..core.utils import add_dilution_inputs, media_dilution_targets
from .reusable_sampler import ReusableSampler

try:
    from cobra.sampling import HopsySampler
except Exception:
    # Not just ImportError: cobra's own hopsy module uses `hopsy.Proposal` as a default
    # argument in a class body, so importing it without the hopsy package raises NameError
    # rather than ImportError. Degrade to "unavailable" instead of taking the process down.
    HopsySampler = None
from optlang.symbolics import Zero

from sigmaflux.pipeline._paths import (
    add_root_argument,
    default_output_dir,
    load_config,
)
from sigmaflux.pipeline._paths import resolve_path as _resolve_path

# Mapping from YAML-friendly names to hopsy proposal class names.
_HOPSY_PROPOSALS = {
    "uniform_coordinate_hit_and_run": "UniformCoordinateHitAndRunProposal",  # default
    "uniform_hit_and_run": "UniformHitAndRunProposal",
    "gaussian_coordinate_hit_and_run": "GaussianCoordinateHitAndRunProposal",
    "gaussian_hit_and_run": "GaussianHitAndRunProposal",
    "billiard_walk": "BilliardWalkProposal",
    "billiard_adaptive_metropolis": "BilliardAdaptiveMetropolisProposal",
    "adaptive_metropolis": "AdaptiveMetropolisProposal",
    "dikin_walk": "DikinWalkProposal",
    "ball_walk": "BallWalkProposal",
}


def resolve_path(path, root=None):
    """Resolve a config path against the config's project root.

    Args:
        path: path from the config, absolute or root-relative.
        root: project root; ``path`` is returned unchanged if omitted.
    """
    if os.path.isabs(path) or root is None:
        return path
    return _resolve_path(path, root)


def output_dir_for(cfg, output_dir_override=None):
    return default_output_dir(cfg, output_dir_override)


def set_bounds(model, reaction_id, lower_bound, upper_bound):
    if reaction_id in model.reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        reaction.bounds = (lower_bound, upper_bound)


def apply_media(model, media_cfg, close_unlisted=True):
    """Constrain the model's boundary fluxes.

    With ``close_unlisted`` (the default) every ``EX_`` reaction is closed first and
    only those named in *media_cfg* are reopened -- correct for a defined medium, but
    wrong for a whole-body/multi-tissue model where ``EX_`` is internal physiology;
    set ``close_unlisted: false`` there so unlisted exchanges keep their model bounds.

    Args:
        model: model whose exchange reactions are constrained.
        media_cfg: mapping of reaction id to ``{lb, ub}`` bounds.
        close_unlisted: close every ``EX_`` reaction not named in ``media_cfg``.
    """
    if close_unlisted:
        for reaction in model.reactions:
            if reaction.id.startswith("EX_"):
                reaction.bounds = (0, 0)

    for rxn_id, bounds in media_cfg.items():
        set_bounds(model, rxn_id, bounds["lb"], bounds["ub"])

    missing = [r for r in media_cfg if r not in model.reactions]
    if missing:
        print(f"WARNING: {len(missing)} media reactions not in the model: {missing[:5]}")

    if close_unlisted:
        print(f"Applied media: {len(media_cfg)} open exchange reactions, all others closed")
    else:
        print(f"Applied media: constrained {len(media_cfg) - len(missing)} measured "
              f"reactions; other exchanges left at their model bounds")


def apply_reaction_bounds(model, bounds_cfg):
    """Override bounds on named non-exchange reactions, before FVA.

    ``media`` only reaches ``EX_`` reactions, but a GEM's internal reaction
    directions (e.g. transporter directionality) can be wrong for the organism
    being modelled.

    Args:
        model: model whose reactions are overridden.
        bounds_cfg: mapping of reaction id to ``{lb, ub}`` bounds; skipped if falsy.
    """
    if not bounds_cfg:
        return
    for reaction_id, bounds in bounds_cfg.items():
        if reaction_id not in model.reactions:
            print(f"  reaction_bounds: '{reaction_id}' not in model, skipped")
            continue
        set_bounds(model, reaction_id, bounds["lb"], bounds["ub"])
    print(f"Applied reaction_bounds overrides: {sorted(bounds_cfg)}")


def constrain_growth(model, growth_cfg):
    set_bounds(model, "BIOMASS", growth_cfg["lb"], growth_cfg["ub"])


def can_consume(reaction, metabolite):
    """Whether *reaction* can consume *metabolite* in some feasible direction.

    Determined by the reaction's bounds, not which side the metabolite sits on: a
    reversible reaction listing it as a product still consumes it running backwards.

    Args:
        reaction: reaction to check.
        metabolite: metabolite the reaction may or may not be able to consume.
    """
    coef = reaction.metabolites[metabolite]
    return (coef < 0 and reaction.upper_bound > 0) or (coef > 0 and reaction.lower_bound < 0)


def drop_skip_constraints(model, skip_constraints):
    """Remove the listed reactions, draining any pool left with no consumer.

    Removing a reaction (e.g. BIOMASS) can strand metabolites it used to consume,
    forcing their producers to zero flux and making sampling infeasible. Stranded
    pools are found via ``can_consume`` (bounds, not stoichiometric sign, since a
    reversible reaction can consume a listed product on reverse flux) and given an
    explicit demand reaction rather than being deleted, which keeps mass balance and
    the atom mapping usable instead of leaving an invisible unbounded sink.

    Args:
        model: model the listed reactions are removed from.
        skip_constraints: reaction ids to remove.
    """
    to_remove = [model.reactions.get_by_id(r) for r in skip_constraints if r in model.reactions]
    model.remove_reactions(to_remove, remove_orphans=True)

    stranded = [m for m in model.metabolites
                if m.reactions and not any(can_consume(r, m) for r in m.reactions)]
    demands = [model.add_boundary(m, type="demand") for m in stranded]

    print(f"Dropped {len(to_remove)} skip-constraints reaction(s): {[r.id for r in to_remove]}")
    if demands:
        print(f"  Added {len(demands)} demand reaction(s) for pools left without a "
              f"consumer: {[r.id for r in demands]}")


def constrain_by_fva(model):
    fva = flux_variability_analysis(model, fraction_of_optimum=0)
    epsilon = 1e-6
    zero_reactions = []
    fixed_count = 0

    for reaction_id, row in fva.iterrows():
        reaction = model.reactions.get_by_id(reaction_id)
        if row["minimum"] > -epsilon and row["maximum"] < epsilon:
            zero_reactions.append(reaction)
        else:
            new_lb = max(reaction.lower_bound, row["minimum"] - epsilon)
            new_ub = min(reaction.upper_bound, row["maximum"] + epsilon)
            reaction.bounds = (new_lb, new_ub)
            if abs(row["maximum"] - row["minimum"]) < 1e-9:
                fixed_count += 1

    model.remove_reactions(zero_reactions)
    orphaned_metabolites = [metabolite for metabolite in model.metabolites if len(metabolite.reactions) == 0]
    model.remove_metabolites(orphaned_metabolites)
    print(f"  Fixed (non-zero, min≈max) reactions kept: {fixed_count}")

    return len(zero_reactions), len(orphaned_metabolites), fva


def check_sampling_coverage(samples, fva, key_reactions=None):
    common = [r for r in samples.columns if r in fva.index]
    obs_min = samples[common].min(axis=0)
    obs_max = samples[common].max(axis=0)
    fva_range = fva.loc[common, "maximum"] - fva.loc[common, "minimum"]
    variable_mask = fva_range > 1e-6
    coverage = (obs_max - obs_min)[variable_mask] / fva_range[variable_mask]

    print("\n--- Bound Coverage Analysis ---")
    print(f"Reactions compared: {len(common)} (FVA variable: {variable_mask.sum()})")
    print(f"Coverage (obs range / FVA range):")
    print(f"  Median : {coverage.median():.3f}")
    print(f"  Mean   : {coverage.mean():.3f}")
    print(f"  > 90%  : {(coverage > 0.9).sum()} reactions")
    print(f"  < 10%  : {(coverage < 0.1).sum()} reactions")

    poor = coverage[coverage < 0.1].sort_values()
    if len(poor) > 0:
        print("\nPoorly covered reactions (coverage < 10%):")
        fva_lo = fva.loc[poor.index, "minimum"]
        fva_hi = fva.loc[poor.index, "maximum"]
        for rxn_id in poor.index:
            print(f"  {rxn_id:<40s}  cov={poor[rxn_id]:.3f}  FVA=[{fva_lo[rxn_id]:.4f}, {fva_hi[rxn_id]:.4f}]  obs=[{obs_min[rxn_id]:.4f}, {obs_max[rxn_id]:.4f}]")

    if key_reactions:
        print("\nKey reaction stats:")
        for rxn_id in key_reactions:
            if rxn_id not in samples.columns:
                print(f"  {rxn_id}: not in samples (pruned as zero-flux)")
                continue
            obs_lo, obs_hi = obs_min[rxn_id], obs_max[rxn_id]
            obs_std = samples[rxn_id].std()
            if rxn_id in fva.index:
                fva_lo = fva.loc[rxn_id, "minimum"]
                fva_hi = fva.loc[rxn_id, "maximum"]
                fva_rng = fva_hi - fva_lo
                cov = (obs_hi - obs_lo) / fva_rng if fva_rng > 1e-6 else float("nan")
                print(f"  {rxn_id}:")
                print(f"    FVA bounds : [{fva_lo:.4f}, {fva_hi:.4f}]  width={fva_rng:.4f}")
                print(f"    Obs range  : [{obs_lo:.4f}, {obs_hi:.4f}]  width={obs_hi - obs_lo:.4f}")
                print(f"    Obs std    : {obs_std:.4f}")
                print(f"    Coverage   : {cov:.3f}")


def diagnose_pool_coverage(samples, fva, reactions, n_bins, label="Pool"):
    """
    Per-reaction bin occupancy diagnostic. Reports observed FVA coverage,
    bin fill rate, and boundary bin counts (first/last bin) for each reaction.

    Args:
        samples: sampled flux points, columns are reactions.
        fva: per-reaction FVA min/max, used as the coverage range.
        reactions: reaction ids to report on.
        n_bins: number of bins per reaction.
        label: label prefixing the printed report.
    """
    present = [r for r in reactions if r in samples.columns and r in fva.index]
    if not present:
        print(f"[{label}] WARNING: no measured reactions found in samples/fva")
        return

    lo = fva.loc[present, "minimum"].values
    hi = fva.loc[present, "maximum"].values
    width = np.where(hi - lo > 1e-9, hi - lo, 1.0)

    vals = samples[present].to_numpy()
    obs_cov = np.clip((vals.max(axis=0) - vals.min(axis=0)) / width, 0, 1)

    norm = np.clip((vals - lo) / width, 0, 1)
    bin_asgn = (norm * n_bins).astype(int).clip(0, n_bins - 1)

    print(f"\n--- {label}: coverage diagnostic ({len(samples)} samples, {n_bins} bins/rxn) ---")
    print(f"  {'Reaction':<30s}  {'ObsCov':>7s}  {'Filled':>8s}  {'Empty':>6s}  {'MinCnt':>7s}  {'MaxCnt':>7s}")
    for i, rxn in enumerate(present):
        counts = np.bincount(bin_asgn[:, i], minlength=n_bins)
        n_occ = (counts > 0).sum()
        print(f"  {rxn:<30s}  {obs_cov[i]:>7.3f}  {n_occ:>4d}/{n_bins:<3d}  {n_bins - n_occ:>6d}  {counts[0]:>7d}  {counts[-1]:>7d}")


def _ipf_weights(bin_asgn, n_bins, n_iter=30):
    """
    Iterative Proportional Fitting: weights w_i such that for every reaction f
    and bin b the weighted fraction equals 1/n_bins (uniform marginal).

    Args:
        bin_asgn: per-sample, per-reaction bin index array.
        n_bins: number of bins per reaction.
        n_iter: number of IPF sweeps.
    """
    n = len(bin_asgn)
    w = np.ones(n, dtype=np.float64) / n
    target = 1.0 / n_bins
    for _ in range(n_iter):
        for f in range(bin_asgn.shape[1]):
            bins_f = bin_asgn[:, f]
            bin_sums = np.bincount(bins_f, weights=w, minlength=n_bins)
            scale = np.where(bin_sums[bins_f] > 1e-15, target / bin_sums[bins_f], 0.0)
            w *= scale
        w /= w.sum()
    return w


def draw_quasi_uniform(samples, fva, n_final, n_bins, reactions):
    """
    IPF-weighted draw from `samples` targeting uniform marginals for each reaction.
    Returns a DataFrame of n_final rows drawn without replacement (with replacement
    only if the pool is smaller than n_final).

    Args:
        samples: pool of sampled flux points to draw from.
        fva: per-reaction FVA min/max, used as the binning range.
        n_final: number of rows to draw.
        n_bins: number of bins per reaction.
        reactions: reaction ids the marginals are targeted on.
    """
    present = [r for r in reactions if r in samples.columns and r in fva.index]
    if not present:
        return samples.sample(min(n_final, len(samples)), random_state=0).reset_index(drop=True)

    lo = fva.loc[present, "minimum"].values
    hi = fva.loc[present, "maximum"].values
    width = np.where(hi - lo > 1e-9, hi - lo, 1.0)

    samples_np = samples.to_numpy()
    present_cols = np.array([samples.columns.get_loc(r) for r in present])
    norm = np.clip((samples_np[:, present_cols] - lo) / width, 0, 1)
    bin_asgn = (norm * n_bins).astype(int).clip(0, n_bins - 1)

    weights = _ipf_weights(bin_asgn, n_bins)
    rng = np.random.default_rng(0)
    replace = len(samples_np) < n_final
    chosen = rng.choice(len(samples_np), n_final, replace=replace, p=weights)
    result_np = samples_np[chosen]

    result_asgn = (np.clip((result_np[:, present_cols] - lo) / width, 0, 1) * n_bins).astype(int).clip(0, n_bins - 1)
    print(f"\n--- Quasi-uniform draw ({len(present)} reactions × {n_bins} bins, n={n_final}) ---")
    print(f"  {'Reaction':<30s}  {'Pool':>10s}  {'Final':>10s}  {'Uniformity':>10s}")
    for f_idx, rxn in enumerate(present):
        pool_cov = (np.bincount(bin_asgn[:, f_idx], minlength=n_bins) > 0).sum()
        final_cnt = np.bincount(result_asgn[:, f_idx], minlength=n_bins).astype(float)
        final_cov = (final_cnt > 0).sum()
        p = final_cnt[final_cnt > 0] / final_cnt.sum()
        uniformity = (-np.sum(p * np.log2(p))) / np.log2(final_cov) if final_cov > 1 else 0.0
        print(f"  {rxn:<30s}  {pool_cov:>4d}/{n_bins}     {final_cov:>4d}/{n_bins}     {uniformity:>10.3f}")

    return pd.DataFrame(result_np, columns=samples.columns)


def summarize_variation(samples):
    stats = pd.DataFrame({
        "min": samples.min(axis=0),
        "max": samples.max(axis=0),
        "std": samples.std(axis=0),
    }).sort_values("std", ascending=False)
    print(f"Variable reactions > 1e-8: {(stats['std'] > 1e-8).sum()} / {samples.shape[1]}")
    print("Flux summary (sorted by std):")
    print(stats.to_string(float_format=lambda x: f"{x:10.4f}"))


def screen_nonessential(model):
    """Return IDs of reactions whose knockout leaves the prepared model feasible."""
    nonessential = []
    essential = []
    for rxn in model.reactions:
        with model:
            rxn.bounds = (0, 0)
            if model.optimize().status == "optimal":
                nonessential.append(rxn.id)
            else:
                essential.append(rxn.id)
    print(f"Essentiality screen: {len(nonessential)} non-essential, {len(essential)} essential / {len(model.reactions)} total reactions")
    print("Essential reactions:")
    for rxn_id in essential:
        print(f"  {rxn_id}")
    return nonessential


_KO_STRATEGIES = ("all", "random", "uncorrelated")


def _normalize_ko_strategy(ko_value):
    """Normalize `sampling.ko` to one of `_KO_STRATEGIES`, or None if disabled.

    Accepts the legacy ``ko: true`` (every non-essential reaction -- now named
    "all") alongside the ``false | all | random | uncorrelated`` string enum.
    """
    if ko_value is False or ko_value is None:
        return None
    if ko_value is True:
        return "all"
    if ko_value in _KO_STRATEGIES:
        return ko_value
    raise ValueError(f"Unknown sampling.ko value {ko_value!r}; expected one of "
                      f"False, True, or {_KO_STRATEGIES}")


def greedy_pivoted_cholesky_select(C, k, tol=1e-10):
    """Greedy pivoted Cholesky: pick up to *k* indices of *C* whose directions carry
    the most variance unexplained by picks already made -- the least mutually
    correlated set. No swaps (a pick is never revisited) and no tuning (runs exactly
    *k* steps, fewer only once remaining residual variance is numerically exhausted).

    Args:
        C: symmetric PSD matrix (e.g. a correlation matrix), shape (n, n).
        k: number of indices to select; capped at n.
        tol: stop once the largest remaining residual diagonal drops below this.
    """
    n = C.shape[0]
    k = min(k, n)
    d = np.diag(C).astype(float).copy()
    L = np.zeros((n, k))
    selected = []
    for t in range(k):
        piv = int(np.argmax(d))
        if d[piv] <= tol:
            break
        L[:, t] = (C[:, piv] - L[:, :t] @ L[piv, :t]) / np.sqrt(d[piv])
        d -= L[:, t] ** 2
        d[piv] = -np.inf  # never reselect; also guards against float noise
        selected.append(piv)
    return selected


def _select_uncorrelated_ko(model, candidates, n_ko, seed):
    """Pick the *n_ko* of *candidates* whose pFBA solutions move least like each other.

    One pFBA solution per candidate KO plus a wild-type baseline; each KO's change
    vector is (KO pFBA - WT pFBA) over all reactions. Correlates the change vectors
    and greedily selects the least mutually correlated set via
    `greedy_pivoted_cholesky_select`.

    Args:
        model: prepared model (post FVA-pruning/skip_constraints); its objective is
            already Zero at the point `main()` calls this, so pfba() finds the
            minimal-total-flux point of the existing feasible region (which already
            encodes media/growth), not a growth-optimizing solution.
        candidates: reaction ids to rank (from `screen_nonessential`).
        n_ko: number of reactions to select.
        seed: used only for the degenerate all-uninformative fallback.
    """
    from cobra.flux_analysis import pfba

    rxn_ids = [r.id for r in model.reactions]
    wt = pfba(model).fluxes.reindex(rxn_ids).to_numpy()

    kept, deltas = [], []
    for rxn_id in candidates:
        rxn = model.reactions.get_by_id(rxn_id)
        with model:
            rxn.bounds = (0, 0)
            try:
                ko_fluxes = pfba(model).fluxes.reindex(rxn_ids).to_numpy()
            except Exception:
                continue  # pFBA can fail even where the FBA feasibility screen passed
        kept.append(rxn_id)
        deltas.append(ko_fluxes - wt)

    if not kept:
        print("  [uncorrelated] no candidate KO had a successful pFBA solve")
        return []

    deltas = np.asarray(deltas)
    informative = deltas.std(axis=1) > 1e-12
    if not informative.any():
        print("  [uncorrelated] every candidate's pFBA solution is unchanged by its "
              "own KO -- falling back to a random n_ko selection")
        rng = np.random.default_rng(seed)
        k = min(n_ko, len(kept))
        return rng.choice(kept, size=k, replace=False).tolist()

    kept = [r for r, keep in zip(kept, informative) if keep]
    C = np.corrcoef(deltas[informative])
    chosen = [kept[i] for i in greedy_pivoted_cholesky_select(C, n_ko)]
    print(f"  [uncorrelated] {len(chosen)}/{len(kept)} least-correlated KO reactions "
          f"selected (target n_ko={n_ko})")
    return chosen


def select_ko_reactions(model, strategy, n_ko, seed):
    """Choose which reactions to knock out, per *strategy* (`_normalize_ko_strategy`
    output). "all": every non-essential reaction. "random": a size-`n_ko` random
    subset. "uncorrelated": the `n_ko` non-essential reactions whose pFBA change
    vectors are least mutually correlated (`_select_uncorrelated_ko`). If `n_ko` is
    at least the number of non-essential candidates, both `random` and
    `uncorrelated` degrade to `all` (nothing to sub-select).

    Args:
        model: prepared model to screen for essentiality and knock out from.
        strategy: one of `_KO_STRATEGIES`.
        n_ko: number of reactions to select; required for "random"/"uncorrelated",
            ignored for "all".
        seed: base seed; "random" draws from it directly, "uncorrelated" only uses
            it for its degenerate fallback.
    """
    nonessential = screen_nonessential(model)
    if strategy == "all":
        return nonessential
    if not n_ko:
        raise ValueError(f"sampling.n_ko is required when sampling.ko is {strategy!r}")
    if n_ko >= len(nonessential):
        print(f"  [{strategy}] n_ko={n_ko} >= {len(nonessential)} non-essential "
              f"reactions -- using all of them")
        return nonessential
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(nonessential, size=n_ko, replace=False).tolist()
    return _select_uncorrelated_ko(model, nonessential, n_ko, seed)


def _do_sample_n(model, s, n_samples, seed):
    return _do_sample(model, dict(s, n_samples=n_samples), seed)


def stratified_sample(sampler, s, fva, reactions, n_bins, n_subsamples, seed):
    """
    For each variable measured reaction, run one constrained MCMC chain per bin.
    Returns a DataFrame containing all strata concatenated.

    Args:
        sampler: ReusableSampler to draw from.
        s: sampling config (method, thinning, processes, ...).
        fva: per-reaction FVA min/max, used to define the bin bounds.
        reactions: reaction ids to stratify on.
        n_bins: number of bins per reaction.
        n_subsamples: samples drawn per stratum.
        seed: base seed; offset per bin.
    """
    model = sampler.model
    all_strata = []
    for rxn_id in reactions:
        if rxn_id not in model.reactions or rxn_id not in fva.index:
            continue
        lo = fva.loc[rxn_id, "minimum"]
        hi = fva.loc[rxn_id, "maximum"]
        if hi - lo < 1e-9:
            continue
        bin_width = (hi - lo) / n_bins
        for b in range(n_bins):
            bin_lo = lo + b * bin_width
            bin_hi = lo + (b + 1) * bin_width
            rxn = model.reactions.get_by_id(rxn_id)
            # The bin window is cut from *this reaction's own* WT FVA range, so it
            # normally sits inside [rxn.lower_bound, rxn.upper_bound] by construction.
            # That stops holding when the caller has separately forced rxn's bounds to
            # (0, 0) -- e.g. sampling.ko knocking this exact reaction out while it is
            # also one of the strata reactions -- and the two windows may not overlap
            # at all, which cobra's bounds setter raises on rather than reports as
            # infeasible. Treat "no overlap" the same as "infeasible": skip the stratum.
            new_lo, new_hi = max(rxn.lower_bound, bin_lo), min(rxn.upper_bound, bin_hi)
            if new_lo > new_hi:
                print(f"  [stratify] skip {rxn_id} bin {b + 1}/{n_bins} — outside current "
                      f"bounds [{rxn.lower_bound:g}, {rxn.upper_bound:g}] "
                      f"(e.g. this reaction is under an active knockout)")
                continue
            with sampler.temporary_bounds({rxn_id: (new_lo, new_hi)}):
                if model.optimize().status != "optimal":
                    print(f"  [stratify] skip {rxn_id} bin {b + 1}/{n_bins} — infeasible")
                    continue
                all_strata.append(_do_sample_n(sampler, s, n_subsamples, seed + b))
        print(f"  [stratify] {rxn_id}: {n_bins} strata × {n_subsamples} samples")
    # Every stratum can be skipped -- e.g. all of `reactions` were knocked out in this
    # call -- and pd.concat([]) raises rather than giving an empty pool; an empty pool
    # is correct here (the caller's pd.concat over KO iterations tolerates it).
    if not all_strata:
        return pd.DataFrame(columns=[r.id for r in model.reactions])
    return pd.concat(all_strata, ignore_index=True)


def _do_sample(model_or_sampler, s, seed):
    # optgp, not hopsy: HopsySampler ships only on the cobrapy devel branch, so a released
    # cobra cannot honour a hopsy default. Every config in paper/configs sets this explicitly.
    method = s.get("method", "optgp")
    if method == "hopsy":
        model = (model_or_sampler.model if isinstance(model_or_sampler, ReusableSampler)
                  else model_or_sampler)
        if HopsySampler is None:
            warnings.warn(
                "method: hopsy was requested but HopsySampler is unavailable — SAMPLING WITH "
                "optgp INSTEAD. The samples will not be the ones the config asked for. To use "
                "hopsy, install the hopsy package and a cobrapy checkout from the devel branch "
                "(HopsySampler is not in any released cobra).",
                RuntimeWarning,
                stacklevel=2,
            )
            method = "optgp"
            return sample(model, n=s["n_samples"], thinning=s["thinning"],
                          processes=s["processes"], method=method, seed=seed)
        import hopsy as _hopsy
        proposal_key = s.get("hopsy_proposal", "uniform_coordinate_hit_and_run")
        if proposal_key not in _HOPSY_PROPOSALS:
            raise ValueError(f"Unknown hopsy_proposal '{proposal_key}'; valid: {list(_HOPSY_PROPOSALS)}")
        proposal_cls = getattr(_hopsy, _HOPSY_PROPOSALS[proposal_key])
        sol = model.optimize()
        assert sol.status == "optimal", f"Model infeasible before sampling: {sol.status}"
        hopsy_sampler = HopsySampler(
            model,
            sampler=proposal_cls,
            thinning=s["thinning"],
            processes=s["processes"],
            nproj=s.get("nproj"),
            seed=seed,
            rounding=s.get("rounding", True),
        )
        return hopsy_sampler.sample(s["n_samples"])
    if isinstance(model_or_sampler, ReusableSampler):
        return model_or_sampler.sample(s["n_samples"], seed=seed)
    return sample(
        model_or_sampler,
        n=s["n_samples"],
        thinning=s["thinning"],
        processes=s["processes"],
        method=method,
        seed=seed,
    )


def main(cfg, output_dir_override=None):
    s = cfg["sampling"]
    ko_strategy = _normalize_ko_strategy(s.get("ko", False))
    output_dir = output_dir_for(cfg, output_dir_override)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {cfg['model_file']}")
    model = load_json_model(resolve_path(cfg["model_file"], cfg.get("_root")))
    model.solver.configuration.tolerances.feasibility = 1e-9
    # TEMPORARY, for a real glpk-vs-gurobi A/B run on e_coli genome-scale -- not a
    # supported config surface, remove after the comparison is decided either way.
    _solver_override = os.environ.get("SIGMAFLUX_SOLVER_OVERRIDE")
    if _solver_override:
        model.solver = _solver_override
        if _solver_override == "gurobi":
            model.solver.configuration.threads = 1
        print(f"  [SIGMAFLUX_SOLVER_OVERRIDE] solver forced to {_solver_override!r}")

    apply_media(model, s["media"], close_unlisted=s.get("close_unlisted", True))
    # Before dilution-target detection: a reversible-transporter override (e.g.
    # ALAt_c_e lb 0 -> -20) only makes its metabolite internally-producible once
    # applied. Detecting on the pre-override bounds silently skipped dilution for
    # any medium component whose only internal route was such an override.
    apply_reaction_bounds(model, s.get("reaction_bounds"))

    # Medium components the cell also produces are mixing nodes whose exchange
    # carries no label, so they cannot be diluted by the medium. Give each one an
    # explicit unlabeled input. Done here rather than in the model files because
    # the medium is a property of the experiment, not of the organism.
    diluted = add_dilution_inputs(model, media_dilution_targets(model, s["media"]))
    if diluted:
        print(f"  Added {len(diluted)} unlabeled media dilution input(s): {diluted}")
    if s.get("growth"):
        constrain_growth(model, s["growth"])

    solution = model.optimize()
    if solution.status != "optimal":
        raise RuntimeError(f"Prepared model is not feasible: {solution.status}")
    print(f"Prepared model status: {solution.status}, objective={solution.objective_value}")

    model.objective = Zero

    n_zero, n_orphan, fva_result = constrain_by_fva(model)
    print(f"Removed {n_zero} zero-flux reactions and {n_orphan} orphaned metabolites; tightened remaining bounds with ±1e-6")

    skip_constraints = cfg.get("skip_constraints", [])
    if skip_constraints:
        drop_skip_constraints(model, skip_constraints)

    measured_rxns = cfg.get("training", {}).get("measured_fluxes", [])
    sub_rxns = s.get("stratified_sampling", {}).get("reactions") or measured_rxns
    if sub_rxns:
        print("\nModel bounds seen by sampler (after FVA tightening):")
        for rxn_id in sub_rxns:
            if rxn_id in model.reactions:
                r = model.reactions.get_by_id(rxn_id)
                fva_lo = fva_result.loc[rxn_id, "minimum"] if rxn_id in fva_result.index else float("nan")
                fva_hi = fva_result.loc[rxn_id, "maximum"] if rxn_id in fva_result.index else float("nan")
                match = abs(r.lower_bound - fva_lo) < 2e-6 and abs(r.upper_bound - fva_hi) < 2e-6
                flag = "" if match else "  *** MISMATCH ***"
                print(f"  {rxn_id:<30s}  bounds=[{r.lower_bound:8.4f}, {r.upper_bound:8.4f}]  FVA=[{fva_lo:8.4f}, {fva_hi:8.4f}]{flag}")
            else:
                print(f"  {rxn_id:<30s}  REMOVED (zero-flux by FVA)")

    prepared_model_path = os.path.join(output_dir, "model_prepared_for_mfa.json")
    cobra.io.save_json_model(model, prepared_model_path)
    print(f"Saved prepared model: {prepared_model_path}")

    if ko_strategy and s.get("growth", {}).get("lb") == 0:
        print("WARNING: ko is enabled but growth lb is 0 — knocked-out models may be trivially feasible at v=0, producing uninformative zero-flux samples")

    strat_cfg = s.get("stratified_sampling", {})
    n_final = s["n_samples"]

    # The sampler is built once here and reused for every KO/stratum below -- only the
    # bounds of the affected reaction(s) change from this point on, via
    # sampler.temporary_bounds(), instead of rebuilding a sampler per bounds change. hopsy
    # is devel-only and unavailable in a released cobra (see _do_sample), so it never
    # actually reaches sampler.sample(); "optgp" here just gives it working warmup/problem
    # bookkeeping to sample from.
    method = s.get("method", "optgp")
    sampler = ReusableSampler(model, method=("optgp" if method == "hopsy" else method),
                              thinning=s["thinning"], processes=s["processes"], seed=cfg["seed"])

    if strat_cfg.get("enabled", False):
        n_bins = strat_cfg.get("n_bins", 5)
        # KO count decides the per-pass sample budget below, so it has to be known
        # before that budget is resolved -- select once here rather than inside the
        # loop that used to follow the WT call.
        ko_reactions = (select_ko_reactions(model, ko_strategy, s.get("n_ko"), cfg["seed"])
                        if ko_strategy else [])
        n_ko_passes = 1 + len(ko_reactions)  # WT + every KO
        n_subsamples_cfg = strat_cfg["n_subsamples"]
        if n_subsamples_cfg == "auto":
            # Each KO reruns the full stratified pass, and draw_quasi_uniform only
            # keeps n_final of whatever gets drawn -- a fixed n_subsamples multiplies
            # by n_ko_passes for no benefit (measured: r_toruloides drew ~6.08M raw
            # samples across 152 passes for a 10k final pool). Spread a ~n_final-sized
            # budget per reaction across its bins and every pass instead, so the total
            # pool stays about `len(sub_rxns) x n_final`, not `... x n_ko_passes`.
            n_subsamples = math.ceil(n_final / n_bins / n_ko_passes)
            print(f"  n_subsamples=auto -> {n_subsamples} (ceil({n_final}/{n_bins}/{n_ko_passes}))")
        else:
            n_subsamples = n_subsamples_cfg
        print(f"Stratified sampling: {len(sub_rxns)} reactions × {n_bins} bins × "
              f"{n_subsamples} samples/stratum × {n_ko_passes} passes "
              f"(WT + {len(ko_reactions)} KO, strategy={ko_strategy})...")
        all_strata = [stratified_sample(sampler, s, fva_result, sub_rxns, n_bins, n_subsamples, cfg["seed"])]
        for i, rxn_id in enumerate(ko_reactions):
            print(f"  KO [{i + 1}/{len(ko_reactions)}]: {rxn_id}")
            with sampler.temporary_bounds({rxn_id: (0, 0)}):
                all_strata.append(stratified_sample(sampler, s, fva_result, sub_rxns, n_bins, n_subsamples, cfg["seed"] + (i + 1) * 1000))
        samples = pd.concat(all_strata, ignore_index=True)
        print(f"Stratified pool: {len(samples)} samples")
        diagnose_pool_coverage(samples, fva_result, sub_rxns, n_bins, label="Stratified pool")
        samples = draw_quasi_uniform(samples, fva_result, n_final, n_bins, sub_rxns)
    else:
        print(f"Sampling {n_final} flux distributions (WT)...")
        all_samples = [_do_sample(sampler, s, cfg["seed"])]
        if ko_strategy:
            ko_reactions = select_ko_reactions(model, ko_strategy, s.get("n_ko"), cfg["seed"])
            for i, rxn_id in enumerate(ko_reactions):
                print(f"  KO [{i + 1}/{len(ko_reactions)}]: {rxn_id}  ({n_final} samples)")
                with sampler.temporary_bounds({rxn_id: (0, 0)}):
                    all_samples.append(_do_sample(sampler, s, cfg["seed"] + i + 1))
        samples = pd.concat(all_samples, ignore_index=True) if len(all_samples) > 1 else all_samples[0]

    samples_path = os.path.join(output_dir, "flux_samples.csv")
    samples.to_csv(samples_path)
    print(f"Saved flux samples: {samples_path} shape={samples.shape}")
    summarize_variation(samples)
    if fva_result is not None:
        check_sampling_coverage(samples, fva_result, key_reactions=["EX_glc__D_e"])


def build_parser():
    parser = argparse.ArgumentParser(description="Sample fluxes from an annotated MFA model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for generated outputs (overrides date-based default)")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    cfg, _ = load_config(args.config, args.root)
    return main(cfg, args.output_dir)


if __name__ == "__main__":
    cli()
