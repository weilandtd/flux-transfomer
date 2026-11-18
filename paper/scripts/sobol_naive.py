"""
Classical (Saltelli) Sobol sensitivity of every predicted flux to every labeling token, over a naive
population of inputs that carries no joint information.

sobol_sensitivity.py resamples tokens from the model's simulated samples, where the MIDs of
different metabolites are correlated: the model can read one fact from several tokens, and a token
swapped on its own contradicts its neighbours. Here every input is independent, so the Sobol
decomposition holds exactly and a token's index is how the model reads that token alone.

The population: the tokens of one metabolite on one tracer, plus the remainder of its MID the model
does not see (M+0 and any isotopologue left out as a token), follow a Dirichlet whose mean and
concentration are matched to that group in the model's simulated samples, drawn as independent
Gamma variables divided by their sum. Each token's Gamma variable is one input factor, so changing
it changes that token and rescales the rest of its MID in proportion, the least change the MID's
sum allows; the remainder's Gamma variable is a factor too (the metabolite's overall labeling).
Tokens constant across the simulated samples keep their value. The measured fluxes are held at their
median over the samples, so the labeling alone moves the prediction, as when an experiment gives the
fluxes and the labeling decides the splits; with ``--vary-measured-fluxes`` each is instead an
independent factor drawn from its own distribution, and then sets most of every flux's variance.
The groups are otherwise independent, so
no correlation between metabolites survives. SALib's Sobol sequence places N(D + 2) points for the
D factors, and ``SALib.analyze.sobol`` gives each flux the first-order index S1 (Saltelli 2010) and
the total index ST (Jansen) with bootstrap confidence intervals, of the ensemble prediction as
``EnsembleSigmaFlux.predict`` reports it: the medoid_convex aggregate of the members
(``uncertainty._medoid_weights``), one set of member weights for the whole population as for one
predict call, solved on the independent base samples (A and B). Solving it per sample instead
lets a small change of one token switch the weights, a jump every token then reads as sensitivity.

The Sobol indices say how much a factor moves a flux, not which way. On the same design, the
direction comes from a rank regression: the standardized ranks of each flux's prediction regressed
on every factor's standardized quantile ``u`` (the factors' ranks, the factors being monotone in
``u``) over all N(D + 2) points give the standardized rank regression coefficient SRRC, whose
square is the factor's share of the flux's rank variance as the inputs are independent (close to
S1 for a monotone model), and the partial rank correlation PRCC. A flux's rank R² (the sum of its
SRRC²) says how much of it a monotone reading explains. Each AB_i row differs from its A row in
factor i alone, so the change of the prediction, signed by the direction of the step, is a radial
elementary effect: its mean mu and mean absolute value mu_star, in units of the flux's standard
deviation, give |mu| / mu_star close to 1 when the factor moves the flux one way everywhere. A
positive sign means a larger token raises the flux; a token's factor raises that token and lowers
the rest of its MID in proportion, and a remainder's factor raises the unseen part of the MID,
M+0 among it, so a positive remainder sign means less labeling raises the flux.

Blocks of members can run as separate jobs with ``--part``, each saving its members' predictions;
``--combine`` stacks every member, aggregates and analyses once. The parts are kept, so the
aggregation can be redone without running the model again.

Per model, in ``paper/.results/cross_attention/<key>_sobol_naive.npz``, in the layout of
sobol_sensitivity.py (so the heatmap and graph read it with ``--sobol _naive``):
  S1, ST, S1_conf, ST_conf  (n_fluxes, n_varied)  per varying token
  varied, token_index, flux_names, fixed_idx
  flux_S1, flux_ST, flux_S1_conf, flux_ST_conf  (n_fluxes, n_measured)  per measured-flux input,
                empty unless ``--vary-measured-fluxes``
  mid_S1, mid_ST, mid_S1_conf, mid_ST_conf  (n_fluxes, n_mids)  per MID remainder, and
  mid_tracer, mid_metabolite  naming them
  SRRC, SRRC_conf, PRCC, mu, mu_star  with the same prefixes and shapes: the signed indices and
                the 95% half-width of SRRC
  rank_R2       (n_fluxes,)  each flux's rank R²
  base_samples, seed, members

Usage:
    python paper/scripts/sobol_naive.py --models m_musculus_t_cell --members 0 --part
    python paper/scripts/sobol_naive.py --models m_musculus_t_cell --combine
"""
import argparse
import pathlib
import time
import warnings

import numpy as np
import torch
from _figure_common import add_common_args, run_dir_for, selected_models
from cross_attention import RESULTS_DIR, sfx_path, validation_inputs
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
from scipy.stats import gamma, rankdata
from sobol_sensitivity import predict

from sigmaflux.ai import load_model
from sigmaflux.ai.uncertainty import _medoid_weights

GRID = np.linspace(1e-6, 1 - 1e-6, 4001)   # quantiles tabulated per Gamma factor
KEYS = ("S1", "ST", "S1_conf", "ST_conf", "SRRC", "SRRC_conf", "PRCC", "mu", "mu_star")


def dirichlet_groups(values, token_index):
    """Per MID group: the varying tokens, the constant tokens' summed value and the Dirichlet
    concentrations (tokens, then the remainder) matched to the group's simulated samples."""
    groups = {}
    for i, (tracer, met, _) in enumerate(token_index):
        groups.setdefault((tracer, met), []).append(i)
    out = []
    for (tracer, met), tokens in groups.items():
        varies = values[:, tokens].max(0) > values[:, tokens].min(0)
        vary = np.array(tokens)[varies]
        fixed = values[0, np.array(tokens)[~varies]].sum()
        # A group whose constant tokens fill its MID has no share left to vary.
        if not len(vary) or fixed > 1 - 1e-6:
            continue
        parts = values[:, vary]
        share = np.column_stack([parts, np.clip(1 - fixed - parts.sum(1), 0, None)]) / (1 - fixed)
        mean, var = share.mean(0), share.var(0)
        ok = var > 0
        total = np.median(mean[ok] * (1 - mean[ok]) / var[ok]) - 1
        out.append({"tracer": tracer, "metabolite": met, "tokens": vary, "fixed": fixed,
                    "alpha": np.maximum(mean * max(total, 0.5), 1e-3)})
    return out


def factor_table(groups, measured, vary_fluxes):
    """Quantile tables (n_factors, len(GRID)) of every factor: per group its tokens and remainder,
    then each measured flux when ``vary_fluxes``, and each factor's (kind, index)."""
    tables, labels = [], []
    for g, group in enumerate(groups):
        for j, a in enumerate(group["alpha"]):
            tables.append(gamma.ppf(GRID, a))
            labels.append(("token", group["tokens"][j]) if j < len(group["tokens"]) else ("mid", g))
    for j in range(measured.shape[1] if vary_fluxes else 0):
        tables.append(np.quantile(measured[:, j], GRID))
        labels.append(("flux", j))
    return np.array(tables), labels


def inputs(u, groups, tables, base):
    """Token fractions (n, n_tok) and measured fluxes from factor quantiles ``u`` (n, n_factors),
    the constant tokens taken from ``base``."""
    x = np.array([np.interp(u[:, f], GRID, tables[f]) for f in range(u.shape[1])]).T
    fracs = np.tile(base, (len(u), 1))
    col = 0
    for group in groups:
        width = len(group["alpha"])
        g = x[:, col:col + width]
        share = g / g.sum(1, keepdims=True) * (1 - group["fixed"])
        fracs[:, group["tokens"]] = share[:, :-1]
        col += width
    return fracs, x[:, col:]


def medoid_convex(stack, n_factors):
    """The medoid_convex aggregate (n, n_fluxes) of member predictions ``stack``
    (n_members, n, n_fluxes): one weight per member, solved on the A and B rows of SALib's
    layout (per base sample A, AB_1..AB_D, B) and applied to every row."""
    step = n_factors + 2
    rows = np.flatnonzero(np.isin(np.arange(stack.shape[1]) % step, (0, step - 1)))
    weights = _medoid_weights(stack[:, rows].reshape(len(stack), -1).astype(float),
                              how="medoid_convex")
    return np.einsum("mnf,m->nf", stack, weights)


def design(d, base_samples, seed):
    """SALib's Sobol sequence (N(D + 2), D) for ``d`` factors, per base sample the rows A,
    AB_1..AB_D, B."""
    problem = {"num_vars": d, "names": [str(i) for i in range(d)], "bounds": [[0, 1]] * d}
    return sobol_sample.sample(problem, base_samples, calc_second_order=False, seed=seed)


def signed_indices(u, y):
    """SRRC, its 95% half-width, PRCC, mu and mu_star (n_fluxes, D) and the rank R² (n_fluxes,)
    of the predictions ``y`` (N(D + 2), n_fluxes) on the design ``u``; a constant flux reads 0."""
    n, d = u.shape
    x = (u - u.mean(0)) / u.std(0)
    ranks = rankdata(y, axis=0)
    spread = ranks.std(0)
    flat = spread == 0
    r = np.divide(ranks - ranks.mean(0), spread, out=np.zeros_like(ranks), where=~flat)
    gram_inv = np.linalg.inv(x.T @ x / n)
    beta = gram_inv @ (x.T @ r / n)
    r2 = np.einsum("df,df->f", beta, x.T @ r / n)
    dof = n - d - 1
    se = np.sqrt(np.outer(np.diag(gram_inv), np.clip(1 - r2, 0, None) * ~flat) / dof)
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    prcc = t / np.sqrt(t ** 2 + dof)
    # A and AB_i of each base sample differ in factor i alone.
    rows = y.reshape(-1, d + 2, y.shape[1])
    steps = u.reshape(-1, d + 2, d)
    direction = np.sign(steps[:, -1] - steps[:, 0])
    scale = np.where(flat, 1, y.std(0))
    effect = (rows[:, 1:-1] - rows[:, :1]) * direction[..., None] / scale
    return {"SRRC": beta.T, "SRRC_conf": 1.96 * se.T, "PRCC": prcc.T,
            "mu": effect.mean(0).T, "mu_star": np.abs(effect).mean(0).T}, r2


def analyze_and_save(dest, y, u, kinds, meta, seed):
    """Sobol and signed indices of every flux of the predictions ``y`` (N(D + 2), n_fluxes) on
    the design ``u``, per factor kind, written with ``meta`` to ``dest``."""
    d = len(kinds)
    problem = {"num_vars": d, "names": [str(i) for i in range(d)], "bounds": [[0, 1]] * d}
    res, r2 = signed_indices(u, y)
    res |= {k: np.zeros((y.shape[1], d)) for k in ("S1", "ST", "S1_conf", "ST_conf")}
    for r in range(y.shape[1]):
        if np.ptp(y[:, r]) > 0:
            out = sobol.analyze(problem, y[:, r], calc_second_order=False, seed=seed)
            for k in ("S1", "ST", "S1_conf", "ST_conf"):
                res[k][r] = out[k]
    np.savez(dest, **{f"{prefix}{k}": res[k][:, kinds == kind] for k in KEYS
                      for prefix, kind in (("", "token"), ("flux_", "flux"), ("mid_", "mid"))},
             rank_R2=r2, **meta)
    print(f"Saved: {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where the .npz files go (default: paper/.results/cross_attention)")
    parser.add_argument("--base-samples", type=int, default=512,
                        help="N of the Sobol sequence; the model runs N(D + 2) times per member")
    parser.add_argument("--members", type=int, nargs="+", default=None,
                        help="Ensemble members to run (default: all)")
    parser.add_argument("--part", action="store_true",
                        help="Save these members' predictions as a part for --combine")
    parser.add_argument("--combine", action="store_true",
                        help="Merge the parts written with --part into the result")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vary-measured-fluxes", action="store_true",
                        help="Make each measured flux a factor instead of holding it at its median")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="Constant values encountered")

    for m in selected_models(args):
        stem = args.results_dir / f"{m['key']}_sobol_naive"
        if args.combine:
            paths = sorted(stem.parent.glob(f"{stem.name}.part*.npz"))
            loaded = [dict(np.load(p)) for p in paths]
            members = np.concatenate([p["members"] for p in loaded])
            kinds = loaded[0]["kinds"]
            y = medoid_convex(np.concatenate([p["y"] for p in loaded]), len(kinds))
            meta = {k: v for k, v in loaded[0].items() if k not in ("y", "kinds", "members")}
            seed = int(meta.get("seed", args.seed))
            u = design(len(kinds), int(meta["base_samples"]), seed)
            assert len(u) == len(y), f"design {len(u)} rows, predictions {len(y)}"
            analyze_and_save(stem.parent / f"{stem.name}.npz", y, u, kinds,
                             meta | {"members": members, "seed": seed}, seed)
            continue

        path = sfx_path(m["key"])
        print(f"{m['label']}: {path.name}", flush=True)
        ens = load_model(str(path), device=args.device)
        ids, fracs, measured, fixed = validation_inputs(ens, path, args.device,
                                                        run_dir=run_dir_for(m))
        values, flux_values = fracs[..., 0].cpu().numpy(), measured.cpu().numpy()
        token_index = np.array(ens.spec.token_index, dtype=str)
        groups = dirichlet_groups(values, token_index)
        tables, labels = factor_table(groups, flux_values, args.vary_measured_fluxes)
        d = len(labels)
        u = design(d, args.base_samples, args.seed)
        print(f"  {d} factors ({sum(k == 'token' for k, _ in labels)} tokens, {len(groups)} MID "
              f"remainders, {sum(k == 'flux' for k, _ in labels)} measured fluxes); {len(u)} model runs per "
              f"member", flush=True)
        x_tok, x_flux = inputs(u, groups, tables, values[0])
        if not args.vary_measured_fluxes:
            x_flux = np.tile(np.median(flux_values, 0), (len(u), 1))
        x_tok = torch.tensor(x_tok, dtype=torch.float32, device=args.device).unsqueeze(-1)
        x_flux = torch.tensor(x_flux, dtype=torch.float32, device=args.device)

        members = args.members if args.members is not None else list(range(len(ens.members)))
        ys = []
        for i in members:
            start = time.time()
            ys.append(predict(ens.members[i]._model.eval(), ids, x_tok, x_flux, fixed,
                              args.batch_size).astype(np.float32))
            print(f"  member {i} ({time.time() - start:.0f} s)", flush=True)
        ys = np.stack(ys)

        kinds = np.array([k for k, _ in labels])
        meta = {"varied": np.array([j for k, j in labels if k == "token"]),
                "token_index": token_index,
                "flux_names": np.array(ens.spec.flux_names, dtype=str),
                "fixed_idx": fixed.cpu().numpy(), "base_samples": args.base_samples,
                "seed": args.seed,
                "mid_tracer": np.array([g["tracer"] for g in groups]),
                "mid_metabolite": np.array([g["metabolite"] for g in groups])}
        if args.part:
            dest = stem.parent / f"{stem.name}.part{members[0]}.npz"
            np.savez(dest, y=ys, kinds=kinds, members=np.array(members), **meta)
            print(f"Saved: {dest}")
        else:
            analyze_and_save(stem.parent / f"{stem.name}.npz", medoid_convex(ys, len(kinds)), u,
                             kinds, meta | {"members": np.array(members)}, args.seed)


if __name__ == "__main__":
    main()
