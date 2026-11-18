"""
Global (Sobol) sensitivity of every predicted flux to every labeling token, over the model's
simulated samples.

Pick-freeze sampling on real inputs (Saltelli): A and B are two disjoint sets of simulated samples,
and for token t the model is run on A with that one token's value taken from B. Only the randomized
token leaves its own sample, so the inputs stay within the range the model saw. SALib's estimators
give the first-order index S1 (Saltelli 2010) and the total index ST (Jansen: with every
interaction) of every flux. Each measured-flux input is randomized on its own too, so a flux set by
the measured fluxes reads as such. Every token that varies across the samples is randomized,
measured in the experiments or not, so a token's index says how informative it is; a token the same
in every sample has no index.

The estimate is repeated over rounds, each drawing fresh A and B. With ``--subset-size K`` each
round also shows the model only K tokens drawn at random, the rest hidden as
``MISSING_LABEL_VALUE`` as inference hides unmeasured ones, and a token's index is its mean over
the rounds that show it. MIDs are correlated: with every token shown, a token whose information
its neighbours also carry reads close to 0, while over subsets it is credited in the rounds where
those neighbours are hidden, so the indices follow how informative a token is in a typical
measurement rather than only what it adds to all the others. The confidence interval is the 95%
half-width of the mean over rounds and ensemble members.

Per model, in ``paper/.results/cross_attention/<key>_sobol<suffix>.npz`` (members can run as
separate jobs with ``--part`` and be merged with ``--combine``):
  S1, ST, S1_conf, ST_conf  (n_fluxes, n_varied)  indices and their 95% half-widths
  varied        (n_varied,)            token_index positions of the randomized tokens
  shown         (n_varied,)            rounds x members that showed each token
  flux_S1, flux_ST, flux_S1_conf, flux_ST_conf  (n_fluxes, n_measured)  the same per measured-flux
                input, in fixed_idx order
  token_index, flux_names, fixed_idx, subset_size (0: every token), rounds, base_samples

Usage:
    python paper/scripts/sobol_sensitivity.py --models m_musculus_t_cell --members 0 1 2 \\
        --all-samples --subset-size 300 --rounds 40
"""
import argparse
import pathlib
import time
import warnings

import numpy as np
import torch
from _figure_common import add_common_args, run_dir_for, selected_models
from cross_attention import RESULTS_DIR, sfx_path, validation_inputs
from SALib.analyze.sobol import first_order, total_order

from sigmaflux.ai import load_model
from sigmaflux.ai.model import MISSING_LABEL_VALUE


def predict(model, ids, fracs, measured, fixed, batch_size):
    """Predicted fluxes (n, n_fluxes) for the given inputs."""
    with torch.no_grad():
        return torch.cat([model(*(i.expand(len(x), -1) for i in ids), x, measured_fluxes=m,
                                measured_flux_ids=fixed)[0].double()
                          for x, m in zip(fracs.split(batch_size), measured.split(batch_size))]
                         ).cpu().numpy()


def one_round(model, ids, fracs, measured, fixed, hidden, swapped, a, b, batch_size):
    """S1 and ST (n_inputs, n_fluxes) of the ``swapped`` tokens, then of every measured-flux input,
    from base samples ``a`` and ``b``, the ``hidden`` tokens (a boolean mask) hidden."""
    hidden = torch.as_tensor(hidden, device=fracs.device)
    xa, xb = fracs[a].clone(), fracs[b].clone()
    xa[:, hidden] = xb[:, hidden] = MISSING_LABEL_VALUE
    ma, mb = measured[a], measured[b]
    ya, yb = (predict(model, ids, x, m, fixed, batch_size) for x, m in ((xa, ma), (xb, mb)))
    n = len(a)
    # Several pick-freeze sets per forward pass, so a batch fills the device.
    per_pass = max(1, batch_size // n)
    swaps = [("token", t) for t in swapped] + [("flux", j) for j in range(measured.shape[1])]
    s1, st = [], []
    for start in range(0, len(swaps), per_pass):
        xs, ms = [], []
        for kind, i in swaps[start:start + per_pass]:
            x, m = xa.clone(), ma.clone()
            if kind == "token":
                x[:, i] = xb[:, i]
            else:
                m[:, i] = mb[:, i]
            xs.append(x)
            ms.append(m)
        yab = predict(model, ids, torch.cat(xs), torch.cat(ms), fixed, batch_size)
        for chunk in np.split(yab, len(xs)):
            s1.append(first_order(ya, chunk, yb))
            st.append(total_order(ya, chunk, yb))
    return np.array(s1), np.array(st)


def finalize(sums, shown_count):
    """Mean S1 and ST over the estimates of each input and their 95% half-widths, inputs last."""
    count = shown_count[:, None]
    out = {}
    for key in ("S1", "ST"):
        mean = np.divide(sums[key], count, out=np.zeros_like(sums[key]), where=count > 0)
        var = np.divide(sums[f"{key}_sq"], count, out=np.zeros_like(mean), where=count > 0) \
            - mean ** 2
        conf = 1.96 * np.sqrt(np.clip(var, 0, None) / np.maximum(count - 1, 1))
        conf[shown_count < 2] = np.nan
        out[key], out[f"{key}_conf"] = mean.T, conf.T
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where the .npz files go (default: paper/.results/cross_attention)")
    parser.add_argument("--all-samples", action="store_true",
                        help="Draw from every simulated sample of the training run, not only the "
                             "validation set stored in the model")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Tokens shown per round, drawn at random (default: every token)")
    parser.add_argument("--rounds", type=int, default=1, help="Estimates per ensemble member")
    parser.add_argument("--base-samples", type=int, default=500,
                        help="Samples in each of A and B per round")
    parser.add_argument("--suffix", default="", help="Appended to the output file name")
    parser.add_argument("--members", type=int, nargs="+", default=None,
                        help="Ensemble members to run (default: all)")
    parser.add_argument("--part", action="store_true",
                        help="Write these members' running sums as a part for --combine")
    parser.add_argument("--combine", action="store_true",
                        help="Merge the parts written with --part into the result, then remove them")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    # A flux constant over a round's samples has no variance to split; its indices stay 0.
    warnings.filterwarnings("ignore", message="Constant values encountered")

    for m in selected_models(args):
        stem = args.results_dir / f"{m['key']}_sobol{args.suffix}"
        if args.combine:
            parts = sorted(stem.parent.glob(f"{stem.name}.part*.npz"))
            loaded = [dict(np.load(p)) for p in parts]
            total = loaded[0] | {key: sum(p[key] for p in loaded)
                                 for key in ("S1", "ST", "S1_sq", "ST_sq", "count")}
            save(stem, total)
            for p in parts:
                p.unlink()
            print(f"Combined {len(parts)} parts")
            continue

        path = sfx_path(m["key"])
        print(f"{m['label']}: {path.name}", flush=True)
        ens = load_model(str(path), device=args.device)
        ids, fracs, measured, fixed = validation_inputs(
            ens, path, args.device, run_dir=run_dir_for(m) if args.all_samples else None)
        values = fracs[..., 0].cpu().numpy()
        varied = np.flatnonzero(values.max(0) > values.min(0))
        n_tok, n_fx = fracs.shape[1], measured.shape[1]
        k = args.subset_size or n_tok
        print(f"  {len(varied)} of {n_tok} tokens vary, {k} shown per round, {n_fx} measured "
              f"fluxes; {args.rounds} rounds of 2 x {args.base_samples} of {len(fracs)} samples",
              flush=True)

        total = {key: np.zeros((len(varied) + n_fx, len(ens.spec.flux_names)))
                 for key in ("S1", "ST", "S1_sq", "ST_sq")}
        total["count"] = np.zeros(len(varied) + n_fx)
        members = args.members if args.members is not None else range(len(ens.members))
        for i in members:
            start = time.time()
            model = ens.members[i]._model.eval()
            rng = np.random.default_rng([args.seed, i])
            for _ in range(args.rounds):
                hidden = np.ones(n_tok, bool)
                hidden[rng.choice(n_tok, k, replace=False)] = False
                ab = rng.choice(len(fracs), 2 * args.base_samples, replace=False)
                s1, st = one_round(model, ids, fracs, measured, fixed, hidden,
                                   varied[~hidden[varied]], ab[:args.base_samples],
                                   ab[args.base_samples:], args.batch_size)
                rows = np.r_[np.flatnonzero(~hidden[varied]), len(varied) + np.arange(n_fx)]
                for key, v in (("S1", s1), ("ST", st)):
                    total[key][rows] += v
                    total[f"{key}_sq"][rows] += v ** 2
                total["count"][rows] += 1
            print(f"  member {i} ({time.time() - start:.0f} s)", flush=True)

        total |= {"varied": varied, "token_index": np.array(ens.spec.token_index, dtype=str),
                  "flux_names": np.array(ens.spec.flux_names, dtype=str),
                  "fixed_idx": fixed.cpu().numpy(), "subset_size": args.subset_size or 0,
                  "rounds": args.rounds, "base_samples": args.base_samples}
        if args.part:
            part = stem.parent / f"{stem.name}.part{'_'.join(map(str, members))}.npz"
            np.savez(part, **total)
            print(f"Saved: {part}")
        else:
            save(stem, total)


def save(stem, total):
    """The result file from the running sums in ``total``."""
    n_var = len(total["varied"])
    out = finalize(total, total["count"])
    dest = stem.parent / f"{stem.name}.npz"
    np.savez(dest, **{key: v[:, :n_var] for key, v in out.items()},
             **{f"flux_{key}": v[:, n_var:] for key, v in out.items()},
             shown=total["count"][:n_var],
             **{key: total[key] for key in ("varied", "token_index", "flux_names", "fixed_idx",
                                            "subset_size", "rounds", "base_samples")})
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
