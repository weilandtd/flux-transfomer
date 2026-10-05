"""
Decompose every predicted flux into exact per-token contributions through the
cross-attention, on each packaged model's validation set or on the experimental data.

After the cross-attention softmax the decoder is linear -- ``output_proj``, the
de-standardization ``x = logit * y_std + y_mean`` and ``v = Z x + offset`` -- so each
flux splits exactly over the tokens the latent queries attend to:

    v_r = sum_t F[r, t] + const_r
    F[r, t] = sum_h (Z diag(y_std) A_h)[r, t] * c_h[t]
    c_h[t]  = w_out^T W_o[:, h] (W_v,h LN(kv_t) + b_v,h)

``A_h`` is head h's latent-to-token attention and ``c_h[t]`` is what token t's value
vector adds to any latent's logit. Z is what turns latent attention into flux
attention; the value term is what makes the sum reproduce the prediction, which is
checked on every forward pass.

``--data validation`` runs the simulated validation set stored in each .sfx.
``--data experimental`` (the default) runs every growth condition of every condition in
the organism's experimental data, one sample per growth condition: replicates are
averaged, and the inputs go through ``predict()`` itself, so tokens the experiment did
not measure enter as the missing-value sentinel exactly as at inference.

Per model, in ``paper/.results/cross_attention/<key>[_experimental].npz``:
  F_abs       (n_fluxes, n_tokens)  mean |F| over samples (or growth conditions) and members
  F_signed    (n_fluxes, n_tokens)  mean F
  F_sd        (n_fluxes, n_tokens)  standard deviation of F over samples, per member, averaged
              over members: how much of a flux's variation passes through the token, which a
              token that is the same in every sample (attended or not) cannot carry
  flux_sd     (n_fluxes,)           standard deviation of the predicted flux, the same way
  token_index labeling tokens as (tracer, metabolite, isotopomer); measured-flux tokens
              follow them in fixed_idx order
  flux_names, fixed_idx, z_zero_rows (fluxes Z cannot move)
  experimental only: conditions, F_abs_by_condition, measured (conditions x labeling
  tokens, whether the experiment measured the token)

Drawing the heatmaps from these is paper/scripts/plot_cross_attention.py.

Usage:
    python paper/scripts/cross_attention.py
    python paper/scripts/cross_attention.py --models e_coli_core
    python paper/scripts/cross_attention.py --data validation --n-samples 100
"""

import argparse
import contextlib
import pathlib

import numpy as np
import torch
from _figure_common import (
    _REPO_ROOT,
    add_common_args,
    resolve_organism,
    run_dir_for,
    selected_models,
)

from sigmaflux.ai import load_model
from sigmaflux.ai._sfx import SfxReader
from sigmaflux.ai.data import load_data

SFX_VERSION = "v1.4.0"
RESULTS_DIR = _REPO_ROOT / "paper" / ".results" / "cross_attention"
# Tolerance on sum_t F + const vs. the model's own flux prediction, relative to the
# largest predicted |flux| in the batch.
REL_TOL = 1e-6


def sfx_path(key):
    return _REPO_ROOT / "models" / "pretrained" / f"{key}-{SFX_VERSION}.sfx"


def results_path(results_dir, key, data):
    return results_dir / (f"{key}.npz" if data == "validation" else f"{key}_{data}.npz")


def validation_inputs(ens, path, device, n_samples=None, run_dir=None):
    """Token tensors and measured fluxes for the stored validation set (export.validate_model),
    or, with ``run_dir``, for every simulated sample of that training run."""
    spec = ens.spec
    if run_dir is None:
        with SfxReader(path) as r:
            fractions = r.read_array("validation/val_fractions.npy")[:n_samples]
            actuals = r.read_array("validation/val_actuals.npy")[:n_samples]
    else:
        labeling, flux, _ = load_data(str(run_dir), max_samples=n_samples)
        fractions = labeling.loc[[tuple(t) for t in spec.token_index]].to_numpy().T
        actuals = flux.loc[list(spec.flux_names)].to_numpy().T
    tables = spec.emb_tables()
    compound_idx = {c: i for i, c in enumerate(tables["compound_vocab"])}
    label_idx = {l: i for i, l in enumerate(tables["label_vocab"])}
    combo_idx = {t: i for i, t in enumerate(spec.tracer_combo_vocab)}

    def ids(lookup, col):
        return torch.tensor([lookup.get(t[col], 0) for t in spec.token_index],
                            dtype=torch.int64, device=device)

    fixed = torch.as_tensor(np.asarray(spec.null_space_data["fixed_idx"], dtype=int),
                            dtype=torch.long, device=device)
    fracs = torch.tensor(fractions, dtype=torch.float32, device=device).unsqueeze(-1)
    measured = torch.tensor(actuals, dtype=torch.float32, device=device)[:, fixed]
    return (ids(compound_idx, 1), ids(label_idx, 2), ids(combo_idx, 0)), fracs, measured, fixed


def contributions(model, q, kv):
    """F (B, n_fluxes, N) and const (n_fluxes,) such that fluxes = F.sum(-1) + const."""
    ca, mha = model.cross_attn, model.cross_attn.attn
    q, kv = ca.q_norm(q), ca.kv_norm(kv)
    _, A = mha(q, kv, kv, need_weights=True, average_attn_weights=False)  # (B, H, R, N)
    A = A.double()
    kv = kv.double()

    d, H = mha.embed_dim, mha.num_heads
    V = kv @ mha.in_proj_weight[2 * d:].double().T + mha.in_proj_bias[2 * d:].double()
    V = V.view(*V.shape[:2], H, d // H)                                   # (B, N, H, dh)
    w = model.output_proj.weight.double()[0]
    u = (w @ mha.out_proj.weight.double()).view(H, d // H)                # (H, dh)
    c = torch.einsum("bnhk,hk->bhn", V, u)                                # (B, H, N)

    n_free = model.cfg.n_free
    G = torch.einsum("bhin,bhn->bin", A[:, :, :n_free], c)               # latent logits by token
    Zs = model.Z_full.double() * model.y_norm_std.double()
    F = torch.einsum("ri,bin->brn", Zs, G)
    bias = w @ mha.out_proj.bias.double() + model.output_proj.bias.double()[0]
    const = (Zs.sum(1) * bias + model.Z_full.double() @ model.y_norm_mean.double()
             + model.flux_offset.double())
    return F, const


@contextlib.contextmanager
def recording(models):
    """Accumulate sum |F|, sum F and the sample count over every forward pass of *models*,
    and per member the sums of F, F², v and v² for the spreads."""
    stats = {"abs": 0.0, "signed": 0.0, "n": 0, "worst": 0.0,
             "member": [{"F": 0.0, "F2": 0.0, "v": 0.0, "v2": 0.0, "n": 0} for _ in models]}
    handles = []
    for model, own in zip(models, stats["member"]):
        captured = {}

        def capture(_module, args, _output, captured=captured):
            captured["q"], captured["kv"] = args[0], args[1]

        def decompose(module, _args, output, captured=captured, own=own):
            with torch.no_grad():
                F, const = contributions(module, captured["q"], captured["kv"])
            fluxes = output[0].double()
            err = (F.sum(-1) + const - fluxes).abs().max().item()
            stats["worst"] = max(stats["worst"], err / max(fluxes.abs().max().item(), 1e-12))
            stats["abs"] = stats["abs"] + F.abs().sum(0).cpu().numpy()
            stats["signed"] = stats["signed"] + F.sum(0).cpu().numpy()
            stats["n"] += F.shape[0]
            for k, x in (("F", F), ("v", fluxes)):
                own[k] = own[k] + x.sum(0).cpu().numpy()
                own[k + "2"] = own[k + "2"] + (x ** 2).sum(0).cpu().numpy()
            own["n"] += F.shape[0]

        handles += [model.cross_attn.register_forward_hook(capture),
                    model.register_forward_hook(decompose)]
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def checked_means(stats):
    if stats["worst"] > REL_TOL:
        raise RuntimeError(f"Decomposition does not reproduce the fluxes: "
                           f"relative error {stats['worst']:.2e} > {REL_TOL:.0e}")
    print(f"  sum_t F + const reproduces fluxes to relative error {stats['worst']:.2e}")
    return stats["abs"] / stats["n"], stats["signed"] / stats["n"]


def spreads(stats):
    """Standard deviations of F and of the predicted fluxes over samples, averaged over members."""
    def sd(m, k):
        mean = m[k] / m["n"]
        return np.sqrt(np.clip(m[k + "2"] / m["n"] - mean ** 2, 0, None))
    return (np.mean([sd(m, "F") for m in stats["member"]], axis=0),
            np.mean([sd(m, "v") for m in stats["member"]], axis=0))


def validation_run(ens, path, args):
    (cids, lids, tids), fracs, measured, fixed = validation_inputs(
        ens, path, args.device, args.n_samples)
    models = [member._model.eval() for member in ens.members]
    with recording(models) as stats:
        for i, model in enumerate(models):
            for s in range(0, fracs.shape[0], args.batch_size):
                sl = slice(s, s + args.batch_size)
                b = fracs[sl].shape[0]
                with torch.no_grad():
                    model(cids.expand(b, -1), lids.expand(b, -1), tids.expand(b, -1),
                          fracs[sl], measured_fluxes=measured[sl], measured_flux_ids=fixed)
            print(f"  member {i + 1}/{len(models)}", flush=True)
    return dict(zip(("F_abs", "F_signed", "F_sd", "flux_sd"),
                    (*checked_means(stats), *spreads(stats))))


def condition_inputs(model):
    """(condition, growth condition, df_L, df_Vm) for every growth condition of every
    condition in the organism's experimental data, one row per replicate."""
    from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path
    from sigmaflux.pipeline.run_inference_workflow import (
        _discover_conditions,
        _find_ext_flux_file,
        _inspect_growth_conditions,
        load_condition_inputs,
    )

    run = run_dir_for(model)
    org = resolve_organism(run)
    cfg = org["cfg"]
    data_dir = resolve_path(org["data"], org["root"])
    mapping_path = resolve_path(mapping_csv_for(cfg), org["root"])
    ext_flux = cfg.get("experiment", {}).get("ext_flux")
    ext_file = resolve_path(ext_flux, org["root"]) if ext_flux else _find_ext_flux_file(data_dir)
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None
    for cond, mid_file in sorted(_discover_conditions(data_dir).items()):
        for gc in _inspect_growth_conditions(mid_file):
            try:
                df_L, df_Vm, _ = load_condition_inputs(
                    mid_file, ext_file, cond, gc, mapping_path, flux_names,
                    unmeasured=cfg.get("experiment", {}).get("unmeasured_isotopologues"))
            except Exception as exc:
                print(f"  {cond}{gc}: skipped — {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]}")
                continue
            yield cond, gc, df_L, df_Vm


def experimental_inputs(model):
    """(label, df_L, df_Vm) for every growth condition, replicates averaged."""
    for cond, gc, df_L, df_Vm in condition_inputs(model):
        df_L = (df_L.groupby(["Tracer Name", "Metabolite ID", "Isotopomer"],
                             as_index=False)["Fraction"].mean())
        df_Vm = df_Vm.groupby("Flux Name", as_index=False)["Value"].mean()
        yield f"{cond}{gc}", df_L, df_Vm


def experimental_run(ens, model):
    models = [member._model.eval() for member in ens.members]
    tracer_id = ens.tracer_ids()
    token_pos = {tuple(t): i for i, t in enumerate(ens.spec.token_index)}
    labels, by_cond, signed, measured = [], [], [], []
    for label, df_L, df_Vm in experimental_inputs(model):
        print(f"  {label}", flush=True)
        with recording(models) as stats:
            ens.predict(df_L, df_Vm, aggregate=False)
        a, s = checked_means(stats)
        mask = np.zeros(len(token_pos), dtype=bool)
        for tracer, met, iso in df_L[["Tracer Name", "Metabolite ID", "Isotopomer"]].itertuples(
                index=False):
            pos = token_pos.get((tracer_id.get(tracer), met, iso))
            if pos is not None:
                mask[pos] = True
        labels.append(label)
        by_cond.append(a)
        signed.append(s)
        measured.append(mask)
    return dict(F_abs=np.mean(by_cond, axis=0), F_signed=np.mean(signed, axis=0),
                F_abs_by_condition=np.stack(by_cond), conditions=np.array(labels),
                measured=np.stack(measured))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--data", choices=["validation", "experimental"], default="experimental")
    parser.add_argument("--results-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Where the .npz files go (default: paper/.results/cross_attention)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Use only the first N validation samples (default: all)")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for m in selected_models(args):
        path = sfx_path(m["key"])
        print(f"{m['label']}: {path.name} ({args.data})", flush=True)
        ens = load_model(str(path), device=args.device)
        if args.data == "validation":
            result = validation_run(ens, path, args)
        else:
            result = experimental_run(ens, m)
        Z = ens.members[0]._model.Z_full.cpu().numpy()
        out = results_path(args.results_dir, m["key"], args.data)
        np.savez(out, **result,
                 token_index=np.array(ens.spec.token_index, dtype=str),
                 flux_names=np.array(ens.spec.flux_names, dtype=str),
                 fixed_idx=np.asarray(ens.spec.null_space_data["fixed_idx"], dtype=int),
                 z_zero_rows=~np.any(Z != 0, axis=1))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
