"""Package a trained SigmaFlux model into one self-contained ``.sfx`` file.

Contrast with :mod:`sigmaflux.ai.checkpoint`, which reads the **training
checkpoint** — a ``.pt`` that only makes sense next to its run directory, config
YAML, COBRA JSON and training HDF5. This module reads and writes the **exported
model**: inference-only, self-describing, and loadable with nothing but the file.

    from sigmaflux.ai import export_model, load_model

    export_model("run/ensemble", name="i_orientalis", model_version="1.0.0")
    sf = load_model("i_orientalis-v1.0.0.sfx")
    sf.predict(df_L, df_Vm)

What is stored
--------------
Once, shared by every member: the vocabularies, null-space basis, token index,
output names, model config, flux normalization, the resolved tracer-name map,
and (optionally) the validation split. Per member: **only the trained weights**.

What is dropped, because it is regenerable or is training-only ballast: the 14
frozen tensors (all five radial embedding tables plus buffers that duplicate
``null_space_data``/``flux_norm_stats``), and ``val_predictions``, ``log``,
``train_indices``, ``data_dir``, ``model_json``.

Weights default to float16, which on the released I. orientalis ensemble leaves
every identifiable reaction intact (#R²>0.9 unchanged at 250, max |ΔR²| 5.8e-3
on those reactions) while cutting 252 MB to ~35 MB. The null-space basis and all
statistics stay float32 regardless — fp16 there would break ``S @ v ≈ 0``.

An export is **always an ensemble**: a single model is an ensemble of one, so
downstream code has exactly one shape to handle.
"""

import datetime
import glob
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import _sfx
from ._sfx import (
    MANIFEST,
    MEMBER_FMT,
    SCHEMA_VERSION,
    SFX_SUFFIX,
    SfxError,
    SfxFormatError,
    SfxIntegrityError,
    SfxReader,
    SfxVersionError,
    SfxWriter,
    read_manifest,
)
from .inference import EnsembleSigmaFlux, ModelSpec

__all__ = [
    "SFX_SUFFIX",
    "SfxError",
    "SfxFormatError",
    "SfxIntegrityError",
    "SfxVersionError",
    "export_model",
    "inspect_model",
    "load_model",
    "validate_model",
]

#: Weight tensors above this magnitude fall back to float32 (fp16 max is 65504).
_FP16_SAFE_MAX = 6.0e4

#: state_dict keys the loader regenerates rather than storing.
_REGENERATED = (
    "compound_emb.weight", "label_emb.weight", "tracer_emb.weight",
    "reaction_emb.weight", "_token_type_emb.weight",
    "Z_full", "token_experiment_ids", "fixed_idx_t",
    "flux_norm_mean", "flux_norm_std",
    "y_norm_mean", "y_norm_std", "rev_norm_mean", "rev_norm_std",
)

#: Keys that must agree across ensemble members.
_SHARED_KEYS = (
    "model_config", "flux_names", "reversibility_names", "measured_fluxes",
    "token_index", "h_reaction", "h_type", "val_indices",
)

_NULL_SPACE_FLOAT = ("Z_free", "y_mean", "y_std", "rev_log_mean", "rev_log_std", "obs_weights")
_NULL_SPACE_INT = ("free_idx", "fixed_idx", "rev_variable_idx")


# ── Source discovery ──────────────────────────────────────────────────────────

def _discover_members(source):
    """Resolve *source* to an ordered list of member checkpoint paths.

    Accepts an ``ensemble/`` directory, a run directory containing one, a single
    ``.pt``, or an explicit sequence of ``.pt`` paths.
    """
    if isinstance(source, (str, os.PathLike)):
        source = [source]
    source = [str(s) for s in source]

    if len(source) == 1:
        one = source[0]
        if os.path.isdir(one):
            members = sorted(glob.glob(os.path.join(one, "model_*.pt")))
            if not members:
                members = sorted(glob.glob(os.path.join(one, "ensemble", "model_*.pt")))
            if not members:
                members = sorted(glob.glob(os.path.join(one, "*.pt")))
            if not members:
                raise FileNotFoundError(f"no .pt checkpoints under {one}")
            return members
        if not os.path.isfile(one):
            raise FileNotFoundError(f"no such checkpoint: {one}")
        return [one]

    for p in source:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"no such checkpoint: {p}")
    return source


def _check_members_agree(checkpoints, paths):
    """Refuse to package members that were not trained as one ensemble."""
    first = checkpoints[0]
    for ckpt, path in zip(checkpoints[1:], paths[1:]):
        for key in _SHARED_KEYS:
            a, b = first.get(key), ckpt.get(key)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                same = np.array_equal(np.asarray(a), np.asarray(b))
            else:
                same = a == b
            if not same:
                raise ValueError(
                    f"ensemble members disagree on {key!r}: {os.path.basename(paths[0])} "
                    f"and {os.path.basename(path)} were not trained as one ensemble"
                )
        za = np.asarray(first["null_space_data"]["Z_free"])
        zb = np.asarray(ckpt["null_space_data"]["Z_free"])
        if not np.array_equal(za, zb):
            raise ValueError(
                f"ensemble members disagree on the null-space basis: "
                f"{os.path.basename(paths[0])} and {os.path.basename(path)}"
            )


# ── Provenance ────────────────────────────────────────────────────────────────

def _git_state(cwd):
    try:
        rev = subprocess.run(["git", "-C", str(cwd), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return None, None
        status = subprocess.run(["git", "-C", str(cwd), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
        return rev.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return _sfx.sha256_bytes(fh.read())


# ── Weight layout ─────────────────────────────────────────────────────────────

def _tensor_layout(state_dict, precision):
    """Describe the trained tensors and their byte offsets within a member blob."""
    layout, offset = [], 0
    for name, tensor in state_dict.items():
        if name in _REGENERATED:
            continue
        arr = tensor.detach().cpu().numpy()
        stored = arr.dtype
        if precision == "fp16" and arr.dtype == np.float32:
            finite = np.isfinite(arr).all()
            if finite and (arr.size == 0 or np.abs(arr).max() <= _FP16_SAFE_MAX):
                stored = np.dtype(np.float16)
        nbytes = int(arr.size * np.dtype(stored).itemsize)
        layout.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "stored_dtype": str(stored),
            "offset": offset,
            "nbytes": nbytes,
        })
        offset += nbytes
    return layout, offset


def _member_blob(state_dict, layout):
    """Concatenate a member's trained tensors into one little-endian buffer."""
    chunks = []
    for spec in layout:
        arr = state_dict[spec["name"]].detach().cpu().numpy()
        chunks.append(np.ascontiguousarray(arr.astype(spec["stored_dtype"], copy=False)).tobytes())
    return b"".join(chunks)


# ── Validation payload ────────────────────────────────────────────────────────

def _read_val_inputs(ckpt, spec, val_indices, data_dir=None, max_samples=None):
    """The MID inputs for the validation split, in training token order.

    This is the only step that touches the training HDF5. Returns
    ``(fractions, rev_actuals)`` or ``(None, None)`` if the data is unreachable.
    """
    from .data import load_data
    resolved = data_dir or ckpt.get("data_dir")
    if not resolved or not os.path.isdir(resolved):
        print(f"  validation inputs: skipped (training data not found at {resolved!r})")
        return None, None
    try:
        labeling_df, _, rev_df = load_data(resolved)
    except (OSError, FileNotFoundError) as exc:
        print(f"  validation inputs: skipped ({exc})")
        return None, None

    token_index = [tuple(t) for t in spec.token_index]
    labeling_df = labeling_df.loc[token_index]
    idx = np.asarray(val_indices, dtype=int)
    if max_samples is not None:
        idx = idx[:max_samples]
    fractions = labeling_df.values[:, idx].T.astype(np.float32)

    rev_names = list(spec.reversibility_names)
    rev_actuals = None
    if rev_names and set(rev_names) <= set(rev_df.index):
        rev_actuals = rev_df.loc[rev_names].values[:, idx].T.astype(np.float32)
    return fractions, rev_actuals


# ── Public API ────────────────────────────────────────────────────────────────

def export_model(source, out_path=None, *, name=None, model_version,
                 precision="fp16", config=None, root=None, description=None,
                 with_validation=True, with_val_inputs="auto", data_dir=None,
                 model_json=None, max_members=None, max_val_samples=None,
                 metrics=True, overwrite=False, verbose=True):
    """Package trained checkpoints into a single ``.sfx`` model.

    Parameters
    ----------
    source : path or sequence of paths
        A run directory, an ``ensemble/`` directory, a single ``.pt``, or an
        explicit list of member ``.pt`` files.
    out_path : path, optional
        Defaults to ``./<name>-v<model_version>.sfx``.
    name : str, optional
        Model name. Defaults to the config's ``name``, else the source basename.
    model_version : str
        Semantic version of the *weights*, e.g. ``"1.0.0"``. Required.
    precision : {"fp16", "fp32"}
        Storage precision for member weights only.
    config : path, optional
        Training YAML. Supplies the human-readable tracer names and the
        inference batch size; without it tracer names fall back to internal ids.
    with_validation, with_val_inputs :
        Store the validation split, and (``"auto"``/True/False) the MID inputs
        needed to recompute R² from the export alone. Reading the inputs is the
        only step that touches the training HDF5.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be 'fp16' or 'fp32', got {precision!r}")

    member_paths = _discover_members(source)
    if max_members is not None:
        member_paths = member_paths[:max_members]
    if verbose:
        print(f"Packaging {len(member_paths)} member(s) from "
              f"{source if isinstance(source, (str, os.PathLike)) else f'{len(member_paths)} files'}")

    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in member_paths]
    _check_members_agree(checkpoints, member_paths)

    cfg = {}
    if config is not None:
        from ..pipeline._paths import load_config
        cfg, _ = load_config(config, root)

    spec = ModelSpec.from_checkpoint(checkpoints[0], config=config,
                                     model_json=model_json, data_dir=data_dir)

    # A tracer name that does not resolve into the vocabulary silently zeroes every
    # token at predict time, so refuse to freeze a broken map into the artifact.
    known = set(spec.vocabs["tracer_vocab"]) | set(spec.tracer_combo_vocab)
    unknown = {k: v for k, v in spec.tracer_display_names.items() if v not in known}
    if unknown:
        raise ValueError(
            f"tracer names do not resolve into the model's tracer vocabulary: {unknown}. "
            f"Known tracers: {sorted(known)}. Check labeling.tracers in {config!r}."
        )

    name = name or cfg.get("name") or Path(member_paths[0]).parent.name
    out_path = Path(out_path) if out_path else Path(f"{name}-v{model_version}{SFX_SUFFIX}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass overwrite=True to replace it")

    layout, member_nbytes = _tensor_layout(checkpoints[0]["model_state_dict"], precision)
    git_commit, git_dirty = _git_state(Path(member_paths[0]).parent)
    ck0 = checkpoints[0]

    from .. import __version__ as sigmaflux_version
    manifest = {
        "format": _sfx.FORMAT,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "model_version": str(model_version),
        "description": description,
        "created_utc": _utc_now(),
        "precision": precision,
        "n_members": len(member_paths),
        "artifact": out_path.name,
        "provenance": {
            "sigmaflux_version": sigmaflux_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "source": str(source if isinstance(source, (str, os.PathLike)) else member_paths),
            "member_sources": [os.path.basename(p) for p in member_paths],
            "trained_utc": datetime.datetime.fromtimestamp(
                max(os.path.getmtime(p) for p in member_paths),
                datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_config": str(config) if config else None,
            "config_sha256": _sha256_file(config),
            "model_config_sha256": _sfx.sha256_bytes(
                json.dumps(spec.model_config, sort_keys=True).encode()),
            "training_data_dir": ck0.get("data_dir"),
            "training_model_json": ck0.get("model_json"),
        },
        "model": {
            "model_config": spec.model_config,
            "h_reaction": spec.h_reaction,
            "h_type": spec.h_type,
            "n_fluxes": len(spec.flux_names),
            "n_rev": len(spec.reversibility_names),
            "n_tokens": len(spec.token_index),
            "n_compounds": len(spec.vocabs["compound_vocab"]),
            "batch_size": spec.batch_size,
            "has_norm_stats": bool(spec.norm_stats),
        },
        "tracers": spec.tracer_display_names,
        "tracers_source": "config" if spec.tracer_display_names else "identity",
        "tensors": layout,
        "member_nbytes": member_nbytes,
        "regenerated_tensors": list(_REGENERATED),
    }

    nsd = spec.null_space_data
    val_indices = np.asarray(ck0.get("val_indices", []), dtype=np.int32)
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    with SfxWriter(out_path, manifest) as w:
        w.write_json("shared/vocabs.json", {
            **{k: spec.vocabs[k] for k in
               ("compound_vocab", "label_vocab", "tracer_vocab", "h_compound", "h_label", "h_tracer")},
            "token_dim": spec.vocabs.get("token_dim"),
            "tracer_combo_vocab": spec.tracer_combo_vocab,
        })
        w.write_json("shared/token_index.json", [list(t) for t in spec.token_index])
        w.write_json("shared/names.json", {
            "flux_names": spec.flux_names,
            "reversibility_names": spec.reversibility_names,
            "measured_fluxes": spec.measured_fluxes,
        })
        if spec.norm_stats:
            w.write_json("shared/norm_stats.json",
                         {k: np.asarray(v).tolist() for k, v in spec.norm_stats.items()})
        w.write_array("shared/token_experiment_ids.npy",
                      np.asarray(spec.token_experiment_ids, dtype=np.int32))
        for key in _NULL_SPACE_FLOAT:
            if nsd.get(key) is not None:
                w.write_array(f"shared/null_space/{key}.npy",
                              np.asarray(nsd[key], dtype=np.float32))
        for key in _NULL_SPACE_INT:
            if nsd.get(key) is not None:
                w.write_array(f"shared/null_space/{key}.npy",
                              np.asarray(nsd[key], dtype=np.int32))
        if spec.flux_norm_stats:
            for key in ("mean", "std"):
                w.write_array(f"shared/flux_norm/{key}.npy",
                              np.asarray(spec.flux_norm_stats[key], dtype=np.float32))

        # ── validation ────────────────────────────────────────────────────────
        vc = ck0.get("val_correlations") or {}
        val_actuals = vc.get("val_actuals")
        included = bool(with_validation and len(val_indices) and val_actuals is not None)
        vblock = {"included": included, "n_val": int(len(val_indices)) if included else 0,
                  "inputs_included": False, "token_order": "training"}
        if included:
            val_actuals = np.asarray(val_actuals, dtype=np.float32)[:len(val_indices)]
            w.write_array("validation/val_indices.npy", val_indices)
            w.write_array("validation/val_actuals.npy", val_actuals)
            vblock["actuals_shape"] = list(val_actuals.shape)

            want_inputs = with_val_inputs if with_val_inputs != "auto" else True
            if want_inputs:
                fractions, rev_actuals = _read_val_inputs(
                    ck0, spec, val_indices, data_dir=data_dir, max_samples=max_val_samples)
                if fractions is not None:
                    w.write_array("validation/val_fractions.npy", fractions)
                    vblock["inputs_included"] = True
                    vblock["fractions_shape"] = list(fractions.shape)
                    if rev_actuals is not None:
                        w.write_array("validation/val_rev_actuals.npy", rev_actuals)
                        vblock["rev_actuals_shape"] = list(rev_actuals.shape)
                elif with_val_inputs is True:
                    raise FileNotFoundError(
                        "with_val_inputs=True but the training data could not be read; "
                        "pass data_dir=... or use with_val_inputs='auto'")
            if metrics:
                vblock["reported"] = [
                    {"member": i, **{k: float(c.get("val_correlations", {}).get(k))
                                     for k in ("flux_mean", "flux_median",
                                               "reversibility_mean", "reversibility_median")
                                     if c.get("val_correlations", {}).get(k) is not None}}
                    for i, c in enumerate(checkpoints)
                ]
                w.write_json("validation/member_metrics.json", [
                    {"member": i,
                     "flux_reaction_names": list(c.get("val_correlations", {})
                                                 .get("flux_reaction_names", [])),
                     "flux_r2_per_reaction": np.asarray(
                         c.get("val_correlations", {}).get("flux_r2_per_reaction", []),
                         dtype=float).tolist()}
                    for i, c in enumerate(checkpoints)
                ])
        manifest["validation"] = vblock

        # ── members ───────────────────────────────────────────────────────────
        for i, ckpt in enumerate(checkpoints):
            w.write_bytes(MEMBER_FMT.format(i), _member_blob(ckpt["model_state_dict"], layout))

    if verbose:
        raw = sum(os.path.getsize(p) for p in member_paths)
        size = out_path.stat().st_size
        print(f"  wrote {out_path}  ({size/1e6:.1f} MB from {raw/1e6:.1f} MB, "
              f"{raw/size:.1f}x smaller, precision={precision})")
    return out_path


def load_model(path, *, device=None):
    """Load a packaged ``.sfx`` model.

    Always returns an :class:`~sigmaflux.ai.inference.EnsembleSigmaFlux`, even for
    a one-member export, so callers have a single shape to handle. Needs no
    config YAML, run directory or COBRA JSON.
    """
    with SfxReader(path) as r:
        m = r.manifest
        vocabs = r.read_json("shared/vocabs.json")
        names = r.read_json("shared/names.json")
        token_index = [tuple(t) for t in r.read_json("shared/token_index.json")]

        nsd = {}
        for key in _NULL_SPACE_FLOAT + _NULL_SPACE_INT:
            entry = f"shared/null_space/{key}.npy"
            if r.has(entry):
                nsd[key] = r.read_array(entry)
        nsd.setdefault("fixed_idx", np.zeros(0, dtype=np.int32))

        flux_norm_stats = None
        if r.has("shared/flux_norm/mean.npy"):
            flux_norm_stats = {"mean": r.read_array("shared/flux_norm/mean.npy"),
                               "std": r.read_array("shared/flux_norm/std.npy")}

        spec = ModelSpec(
            model_config         = m["model"]["model_config"],
            vocabs               = {k: vocabs[k] for k in
                                    ("compound_vocab", "label_vocab", "tracer_vocab",
                                     "h_compound", "h_label", "h_tracer")},
            tracer_combo_vocab   = vocabs.get("tracer_combo_vocab", []),
            token_index          = token_index,
            token_experiment_ids = r.read_array("shared/token_experiment_ids.npy").tolist(),
            flux_names           = names["flux_names"],
            reversibility_names  = names["reversibility_names"],
            measured_fluxes      = names["measured_fluxes"],
            null_space_data      = nsd,
            flux_norm_stats      = flux_norm_stats,
            norm_stats           = r.read_json("shared/norm_stats.json", default={}),
            h_reaction           = m["model"]["h_reaction"],
            h_type               = m["model"]["h_type"],
            tracer_display_names = m.get("tracers", {}),
            batch_size           = m["model"].get("batch_size", 64),
        )

        regenerated = set(m.get("regenerated_tensors", ()))
        state_dicts = []
        for i in range(m["n_members"]):
            arrays = r.member_arrays(i)
            state_dicts.append({k: torch.from_numpy(v) for k, v in arrays.items()})

    model = EnsembleSigmaFlux.from_spec(
        spec, state_dicts, device=device,
        member_ids=[MEMBER_FMT.format(i) for i in range(m["n_members"])],
        strict=False,   # the regenerated tensors are deliberately absent
    )
    # strict=False would also hide a renamed key, so check the difference exactly.
    expected = set(model.members[0]._model.state_dict()) - set(state_dicts[0])
    if expected != regenerated:
        raise SfxFormatError(
            f"{path}: weights do not match the model. Missing but not listed as "
            f"regenerated: {sorted(expected - regenerated)}; listed but present: "
            f"{sorted(regenerated - expected)}"
        )

    model.manifest = m
    model.name = m.get("name")
    model.model_version = m.get("model_version")
    model.source_path = str(path)
    return model


def inspect_model(path):
    """Return a packaged model's manifest without loading any weights."""
    return read_manifest(path)


def validate_model(path, *, device=None, per_member=False):
    """Recompute validation R² from the export alone.

    Requires an export written with the validation inputs (``with_val_inputs``);
    reproduces the number training reported, in training token order.
    """
    import pandas as pd

    from .train import per_reaction_r2

    manifest = read_manifest(path)
    vblock = manifest.get("validation", {})
    if not vblock.get("inputs_included"):
        raise SfxFormatError(
            f"{path} has no stored validation inputs, so R² cannot be recomputed. "
            f"Re-export with with_val_inputs=True."
        )

    model = load_model(path, device=device)
    with SfxReader(path) as r:
        fractions = r.read_array("validation/val_fractions.npy")
        actuals = r.read_array("validation/val_actuals.npy")

    dev = model.device
    n_samples, n_tokens = fractions.shape
    spec = model.spec
    tables = spec.emb_tables()
    compound_idx = {c: i for i, c in enumerate(tables["compound_vocab"])}
    label_idx = {l: i for i, l in enumerate(tables["label_vocab"])}
    combo_idx = {t: i for i, t in enumerate(spec.tracer_combo_vocab)}

    cids = torch.tensor([[compound_idx.get(t[1], 0) for t in spec.token_index]] * n_samples,
                        dtype=torch.int64, device=dev)
    lids = torch.tensor([[label_idx.get(t[2], 0) for t in spec.token_index]] * n_samples,
                        dtype=torch.int64, device=dev)
    tids = torch.tensor([[combo_idx.get(t[0], 0) for t in spec.token_index]] * n_samples,
                        dtype=torch.int64, device=dev)
    fracs = torch.tensor(fractions, dtype=torch.float32, device=dev).unsqueeze(-1)
    targets = torch.tensor(actuals, dtype=torch.float32, device=dev)

    fixed = np.asarray(spec.null_space_data.get("fixed_idx", []), dtype=int)
    fixed_t = torch.as_tensor(fixed, dtype=torch.long, device=dev)
    measured = targets.index_select(1, fixed_t) if len(fixed) else None

    preds = []
    with torch.no_grad():
        for member in model.members:
            preds.append(member._model(cids, lids, tids, fracs,
                                       measured_fluxes=measured,
                                       measured_flux_ids=fixed_t if len(fixed) else None
                                       )[0].cpu().numpy())
    stacked = np.stack(preds)
    out = pd.DataFrame({"reaction": spec.flux_names,
                        "r2": per_reaction_r2(np.median(stacked, axis=0), actuals)})
    if per_member:
        for i, p in enumerate(stacked):
            out[f"r2_member_{i}"] = per_reaction_r2(p, actuals)
    return out
