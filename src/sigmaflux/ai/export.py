"""Package a trained SigmaFlux model into one self-contained ``.sfx`` file.

Contrast with :mod:`sigmaflux.ai.checkpoint`, which reads the **training
checkpoint** — a ``.pt`` that only makes sense next to its run directory, config
YAML, COBRA JSON and training HDF5. This module reads and writes the **exported
model**: inference-only, self-describing, and loadable with nothing but the file.

    from sigmaflux.ai import export_model, load_model

    export_model("run/ensemble", name="i_orientalis", model_version="1.0.0")
    sf = load_model("i_orientalis-v1.2.0.sfx")
    sf.predict(df_L, df_Vm)

What is stored
--------------
Once, shared by every member: the vocabularies, null-space basis, token index,
output names, model config, flux normalization, the resolved tracer-name map,
the prepared COBRA network, and (optionally) the validation split. Per member:
**only the trained weights**.

The network is the MFA-prepared model the run was trained on — the one whose
reaction order matches ``flux_names``, not the annotated model it came from. It
costs ~30 KB packed and is what lets a consumer check ``S @ v ≈ 0``, look up a
reaction's metabolites or subsystem, and draw a map, none of which the names
alone support. :func:`read_stoichiometry` returns it already aligned to the
model's outputs.

Alongside it, ``shared/metabolites.json`` lists every network metabolite with its
name and whether it is *observable* -- a metabolite the model takes MIDs for, i.e.
one in the token index. :func:`read_metabolites` returns it as a table. Given the
training config, names are refreshed from its ``model_file``, so a run prepared
before a model's names were fixed still exports the current ones.

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
from dataclasses import replace
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
from .checkpoint import promote_norm_buffers
from .inference import EnsembleSigmaFlux, ModelSpec, SigmaFlux

__all__ = [
    "SFX_SUFFIX",
    "SfxError",
    "SfxFormatError",
    "SfxIntegrityError",
    "SfxVersionError",
    "export_model",
    "inspect_model",
    "load_model",
    "read_cobra_model",
    "read_metabolites",
    "read_stoichiometry",
    "validate_model",
]

#: Entry holding the prepared COBRA network, as written by ``sigmaflux-sample-flux``.
NETWORK_ENTRY = "shared/cobra_model.json"
#: Entry listing every network metabolite: id, name, and whether it is observable.
METABOLITES_ENTRY = "shared/metabolites.json"
#: Filename of the prepared network inside a training run directory.
PREPARED_MODEL_JSON = "model_prepared_for_mfa.json"

#: Weight tensors above this magnitude fall back to float32 (fp16 max is 65504).
_FP16_SAFE_MAX = 6.0e4

#: state_dict keys the loader regenerates rather than storing. This is the
#: *maximal* set — which of them a given model actually has depends on its
#: config (a model trained without flux normalization has no flux_norm_* buffer),
#: so the manifest records the subset that was really dropped.
_REGENERATED = (
    "compound_emb.weight", "label_emb.weight", "tracer_emb.weight",
    "reaction_emb.weight", "_token_type_emb.weight",
    "Z_full", "flux_offset", "token_experiment_ids", "fixed_idx_t",
    "flux_norm_mean", "flux_norm_std",
    "y_norm_mean", "y_norm_std", "rev_norm_mean", "rev_norm_std",
)

#: Regenerated buffers added after the shipped bundles were written. Each defaults to a
#: neutral value -- ``flux_offset`` to zeros, which is exactly what a model trained with
#: no pinned basis directions had implicitly -- so a manifest that predates one is still
#: a match. Only these are forgiven, so a *renamed* key is still caught.
_OPTIONAL_REGENERATED = frozenset({"flux_offset"})

#: Keys that must agree across ensemble members.
_SHARED_KEYS = (
    "model_config", "flux_names", "reversibility_names", "measured_fluxes",
    "token_index", "h_reaction", "h_type", "val_indices",
)

_NULL_SPACE_FLOAT = ("Z_free", "flux_offset", "y_mean", "y_std", "rev_log_mean",
                     "rev_log_std", "obs_weights")
_NULL_SPACE_INT = ("free_idx", "fixed_idx", "rev_variable_idx")


# ── Source discovery ──────────────────────────────────────────────────────────

def _discover_members(source):
    """Resolve *source* to an ordered list of member checkpoint paths: an ``ensemble/`` directory,
    a run directory containing one, a single ``.pt``, or an explicit sequence of ``.pt`` paths.

    Parameters
    ----------
    source : path or sequence of paths
        An ``ensemble/`` directory, a run directory containing one, a single ``.pt``,
        or an explicit sequence of ``.pt`` paths.
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
    """Refuse to package members that were not trained as one ensemble.

    Parameters
    ----------
    checkpoints : list of dict
        Loaded checkpoint dicts for every member.
    paths : list of str
        Matching checkpoint file paths, for error messages.

    Returns ``True`` when every member shares one null-space basis; a seed-varied
    ICA basis legitimately differs per member, so that case is reported (not rejected)
    and the basis is then stored per member.
    """
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

    ref = np.asarray(first["null_space_data"]["Z_free"])
    divergent = []
    for ckpt, path in zip(checkpoints[1:], paths[1:]):
        other = np.asarray(ckpt["null_space_data"]["Z_free"])
        if other.shape != ref.shape:
            raise ValueError(
                f"ensemble members disagree on the null-space *shape*: "
                f"{os.path.basename(paths[0])} has {ref.shape}, "
                f"{os.path.basename(path)} has {other.shape}"
            )
        if not np.array_equal(other, ref):
            divergent.append(os.path.basename(path))
    if divergent:
        print(f"  note: {len(divergent)} member(s) use a different null-space basis "
              f"({', '.join(divergent[:4])}{'...' if len(divergent) > 4 else ''}); "
              f"storing the basis per member")
    return not divergent


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
    """Describe the trained tensors and their byte offsets within a member blob.

    Parameters
    ----------
    state_dict : dict
        Model state dict to lay out.
    precision : {"fp16", "fp32"}
        Storage precision; finite fp32 tensors within range are downcast to fp16.
    """
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
    """Concatenate a member's trained tensors into one little-endian buffer.

    Parameters
    ----------
    state_dict : dict
        Model state dict holding the tensors.
    layout : list of dict
        Tensor layout from :func:`_tensor_layout`.
    """
    chunks = []
    for spec in layout:
        arr = state_dict[spec["name"]].detach().cpu().numpy()
        chunks.append(np.ascontiguousarray(arr.astype(spec["stored_dtype"], copy=False)).tobytes())
    return b"".join(chunks)


# ── Prepared network ──────────────────────────────────────────────────────────

def _find_network_json(ckpt, model_json=None, data_dir=None, config=None, root=None):
    """Locate the prepared COBRA network for a checkpoint. The run directory's
    ``model_prepared_for_mfa.json`` is preferred over the checkpoint's ``model_json``, since
    only the prepared model's reaction order and set match ``flux_names``.

    Parameters
    ----------
    ckpt : dict
        Loaded checkpoint dict.
    model_json : str, optional
        Explicit network path, tried first.
    data_dir : str, optional
        Run directory to look for the prepared network in.
    config : path, optional
        Training YAML, to resolve ``model_file`` as a last resort.
    root : path, optional
        Root for resolving *config*'s relative paths.
    """
    candidates = [model_json]
    run_dir = data_dir or ckpt.get("data_dir")
    if run_dir:
        candidates.append(os.path.join(run_dir, PREPARED_MODEL_JSON))
    candidates.append(ckpt.get("model_json"))
    if config is not None:
        from ..pipeline._paths import load_config, resolve_path
        cfg, cfg_root = load_config(config, root)
        candidates.append(resolve_path(cfg.get("model_file"), cfg_root))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _read_network(path, flux_names):
    """Read a COBRA JSON; returns ``(payload, block, doc)`` where *payload* is the raw file bytes,
    *block* is the manifest entry and *doc* the parsed network. ``block["covers_flux_names"]`` decides usability: without
    every predicted reaction present, a consumer cannot build ``S`` for the prediction vector.

    Parameters
    ----------
    path : str
        Path to the COBRA network JSON.
    flux_names : list of str
        Predicted reaction ids, checked for coverage against the network.
    """
    with open(path, "rb") as fh:
        payload = fh.read()
    doc = json.loads(payload.decode("utf-8"))
    reaction_ids = [r["id"] for r in doc.get("reactions", [])]

    return payload, {
        "included": True,
        "source": os.path.basename(path),
        "sha256": _sfx.sha256_bytes(payload),
        "n_reactions": len(reaction_ids),
        "n_metabolites": len(doc.get("metabolites", [])),
        "covers_flux_names": set(flux_names) <= set(reaction_ids),
        # False is fine — read_stoichiometry reorders — but it flags an export
        # built from a model other than the one training actually used.
        "reaction_order_matches": reaction_ids == list(flux_names),
    }, doc


def _refresh_metabolite_names(doc, config, root):
    """Overwrite *doc*'s metabolite names, by id, with those of *config*'s ``model_file``.
    Returns the model file's basename, or ``None`` when *config* names no readable model.

    Parameters
    ----------
    doc : dict
        Parsed COBRA network, modified in place.
    config : path
        Training YAML whose ``model_file`` is the annotated source model.
    root : path, optional
        Root for resolving *config*'s relative paths.
    """
    from ..pipeline._paths import load_config, resolve_path
    cfg, cfg_root = load_config(config, root)
    path = resolve_path(cfg.get("model_file"), cfg_root)
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        names = {m["id"]: m.get("name", "") for m in json.load(fh).get("metabolites", [])}
    for met in doc.get("metabolites", []):
        if met["id"] in names:
            met["name"] = names[met["id"]]
    return os.path.basename(path)


def _metabolite_rows(doc, token_index):
    """One ``{metabolite_id, description, observable}`` record per network metabolite, in network
    order; observable means the metabolite appears in the token index.

    Parameters
    ----------
    doc : dict
        Parsed COBRA network.
    token_index : list of tuple
        ``(Tracer, Metabolite, Isotopomer)`` tokens the model was trained on.
    """
    observed = {t[1] for t in token_index}
    met_ids = [m["id"] for m in doc.get("metabolites", [])]
    missing = sorted(observed - set(met_ids))
    if missing:
        raise ValueError(f"token metabolites missing from the network: {missing}")
    return [{"metabolite_id": m["id"], "description": m.get("name", ""),
             "observable": m["id"] in observed}
            for m in doc.get("metabolites", [])]


# ── Validation payload ────────────────────────────────────────────────────────

def _read_val_data(ckpt, spec, val_indices, data_dir=None, max_samples=None):
    """The validation split's MID inputs and targets, in training token order — the only step
    that touches the training HDF5. Returns ``(fractions, flux_actuals, rev_actuals)``, or
    ``(None, None, None)`` when the training data is unreachable.

    Parameters
    ----------
    ckpt : dict
        Loaded checkpoint dict.
    spec : ModelSpec
        Model spec, for the token index and output names.
    val_indices : array-like
        Sample columns making up the validation split.
    data_dir : str, optional
        Override the training data directory recorded in *ckpt*.
    max_samples : int, optional
        Cap the number of validation samples read.
    """
    from .data import load_data
    resolved = data_dir or ckpt.get("data_dir")
    if not resolved or not os.path.isdir(resolved):
        print(f"  validation inputs: skipped (training data not found at {resolved!r})")
        return None, None, None
    try:
        labeling_df, flux_df, rev_df = load_data(resolved)
    except (OSError, FileNotFoundError, KeyError) as exc:
        print(f"  validation inputs: skipped ({exc})")
        return None, None, None

    idx = np.asarray(val_indices, dtype=int)
    if max_samples is not None:
        idx = idx[:max_samples]

    labeling_df = labeling_df.loc[[tuple(t) for t in spec.token_index]]
    fractions = labeling_df.values[:, idx].T.astype(np.float32)

    flux_names = list(spec.flux_names)
    flux_actuals = (flux_df.loc[flux_names].values[:, idx].T.astype(np.float32)
                    if flux_names and set(flux_names) <= set(flux_df.index) else None)

    rev_names = list(spec.reversibility_names)
    rev_actuals = (rev_df.loc[rev_names].values[:, idx].T.astype(np.float32)
                   if rev_names and set(rev_names) <= set(rev_df.index) else None)
    return fractions, flux_actuals, rev_actuals


# ── Public API ────────────────────────────────────────────────────────────────

def export_model(source, out_path=None, *, name=None, model_version,
                 precision="fp16", config=None, root=None, description=None,
                 with_validation=True, with_val_inputs="auto", with_network="auto",
                 data_dir=None, model_json=None, max_members=None, max_val_samples=None,
                 metrics=True, overwrite=False, verbose=True):
    """Package trained checkpoints into a single ``.sfx`` model; returns the written path.

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
    root : path, optional
        Root for resolving *config*'s relative paths.
    description : str, optional
        Free-text description recorded in the manifest.
    with_validation, with_val_inputs :
        Store the validation split, and (``"auto"``/True/False) the MID inputs
        needed to recompute R² from the export alone.
    with_network : {"auto", True, False}
        Store the prepared COBRA network (default ``"auto"``: embed it when found
        and covering every predicted reaction, else warn and carry on). ``True``
        makes a missing/mismatched network an error; ``False`` skips it.
    data_dir, model_json : path, optional
        Override the network/training-data paths recorded in the checkpoint.
    max_members : int, optional
        Cap the number of ensemble members packaged.
    max_val_samples : int, optional
        Cap the number of validation samples stored.
    metrics : bool
        Record each member's reported validation R² in the manifest.
    overwrite : bool
        Allow replacing an existing file at *out_path*.
    verbose : bool
        Print progress and the final artifact size.
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
    shared_null_space = _check_members_agree(checkpoints, member_paths)

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
    # Record the commit of the *code* that built this, not of the checkpoint's
    # directory: run dirs usually live on scratch outside any repo, which would
    # silently leave the commit null exactly when provenance matters most.
    git_commit, git_dirty = _git_state(Path(__file__).parent)
    ck0 = checkpoints[0]

    network_payload, network, network_doc = None, {"included": False}, None
    if with_network is not False:
        found = _find_network_json(ck0, model_json=model_json, data_dir=data_dir,
                                   config=config, root=root)
        if found is None:
            if with_network is True:
                raise FileNotFoundError(
                    f"with_network=True but no COBRA network was found; pass "
                    f"model_json=..., or data_dir=... containing {PREPARED_MODEL_JSON}"
                )
            print("  network: skipped (no prepared COBRA JSON found)")
        else:
            network_payload, candidate, network_doc = _read_network(found, spec.flux_names)
            if candidate["covers_flux_names"]:
                network = candidate
            elif with_network is True:
                raise ValueError(
                    f"{found} is missing reactions this model predicts, so it cannot "
                    f"describe its output: it has {candidate['n_reactions']} reactions "
                    f"and the model predicts {len(spec.flux_names)}. Point model_json "
                    f"at the run's {PREPARED_MODEL_JSON}."
                )
            else:
                network_payload = network_doc = None
                print(f"  network: skipped ({os.path.basename(found)} does not cover "
                      f"every predicted reaction)")

    metabolite_rows, metabolites = None, {"included": False}
    if network_doc is not None:
        names_source = (_refresh_metabolite_names(network_doc, config, root)
                        if config is not None else None)
        if names_source is not None:
            network_payload = json.dumps(network_doc).encode("utf-8")
            network["sha256"] = _sfx.sha256_bytes(network_payload)
        network["names_source"] = names_source or "network"
        metabolite_rows = _metabolite_rows(network_doc, spec.token_index)
        metabolites = {"included": True, "n_metabolites": len(metabolite_rows),
                       "n_observable": sum(r["observable"] for r in metabolite_rows)}

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
            # Basenames only, never absolute paths: an exported model is shared,
            # and the producer's directory layout is both useless to a consumer
            # and identifying (usernames, cluster project names, scratch mounts).
            "source": os.path.basename(str(source).rstrip("/")) if isinstance(
                source, (str, os.PathLike)) else f"{len(member_paths)} checkpoints",
            "member_sources": [os.path.basename(p) for p in member_paths],
            "trained_utc": datetime.datetime.fromtimestamp(
                max(os.path.getmtime(p) for p in member_paths),
                datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_config": str(config) if config else None,
            "config_sha256": _sha256_file(config),
            "model_config_sha256": _sfx.sha256_bytes(
                json.dumps(spec.model_config, sort_keys=True).encode()),
            "training_run": os.path.basename(str(ck0.get("data_dir") or "").rstrip("/")) or None,
            "training_model_json": os.path.basename(str(ck0.get("model_json") or "")) or None,
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
        "network": network,
        "metabolites": metabolites,
        "tensors": layout,
        "member_nbytes": member_nbytes,
        "regenerated_tensors": [k for k in _REGENERATED
                                if k in checkpoints[0]["model_state_dict"]],
        "shared_null_space": shared_null_space,
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
        if spec.flux_bounds:
            # float64: the projection's ratio test divides by the slack to a bound, and
            # float32 rounding there can put a state marginally outside the bound it was
            # just projected onto.
            for key in ("lb", "ub"):
                w.write_array(f"shared/flux_bounds/{key}.npy",
                              np.asarray(spec.flux_bounds[key], dtype=np.float64))
        if spec.flux_norm_stats:
            for key in ("mean", "std"):
                w.write_array(f"shared/flux_norm/{key}.npy",
                              np.asarray(spec.flux_norm_stats[key], dtype=np.float32))
        if network_payload is not None:
            w.write_bytes(NETWORK_ENTRY, network_payload)
        if metabolite_rows is not None:
            w.write_json(METABOLITES_ENTRY, metabolite_rows)

        # ── validation ────────────────────────────────────────────────────────
        vblock = {"included": False, "n_val": 0, "inputs_included": False,
                  "token_order": "training"}
        if with_validation and len(val_indices):
            fractions = flux_actuals = rev_actuals = None
            if with_val_inputs is not False:
                fractions, flux_actuals, rev_actuals = _read_val_data(
                    ck0, spec, val_indices, data_dir=data_dir, max_samples=max_val_samples)
                if fractions is None and with_val_inputs is True:
                    raise FileNotFoundError(
                        "with_val_inputs=True but the training data could not be read; "
                        "pass data_dir=... or use with_val_inputs='auto'")

            # Fall back to the targets training recorded when the HDF5 is out of reach.
            if flux_actuals is None:
                recorded = (ck0.get("val_correlations") or {}).get("val_actuals")
                if recorded is not None:
                    flux_actuals = np.asarray(recorded, dtype=np.float32)[:len(val_indices)]

            if flux_actuals is not None:
                if max_val_samples is not None:
                    val_indices = val_indices[:max_val_samples]
                w.write_array("validation/val_indices.npy", val_indices)
                w.write_array("validation/val_actuals.npy", flux_actuals)
                vblock.update(included=True, n_val=int(len(val_indices)),
                              actuals_shape=list(flux_actuals.shape))
                if fractions is not None:
                    w.write_array("validation/val_fractions.npy", fractions)
                    vblock["inputs_included"] = True
                    vblock["fractions_shape"] = list(fractions.shape)
                if rev_actuals is not None:
                    w.write_array("validation/val_rev_actuals.npy", rev_actuals)
                    vblock["rev_actuals_shape"] = list(rev_actuals.shape)
            if metrics and vblock["included"]:
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
            if not shared_null_space:
                # A seed-varied ICA basis differs per member; the decoded fluxes
                # are physical either way, so keep each member's own basis.
                member_nsd = promote_norm_buffers(ckpt["null_space_data"],
                                                  ckpt["model_state_dict"])
                for key in _NULL_SPACE_FLOAT:
                    if member_nsd.get(key) is not None:
                        w.write_array(f"members/{i:03d}_null_space/{key}.npy",
                                      np.asarray(member_nsd[key], dtype=np.float32))

    if verbose:
        raw = sum(os.path.getsize(p) for p in member_paths)
        size = out_path.stat().st_size
        print(f"  wrote {out_path}  ({size/1e6:.1f} MB from {raw/1e6:.1f} MB, "
              f"{raw/size:.1f}x smaller, precision={precision})")
    return out_path


def load_model(path, *, device=None):
    """Load a packaged ``.sfx`` model. Always returns an
    :class:`~sigmaflux.ai.inference.EnsembleSigmaFlux`, even for a one-member export, so callers
    have a single shape to handle. Needs no config YAML, run directory, or COBRA JSON.

    Parameters
    ----------
    path : str
        Path to the ``.sfx`` file.
    device : optional
        Device to load member weights onto.
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

        # Bounds are required to predict. An export written before they were stored can
        # still supply them from the prepared network it embeds, which is the same file
        # they were read off in the first place.
        flux_bounds = None
        if r.has("shared/flux_bounds/lb.npy"):
            flux_bounds = {"lb": r.read_array("shared/flux_bounds/lb.npy"),
                           "ub": r.read_array("shared/flux_bounds/ub.npy")}
        elif r.has(NETWORK_ENTRY):
            from cobra.io.json import from_json

            from .inference import bounds_from_cobra
            flux_bounds = bounds_from_cobra(
                from_json(r.read_bytes(NETWORK_ENTRY).decode("utf-8")),
                names["flux_names"], source=f"{path}'s embedded network")

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
            flux_bounds          = flux_bounds,
            flux_norm_stats      = flux_norm_stats,
            norm_stats           = r.read_json("shared/norm_stats.json", default={}),
            h_reaction           = m["model"]["h_reaction"],
            h_type               = m["model"]["h_type"],
            tracer_display_names = m.get("tracers", {}),
            batch_size           = m["model"].get("batch_size", 64),
        )

        regenerated = set(m.get("regenerated_tensors", ()))
        state_dicts, specs = [], []
        for i in range(m["n_members"]):
            state_dicts.append({k: torch.from_numpy(v) for k, v in r.member_arrays(i).items()})
            if m.get("shared_null_space", True):
                specs.append(spec)
            else:
                member_nsd = dict(nsd)
                for key in _NULL_SPACE_FLOAT:
                    entry = f"members/{i:03d}_null_space/{key}.npy"
                    if r.has(entry):
                        member_nsd[key] = r.read_array(entry)
                # Same spec but this member's basis; the embedding-table cache is
                # shared because it does not depend on the null space.
                specs.append(replace(spec, null_space_data=member_nsd, _cache=spec._cache))

    members = [SigmaFlux.from_spec(s, sd, device=device, strict=False)
               for s, sd in zip(specs, state_dicts)]
    model = EnsembleSigmaFlux.__new__(EnsembleSigmaFlux)
    model.member_paths = [MEMBER_FMT.format(i) for i in range(m["n_members"])]
    model._adopt(members, split_source=None)
    # strict=False would also hide a renamed key, so check the difference exactly.
    expected = set(model.members[0]._model.state_dict()) - set(state_dicts[0])
    if expected - _OPTIONAL_REGENERATED != regenerated - _OPTIONAL_REGENERATED:
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


def read_cobra_model(path):
    """The prepared COBRA network stored in *path*, as a :class:`cobra.Model`; needs no run
    directory. Raises :class:`SfxFormatError` if the model was exported without one — check
    ``inspect_model(path)["network"]["included"]`` first to branch without catching.
    """
    from cobra.io.json import from_json

    with SfxReader(path) as r:
        if not r.has(NETWORK_ENTRY):
            raise SfxFormatError(
                f"{path} carries no COBRA network. Re-export it with with_network=True."
            )
        return from_json(r.read_bytes(NETWORK_ENTRY).decode("utf-8"))


def read_metabolites(path):
    """Every network metabolite of *path* as a DataFrame with columns ``metabolite_id``,
    ``description`` and ``observable`` (the model takes MIDs for it). Raises
    :class:`SfxFormatError` if the model was exported without the list.
    """
    import pandas as pd

    with SfxReader(path) as r:
        if not r.has(METABOLITES_ENTRY):
            raise SfxFormatError(
                f"{path} carries no metabolite list. Re-export it with a current sigmaflux "
                f"and with_network=True."
            )
        return pd.DataFrame(r.read_json(METABOLITES_ENTRY),
                            columns=["metabolite_id", "description", "observable"])


def read_stoichiometry(path):
    """``(S, metabolite_ids, cobra_model)`` with ``S`` columns reindexed to ``flux_names`` order,
    so a consumer can multiply straight into a prediction (``S @ predicted_fluxes ≈ 0``).
    """
    from cobra.util.array import create_stoichiometric_matrix

    model = read_cobra_model(path)
    with SfxReader(path) as r:
        flux_names = r.read_json("shared/names.json")["flux_names"]

    S = create_stoichiometric_matrix(model, array_type="dense")
    order = {rxn.id: i for i, rxn in enumerate(model.reactions)}
    missing = [name for name in flux_names if name not in order]
    if missing:
        raise SfxFormatError(
            f"{path}: the stored network is missing {len(missing)} predicted "
            f"reaction(s), e.g. {missing[:5]}"
        )
    return (S[:, [order[name] for name in flux_names]],
            [met.id for met in model.metabolites],
            model)


def validate_model(path, *, device=None, per_member=False):
    """Recompute validation R² from the export alone (requires ``with_val_inputs`` at export time);
    reproduces the number training reported, in training token order.

    Parameters
    ----------
    path : str
        Path to the ``.sfx`` file.
    device : optional
        Device to run inference on.
    per_member : bool
        Also return each ensemble member's per-reaction R² as extra columns.
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
