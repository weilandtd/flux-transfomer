"""Rebuild a trained FluxDecoder — and the dataset it was scored on — from a checkpoint.

:mod:`sigmaflux.pipeline.train_ai_model` writes everything needed to reconstruct
both the model and its exact validation view: the null-space basis, the token
index, the vocabulary harmonics, the train/val split, and the paths to the
training HDF5 and the COBRA JSON. This module is the single reader of that
format, so analysis code never has to re-derive it.

Typical single-model use::

    ctx   = load_eval_context('run/model.pt')
    model = build_flux_decoder(ctx, device=resolve_device())
    (cids, lids, tids, fracs), targets = ctx.full_batch(device=model.fixed_idx_t.device)

An ensemble shares one context across members — the data, vocabularies, dataset
and targets are identical by construction, so only the weights differ::

    ctx    = load_eval_context(members[0])
    models = [build_flux_decoder(ctx, m) for m in members]

This module deliberately does *not* handle experimental MID input; that is
:class:`sigmaflux.ai.inference.SigmaFlux`, which builds a token skeleton from
user measurements rather than a dataset from training data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_data,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
)
from .model import FluxDecoder, FluxDecoderConfig

#: Use every sample column in the training data, ignoring the saved split.
ALL = "all"
#: Use the checkpoint's own ``val_indices`` (the split training scored on).
VAL = "val"

__all__ = [
    "ALL",
    "VAL",
    "EvalContext",
    "build_flux_decoder",
    "flux_decoder_config",
    "load_eval_context",
    "promote_norm_buffers",
    "resolve_device",
]


def resolve_device(device=None):
    """``'cuda'`` when available, else ``'cpu'``; pass a value through unchanged."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def flux_decoder_config(model_config):
    """Filter a saved ``model_config`` dict down to :class:`FluxDecoderConfig` fields.

    Checkpoints carry extra bookkeeping keys alongside the architecture; passing
    the raw dict to the dataclass would raise on the unexpected ones.
    """
    return FluxDecoderConfig(**{
        k: model_config[k]
        for k in FluxDecoderConfig.__dataclass_fields__
        if k in model_config
    })


def promote_norm_buffers(null_space_data, state_dict):
    """Fill in ``y_mean``/``y_std``/``rev_log_mean``/``rev_log_std`` from *state_dict* when
    ``null_space_data`` omits them, since ``load_state_dict`` cannot promote a ``None`` buffer
    to a tensor. Returns a new dict; the input is not mutated.

    Parameters
    ----------
    null_space_data : dict
        Null-space basis and normalization stats to fill in.
    state_dict : dict
        Model state dict to promote the missing buffers from.
    """
    nsd = dict(null_space_data)
    if nsd.get("y_mean") is None and state_dict.get("y_norm_mean") is not None:
        nsd["y_mean"] = state_dict["y_norm_mean"].numpy()
        nsd["y_std"] = state_dict["y_norm_std"].numpy()
    if nsd.get("rev_log_mean") is None and state_dict.get("rev_norm_mean") is not None:
        nsd["rev_log_mean"] = state_dict["rev_norm_mean"].numpy()
        nsd["rev_log_std"] = state_dict["rev_norm_std"].numpy()
    return nsd


def _with_flux_bounds(null_space_data, checkpoint):
    """Ensure ``flux_bounds`` is reachable from *null_space_data*.

    Training stores the bounds twice, at the checkpoint top level and inside
    ``null_space_data``, but only the top-level copy is unconditional in older runs.
    :class:`~sigmaflux.ai.flux_projection.FluxProjection` is built from
    ``null_space_data`` alone, so the fallback is what lets a projection model be rebuilt
    against a checkpoint written before the second copy existed.
    """
    if null_space_data.get("flux_bounds") is not None:
        return null_space_data
    bounds = checkpoint.get("flux_bounds")
    if bounds is None:
        return null_space_data
    return {**null_space_data, "flux_bounds": bounds}


def _flux_norm_stats(checkpoint, state_dict):
    """Per-reaction flux normalization, from the checkpoint or else the weights.

    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint dict, checked first for ``flux_norm_stats``.
    state_dict : dict
        Model state dict, checked as a fallback.
    """
    stats = checkpoint.get("flux_norm_stats")
    if stats is not None:
        return stats
    if state_dict.get("flux_norm_mean") is not None:
        return {
            "mean": state_dict["flux_norm_mean"].numpy(),
            "std": state_dict["flux_norm_std"].numpy(),
        }
    return None


def _as_checkpoint(checkpoint, map_location="cpu"):
    """Accept an already-loaded checkpoint dict or a path to one.

    Parameters
    ----------
    checkpoint : str or dict
        Path to a ``.pt`` file, or an already-loaded checkpoint dict.
    map_location : str
        Passed to :func:`torch.load` when *checkpoint* is a path.
    """
    if isinstance(checkpoint, dict):
        return checkpoint
    return torch.load(checkpoint, map_location=map_location, weights_only=False)


@dataclass
class EvalContext:
    """Everything needed to score a trained checkpoint, loaded once. Built by
    :func:`load_eval_context`; an ensemble reuses one instance across all its members.
    """

    ckpt: dict
    config: FluxDecoderConfig
    null_space_data: dict
    flux_names: list
    reversibility_names: list
    sample_indices: Optional[np.ndarray]
    labeling_df: pd.DataFrame
    flux_df: pd.DataFrame
    reversibility_df: pd.DataFrame
    cobra_model: Any
    stoichiometric_matrix: Any
    vocabs: dict
    tracer_combo_vocab: Any
    tracer_emb_table: Any
    token_experiment_ids: Any
    dataset: Any
    norm_stats: Optional[dict]
    data_dir: str
    model_json: str
    _batch_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def loader(self, batch_size=None, shuffle=False):
        """DataLoader over :attr:`dataset`; ``batch_size=None`` yields one full batch.

        Parameters
        ----------
        batch_size : int, optional
            Batch size; ``None`` yields the whole dataset as one batch.
        shuffle : bool
            Shuffle sample order.
        """
        if self.dataset is None:
            raise ValueError("EvalContext was built with build_dataset=False")
        return DataLoader(
            self.dataset,
            batch_size=len(self.dataset) if batch_size is None else batch_size,
            shuffle=shuffle,
        )

    def full_batch(self, device=None):
        """The whole dataset as one batch: ``((cids, lids, tids, fracs), targets)``, moved to
        *device* if given. The CPU batch is cached, so repeated calls do not re-tokenize.
        """
        if "cpu" not in self._batch_cache:
            self._batch_cache["cpu"] = next(iter(self.loader()))
        (cids, lids, tids, fracs), targets = self._batch_cache["cpu"]
        if device is None:
            return (cids, lids, tids, fracs), targets
        device = torch.device(device)
        moved = tuple(t.to(device) for t in (cids, lids, tids, fracs))
        return moved, targets.to(device)


def load_eval_context(
    checkpoint,
    *,
    samples=VAL,
    max_samples=None,
    data=None,
    data_dir=None,
    model_json=None,
    measured_fluxes=None,
    normalize=False,
    normalize_mids=False,
    build_dataset=True,
    map_location="cpu",
):
    """Load a checkpoint together with the data view it was scored on; returns an :class:`EvalContext`.

    Parameters
    ----------
    checkpoint : str or dict
        Path to a ``.pt`` file, or an already-loaded checkpoint dict.
    samples : {VAL, ALL} or array-like
        Which sample columns to build the dataset from: :data:`VAL` (default) uses
        the checkpoint's own ``val_indices``, :data:`ALL` uses every column, or
        pass an explicit index array.
    max_samples : int, optional
        Forwarded to :func:`~sigmaflux.ai.data.load_data`; caps the HDF5 *load*, so
        explicit ``samples`` indices must fall inside it.
    data : tuple, optional
        ``(labeling_df, flux_df, reversibility_df)`` to use instead of reading the HDF5.
    data_dir, model_json : str, optional
        Override the paths recorded in the checkpoint.
    measured_fluxes : list, optional
        Defaults to the checkpoint's ``measured_fluxes`` (the bypass tokens).
    normalize : bool
        Z-score the flux/reversibility targets.
    normalize_mids : bool
        Z-score the labeling (MID) fractions per token.
    build_dataset : bool
        Set ``False`` to skip dataset construction when only the model is wanted.
    map_location : str
        Passed to :func:`torch.load` when *checkpoint* is a path.
    """
    ckpt = _as_checkpoint(checkpoint, map_location=map_location)
    cfg_dict = ckpt["model_config"]
    state_dict = ckpt["model_state_dict"]

    resolved_data_dir = data_dir or ckpt.get("data_dir")
    resolved_model_json = model_json or ckpt.get("model_json")

    if data is not None:
        labeling_df, flux_df, reversibility_df = data
    else:
        labeling_df, flux_df, reversibility_df = load_data(
            resolved_data_dir, max_samples=max_samples
        )

    # Restrict and order the labeling rows to the exact token sequence trained on.
    token_index = ckpt.get("token_index")
    if token_index is not None:
        labeling_df = labeling_df.loc[token_index]

    if isinstance(samples, str):
        if samples == VAL:
            sample_indices = np.asarray(ckpt["val_indices"], dtype=int)
        elif samples == ALL:
            sample_indices = None
        else:
            raise ValueError(f"samples must be VAL, ALL or an index array, got {samples!r}")
    else:
        sample_indices = np.asarray(samples, dtype=int)

    _, base_tracer_names = create_tracer_mapping(labeling_df)
    S, cobra_model = load_stoichiometric_matrix(resolved_model_json)
    vocabs = build_radial_vocabs(
        cobra_model,
        labeling_df,
        base_tracer_names,
        h_compound=cfg_dict.get("d_comp", 32) // 2,
        h_label=cfg_dict.get("d_label", 10) // 2,
        h_tracer=cfg_dict.get("d_tracer", 2) // 2,
    )

    dataset = tracer_combo_vocab = tracer_emb_table = token_experiment_ids = None
    norm_stats = None
    if build_dataset:
        (
            dataset,
            norm_stats,
            token_experiment_ids,
            tracer_combo_vocab,
            tracer_emb_table,
        ) = prepare_id_dataset(
            labeling_df,
            flux_df,
            reversibility_df,
            vocabs=vocabs,
            train_indices=sample_indices,
            measured_fluxes=(
                ckpt.get("measured_fluxes") if measured_fluxes is None else measured_fluxes
            ),
            norm_stats_input=ckpt.get("norm_stats"),
            normalize=normalize,
            normalize_mids=normalize_mids,
        )

    flux_names = list(ckpt.get("flux_names", []))
    return EvalContext(
        ckpt=ckpt,
        config=flux_decoder_config(cfg_dict),
        null_space_data=_with_flux_bounds(
            promote_norm_buffers(ckpt["null_space_data"], state_dict), ckpt),
        flux_names=flux_names,
        reversibility_names=list(ckpt.get("reversibility_names", flux_names)),
        sample_indices=sample_indices,
        labeling_df=labeling_df,
        flux_df=flux_df,
        reversibility_df=reversibility_df,
        cobra_model=cobra_model,
        stoichiometric_matrix=S,
        vocabs=vocabs,
        tracer_combo_vocab=tracer_combo_vocab,
        tracer_emb_table=tracer_emb_table,
        token_experiment_ids=token_experiment_ids,
        dataset=dataset,
        norm_stats=norm_stats,
        data_dir=resolved_data_dir,
        model_json=resolved_model_json,
    )


def build_flux_decoder(ctx, checkpoint=None, *, device=None, strict=True):
    """Reconstruct one :class:`FluxDecoder`, in ``eval()`` mode on *device*, against *ctx*'s
    shared config/vocabularies/basis and *checkpoint*'s weights (path, dict, or ``None`` for
    ``ctx.ckpt``) — letting an ensemble load N weight sets against one shared context.

    Parameters
    ----------
    ctx : EvalContext
        Shared config/vocabularies/null-space basis to build the model against.
    checkpoint : str, dict, or None
        Member weights: a path, a loaded checkpoint dict, or ``None`` for ``ctx.ckpt``.
    device : optional
        Device to move the model to.
    strict : bool
        Passed to ``load_state_dict``.
    """
    ckpt = ctx.ckpt if checkpoint is None else _as_checkpoint(checkpoint)
    state_dict = ckpt["model_state_dict"]
    n_reactions = len(ctx.flux_names) or ctx.config.n_fluxes

    model = FluxDecoder(
        ctx.config,
        token_experiment_ids=state_dict["token_experiment_ids"],
        compound_emb_table=ctx.vocabs["compound_emb_table"],
        label_emb_table=ctx.vocabs["label_emb_table"],
        tracer_emb_table=ctx.tracer_emb_table,
        reaction_emb_table=radial_embedding_table(n_reactions, ckpt["h_reaction"]),
        type_emb_table=radial_embedding_table(2, ckpt["h_type"]),
        null_space_data=ctx.null_space_data,
        flux_norm_stats=_flux_norm_stats(ckpt, state_dict),
    )
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    model.to(resolve_device(device))
    return model
