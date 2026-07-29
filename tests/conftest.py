"""Shared fixtures for the AI tests.

``toy_run`` builds a complete but minimal *run directory* on the toy one-carbon
model: the three training HDF5 batches, a trained-shaped checkpoint (random
weights, correct stoichiometric structure), and a two-member ensemble. It is
what a real ``sigmaflux-train`` run produces, minus the training.

Both :mod:`test_ai_inference` (which needs only the checkpoint) and
:mod:`test_checkpoint` (which needs the HDF5 data and the split as well) build
on it, so the checkpoint schema lives in exactly one place.

``_no_repo_root_litter`` is an autouse guard: tests must write to ``tmp_path``,
never to the working tree.
"""

import os
import subprocess
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from sigmaflux.ai.data import (
    build_radial_vocabs,
    create_tracer_mapping,
    load_stoichiometric_matrix,
    prepare_id_dataset,
    radial_embedding_table,
)
from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
from sigmaflux.ai.utils import compute_sparse_nullspace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY_MODEL_JSON = os.path.join(REPO_ROOT, "models", "toy", "one_carbon_annotated.json")
TOY_CONFIG_YAML = os.path.join(REPO_ROOT, "data", "test", "toy_config.yaml")

#: Internal tracer ID: f"{tracer_id}_{labeling}_{enrichment}" (matches toy_config.yaml)
TRACER_ID = "Ser_in_111_0.1"

N_SAMPLES = 30
N_VAL = 8
H_REACTION = 4      # d_reaction = 8, fits in d_entity_lab = 14
H_TYPE = 2          # d_type = 4

#: Generated-artifact suffixes a test might leave in the working tree. Only these
#: are swept, so an unrelated file someone creates mid-run is never touched.
_ARTIFACT_SUFFIXES = (".pkl", ".pt", ".sfx", ".h5", ".npz", ".npy", ".csv", ".out")


@pytest.fixture(scope="session", autouse=True)
def _no_repo_root_litter():
    """Delete generated artifacts a test drops in the repo root, and say so.

    Tests should write only under ``tmp_path``. Some library paths still default
    to a relative filename — ``MFAModel.prepare(with_backward=True)`` reads
    ``dAs.pkl``/``dBs.pkl`` from the current directory, for instance — so a test
    that grows such a call would silently start littering the working tree.

    The sweep is deliberately narrow: repo root only, non-recursive, only files
    that did not exist when the session started, and only known artifact
    suffixes. It warns rather than deleting silently, because a test writing to
    the working tree is a bug in the test, not something to paper over.
    """
    def snapshot():
        return {e.name: e.stat().st_mtime
                for e in os.scandir(REPO_ROOT)
                if e.is_file() and e.name.endswith(_ARTIFACT_SUFFIXES)}

    # Never touch a file git tracks: a tracked artifact is somebody's deliberate
    # choice, not litter, and deleting one would be a silent, damaging surprise.
    try:
        tracked = set(subprocess.run(
            ["git", "-C", REPO_ROOT, "ls-files", "-z"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split("\0"))
    except (OSError, subprocess.SubprocessError):
        tracked = set()

    before = snapshot()
    yield
    strays = [name for name, mtime in snapshot().items()
              if before.get(name) != mtime and name not in tracked]
    for name in sorted(strays):
        os.remove(os.path.join(REPO_ROOT, name))
    if strays:
        warnings.warn(
            f"tests wrote {len(strays)} artifact(s) into the repo root and they were "
            f"removed: {', '.join(sorted(strays))}. Write to tmp_path instead.",
            stacklevel=1,
        )


@pytest.fixture(scope="session")
def toy_run(tmp_path_factory):
    """A complete toy run directory.

    Returns a dict with ``dir``, ``checkpoint``, ``ensemble_dir``,
    ``member_paths``, ``config``, ``model_json``, ``flux_names``,
    ``measured_fluxes``, ``val_indices`` and ``n_fluxes``.
    """
    run_dir = tmp_path_factory.mktemp("toy_run")

    S, cobra_model = load_stoichiometric_matrix(TOY_MODEL_JSON)
    flux_names = [r.id for r in cobra_model.reactions]
    n_fluxes = len(flux_names)

    measured_fluxes = ["EX_Ser_in", "EX_Gly_src"]
    fixed_idx = [flux_names.index(r) for r in measured_fluxes]
    free_idx = [i for i in range(n_fluxes) if i not in set(fixed_idx)]

    # ── Synthetic training data: 2 metabolites × isotopomers × 1 tracer ────────
    rows = [
        (TRACER_ID, "Ser", "M+1"), (TRACER_ID, "Ser", "M+2"), (TRACER_ID, "Ser", "M+3"),
        (TRACER_ID, "Gly", "M+1"), (TRACER_ID, "Gly", "M+2"),
    ]
    midx = pd.MultiIndex.from_tuples(rows, names=["Tracer", "Metabolite", "Isotopomer"])
    rng = np.random.default_rng(42)
    ser_fracs = rng.dirichlet(np.ones(3), N_SAMPLES).T.astype(np.float32)
    gly_fracs = rng.dirichlet(np.ones(2), N_SAMPLES).T.astype(np.float32)
    labeling_df = pd.DataFrame(
        np.vstack([ser_fracs, gly_fracs]), index=midx, columns=list(range(N_SAMPLES))
    )
    flux_df = pd.DataFrame(
        np.abs(rng.normal(2.0, 1.0, (n_fluxes, N_SAMPLES))).astype(np.float32),
        index=flux_names, columns=list(range(N_SAMPLES)),
    )
    rev_df = pd.DataFrame(
        np.clip(rng.normal(0.5, 0.2, (n_fluxes, N_SAMPLES)), 0, 1).astype(np.float32),
        index=flux_names, columns=list(range(N_SAMPLES)),
    )

    # Written in the batch layout load_data() discovers: <base>_<start>_<end>.h5
    suffix = f"_0_{N_SAMPLES}"
    labeling_df.to_hdf(run_dir / f"wb_model_labeling_data{suffix}.h5", key="labeling", mode="w")
    flux_df.to_hdf(run_dir / f"wb_model_flux_data{suffix}.h5", key="fluxes", mode="w")
    rev_df.to_hdf(run_dir / f"wb_model_reversibility_data{suffix}.h5", key="reversibility", mode="w")

    # ── Vocabs and dataset (no bypass tokens — matches inference.py) ──────────
    _, base_tracer_names = create_tracer_mapping(labeling_df)
    vocabs = build_radial_vocabs(cobra_model, labeling_df, base_tracer_names)
    (dataset, norm_stats, token_experiment_ids,
     tracer_combo_vocab, tracer_combo_emb_table) = prepare_id_dataset(
        labeling_df, flux_df, rev_df,
        vocabs=vocabs,
        train_indices=list(range(N_SAMPLES)),
        normalize=False,
        measured_fluxes=None,
    )

    # ── Null-space basis and y normalisation ──────────────────────────────────
    Z_full = compute_sparse_nullspace(S)
    n_free = Z_full.shape[1]
    y_train = dataset.targets[:, :n_fluxes] @ torch.tensor(Z_full, dtype=torch.float32)
    null_space_data = dict(
        Z_free=Z_full,
        free_idx=free_idx,
        fixed_idx=fixed_idx,
        y_mean=y_train.mean(dim=0).numpy(),
        y_std=y_train.std(dim=0).clamp_min(1e-6).numpy(),
        obs_weights=np.ones(n_free, dtype=np.float32),
    )

    cfg = FluxDecoderConfig(
        d_model=16, n_heads=1, n_layers=1, dropout=0.0,
        n_compounds=len(vocabs["compound_vocab"]),
        n_labels=len(vocabs["label_vocab"]),
        n_tracers=len(tracer_combo_vocab),
        d_comp=vocabs["h_compound"] * 2,
        d_label=vocabs["h_label"] * 2,
        d_tracer=vocabs["h_tracer"] * 2,
        n_fluxes=n_fluxes, n_rev=n_fluxes, n_free=n_free,
        d_reaction=H_REACTION * 2, d_type=H_TYPE * 2,
    )

    def build(seed):
        torch.manual_seed(seed)
        return FluxDecoder(
            cfg,
            token_experiment_ids=token_experiment_ids,
            compound_emb_table=vocabs["compound_emb_table"],
            label_emb_table=vocabs["label_emb_table"],
            tracer_emb_table=tracer_combo_emb_table,
            null_space_data=null_space_data,
            reaction_emb_table=radial_embedding_table(n_fluxes, H_REACTION),
            type_emb_table=radial_embedding_table(2, H_TYPE),
        )

    all_idx = list(range(N_SAMPLES))
    val_indices = all_idx[-N_VAL:]
    train_indices = all_idx[:-N_VAL]

    def checkpoint_for(model):
        return {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(cfg),
            "vocabs": {k: v for k, v in vocabs.items()
                       if k not in ("compound_emb_table", "label_emb_table", "tracer_emb_table")},
            "tracer_combo_vocab": tracer_combo_vocab,
            "token_experiment_ids": token_experiment_ids.tolist(),
            "token_index": list(labeling_df.index),
            "norm_stats": norm_stats,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "log": {"train_loss": [], "val_loss": [], "grad_norm": []},
            "val_correlations": {},
            "flux_names": flux_names,
            "reversibility_names": flux_names,
            "measured_fluxes": measured_fluxes,
            "null_space_data": null_space_data,
            "h_reaction": H_REACTION,
            "h_type": H_TYPE,
            "data_dir": str(run_dir),
            "model_json": TOY_MODEL_JSON,
        }

    ckpt_path = run_dir / "toy_ckpt.pt"
    torch.save(checkpoint_for(build(0)), ckpt_path)

    # Two seed-varied members, exactly as train_ai_model.py --ensemble-size writes them.
    ensemble_dir = run_dir / "ensemble"
    ensemble_dir.mkdir()
    member_paths = []
    for i in range(2):
        p = ensemble_dir / f"model_{i}.pt"
        torch.save(checkpoint_for(build(100 + i)), p)
        member_paths.append(str(p))

    return {
        "dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "ensemble_dir": str(ensemble_dir),
        "member_paths": member_paths,
        "config": TOY_CONFIG_YAML,
        "model_json": TOY_MODEL_JSON,
        "flux_names": flux_names,
        "measured_fluxes": measured_fluxes,
        "val_indices": val_indices,
        "n_fluxes": n_fluxes,
    }
