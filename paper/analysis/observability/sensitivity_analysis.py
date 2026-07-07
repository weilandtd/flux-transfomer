"""
HSIC global sensitivity analysis for SigmaFlux labeling data.

Quantifies which fluxes and reversibilities most influence the isotope
labeling pattern using De Veiga (2015) HSIC sensitivity indices.

Example:
    python sensitivity_analysis.py --data-dir I_orientalis_20260512 --top-n 20

    # Subsample 5 000 rows and use all CPU cores (fast, ~50 K dataset):
    python sensitivity_analysis.py --data-dir I_orientalis_20260512 --top-n 20 --max-samples 5000 --n-jobs -1

    # Include model R² comparison and scatter plot:
    python sensitivity_analysis.py --data-dir I_orientalis_20260512 --top-n 20 --max-samples 5000 --n-jobs -1 --model I_orientalis_20260512/imisor_transformer.pt

    # Save full results to CSV:
    python sensitivity_analysis.py --data-dir I_orientalis_20260512 --top-n 20 --max-samples 5000 --n-jobs -1 --output I_orientalis_20260512/hsic_sensitivity.csv
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from sigmaflux.ai.data import load_data
from sigmaflux.analysis.sensitivity import hsic_indices

COLORS = {
    "main":      "#2C7BB6",
    "reference": "#AAAAAA",
}

OUTPUT_DIR = pathlib.Path("output")


def main(args: argparse.Namespace) -> None:
    labeling_df, flux_df, rev_df = load_data(args.data_dir, max_samples=args.max_samples)

    # HDF5 files are stored transposed: (features, samples) → transpose to (samples, features)
    Y = labeling_df.T.reset_index(drop=True).values.astype(np.float64)
    F = flux_df.T.reset_index(drop=True)
    R = rev_df.T.reset_index(drop=True)

    # Drop reversibilities with near-zero variance — irreversible reactions are
    # masked to 0 in MFAModel and carry no information.
    rev_variable = R.std(axis=0) > 1e-8
    R = R.loc[:, rev_variable]

    # ── Subsample once so HSIC and model inference use the EXACT same rows ────
    n_total = Y.shape[0]
    if args.max_samples is not None and n_total > args.max_samples:
        sample_idx = np.random.default_rng(0).choice(
            n_total, size=args.max_samples, replace=False
        )
    else:
        sample_idx = np.arange(n_total)

    n_sub = len(sample_idx)
    print(f"\nSamples: {n_sub}{f' (of {n_total})' if n_sub < n_total else ''}  |  Labeling dim: {Y.shape[1]}")
    print(f"Flux inputs: {F.shape[1]}  |  Reversibility inputs: {R.shape[1]}\n")

    Y_sub = Y[sample_idx]
    F_sub = F.iloc[sample_idx]
    R_sub = R.iloc[sample_idx]

    print("Computing HSIC indices for fluxes...")
    flux_results = hsic_indices(
        F_sub.values.astype(np.float64),
        Y_sub,
        input_names=F.columns.tolist(),
        n_jobs=args.n_jobs,
    )

    print("Computing HSIC indices for reversibilities...")
    rev_results = hsic_indices(
        R_sub.values.astype(np.float64),
        Y_sub,
        input_names=[c + "_rev" for c in R.columns.tolist()],
        n_jobs=args.n_jobs,
    )

    _print_top(flux_results, "Fluxes", args.top_n, args.by)
    _print_top(rev_results, "Reversibilities", args.top_n, args.by)

    if args.model:
        # Pass full DataFrames so build_radial_vocabs sees the complete vocabulary;
        # sample_idx is forwarded to prepare_id_dataset as train_indices so inference
        # runs on the identical rows used for HSIC above.
        model_r2 = _run_model_inference(
            args.model,
            labeling_df, flux_df, rev_df,
            sample_idx=sample_idx,
            model_json=args.model_json,
            device=args.device,
        )
        if model_r2 is not None:
            comparison = flux_results.join(model_r2, how="inner")
            print(f"Matched {len(comparison)} reactions between HSIC and model.")
            OUTPUT_DIR.mkdir(exist_ok=True)
            _plot_sensitivity_vs_r2(comparison, OUTPUT_DIR)
            _print_comparison_table(comparison, args.top_n)
            if args.output:
                out = pathlib.Path(args.output)
                comp_path = out.with_stem(out.stem + "_model_comparison")
                comparison[["hsic", "eta", "R", "model_r2"]].sort_values(
                    "R", ascending=False
                ).to_csv(comp_path)
                print(f"Comparison table saved to {comp_path}")

    if args.output:
        flux_results["source"] = "flux"
        rev_results["source"] = "reversibility"
        combined = pd.concat([flux_results, rev_results]).sort_values(
            args.by, ascending=False
        )
        combined.to_csv(args.output)
        print(f"\nFull results saved to {args.output}")


def _run_model_inference(
    ckpt_path: str,
    labeling_df,
    flux_df,
    rev_df,
    sample_idx: np.ndarray,
    model_json: str | None = None,
    device: str = "cpu",
) -> pd.DataFrame | None:
    """Run FluxDecoder on sample_idx samples, return per-reaction R² DataFrame."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from sigmaflux.ai.data import (
            build_radial_vocabs,
            create_tracer_mapping,
            load_stoichiometric_matrix,
            prepare_id_dataset,
        )
        from sigmaflux.ai.model import FluxDecoder, FluxDecoderConfig
    except ImportError as exc:
        print(f"Warning: missing dependency for model inference ({exc}); skipping.")
        return None

    print(f"\nLoading model from {ckpt_path!r} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Resolve COBRA model JSON
    if model_json is None:
        model_json = ckpt.get("model_json", "")
    if not os.path.isabs(model_json):
        model_json = os.path.join(_REPO_ROOT, model_json)
    if not os.path.exists(model_json):
        print(
            f"Warning: COBRA model JSON not found at {model_json!r}. "
            "Pass --model-json to specify the path."
        )
        return None

    stoichiometric_matrix, cobra_model = load_stoichiometric_matrix(model_json)

    _, base_tracer_names = create_tracer_mapping(labeling_df)
    vcabs = ckpt["vocabs"]
    vocabs = build_radial_vocabs(
        cobra_model, labeling_df, base_tracer_names,
        h_compound=vcabs["h_compound"],
        h_label=vcabs["h_label"],
        h_tracer=vcabs["h_tracer"],
    )

    norm_stats = ckpt["norm_stats"]
    measured_fluxes = norm_stats.get("measured_fluxes_feature_names") or None

    dataset, _, token_exp_ids, _, tracer_combo_emb = prepare_id_dataset(
        labeling_df, flux_df, rev_df,
        vocabs=vocabs,
        train_indices=sample_idx,
        normalize=True,
        measured_fluxes=measured_fluxes,
        norm_stats_input=norm_stats,
    )

    cfg = FluxDecoderConfig(**ckpt["model_config"])
    null_space_data = ckpt["null_space_data"]
    fixed_flux_indices = null_space_data.get("fixed_idx") or None

    model = FluxDecoder(
        config=cfg,
        stoichiometric_matrix=stoichiometric_matrix,
        token_experiment_ids=token_exp_ids,
        compound_emb_table=vocabs["compound_emb_table"],
        label_emb_table=vocabs["label_emb_table"],
        tracer_emb_table=tracer_combo_emb,
        fixed_flux_indices=fixed_flux_indices,
        null_space_data=null_space_data,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    flux_dim = cfg.n_fluxes
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    flux_preds_list, flux_tgts_list = [], []
    with torch.no_grad():
        for (cids, lids, tids, fracs), tgts in loader:
            cids  = cids.to(device)
            lids  = lids.to(device)
            tids  = tids.to(device)
            fracs = fracs.to(device)
            fixed_fluxes = None
            if cfg.n_measured > 0:
                bypass_mask  = token_exp_ids.to(device) == -1
                fixed_fluxes = fracs[:, bypass_mask, 0]
            pred_flux, _ = model(cids, lids, tids, fracs, None, fixed_fluxes)
            flux_preds_list.append(pred_flux.cpu())
            flux_tgts_list.append(tgts[:, :flux_dim])

    all_preds = torch.cat(flux_preds_list)
    all_tgts  = torch.cat(flux_tgts_list)

    variable = all_tgts.std(dim=0) > 1e-8
    if not variable.any():
        print("Warning: no variable flux reactions found; skipping R² computation.")
        return None

    fp     = all_preds[:, variable]
    ft     = all_tgts[:, variable]
    ss_res = ((fp - ft) ** 2).sum(dim=0)
    ss_tot = ((ft - ft.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2     = torch.nan_to_num(
        1.0 - ss_res / ss_tot.clamp_min(1e-12), nan=0.0
    ).numpy()

    flux_names = ckpt.get("flux_names", [])
    var_names  = (
        [flux_names[i] for i, v in enumerate(variable.tolist()) if v]
        if flux_names
        else [f"flux_{i}" for i in range(int(variable.sum()))]
    )

    print(
        f"Model R² on {n_sub} samples: {int(variable.sum())} variable reactions, "
        f"mean={r2.mean():.4f}, median={float(np.median(r2)):.4f}"
    )
    return pd.DataFrame({"model_r2": r2}, index=pd.Index(var_names))


def _plot_sensitivity_vs_r2(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    from scipy.stats import pearsonr

    mpl.rcParams.update({"font.family": "Arial", "font.size": 12})

    plot_df = df.reset_index().copy()
    plot_df["model_r2"] = plot_df["model_r2"].clip(lower=0.0)

    r, _ = pearsonr(plot_df["R"], plot_df["model_r2"])

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(
        data=plot_df, x="R", y="model_r2",
        ax=ax, color=COLORS["main"], s=20, linewidth=0, alpha=0.7,
    )
    ax.axhline(0.0, color=COLORS["reference"], linestyle="--", linewidth=0.8)
    ax.text(
        0.05, 0.95, f"$r$ = {r:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
    )
    ax.set_xlabel("HSIC sensitivity $R$")
    ax.set_ylabel("Model $R^2$ (clipped at 0)")
    sns.despine(ax=ax)
    fig.tight_layout()
    path = output_dir / "sensitivity_vs_model_r2.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to {path}")


def _print_comparison_table(df: pd.DataFrame, top_n: int) -> None:
    tbl = (
        df[["R", "eta", "model_r2"]]
        .rename(columns={"R": "HSIC_R", "eta": "HSIC_eta", "model_r2": "model_R2"})
        .sort_values("HSIC_R", ascending=False)
        .head(top_n)
    )
    print(f"\n── Top {top_n} fluxes: HSIC sensitivity vs. model R² ──")
    print(tbl.to_string(float_format="%.4f"))


def _print_top(df: pd.DataFrame, label: str, top_n: int, by: str) -> None:
    sorted_df = df.sort_values(by, ascending=False).head(top_n)
    print(f"\n── Top {top_n} {label} by {by} ──")
    print(sorted_df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HSIC global sensitivity analysis for SigmaFlux labeling data."
    )
    parser.add_argument("--data-dir",   required=True,  help="Directory containing wb_model_*.h5 files")
    parser.add_argument("--output",     default=None,   help="Path to write full results CSV")
    parser.add_argument("--top-n",      type=int, default=20, dest="top_n",   help="Rows to print per table (default: 20)")
    parser.add_argument("--by",         default="eta", choices=["eta", "R", "hsic"], help="Sort column (default: eta)")
    parser.add_argument("--max-samples",type=int, default=None, dest="max_samples", help="Randomly subsample this many rows for analysis")
    parser.add_argument("--n-jobs",     type=int, default=1, dest="n_jobs",   help="Parallel workers for chunk loop (-1 = all CPU cores, default: 1)")
    parser.add_argument("--model",      default=None,   help="Path to trained model checkpoint (.pt) for R² comparison")
    parser.add_argument("--model-json", default=None, dest="model_json",      help="Path to COBRA model JSON (overrides checkpoint path)")
    parser.add_argument("--device",     default="auto", help="Inference device: auto | cpu | cuda (default: auto)")
    args = parser.parse_args()
    main(args)
