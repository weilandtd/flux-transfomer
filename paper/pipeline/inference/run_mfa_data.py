"""
Traditional MFA on real 13C labeling data.

Reads organism config (same YAML used for AI training), loads MID and external
flux measurements, fits MFAModel by gradient descent, and writes per-condition
flux predictions.

Usage (I. orientalis, conditions C N P):
    python -u paper/pipeline/inference/run_mfa_data.py \\
        --config paper/configs/i_orientalis.yaml \\
        --mid-file "data/I_orientalis/I_orientalis_{condition}.xlsx" \\
        --extflux-file data/I_orientalis/I_orientalis_ExtFlux.xlsx \\
        --mapping-csv models/I_orientalis/I_orientalis_mapping.csv \\
        --output-dir paper/.results/mfa_data_io_YYYYMMDD/ \\
        --conditions C N P
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import yaml
from cobra.io.json import load_json_model
from torch.nn import MSELoss

_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))

_REPO_ROOT = str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

torch.set_num_threads(_n_threads)

from sigmaflux.core.data import FluxData, MassIsotopomerData
from sigmaflux.core.experiment import Experiment, Tracer
from sigmaflux.core.mfa import fit_mfa_model
from sigmaflux.core.model import MFAModel
from sigmaflux.utils import (
    extflux_df_to_flux_table,
    mid_df_to_mass_isotopomer_table,
    read_xlsx_extflux,
    read_xlsx_mid_data,
)


def parse_args():
    p = argparse.ArgumentParser(description="Traditional MFA on real 13C labeling data")
    p.add_argument("--config", required=True, help="Path to organism YAML config (relative to repo root)")
    p.add_argument("--mid-file", required=True,
                   help="MID xlsx path with {condition} placeholder, e.g. data/IO/IO_{condition}.xlsx")
    p.add_argument("--extflux-file", required=True, help="External flux xlsx path (relative to repo root)")
    p.add_argument("--mapping-csv", required=True, help="Metabolite name mapping CSV (relative to repo root)")
    p.add_argument("--output-dir", required=True, help="Directory for results (relative to repo root)")
    p.add_argument("--conditions", nargs="+", default=["C", "N", "P"])
    p.add_argument("--gc", type=int, default=1, help="Growth condition number (default: 1)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flux-scaler", type=float, default=200.0)
    p.add_argument("--lambda0", type=float, default=1.0)
    p.add_argument("--max-sampling-flux", type=float, default=50.0,
                   help="Hard cap on COBRA reaction bounds during flux sampling (mmol/gDW/h)")
    p.add_argument("--seed", type=int, default=256)
    return p.parse_args()


def _p(rel):
    return os.path.join(_REPO_ROOT, rel)


def _apply_media_bounds(cobra_model, media_cfg):
    """Close all exchange reactions, then open only those defined in media_cfg."""
    for rxn in cobra_model.reactions:
        if rxn.id.startswith("EX_"):
            rxn.lower_bound = 0.0
            rxn.upper_bound = 0.0
    for rxn_id, bounds in media_cfg.items():
        try:
            rxn = cobra_model.reactions.get_by_id(rxn_id)
            rxn.lower_bound = bounds["lb"]
            rxn.upper_bound = bounds["ub"]
        except KeyError:
            print(f"  Warning: media reaction {rxn_id} not in model, skipping")


def _feasible_init(constrained, fallback):
    """Parsimonious feasible flux vector to initialize the MFA fit.

    The measured-flux bounds (mean±2σ) are frequently over-determined and make
    the polytope infeasible (e.g. measured glucose uptake can't supply all the
    measured biomass-precursor demands under mass balance) — this is normal in
    13C-MFA and is reconciled by the weighted least-squares fit, NOT by hard
    bounds. So for the *initial point* only, fall back to the feasible
    media-only model. Measurements are still applied as soft targets via FluxData.
    """
    from cobra.flux_analysis import pfba
    for mdl, label in ((constrained, "measured-bounds"), (fallback, "media-only")):
        try:
            return pfba(mdl).fluxes, label
        except Exception:
            continue
    return fallback.optimize().fluxes, "FBA"


def run_condition(cobra_model, condition, gc, args, tracer_cfg, timestamp, output_dir):
    """Fit MFA for one condition/replicate. cobra_model must already have media bounds applied."""
    print(f"\n{'='*60}")
    print(f"  Condition {condition}, GC {gc}")
    print(f"{'='*60}")

    # ── 1. Load measurement data ──────────────────────────────────────────────
    df_L = read_xlsx_mid_data(
        xlsx_path=_p(args.mid_file.format(condition=condition)),
        mapping_csv=_p(args.mapping_csv),
        growth_condition=gc,
    )
    df_Vm, df_err = read_xlsx_extflux(
        xlsx_path=_p(args.extflux_file),
        condition=condition,
        growth_condition=gc,
        return_errors=True,
    )

    # ── 2. Convert to [id/reaction, mid, mean, std] tables ───────────────────
    mid_table = mid_df_to_mass_isotopomer_table(df_L)
    flux_table = extflux_df_to_flux_table(df_Vm, df_err)

    model_rxn_ids = {r.id for r in cobra_model.reactions}
    flux_table = flux_table[flux_table["reaction"].isin(model_rxn_ids)].reset_index(drop=True)

    # Fill NaN std from single-replicate aggregation with a small default
    mid_table["std"] = mid_table["std"].fillna(0.01)
    flux_table["std"] = flux_table["std"].fillna(flux_table["mean"].abs() * 0.05 + 0.01)

    # ── 3. Sign-flip uptake reactions: xlsx positive → COBRA negative ─────────
    # After applying media bounds, uptake-only exchanges have ub <= 0 in the config
    # (e.g. EX_glc__D_e: {lb: -12, ub: -1}). Use this to detect which direction
    # the model expects negative flux for the xlsx-reported positive uptake value.
    for idx, row in flux_table.iterrows():
        rxn = cobra_model.reactions.get_by_id(row["reaction"])
        if rxn.upper_bound <= 0 and row["mean"] > 0:
            flux_table.at[idx, "mean"] = -row["mean"]

    # ── 4. Constrain COBRA bounds for sampling using measurements ─────────────
    model_for_sampling = copy.deepcopy(cobra_model)

    for _, row in flux_table.iterrows():
        rxn = model_for_sampling.reactions.get_by_id(row["reaction"])
        mean, std = float(row["mean"]), float(row["std"])
        lb = max(mean - 2 * std, rxn.lower_bound)
        ub = min(mean + 2 * std, rxn.upper_bound)
        if lb < ub:
            rxn.lower_bound, rxn.upper_bound = lb, ub

    for rxn in model_for_sampling.reactions:
        rxn.lower_bound = max(rxn.lower_bound, -args.max_sampling_flux)
        rxn.upper_bound = min(rxn.upper_bound, args.max_sampling_flux)

    # ── 5. Create MFAModel with a feasible initial point ─────────────────────
    # (the measured-bounds polytope is often infeasible; init from a feasible
    #  relaxation, measurements remain soft targets in the fit — see _feasible_init)
    init_series, init_src = _feasible_init(model_for_sampling, cobra_model)
    if init_src != "measured-bounds":
        print(f"  Note: measured-bounds model infeasible for init; using {init_src} "
              f"init (measurements still applied as soft targets in the fit).")
    mfa_model = MFAModel(
        model_for_sampling,
        f"MFA_{condition}{gc}",
        lambda0=args.lambda0,
        seed=args.seed,
        initial_flux_distribution=init_series,
    )

    init_vals = mfa_model.initial_flux_values.cpu().numpy()
    glc_ix = mfa_model._flux_hash.get("EX_glc__D_e")
    bio_ix = mfa_model._flux_hash.get("BIOMASS_c") or mfa_model._flux_hash.get("BIOMASS")
    if glc_ix is not None:
        print(f"  Init EX_glc__D_e: {init_vals[glc_ix]:.4f}")
    if bio_ix is not None:
        print(f"  Init BIOMASS:     {init_vals[bio_ix]:.4f}")

    # ── 6. Build experiment and prepare EMU network ───────────────────────────
    tracers = [
        Tracer(
            mfa_model.metabolites.get_by_id(t["tracer_id"]),
            t["enrichment"],
            t["labeling"],
        )
        for t in tracer_cfg
    ]

    mid_data = MassIsotopomerData(model=mfa_model, table=mid_table)
    flux_data = FluxData(mfa_model, flux_table)
    experiment = Experiment(mid_data, tracers, fluxes=flux_data)
    mfa_model.prepare([experiment])

    # ── 7. Fit ────────────────────────────────────────────────────────────────
    tag = f"{condition}{gc}_{timestamp}"
    ckpt = os.path.join(output_dir, f"ckpt_{tag}.pt")
    optimizer = torch.optim.Adam(mfa_model.parameters(), lr=args.lr)
    loss_log = fit_mfa_model(
        mfa_model,
        MSELoss(),
        MSELoss(),
        optimizer,
        epochs=args.epochs,
        flux_scaler=args.flux_scaler,
        checkpoint_file=ckpt,
        labeling_checkpoint_file=os.path.join(output_dir, f"ckpt_{tag}_labeling.pt"),
        optimizer_checkpoint=os.path.join(output_dir, f"ckpt_{tag}_optim.pt"),
    )

    # ── 8. Save predictions and loss curve ───────────────────────────────────
    reaction_ids = [r.id for r in mfa_model.reactions]
    flux_values = mfa_model.fluxes.detach().cpu().numpy()
    predictions = pd.DataFrame({"Flux Name": reaction_ids, "Value": flux_values})

    pred_path = os.path.join(output_dir, f"predictions_{tag}.csv")
    predictions.to_csv(pred_path, index=False)

    loss_df = pd.DataFrame({"epoch": range(len(loss_log)), "loss": loss_log})
    loss_df.to_csv(os.path.join(output_dir, f"loss_{tag}.csv"), index=False)

    residual = (mfa_model.stoichiometry @ mfa_model.fluxes).detach().abs().max().item()
    print(f"  Saved → {pred_path}")
    print(f"  Loss:  {loss_log[0]:.5f} (start)  →  {loss_log[-1]:.5f} (final)")
    print(f"  Max mass balance residual: {residual:.4e}")

    return predictions, loss_log


def main():
    args = parse_args()

    with open(_p(args.config)) as f:
        cfg = yaml.safe_load(f)

    tracer_cfg = cfg["labeling"]["tracers"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _p(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {cfg['model_file']}")
    cobra_model = load_json_model(_p(cfg["model_file"]))
    print(f"  {len(cobra_model.reactions)} reactions, {len(cobra_model.metabolites)} metabolites")

    # Apply organism media bounds once (shared across conditions)
    _apply_media_bounds(cobra_model, cfg["sampling"]["media"])

    results = {}
    for condition in args.conditions:
        predictions, _ = run_condition(
            cobra_model, condition, args.gc, args, tracer_cfg, timestamp, output_dir
        )
        results[condition] = predictions.set_index("Flux Name")["Value"]

    # ── Write aggregated Excel workbook ───────────────────────────────────────
    excel_path = os.path.join(output_dir, f"predictions_{timestamp}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for condition, series in results.items():
            df = series.reset_index()
            df.columns = ["Flux Name", "Value"]
            df.to_excel(writer, sheet_name=condition, index=False)

    print(f"\nAll conditions done. Predictions → {excel_path}")


if __name__ == "__main__":
    main()
