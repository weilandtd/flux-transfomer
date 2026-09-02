"""
SigmaFlux's predicted labeling against the measured ¹³C data it was inferred from.

Per model:
  experimental_mid_<key>.svg      predicted vs measured MID fraction, log-log, one point
                                   per (condition, growth condition, tracer, metabolite,
                                   isotopomer), with error bars on both axes
Across all models:
  experimental_mid_summary.csv    per-(model, condition, growth condition) R², RMSE, MAE,
                                   worst error, median |z| against replicate spread, the
                                   fraction of tokens inside one replicate SD, and the
                                   fraction whose measurement falls in the predicted
                                   interval
  experimental_mid_tokens.csv     every token behind the panels

This closes the loop the flux panels leave open. plot_experimental_predictions.py asks
whether our fluxes agree with a *published MFA fit* -- a second estimate, with its own
identifiability problems, on the same data. This asks whether they reproduce the
measurement: each condition's predicted flux state is pushed back through the EMU
forward pass and the simulated MIDs are compared to what the instrument reported.
Labeling depends on flux *ratios*, so a state can be far from another estimate in flux
space and still explain the data, or close to it and not -- which is exactly why both
panels are needed.

Both axes carry an uncertainty. The x bar is the measured replicate spread -- the floor
on how well any method can do. The y bar is the same bootstrap the flux predictions
report, propagated all the way into labeling space: the measured replicates are resampled
across tracers (``sample_inputs``), every draw is predicted by every ensemble member, and
each of those flux states is simulated through the EMU pass in full. ``summarize_draws``
then reduces the resulting fractions exactly as it does for fluxes -- ensemble mean at
the maximum-likelihood input, bias-corrected bootstrap percentile interval over the
draws. Propagating in MID space rather than mapping the flux interval through is the
point: the EMU pass is nonlinear, so the endpoints of a flux interval are not the
endpoints of the labeling interval.

That costs ``(n_bootstrap + 1) x n_members`` EMU passes per growth condition -- minutes
per condition, not seconds (``--n-bootstrap`` trades runtime against how finely the tails
resolve). It also means this script needs the run's ensemble, unlike the flux panels,
which read the predictions CSV.

Axes are log-log: measured fractions run from ~1e-5 to ~1, and a linear panel puts most
tokens in the bottom corner. A fraction of exactly zero has no position on a log axis and
is dropped rather than floored (``--min-fraction`` raises the cut), and the count is
reported.

Conditions and growth conditions are the same ``PREDICTION_SPECS`` selection the flux
panels use, so the two figures describe the identical set of measured states. M+0 is
excluded on both sides -- the training labeling index carries M+1..M+N only, and M+0 is
one minus the rest rather than an independent measurement.

Each panel reports R² per condition, on the fractions in linear space -- the same
definition the flux panels and the simulation benchmark use, so R² means one thing across
the paper. RMSE and the interval-coverage fraction stay in the summary CSV.

A model whose ensemble, prepared model or experimental workbooks are missing is skipped
with a warning rather than failing the run, so the figure set fills itself in as runs
land.

Usage:
    python paper/scripts/plot_experimental_mid.py
    python paper/scripts/plot_experimental_mid.py --models e_coli_core
    python paper/scripts/plot_experimental_mid.py --run-suffix 20260829
    python paper/scripts/plot_experimental_mid.py --models e_coli_core --n-bootstrap 20
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sigmaflux.ai.inference import load_predictor
from sigmaflux.ai.uncertainty import DEFAULT_N_BOOTSTRAP, sample_inputs, summarize_draws
from sigmaflux.pipeline._benchmark import (
    build_mfa_context, load_prepared_model, pin_threads)
from sigmaflux.pipeline._paths import mapping_csv_for, resolve_path
from sigmaflux.pipeline.run_inference_workflow import (
    _discover_conditions, _find_ext_flux_file, _inspect_growth_conditions,
    load_condition_inputs, predict_draws)
from sigmaflux.utils import read_reference_fluxes

from _figure_common import (
    CONDITION_COLORS, CONDITION_LABEL, COLORS, PREDICTION_SPECS, UNBOUNDED_FLUX,
    _REPO_ROOT, add_common_args, apply_style, axis_floor, error_offsets, growth_rates,
    reference_path, resolve_organism, run_dir_for, save_figure, selected_models)

sys.path.insert(0, str(_REPO_ROOT / "paper" / "analysis" / "diagnostics"))
from validate_inferred_fluxes import (  # noqa: E402
    agreement, measured_table, simulate, split_predictions)

#: MID token keys -- one predicted fraction is identified by all three.
TOKEN = ["Tracer", "Metabolite", "Isotopomer"]

#: Hard floor for the log axes. Only a backstop: the limit is set from the data
#: (:func:`_figure_common.axis_floor`), and this stops a single near-zero fraction from
#: stretching the panel over a dozen empty decades.
MID_FLOOR = 1e-5

apply_style()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n-bootstrap", type=int, default=None,
                   help="Replicate resamples per growth condition (default: the config's "
                        "experiment.n_bootstrap). Every draw costs one EMU pass per "
                        "ensemble member.")
    p.add_argument("--min-fraction", type=float, default=0.0,
                   help="Drop tokens whose measured or predicted fraction is at or below "
                        "this. Anything non-positive is dropped regardless -- a log axis "
                        "cannot place it (default: 0.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the replicate bootstrap (default: 42)")
    return p.parse_args()


def tracer_name_map(tracer_cfg, experiments):
    """Config display name -> the internal tracer string the labeling index is keyed on.

    ``build_labeling_index`` names each experiment by its tracers' repr, and
    ``build_experiments`` builds one experiment per config tracer in order, so zipping
    the two is exact -- and needs no checkpoint, unlike ``SigmaFlux.tracer_ids()``.
    """
    return {cfg_entry.get("name", str(exp.tracers[0])):
            "|".join(str(t) for t in exp.tracers)
            for cfg_entry, exp in zip(tracer_cfg, experiments)}


def _growth_conditions(mid_file):
    """Growth-condition numbers present in a MID workbook."""
    return _inspect_growth_conditions(mid_file)


def selected_growth_conditions(org, cond, available, select):
    """The growth conditions of one condition that go into the panel.

    ``max_D`` reads the dilution rate off the reference's own BIOMASS row rather than
    assuming an index, matching how the flux panels pick the same states.
    """
    if select == "wt":
        return [gc for gc in available if gc == 1]
    if select != "max_D":
        return list(available)
    rates = growth_rates(read_reference_fluxes(reference_path(org), cond,
                                               unbounded=UNBOUNDED_FLUX))
    rated = {gc: rates[gc] for gc in available if gc in rates}
    if not rated:
        print(f"  condition {cond}: no BIOMASS row in the reference — keeping every "
              f"growth condition")
        return list(available)
    return [max(rated, key=rated.get)]


def simulate_draws(mfa_model, labeling_to, labeling_index, flux_wide, rev_wide, keep):
    """One EMU pass per (draw, ensemble member), as the long frame summarize_draws takes.

    The uncertainty is propagated in labeling space rather than mapped from the flux
    interval: the EMU pass is nonlinear, so a flux interval's endpoints are not the
    labeling interval's endpoints. Each (draw, member) flux state is simulated in full
    and the spread of the resulting fractions is what gets summarized.

    This is the expensive step -- (n_bootstrap + 1) x n_members passes per growth
    condition, ~46 ms each on E. coli core and ~137 ms on I. orientalis.

    Results accumulate in one preallocated ``(n_states, n_kept_tokens)`` array and are
    turned into the long frame once, at the end. Appending a small DataFrame per pass and
    concatenating four thousand of them instead costs gigabytes in per-frame pandas
    overhead alone, and was enough to get the job killed on a login node.

    Args:
        keep: boolean mask over ``labeling_index`` -- the tokens the measurement actually
            has, so the array holds nothing that would be dropped by the join anyway.
    """
    index = labeling_index[keep]
    values = np.empty((len(flux_wide), len(index)), dtype=float)
    started = time.perf_counter()
    for i, (key, flux_row) in enumerate(flux_wide.iterrows()):
        sim = simulate(mfa_model, flux_row, rev_wide.loc[key], labeling_to, labeling_index)
        values[i] = sim.to_numpy()[keep]
        if (i + 1) % 500 == 0 or (i + 1) == len(flux_wide):
            per = (time.perf_counter() - started) / (i + 1)
            print(f"    {i + 1}/{len(flux_wide)} passes ({per * 1000:.0f} ms each, "
                  f"~{per * (len(flux_wide) - i - 1):.0f}s left)", flush=True)

    n_tokens = len(index)
    out = pd.DataFrame({level: np.tile(index.get_level_values(level).to_numpy(), len(values))
                        for level in TOKEN})
    out["Value"] = values.ravel()
    for pos, name in enumerate(("Sample", "Member")):
        out[name] = np.repeat(flux_wide.index.get_level_values(pos).to_numpy(), n_tokens)
    return out


def condition_tokens(sf, mfa_model, labeling_to, labeling_index, name_to_id, reaction_ids,
                     mid_file, ext_file, mapping_path, cond, gc, flux_names,
                     n_bootstrap, rng):
    """Predicted-vs-measured MID tokens for one growth condition, with intervals.

    Inputs, draws and prediction all go through run_inference_workflow's own helpers, so
    this reports the pipeline's numbers rather than a parallel copy of them.
    """
    df_L, df_Vm, df_err = load_condition_inputs(mid_file, ext_file, cond, gc, mapping_path,
                                                flux_names)
    df_L_draws, df_Vm_draws = sample_inputs(df_L, df_Vm, df_err, rng=rng,
                                            n_bootstrap=n_bootstrap)
    n_draws = int(df_L_draws["Sample"].max())      # the MLE row sits at the top index
    raw = predict_draws(sf, df_L_draws, df_Vm_draws, len(sf.flux_names))
    flux_wide, rev_wide, _ = split_predictions(raw, sf, reaction_ids)

    measured = measured_table(df_L, name_to_id)
    # Simulate only what the measurement can be compared against: the join drops the rest
    # anyway, and carrying them multiplies the array by every unmeasured token.
    keep = pd.MultiIndex.from_frame(measured[TOKEN]).isin(labeling_index)
    wanted = pd.MultiIndex.from_frame(measured.loc[keep, TOKEN])
    mask = labeling_index.isin(wanted)
    mids = simulate_draws(mfa_model, labeling_to, labeling_index, flux_wide, rev_wide, mask)
    # Same reduction the flux predictions report: ensemble mean at the maximum-likelihood
    # input, and a bias-corrected bootstrap percentile interval over the draws.
    predicted = summarize_draws(mids, n_draws, group_cols=TOKEN)

    tokens = agreement(predicted.set_index(TOKEN)["Value"].rename("sim"), measured)
    return tokens.merge(predicted[TOKEN + ["Value_lo", "Value_hi", "sigma_total"]],
                        on=TOKEN, how="left")


def model_tokens(model, run, org, spec, args):
    """Every measured MID token for one model, paired with its predicted value.

    Returns None when the run has nothing to compare. The predictor and the EMU network
    are built once per model and reused for every (condition, growth condition); the
    network's targets are the metabolites the platform maps, so the expansion never walks
    past what is measured.
    """
    cfg = org["cfg"]
    data_dir = resolve_path(org["data"], org["root"])
    mapping_path = resolve_path(mapping_csv_for(cfg), org["root"])
    ext_flux = cfg.get("experiment", {}).get("ext_flux")
    ext_file = resolve_path(ext_flux, org["root"]) if ext_flux else _find_ext_flux_file(data_dir)
    if ext_file is None:
        raise FileNotFoundError(f"no *_ExtFlux.xlsx in {data_dir}")
    flux_names = cfg.get("training", {}).get("measured_fluxes") or None
    n_bootstrap = (args.n_bootstrap if args.n_bootstrap is not None
                   else cfg.get("experiment", {}).get("n_bootstrap", DEFAULT_N_BOOTSTRAP))
    available = _discover_conditions(data_dir)

    cobra_model, mapped_met_ids = load_prepared_model(str(run), cfg, org["root"])
    reaction_ids = [r.id for r in cobra_model.reactions]
    sf = load_predictor(cfg["_config_path"], None, data_dir=str(run))

    mfa_model, _, experiments, labeling_to, labeling_index = build_mfa_context(
        cobra_model, cfg["labeling"]["tracers"], pd.Series(0.0, index=reaction_ids),
        "experimental_mid", with_backward=False, restrict_targets_to=mapped_met_ids)
    name_to_id = tracer_name_map(cfg["labeling"]["tracers"], experiments)

    rng = np.random.default_rng(args.seed)
    parts = []
    for cond in spec["conditions"]:
        if cond not in available:
            print(f"  condition {cond}: no MID workbook in {data_dir} — skipping")
            continue
        gcs = selected_growth_conditions(org, cond, _growth_conditions(available[cond]),
                                         spec["select"])
        for gc in gcs:
            started = time.perf_counter()
            tokens = condition_tokens(
                sf, mfa_model, labeling_to, labeling_index, name_to_id, reaction_ids,
                available[cond], ext_file, mapping_path, cond, gc, flux_names,
                n_bootstrap, rng)
            print(f"  {cond}{gc}: {len(tokens)} MID tokens, {n_bootstrap} draws "
                  f"[{time.perf_counter() - started:.0f}s]")
            parts.append(tokens.assign(condition=cond, gc=gc))

    return pd.concat(parts, ignore_index=True) if parts else None


def drop_unplottable(df, min_fraction):
    """Tokens a log axis can show, plus how many were dropped.

    A measured fraction of exactly zero -- the instrument saw nothing at that mass -- has
    no position on a log axis. It is dropped rather than floored, because flooring would
    draw it at whatever the axis bottom happens to be and invite reading a value off it.
    """
    keep = (df["measured"] > min_fraction) & (df["sim"] > min_fraction)
    return df[keep].copy(), int((~keep).sum())


def _r2(pred, measured):
    """Linear R² of predicted against measured fractions.

    Linear, not on log10 of the fractions the axes show: this is the same definition as
    ``_r2_linear`` in plot_experimental_predictions.py and ``per_flux_r2`` in the
    benchmark, so R² means one thing across the paper. It is therefore carried by the
    large fractions, which the log axes deliberately are not.
    """
    if len(measured) < 2:
        return float("nan")
    ss_res = ((pred - measured) ** 2).sum()
    ss_tot = ((measured - measured.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def summarize(model, df):
    """Per-(condition, growth condition) agreement statistics."""
    rows = []
    for (cond, gc), grp in df.groupby(["condition", "gc"], sort=False):
        err = grp["error"].to_numpy()
        sd = grp["measured_std"].to_numpy()
        inside = ((grp["Value_lo"] <= grp["measured"]) &
                  (grp["measured"] <= grp["Value_hi"]))
        rows.append(dict(
            model=model["label"], key=model["key"], condition=cond, gc=int(gc),
            label=CONDITION_LABEL.get(model["key"], {}).get(cond, cond),
            n_tokens=len(grp),
            r2=_r2(grp["sim"].to_numpy(), grp["measured"].to_numpy()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            mae=float(np.mean(np.abs(err))),
            max_abs_error=float(np.max(np.abs(err))),
            median_abs_z=float(np.median(np.abs(grp["z"].to_numpy()))),
            frac_within_1sd=float(np.mean(np.abs(err) <= np.nan_to_num(sd, nan=np.inf))),
            frac_measured_in_interval=float(inside.mean()),
            median_interval_width=float((grp["Value_hi"] - grp["Value_lo"]).median()),
        ))
    return rows


def plot_experimental_mid(model, df, summary_rows, out_dir):
    """Predicted vs measured MID fraction, log-log, one figure per model."""
    names = CONDITION_LABEL.get(model["key"], {})
    tidy = df.assign(label=[names.get(c, c) for c in df["condition"]])
    palette = {names.get(c, c): CONDITION_COLORS.get(c, "#777777")
               for c in tidy["condition"].unique()}

    lim_lo = axis_floor(tidy["measured"], tidy["sim"], floor=MID_FLOOR)
    # A simulated fraction above 1 is not a MID -- the flux state that produced it is
    # infeasible. The axis stretches to show it rather than clipping it out of sight.
    lim_hi = float(max(1.0, tidy["measured"].max(), tidy["sim"].max()))
    n_over = int((tidy["sim"] > 1.0).sum())
    if n_over:
        print(f"  {n_over} simulated fraction(s) above 1 — those flux states are not "
              f"feasible; axis extended to {lim_hi:.3g}")

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=COLORS["reference"], lw=1.0, ls="--", zorder=1)

    # Error bars first, so markers sit on top. x is the measured replicate spread -- the
    # floor on how well any method can do -- and y the bootstrap interval of the
    # prediction, so both sides show the uncertainty they actually carry.
    for cond, grp in tidy.groupby("condition", sort=False):
        x, y = grp["measured"].to_numpy(), grp["sim"].to_numpy()
        sd = grp["measured_std"].fillna(0.0).to_numpy()
        xerr = error_offsets(x, x - sd, x + sd, lim_lo, lim_hi)
        yerr = error_offsets(y, grp["Value_lo"].to_numpy(), grp["Value_hi"].to_numpy(),
                             lim_lo, lim_hi)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=CONDITION_COLORS.get(cond, "#777777"), elinewidth=0.6,
                    alpha=0.35, capsize=0, zorder=1.5)

    sns.scatterplot(data=tidy, x="measured", y="sim", hue="label", palette=palette,
                    s=14, linewidth=0, alpha=0.8, ax=ax, zorder=2)

    # Pooled per condition, not per growth condition: a panel legend with one line per
    # (condition, growth condition) would not fit, and the CSV keeps the finer split.
    lines = []
    for cond, grp in tidy.groupby("condition", sort=False):
        r2 = _r2(grp["sim"].to_numpy(), grp["measured"].to_numpy())
        lines.append(f"{names.get(cond, cond)}: $R^2$ = {r2:.2f}")
    ax.text(0.97, 0.04, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured isotopologue fraction")
    ax.set_ylabel("SigmaFlux simulated isotopologue fraction")
    # A one-condition panel already names it in the RMSE label; a legend there is one
    # stray marker in the corner saying nothing.
    if len(palette) > 1:
        ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=1.6)
    else:
        ax.legend_.remove()
    sns.despine(ax=ax)
    save_figure(fig, out_dir / f"experimental_mid_{model['key']}.svg")
    return summary_rows


def main():
    args = parse_args()
    pin_threads()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in selected_models(args) if m["key"] in PREDICTION_SPECS]

    print("── Simulated vs measured MIDs ──")
    summary, tokens = [], []
    for model in models:
        run = run_dir_for(model, args.run_suffix)
        if not list(run.glob("predictions/predictions_*.csv")):
            print(f"  {model['label']}: skipped — no predictions/ in {run.name}; "
                  f"run paper/submit/submit_inference.sh {run}")
            continue
        print(f"\n{model['label']}  ({run.name})")
        try:
            org = resolve_organism(run)
            df = model_tokens(model, run, org, PREDICTION_SPECS[model["key"]], args)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"  {model['label']}: skipped — {type(exc).__name__}: {exc}")
            continue
        if df is None or df.empty:
            print(f"  {model['label']}: skipped — no MID token matched between the "
                  f"simulation and the workbook")
            continue
        df, n_dropped = drop_unplottable(df, args.min_fraction)
        if n_dropped:
            print(f"  dropped {n_dropped} token(s) at or below {args.min_fraction:g} on "
                  f"one axis — a log axis cannot place them")
        if df.empty:
            print(f"  {model['label']}: skipped — every token was dropped")
            continue
        summary.extend(plot_experimental_mid(model, df, summarize(model, df), out_dir))
        tokens.append(df.assign(model=model["label"], key=model["key"]))

    if summary:
        path = out_dir / "experimental_mid_summary.csv"
        pd.DataFrame(summary).to_csv(path, index=False)
        print(f"Saved: {path}")
        path = out_dir / "experimental_mid_tokens.csv"
        pd.concat(tokens, ignore_index=True).to_csv(path, index=False)
        print(f"Saved: {path}")
    else:
        print("No model had usable prediction data — nothing written.")


if __name__ == "__main__":
    main()
