"""
Extract informative labeling tokens for flux estimation — pipeline step 3.

Sits between generate_training_data.py (step 2) and train_ai_model.py (step 4).
Loads the generated HDF5 data, optionally filters to experimentally measurable
metabolites via the mapping CSV, runs Fisher-information significance token
selection, and writes features.csv to the run directory.

The training script requires this CSV and reads it automatically; the config's
features: section must set
  token_selection_method: fisher       # exact, cross-fit sensitivity Gram
or
  token_selection_method: fisher_fast  # approximate, single-shot Gram
-- both run the identical significance test, differing only in how the
sensitivity Gram is estimated -- or
  token_selection_method: all          # no significance test: every token that
                                        # survives restrict_to_mapped/exclude_extracellular
                                        # is included. The no-feature-selection ablation.

Usage:
    sigmaflux-select-features --config paper/configs/i_orientalis.yaml --data-dir <run_dir>
"""
import argparse
import os


from sigmaflux.ai.data import load_data
from sigmaflux.ai import features as feat_lib
from sigmaflux.pipeline._paths import (
    add_root_argument, load_config, mapping_csv_for, resolve_path,
)

_METHODS = feat_lib.TOKEN_SELECTION_METHODS


def main(args):
    cfg, root = load_config(args.config, getattr(args, "root", None))

    fc                  = cfg.get("features", {})
    n_samples           = fc.get("n_samples", 10000)
    subsample_n         = fc.get("subsample_n", None)
    noise_std           = fc.get("noise_std", 0.02)
    n_permutations      = fc.get("n_permutations", 5)
    include_rev         = fc.get("include_reversibilities", True)
    run_mi_audit        = fc.get("run_mi_audit", False)
    restrict            = fc.get("restrict_to_mapped", False)
    exclude_extracellular = fc.get("exclude_extracellular", False)
    out_name            = fc.get("output", "features.csv")
    method              = fc.get("token_selection_method", "fisher")
    seed                = cfg.get("seed", 42)

    if method not in _METHODS:
        raise ValueError(
            f"features.token_selection_method must be one of {sorted(_METHODS)}, got {method!r}"
        )

    output_path = os.path.join(args.data_dir, out_name)

    if os.path.isfile(output_path):
        print(f"[extract_informative_features] {output_path} already exists — skipping.")
        return

    print(f"Config   : {args.config}")
    print(f"Data dir : {args.data_dir}")
    print(f"Output   : {output_path}")

    def _abs(p):
        return resolve_path(p, root)

    model_json = _abs(cfg["model_file"])
    prepared = os.path.join(args.data_dir, "model_prepared_for_mfa.json")
    if os.path.isfile(prepared):
        model_json = prepared

    # experiment.mapping_csv wins where it is set: selecting tokens the measurement
    # platform cannot resolve trains the model on inputs no real experiment can supply,
    # and drops measured metabolites the training mapping happens not to list.
    mapping_csv = _abs(mapping_csv_for(cfg))

    print(f"\nLoading generated data (up to {n_samples} samples) ...")
    labeling_df, flux_df, reversibility_df = load_data(args.data_dir, max_samples=n_samples)

    if exclude_extracellular:
        labeling_df = feat_lib.drop_extracellular_tokens(labeling_df)

    # method 'all' never reads theta -- select_all_tokens takes token_states alone
    # and reports n_free_flux_directions: None -- so do not pay for the null-space
    # rotation to build it. That rotation is O(max_iter * d^2) in Varimax pairwise
    # sweeps and dominates everything else at genome scale: on the mouse whole-body
    # model (d = 1790) it ran 16 h without converging, for a result that was then
    # discarded.
    need_theta = method != "all"
    theta, token_states, d = feat_lib.build_theta_and_token_states(
        labeling_df, flux_df, model_json, n_samples,
        reversibility_df=reversibility_df if include_rev else None,
        subsample_n=subsample_n, seed=seed, need_theta=need_theta,
    )
    n_used = theta.shape[0] if theta is not None else min(
        n_samples, flux_df.shape[1], labeling_df.shape[1],
        subsample_n if subsample_n is not None else n_samples)
    print(f"  free-flux directions d = {d if need_theta else 'not computed (method=all)'}; "
          f"candidate tokens = {len(token_states)}; samples = {n_used}")

    if restrict:
        if mapping_csv and os.path.isfile(mapping_csv):
            token_states = feat_lib.apply_mapping_filter(token_states, mapping_csv)
        else:
            print("WARNING: restrict_to_mapped=true but mapping_csv not found — skipping filter")

    if method == "all":
        print("\nToken selection: all (no significance test -- every measurable token) ...")
        df, summary = feat_lib.select_all_tokens(token_states)
    else:
        print(f"\nToken selection: {method} "
              f"({'exact cross-fit' if method == 'fisher' else 'approximate single-shot'} "
              f"sensitivity Gram) ...")
        df, summary = feat_lib.select_significant_tokens(
            theta, token_states,
            noise_std=noise_std,
            n_permutations=n_permutations,
            seed=seed,
            exact=(method != "fisher_fast"),
            run_mi_audit=run_mi_audit,
        )

    df.to_csv(output_path, index=False)

    n_included = int(df["included"].sum())

    print("\n=======================  FEATURE SELECTION SUMMARY  =======================")
    print(f"Free-flux directions                     : {summary['n_free_flux_directions']}")
    eff_rank = summary.get("effective_rank")
    if eff_rank is not None:
        print(f"Effective rank                           : {eff_rank:.2f}")
    print(f"Noise std (σ)                            : {noise_std}")
    null_floor = summary.get("null_floor_nats")
    if null_floor is not None:
        print(f"Null floor (nats, {n_permutations} permutations)      : {null_floor:.4f}")
    print("-" * 74)
    print(f"TOTAL candidate tokens                   : {len(df)}")
    print(f"  ... INCLUDED in training ({'all' if method == 'all' else 'significant'})"
          f"{'  ' if method == 'all' else ''} : {n_included}")
    if n_included > 0:
        print(f"  unique metabolites included            : "
              f"{df.loc[df['included'], 'metabolite_id'].nunique()}")
    print("-" * 74)

    ev = summary.get("eigenvalues", [])
    if len(ev):
        print("Top eigenvalues: " + ", ".join(f"{e:.1f}" for e in ev[:10]))

    print(f"\nFeatures written to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--data-dir", required=True,
                        help="Run directory containing generated *.h5 files")
    return add_root_argument(parser)


def cli(argv=None):
    """Console-script entry point."""
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    cli()
