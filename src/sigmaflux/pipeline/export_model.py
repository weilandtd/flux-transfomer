"""
Package trained checkpoints into a single self-contained ``.sfx`` model.

An exported model carries everything inference needs — vocabularies, null-space
basis, token index, output names, the resolved tracer-name map — so it loads
with no config YAML, no run directory and no COBRA JSON. It is always an
ensemble; a single checkpoint becomes an ensemble of one.

The run's prepared COBRA network travels with it too (~30 KB packed), so a
consumer can check ``S @ v ≈ 0`` and resolve reactions to metabolites without
the run directory: ``sigmaflux.ai.read_stoichiometry`` returns it aligned to the
model's outputs. Pass ``--no-model-json`` to leave it out.

Example:
    sigmaflux-export --source <run>/ensemble \\
        --config paper/configs/i_orientalis.yaml \\
        --model-version 1.0.0

Weights default to float16, which leaves every identifiable reaction intact
while cutting a 20-member ensemble from ~252 MB to ~35 MB. Use
``--precision fp32`` for a bit-exact artifact.
"""

import argparse

from ..ai.export import export_model, inspect_model, load_model
from ._paths import add_root_argument


def build_parser():
    p = argparse.ArgumentParser(
        prog="sigmaflux-export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, nargs="+",
                   help="Run directory, ensemble/ directory, a .pt, or several .pt files")
    p.add_argument("--out", default=None,
                   help="Output path (default: ./<name>-v<model-version>.sfx)")
    p.add_argument("--name", default=None,
                   help="Model name (default: config 'name', else the source directory)")
    p.add_argument("--model-version", required=True,
                   help="Semantic version of the weights, e.g. 1.0.0")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                   help="Storage precision for member weights (default: fp16)")
    p.add_argument("--config", default=None,
                   help="Training YAML: human-readable tracer names and batch size")
    p.add_argument("--description", default=None)
    p.add_argument("--data-dir", default=None,
                   help="Override the checkpoint's data_dir when reading validation inputs")
    p.add_argument("--model-json", default=None,
                   help="Prepared COBRA JSON to package (default: the run directory's "
                        "model_prepared_for_mfa.json); also rebuilds the vocabularies "
                        "for checkpoints predating stored vocabs")
    p.add_argument("--no-model-json", dest="with_network", action="store_false",
                   help="Omit the COBRA network (no stoichiometry for consumers)")
    p.add_argument("--with-model-json", dest="with_network",
                   action="store_const", const=True, default="auto",
                   help="Require the COBRA network: fail rather than export without it")
    p.add_argument("--no-validation", dest="with_validation", action="store_false",
                   help="Omit the validation split entirely")
    p.add_argument("--with-val-inputs", dest="with_val_inputs",
                   action="store_const", const=True, default="auto",
                   help="Require the validation MID inputs (reads the training HDF5)")
    p.add_argument("--no-val-inputs", dest="with_val_inputs",
                   action="store_const", const=False,
                   help="Skip the validation MID inputs (no training HDF5 read)")
    p.add_argument("--max-members", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--no-metrics", dest="metrics", action="store_false",
                   help="Omit the per-member validation metrics")
    p.add_argument("--check", action="store_true",
                   help="Reload the written file and report a round-trip summary")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    add_root_argument(p)
    return p


def main(args):
    path = export_model(
        args.source if len(args.source) > 1 else args.source[0],
        args.out,
        name=args.name,
        model_version=args.model_version,
        precision=args.precision,
        config=args.config,
        root=args.root,
        description=args.description,
        with_validation=args.with_validation,
        with_val_inputs=args.with_val_inputs,
        with_network=args.with_network,
        data_dir=args.data_dir,
        model_json=args.model_json,
        max_members=args.max_members,
        max_val_samples=args.max_val_samples,
        metrics=args.metrics,
        overwrite=args.force,
    )

    manifest = inspect_model(path)
    val = manifest.get("validation", {})
    print(f"  name        : {manifest['name']} v{manifest['model_version']} "
          f"(schema {manifest['schema_version']})")
    print(f"  members     : {manifest['n_members']}   precision: {manifest['precision']}")
    print(f"  outputs     : {manifest['model']['n_fluxes']} fluxes, "
          f"{manifest['model']['n_rev']} reversibilities, "
          f"{manifest['model']['n_tokens']} tokens")
    print(f"  validation  : {'included' if val.get('included') else 'omitted'}"
          f"{', inputs stored' if val.get('inputs_included') else ''}")
    net = manifest.get("network", {})
    detail = (f"{net['n_reactions']} reactions, {net['n_metabolites']} metabolites"
              if net.get("included") else "omitted")
    print(f"  network     : {detail}")

    if args.check:
        model = load_model(path, device="cpu")
        print(f"  round-trip  : loaded {len(model.members)} member(s) as "
              f"{type(model).__name__}; {len(model.flux_names)} flux outputs")
    return 0


def cli(argv=None):
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
