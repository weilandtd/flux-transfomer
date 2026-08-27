"""
Show what is inside a packaged ``.sfx`` model.

Reads only the manifest — no weights, no torch, no COBRA — so it is instant and
works anywhere numpy is available.

Example:
    sigmaflux-inspect i_orientalis-v1.2.0.sfx
    sigmaflux-inspect i_orientalis-v1.2.0.sfx --verify
"""

import argparse
import json
import zipfile

from ..ai._sfx import SfxReader, read_manifest


def build_parser():
    p = argparse.ArgumentParser(
        prog="sigmaflux-inspect",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Path to a .sfx model")
    p.add_argument("--json", action="store_true", help="Dump the raw manifest")
    p.add_argument("--tensors", action="store_true", help="List the stored weight tensors")
    p.add_argument("--entries", action="store_true", help="List zip entries with sizes")
    p.add_argument("--verify", action="store_true",
                   help="Check every entry against its checksum and the zip CRCs")
    return p


def _summary(m, path):
    """Print a human-readable summary of a model manifest.

    Args:
        m: parsed manifest dict.
        path: path to the .sfx file (used only to report its size).
    """
    prov, model, val = m.get("provenance", {}), m.get("model", {}), m.get("validation", {})
    import os
    print(f"{m['name']}  v{m['model_version']}   ({os.path.getsize(path)/1e6:.1f} MB)")
    if m.get("description"):
        print(f"  {m['description']}")
    print(f"  schema        : {m['schema_version']}")
    print(f"  precision     : {m['precision']}   members: {m['n_members']}")
    print(f"  created       : {m.get('created_utc')}")
    print(f"  outputs       : {model.get('n_fluxes')} fluxes, {model.get('n_rev')} "
          f"reversibilities, {model.get('n_tokens')} tokens, "
          f"{model.get('n_compounds')} compounds")
    tracers = m.get("tracers", {})
    print(f"  tracers       : {len(tracers)} ({m.get('tracers_source')})")
    for name in tracers:
        print(f"                  {name}")
    if val.get("included"):
        detail = f"{val.get('n_val')} samples"
        if val.get("inputs_included"):
            detail += ", inputs stored (R2 recomputable)"
        print(f"  validation    : included ({detail})")
    else:
        print("  validation    : omitted")
    net = m.get("network", {})
    if net.get("included"):
        order = "" if net.get("reaction_order_matches", True) else ", reordered on read"
        print(f"  network       : {net['n_reactions']} reactions, "
              f"{net['n_metabolites']} metabolites ({net.get('source')}{order})")
    else:
        print("  network       : omitted (no stoichiometry available)")
    reported = val.get("reported") or []
    medians = [r["flux_median"] for r in reported if "flux_median" in r]
    if medians:
        print(f"  reported R2   : median {min(medians):.4f}-{max(medians):.4f} across members")
    print(f"  built with    : sigmaflux {prov.get('sigmaflux_version')}, "
          f"torch {prov.get('torch_version')}, python {prov.get('python_version')}")
    commit = prov.get("git_commit")
    if commit:
        print(f"  git           : {commit[:12]}{' (dirty)' if prov.get('git_dirty') else ''}")
    print(f"  trained from  : {prov.get('training_config') or 'unknown config'}")


def main(args):
    manifest = read_manifest(args.model)

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    _summary(manifest, args.model)

    if args.tensors:
        print(f"\n  {'tensor':<34} {'shape':<18} {'stored':<9} {'bytes':>10}")
        for t in manifest["tensors"]:
            print(f"  {t['name']:<34} {str(tuple(t['shape'])):<18} "
                  f"{t['stored_dtype']:<9} {t['nbytes']:>10,}")
        print(f"  {'':<34} {'':<18} {'total':<9} {manifest['member_nbytes']:>10,} per member")
        print(f"  regenerated on load: {len(manifest.get('regenerated_tensors', []))} tensors")

    if args.entries:
        print(f"\n  {'entry':<40} {'raw':>12} {'stored':>12}")
        with zipfile.ZipFile(args.model) as zf:
            for info in zf.infolist():
                print(f"  {info.filename:<40} {info.file_size:>12,} {info.compress_size:>12,}")

    if args.verify:
        with SfxReader(args.model) as r:
            checked = r.verify()
        print(f"\n  verified {len(checked)} entries: all checksums and CRCs match")
    return 0


def cli(argv=None):
    return main(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
