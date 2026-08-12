"""Console-script entry for the MFA benchmark, ahead of the heavy imports.

This exists for one reason: import order. ``cyipopt`` links ``libspral``, which
needs ``CXXABI_1.3.15`` from the conda env's ``libstdc++``. If torch loads first it
pulls in the system ``/lib64`` copy, the dynamic loader reuses that SONAME, and a
later ``import x3cflux`` fails outright. Loading x3cflux first makes the process
resolve the env's copy and both coexist happily.

So this module must stay free of torch, cobra and anything that drags them in --
it may only look at ``sys.argv``, decide whether the x3cflux backend was asked for,
load it, and only then import the real driver. The alternative fix is to export
``LD_LIBRARY_PATH`` around every invocation, which pushes an environment detail
onto every caller and every submit script; this keeps it in one place.

:mod:`sigmaflux.pipeline.run_mfa_benchmark` remains directly runnable
(``python -m``) for the native backend, where none of this applies.
"""

import sys


def _wants_x3cflux(argv):
    """Whether ``--backend x3cflux`` (in either spelling) appears in *argv*."""
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            return argv[i + 1] == "x3cflux"
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1] == "x3cflux"
    return False


def cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if _wants_x3cflux(argv):
        from ..x3cflux import preimport, require
        if not preimport():
            require()          # not installed: raise with the install instructions
    from .run_mfa_benchmark import cli as run_cli
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
