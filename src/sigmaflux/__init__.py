"""SigmaFlux: EMU-framework MFA and AI-based flux prediction.

The public names below are re-exported lazily (PEP 562). ``from sigmaflux import
MFAModel`` works exactly as before, but bare ``import sigmaflux`` no longer pulls
in torch -- which is what lets a console script decide what to load first.

That matters for one case in particular: ``cyipopt`` links ``libspral``, which
needs ``CXXABI_1.3.15`` from the conda env's ``libstdc++``. If torch loads the
system copy first the loader reuses that SONAME and ``import x3cflux`` then fails,
so :mod:`sigmaflux.pipeline.mfa_entry` imports x3cflux before anything heavy. An
eager re-export here would have made torch unavoidable and forced the alternative
fix, ``LD_LIBRARY_PATH``, on every caller.
"""

import importlib

__version__ = "0.4.2"

#: Public name -> module it lives in.
_EXPORTS = {
    "MFAModel": "sigmaflux.core.model",
    "fit_mfa_model": "sigmaflux.core.mfa",
    "Experiment": "sigmaflux.core.experiment",
    "Tracer": "sigmaflux.core.experiment",
    "MassIsotopomerData": "sigmaflux.core.data",
    "FluxData": "sigmaflux.core.data",
    "EMUExpansion": "sigmaflux.expansion.emu_expansion",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value        # cache, so this runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
