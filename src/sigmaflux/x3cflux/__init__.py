"""x3cflux (13CFLUX) as a second MFA backend.

The MFA arm of the benchmark otherwise rests on one in-house optimiser, which
leaves open whether a result measures MFA or measures our L-BFGS-B. This
subpackage runs the same fits through x3cflux 3.0.4 -- an independent,
published, citable implementation -- on identical data, and writes identical
output, so the two are directly comparable.

The pieces:

``convert``
    cobra model + ``annotation.atom_mapping`` -> x3cflux ``NetworkData``.
``check_network``
    The pre-flight validator. x3cflux ships none, and a violated invariant is a
    SIGSEGV rather than an exception, so this is what separates a clear error
    message from bisecting a dead interpreter.
``fluxmap``
    cobra ``(v, r)`` <-> x3cflux free parameters, in both directions.
``fit``
    Gauss-Newton Ipopt driver with per-start subprocess isolation.

``x3cflux`` itself is an optional dependency (``pip install sigmaflux[x3cflux]``;
the wheel is cp311-only), so nothing here is imported at package-import time --
importing this module must stay free of it, exactly as ``sigmaflux.pipeline``
stays free of torch. Use :func:`available` to test for it.

**Import order matters.** ``cyipopt`` links ``libspral``, which needs
``CXXABI_1.3.15`` -- present in the conda env's ``libstdc++.so.6`` (conda-forge
``libstdcxx-ng``, the reason ``environment.yml`` pins it) but not in the system
``/lib64`` copy. Importing torch first loads the system one, and the dynamic
loader then reuses that SONAME, so a later ``import x3cflux`` fails. Importing
x3cflux first makes the process resolve the env's copy and everything coexists.
:func:`preimport` does that, and :mod:`sigmaflux.pipeline.mfa_entry` calls it
before any heavy module loads. Do not paper over this with ``LD_LIBRARY_PATH``.
"""

__all__ = ["available", "preimport", "require"]


def available():
    """Whether the optional ``x3cflux`` dependency can be imported."""
    import importlib.util

    return importlib.util.find_spec("x3cflux") is not None


def preimport():
    """Load x3cflux and its Ipopt stack before torch can claim ``libstdc++``.

    Call this at process entry, before importing anything that pulls in torch;
    afterwards the order no longer matters. Returns True if x3cflux was loaded,
    False if it is not installed -- absence is not an error here, since the caller
    may not have asked for this backend at all.
    """
    if not available():
        return False
    import cyipopt  # noqa: F401  -- the libspral consumer; must precede torch
    import x3cflux  # noqa: F401
    import x3cflux.utilities  # noqa: F401  -- pulls cvxpy, also pre-torch
    return True


def require():
    """Import and return ``x3cflux``, with an actionable message when it is absent."""
    try:
        import x3cflux
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the x3cflux backend needs the optional 'x3cflux' dependency: "
            "pip install 'sigmaflux[x3cflux]'. The wheel is cp311-only, so it "
            "requires a Python 3.11 environment."
        ) from exc
    return x3cflux
