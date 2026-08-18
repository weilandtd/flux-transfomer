"""The ``.sfx`` container: a zip holding one packaged SigmaFlux model.

This module owns the *file format* only — entry names, the manifest, schema
versioning, blob encoding and checksums. It deliberately imports **no torch,
no cobra, no pandas**, so the manifest of a model can be read anywhere numpy is
available. :mod:`sigmaflux.ai.export` owns the semantics on top of it.

Layout
------
::

    manifest.json                     STORED, first entry — inspectable without inflating
    shared/vocabs.json                compound/label/tracer vocabularies + harmonics
    shared/token_index.json           (Tracer, Metabolite, Isotopomer), training order
    shared/names.json                 flux / reversibility / measured-flux names
    shared/norm_stats.json            omitted when the model trained without MID z-scoring
    shared/token_experiment_ids.npy
    shared/null_space/*.npy           Z_free, y_*, rev_log_*, obs_weights, *_idx
    shared/flux_norm/{mean,std}.npy
    validation/*.npy                  val split: indices, actuals, MID inputs
    validation/member_metrics.json
    members/000.bin …                 one flat little-endian blob per ensemble member

Member weights are a single concatenated blob rather than one entry per tensor:
a 20-member ensemble has ~67 tensors each, and 1,340 independently-deflated
entries would both lose cross-tensor redundancy and add ~134 KB of zip headers.
The layout is described once in ``manifest["tensors"]`` and is identical for
every member, so reading one is a single ``read`` plus ``np.frombuffer`` slices.

Nothing in an ``.sfx`` is pickled. Unlike a ``.pt``, loading one cannot execute
arbitrary code.
"""

import hashlib
import json
import zipfile
import zlib

import numpy as np

#: Schema this build writes.
SCHEMA_VERSION = 1
#: Oldest schema this build can still read.
MIN_READABLE_SCHEMA = 1
#: Value of ``manifest["format"]``.
FORMAT = "sigmaflux-model"
#: File suffix for a packaged model.
SFX_SUFFIX = ".sfx"
#: Written into the zip end-of-central-directory comment, as a greppable magic.
ARCHIVE_COMMENT = b"sigmaflux-sfx/1"

MANIFEST = "manifest.json"
MEMBER_FMT = "members/{:03d}.bin"

#: schema version -> function upgrading a manifest of that version to the next.
_UPGRADERS = {}

__all__ = [
    "ARCHIVE_COMMENT",
    "FORMAT",
    "MANIFEST",
    "MEMBER_FMT",
    "MIN_READABLE_SCHEMA",
    "SCHEMA_VERSION",
    "SFX_SUFFIX",
    "SfxError",
    "SfxFormatError",
    "SfxIntegrityError",
    "SfxReader",
    "SfxVersionError",
    "SfxWriter",
    "check_schema",
    "read_manifest",
    "sha256_bytes",
]


class SfxError(ValueError):
    """Base class for every ``.sfx`` failure."""


class SfxFormatError(SfxError):
    """The file is not an ``.sfx``, or a required entry is missing."""


class SfxVersionError(SfxError):
    """The file's schema version is outside what this build can read."""


class SfxIntegrityError(SfxError):
    """An entry's checksum does not match the manifest."""


def sha256_bytes(data):
    """Hex sha256 of a bytes-like object."""
    return hashlib.sha256(data).hexdigest()


def check_schema(manifest, *, source="<bytes>"):
    """Validate the format marker and schema version, upgrading in place if needed.
    Bump the schema version only when an entry is renamed/removed or a field's
    meaning changes — adding new keys does not require one, since readers ignore
    unknown keys."""
    if manifest.get("format") != FORMAT:
        raise SfxFormatError(
            f"{source} is not a SigmaFlux model: expected format {FORMAT!r}, "
            f"got {manifest.get('format')!r}"
        )
    try:
        version = int(manifest["schema_version"])
    except (KeyError, TypeError, ValueError):
        raise SfxFormatError(f"{source} has no usable schema_version") from None

    if version > SCHEMA_VERSION:
        raise SfxVersionError(
            f"{source} uses .sfx schema {version}, but this sigmaflux reads up to "
            f"{SCHEMA_VERSION}. Upgrade sigmaflux to load it."
        )
    if version < MIN_READABLE_SCHEMA:
        raise SfxVersionError(
            f"{source} uses .sfx schema {version}, which this sigmaflux no longer "
            f"reads (minimum {MIN_READABLE_SCHEMA}). Re-export it from the training "
            f"checkpoints with a current sigmaflux."
        )
    while version < SCHEMA_VERSION:
        manifest = _UPGRADERS[version](manifest)
        version += 1
        manifest["schema_version"] = version
    return manifest


def read_manifest(path):
    """Return the manifest of *path*, reading nothing else — cheap since the
    manifest is the first entry and is STORED, so nothing is inflated."""
    with zipfile.ZipFile(path) as zf:
        try:
            raw = zf.read(MANIFEST)
        except KeyError:
            raise SfxFormatError(f"{path} has no {MANIFEST}; not a SigmaFlux model") from None
    return check_schema(json.loads(raw.decode("utf-8")), source=str(path))


class SfxWriter:
    """Write a ``.sfx``. Use as a context manager; the manifest is written first."""

    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest
        self._entries = {}
        self._zf = None

    def __enter__(self):
        self._zf = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED)
        self._zf.comment = ARCHIVE_COMMENT
        # Placeholder so the manifest occupies the first local header; rewritten
        # in __exit__ once every entry's checksum is known.
        self._zf.writestr(zipfile.ZipInfo(MANIFEST), b"{}", compress_type=zipfile.ZIP_STORED)
        return self

    def write_bytes(self, name, data, *, store=False):
        """Add a raw entry and record its size and checksum."""
        self._zf.writestr(
            name, data,
            compress_type=zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED,
        )
        self._entries[name] = {"sha256": sha256_bytes(data), "size": len(data)}
        return len(data)

    def write_json(self, name, obj):
        return self.write_bytes(name, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))

    def write_array(self, name, array):
        """Store a numpy array as a `.npy` payload (no pickle)."""
        import io
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
        return self.write_bytes(name, buf.getvalue())

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.manifest["entries"] = self._entries
            payload = json.dumps(self.manifest, indent=1, sort_keys=True).encode("utf-8")
            self._zf.close()
            _replace_manifest(self.path, payload)
        else:
            self._zf.close()
        self._zf = None
        return False


def _replace_manifest(path, payload):
    """Rewrite the archive with the real manifest first and STORED. ``zipfile``
    cannot update an entry in place, so the whole archive is rebuilt once;
    everything else is copied compressed-as-is (no recompression)."""
    import os
    import shutil
    import tempfile

    tmp = f"{path}.tmp"
    with zipfile.ZipFile(path) as src, \
         zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        dst.comment = src.comment or ARCHIVE_COMMENT
        dst.writestr(zipfile.ZipInfo(MANIFEST), payload, compress_type=zipfile.ZIP_STORED)
        for info in src.infolist():
            if info.filename == MANIFEST:
                continue
            dst.writestr(info, src.read(info.filename), compress_type=info.compress_type)
    shutil.move(tmp, path)


class SfxReader:
    """Read a ``.sfx``. Use as a context manager."""

    def __init__(self, path):
        self.path = path
        self._zf = zipfile.ZipFile(path)
        try:
            self.manifest = check_schema(
                json.loads(self._zf.read(MANIFEST).decode("utf-8")), source=str(path))
        except KeyError:
            self._zf.close()
            raise SfxFormatError(f"{path} has no {MANIFEST}; not a SigmaFlux model") from None
        except Exception:
            self._zf.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    # ── Entries ───────────────────────────────────────────────────────────────

    def names(self):
        return self._zf.namelist()

    def has(self, name):
        return name in self._zf.namelist()

    def read_bytes(self, name, *, verify=True):
        try:
            data = self._zf.read(name)
        except KeyError:
            raise SfxFormatError(f"{self.path} has no entry {name!r}") from None
        except (zlib.error, zipfile.BadZipFile) as exc:
            # A damaged deflate stream fails before any checksum can run; report
            # it as corruption rather than leaking zlib's exception type.
            raise SfxIntegrityError(f"{name} in {self.path} is corrupt: {exc}") from None
        if verify:
            expected = self.manifest.get("entries", {}).get(name, {}).get("sha256")
            if expected is not None and sha256_bytes(data) != expected:
                raise SfxIntegrityError(
                    f"{name} in {self.path} is corrupt: sha256 does not match the manifest"
                )
        return data

    def read_json(self, name, default=None):
        if default is not None and not self.has(name):
            return default
        return json.loads(self.read_bytes(name).decode("utf-8"))

    def read_array(self, name, default=None):
        if default is not None and not self.has(name):
            return default
        import io
        return np.load(io.BytesIO(self.read_bytes(name)), allow_pickle=False)

    def member_arrays(self, index):
        """Every trained tensor of member *index*, as ``{name: ndarray}`` — one read
        plus ``frombuffer`` slices, using the per-tensor layout shared across
        members in the manifest."""
        blob = self.read_bytes(MEMBER_FMT.format(index))
        expected = self.manifest.get("member_nbytes")
        if expected is not None and len(blob) != expected:
            raise SfxIntegrityError(
                f"member {index} of {self.path} is {len(blob)} bytes, expected {expected}"
            )
        out = {}
        for spec in self.manifest["tensors"]:
            start, nbytes = spec["offset"], spec["nbytes"]
            chunk = np.frombuffer(blob, dtype=np.dtype(spec["stored_dtype"]),
                                  count=nbytes // np.dtype(spec["stored_dtype"]).itemsize,
                                  offset=start)
            out[spec["name"]] = chunk.reshape(spec["shape"]).astype(spec["dtype"], copy=True)
        return out

    def verify(self):
        """Check every entry against the manifest checksums and the zip CRCs.

        Returns the list of entry names checked; raises :class:`SfxIntegrityError`
        on the first mismatch.
        """
        bad = self._zf.testzip()
        if bad is not None:
            raise SfxIntegrityError(f"{self.path}: CRC failure in {bad}")
        checked = []
        for name, meta in self.manifest.get("entries", {}).items():
            data = self._zf.read(name)
            if sha256_bytes(data) != meta["sha256"]:
                raise SfxIntegrityError(f"{self.path}: sha256 mismatch for {name}")
            if len(data) != meta["size"]:
                raise SfxIntegrityError(f"{self.path}: size mismatch for {name}")
            checked.append(name)
        return checked
