"""The null-space basis must be shared across ensemble members.

An ICA basis is not bit-reproducible: FastICA's fixed point depends on the
sample covariance's last bits, which change with the BLAS thread count. Members
trained on differently sized nodes therefore land on different — equally valid —
rotations of the same subspace. Predicted fluxes are unaffected because they are
physical, but the members' basis coordinates become incomparable, which silently
breaks anything that compares members in basis space.

``share_basis`` is the guard: member 0 writes the basis, the rest load it.
"""

import numpy as np
import pytest

from sigmaflux.pipeline.train_ai_model import share_basis


def _basis(seed, shape=(12, 4)):
    return np.random.default_rng(seed).normal(size=shape)


def test_first_member_writes_and_later_members_load(tmp_path):
    """The decisive property: a differing recomputation is overridden."""
    path = str(tmp_path / "basis.npz")
    member0 = _basis(0)

    assert share_basis(member0, path) is member0
    assert (tmp_path / "basis.npz").exists()

    # Member 1 computed a *different* basis, as a different BLAS thread count would
    member1 = _basis(1)
    assert not np.array_equal(member0, member1)
    assert np.array_equal(share_basis(member1, path), member0)


def test_all_members_agree(tmp_path):
    path = str(tmp_path / "basis.npz")
    used = [share_basis(_basis(i), path) for i in range(5)]
    for other in used[1:]:
        assert np.array_equal(other, used[0])


def test_shape_mismatch_is_rejected(tmp_path):
    """A basis from a different model must not be silently adopted."""
    path = str(tmp_path / "basis.npz")
    share_basis(_basis(0, (12, 4)), path)
    with pytest.raises(ValueError, match="belongs to a different model"):
        share_basis(_basis(1, (12, 5)), path)


def test_none_disables_sharing(tmp_path):
    original = _basis(0)
    assert share_basis(original, None) is original
    assert not list(tmp_path.iterdir())


def test_parent_directory_is_created(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "basis.npz")
    Z = _basis(0)
    assert np.array_equal(share_basis(Z, path), Z)
    assert np.array_equal(share_basis(_basis(1), path), Z)


def test_roundtrip_preserves_values_exactly(tmp_path):
    """Sharing must be lossless — a float32 round-trip would perturb the basis."""
    path = str(tmp_path / "basis.npz")
    Z = _basis(0).astype(np.float64)
    share_basis(Z, path)
    loaded = share_basis(_basis(1), path)
    assert loaded.dtype == Z.dtype
    assert np.array_equal(loaded, Z)
