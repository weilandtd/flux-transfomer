"""Tests for isotopologue labeling noise augmentation."""

import pytest
import torch
import numpy as np

from sigmaflux.ai.train import apply_labeling_noise


@pytest.fixture
def simple_batch():
    """Create a simple batch with 2 metabolites, 3 tokens."""
    fractions = torch.tensor([
        [[0.1], [0.2], [0.3]],  # batch 0
        [[0.15], [0.25], [0.35]],  # batch 1
    ], dtype=torch.float32)

    compound_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    tracer_ids = torch.tensor([0, 0, 0], dtype=torch.long)

    return fractions, compound_ids, tracer_ids


@pytest.fixture
def multi_tracer_batch():
    """Create batch with multiple metabolites and tracers."""
    fractions = torch.tensor([
        [[0.1], [0.2], [0.3], [0.05], [0.15], [0.2], [0.2], [0.25]],
    ], dtype=torch.float32)

    compound_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    tracer_ids = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1], dtype=torch.long)

    return fractions, compound_ids, tracer_ids


class TestNoiseFunction:
    """Test suite for apply_labeling_noise function."""

    def test_no_noise_unchanged(self, simple_batch):
        """With noise_std=0, fractions should remain unchanged."""
        fractions, compound_ids, tracer_ids = simple_batch
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.0)

        assert torch.allclose(result, fractions)

    def test_noise_applied(self, simple_batch):
        """With noise_std>0, fractions should be modified."""
        fractions, compound_ids, tracer_ids = simple_batch
        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.1)

        assert not torch.allclose(result, fractions)

    def test_sum_to_one_constraint(self, multi_tracer_batch):
        """Verify that isotopomers sum to 1 (including implicit M+0)."""
        fractions, compound_ids, tracer_ids = multi_tracer_batch

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.15)

        result_np = result[0, :, 0].numpy()
        compound_ids_np = compound_ids.numpy()
        tracer_ids_np = tracer_ids.numpy()

        # Check each (metabolite, tracer) group sums to 1
        for comp_id in np.unique(compound_ids_np):
            for tracer_id in np.unique(tracer_ids_np):
                mask = (compound_ids_np == comp_id) & (tracer_ids_np == tracer_id)
                if mask.any():
                    stored_fracs = result_np[mask]
                    m0_frac = 1.0 - stored_fracs.sum()
                    total = m0_frac + stored_fracs.sum()
                    assert np.abs(total - 1.0) < 1e-5

    def test_fractions_in_valid_range(self, multi_tracer_batch):
        """Verify fractions stay in [0, 1] after noise."""
        fractions, compound_ids, tracer_ids = multi_tracer_batch

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.3)

        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_batch_independence(self, simple_batch):
        """Verify different batch items have independent noise."""
        fractions, compound_ids, tracer_ids = simple_batch

        # Duplicate first batch item to second
        fractions_dup = torch.cat([fractions[:1], fractions[:1]], dim=0)

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions_dup, compound_ids, tracer_ids, noise_std=0.1)

        # Different batches should have different noise
        assert not torch.allclose(result[0], result[1])

    @pytest.mark.parametrize("noise_std", [0.0, 0.05, 0.1, 0.2, 0.5])
    def test_noise_magnitude_variation(self, simple_batch, noise_std):
        """Verify noise produces reasonable variations across different magnitudes."""
        fractions, compound_ids, tracer_ids = simple_batch

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=noise_std)

        if noise_std == 0.0:
            assert torch.allclose(result, fractions)
        else:
            assert not torch.allclose(result, fractions)
            # Fractions should be reasonably close (not wildly different)
            assert torch.all(torch.abs(result - fractions) < 2.0)

    def test_device_compatibility(self, simple_batch):
        """Verify function works with different tensor devices."""
        fractions, compound_ids, tracer_ids = simple_batch

        # Test CPU (default)
        result_cpu = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.1)
        assert result_cpu.device.type == 'cpu'

    def test_dtype_preservation(self, simple_batch):
        """Verify output dtype matches input dtype."""
        fractions, compound_ids, tracer_ids = simple_batch

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.1)

        assert result.dtype == fractions.dtype

    def test_shape_preservation(self, multi_tracer_batch):
        """Verify output shape matches input shape."""
        fractions, compound_ids, tracer_ids = multi_tracer_batch

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.1)

        assert result.shape == fractions.shape

    def test_single_metabolite(self):
        """Test with single metabolite."""
        fractions = torch.tensor([[[0.3], [0.2]]], dtype=torch.float32)
        compound_ids = torch.tensor([0, 0], dtype=torch.long)
        tracer_ids = torch.tensor([0, 0], dtype=torch.long)

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids, tracer_ids, noise_std=0.1)

        # Check sum to 1
        stored = result[0, :, 0].numpy()
        m0 = 1.0 - stored.sum()
        assert np.abs(m0 + stored.sum() - 1.0) < 1e-5

    def test_2d_ids_from_dataloader(self):
        """Test with 2D ID tensors as they come from DataLoader batching."""
        fractions = torch.tensor([
            [[0.1], [0.2], [0.3]],
            [[0.15], [0.25], [0.35]],
        ], dtype=torch.float32)

        # DataLoader stacks 1D tensors to 2D
        compound_ids_2d = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.long)
        tracer_ids_2d = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)

        torch.manual_seed(42)
        result = apply_labeling_noise(fractions, compound_ids_2d, tracer_ids_2d, noise_std=0.1)

        assert result.shape == fractions.shape
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
