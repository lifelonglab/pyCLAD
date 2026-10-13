"""Structural tests for the paper-aligned COMMAND architecture."""

from __future__ import annotations

import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="COMMAND requires the optional torch dependency")
class TestCommandArchitecture:

    def _small_config(self):
        from pyclad.video.models.command.config import CommandArchitectureConfig

        return CommandArchitectureConfig(
            appearance_dim=4,
            motion_dim=4,
            hidden_dim=8,
            state_dim=3,
            temporal_kernel_size=5,
            mamba_blocks=1,
            memory_size=5,
            projection_dim=6,
            cbam_reduction=2,
            cbam_temporal_kernel_size=7,
            dropout=0.0,
        )

    def test_published_defaults_are_fixed_in_config(self):
        from pyclad.video.models.command.config import CommandArchitectureConfig

        config = CommandArchitectureConfig()
        assert config.appearance_dim == 1024
        assert config.motion_dim == 1024
        assert config.fused_dim == 2048
        assert config.temporal_kernel_size == 5
        assert config.memory_size == 256

    def test_default_parameter_count_is_locked(self):
        from pyclad.video.models.command.architecture import CommandNetwork

        model = CommandNetwork()
        assert sum((parameter.numel() for parameter in model.parameters())) == 15036943

    def test_complete_sequence_reaches_every_paper_module(self):
        import torch

        from pyclad.video.models.command.architecture import CommandNetwork

        config = self._small_config()
        model = CommandNetwork(config).eval()
        appearance = torch.randn(2, 32, config.appearance_dim)
        motion = torch.randn(2, 32, config.motion_dim)
        with torch.no_grad():
            output = model(appearance, motion)
        assert output.logits.shape == (2, 32)
        assert output.probabilities.shape == (2, 32)
        assert output.projections.shape == (2, 32, config.projection_dim)
        assert output.temporal_features.shape == (2, 32, config.fused_dim)
        assert output.memory.primary_distances.shape == (2, 32, config.memory_size)
        assert output.memory.secondary_distances.shape == (2, 32, config.memory_size)
        assert output.memory.dual_memory_deviation.shape == (2, 32)
        assert torch.isfinite(output.logits).all()

    def test_augfusenet_is_exact_modality_concatenation(self):
        import torch

        from pyclad.video.models.command.architecture import AugFuseNet

        appearance = torch.tensor([[[1.0, 2.0]]])
        motion = torch.tensor([[[3.0, 4.0, 5.0]]])
        fused = AugFuseNet(2, 3)(appearance, motion)
        torch.testing.assert_close(fused, torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]))

    def test_temporal_convolution_has_published_kernel_and_is_depthwise(self):
        from pyclad.video.models.command.architecture import CommandNetwork

        config = self._small_config()
        model = CommandNetwork(config)
        convolution = model.temporal.blocks[0].depthwise_convolution
        assert convolution.kernel_size == (5,)
        assert convolution.groups == config.hidden_dim
        assert model.temporal.cbam.output_norm.normalized_shape[0] == config.fused_dim

    def test_memdualnet_uses_exact_euclidean_minimum(self):
        import torch

        from pyclad.video.models.command.architecture import MemDualNet

        memory = MemDualNet(feature_dim=2, memory_size=2)
        with torch.no_grad():
            memory.primary_memory.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
            memory.secondary_memory.copy_(torch.tensor([[2.0, 0.0], [20.0, 20.0]]))
        output = memory(torch.tensor([[[1.0, 0.0]]]))
        torch.testing.assert_close(output.primary_nearest_distance, torch.tensor([[1.0]]))
        torch.testing.assert_close(output.secondary_nearest_distance, torch.tensor([[1.0]]))
        torch.testing.assert_close(output.dual_memory_deviation, torch.tensor([[1.0]]))

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_small_memdualnet_distances_and_gradients_match_euclidean_reference(self, device):
        import torch

        from pyclad.video.models.command.architecture import MemDualNet

        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable")
        features = torch.linspace(-1.0, 1.0, 128).reshape(2, 8, 8)
        primary = torch.linspace(2.0, 3.0, 40).reshape(5, 8)
        secondary = torch.linspace(-3.0, -2.0, 40).reshape(5, 8)
        reference_features = features.clone().requires_grad_()
        reference_primary = primary.clone().requires_grad_()
        reference_secondary = secondary.clone().requires_grad_()
        reference_distances = [
            torch.linalg.vector_norm(reference_features.unsqueeze(-2) - bank, dim=-1)
            for bank in (reference_primary, reference_secondary)
        ]
        sum(distances.sum() for distances in reference_distances).backward()

        memory = MemDualNet(feature_dim=8, memory_size=5).to(device)
        with torch.no_grad():
            memory.primary_memory.copy_(primary.to(device))
            memory.secondary_memory.copy_(secondary.to(device))
        inputs = features.to(device).requires_grad_()
        output = memory(inputs)
        torch.testing.assert_close(output.primary_distances.cpu(), reference_distances[0])
        torch.testing.assert_close(output.secondary_distances.cpu(), reference_distances[1])
        (output.primary_distances.sum() + output.secondary_distances.sum()).backward()
        for actual, reference in (
            (inputs, reference_features),
            (memory.primary_memory, reference_primary),
            (memory.secondary_memory, reference_secondary),
        ):
            assert actual.grad is not None
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad.cpu(), reference.grad, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_default_memdualnet_reports_zero_and_increasing_distance_from_temporal_prototype(self, device):
        import numpy as np
        import torch

        from pyclad.video.models.command.architecture import CommandVideoNetwork
        from pyclad.video.models.command.config import CommandArchitectureConfig

        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable")
        torch.manual_seed(42)
        model = CommandVideoNetwork(CommandArchitectureConfig()).to(device).eval()
        inputs = torch.as_tensor(np.random.default_rng(3).normal(size=(1, 32, 2048)).astype(np.float32), device=device)
        with torch.no_grad():
            prototype = model(inputs).temporal_features[0, 0]
            memory = model.network.memory
            primary_slot = memory.replace_least_accessed("primary", prototype)
            secondary_slot = memory.replace_least_accessed("secondary", prototype)
            scales = torch.tensor([0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3], device=device)
            features = prototype[None, None, :] + scales[None, :, None]
            output = memory(features)
        expected = [
            torch.linalg.vector_norm(features.cpu().unsqueeze(-2) - bank.cpu(), dim=-1)
            for bank in (memory.primary_memory, memory.secondary_memory)
        ]
        assert output.dual_memory_deviation[0, 0].item() == 0.0
        assert torch.all(output.dual_memory_deviation[:, 1:] > output.dual_memory_deviation[:, :-1])
        assert torch.all(output.primary_nearest_slot == primary_slot)
        assert torch.all(output.secondary_nearest_slot == secondary_slot)
        torch.testing.assert_close(output.primary_distances.cpu(), expected[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(output.secondary_distances.cpu(), expected[1], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    @pytest.mark.parametrize("feature_dim,memory_size,temporal_steps", [(8, 5, 4), (2048, 256, 40)])
    def test_memdualnet_zero_and_near_prototype_gradients_match_direct_norm(
        self, device, feature_dim, memory_size, temporal_steps
    ):
        import torch

        from pyclad.video.models.command.architecture import MemDualNet

        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable")
        prototype = torch.linspace(-1.0, 1.0, feature_dim)
        scales = torch.tensor([0.0, 1e-5, 1e-4, 1e-3]).repeat(temporal_steps // 4)
        features = prototype[None, None, :] + scales[None, :, None]
        primary = torch.linspace(2.0, 3.0, memory_size * feature_dim).reshape(memory_size, feature_dim)
        secondary = -primary.clone()
        primary[0] = prototype
        secondary[1] = prototype
        reference_features = features.clone().requires_grad_()
        reference_primary = primary.clone().requires_grad_()
        reference_secondary = secondary.clone().requires_grad_()
        reference_distances = [
            torch.linalg.vector_norm(reference_features.unsqueeze(-2) - bank, dim=-1)
            for bank in (reference_primary, reference_secondary)
        ]
        weights = torch.linspace(0.1, 1.0, features.shape[1] * memory_size).reshape(1, features.shape[1], memory_size)
        sum((distances * weights).sum() for distances in reference_distances).backward()

        memory = MemDualNet(feature_dim, memory_size).to(device)
        with torch.no_grad():
            memory.primary_memory.copy_(primary.to(device))
            memory.secondary_memory.copy_(secondary.to(device))
        inputs = features.to(device).requires_grad_()
        output = memory(inputs)
        torch.testing.assert_close(output.primary_distances.cpu(), reference_distances[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(output.secondary_distances.cpu(), reference_distances[1], rtol=1e-5, atol=1e-6)
        ((output.primary_distances + output.secondary_distances) * weights.to(device)).sum().backward()
        for actual, reference in (
            (inputs, reference_features),
            (memory.primary_memory, reference_primary),
            (memory.secondary_memory, reference_secondary),
        ):
            assert actual.grad is not None
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad.cpu(), reference.grad, rtol=1e-5, atol=1e-5)

    def test_frequency_aware_fifo_lru_replacement_is_deterministic(self):
        import torch

        from pyclad.video.models.command.architecture import MemDualNet

        memory = MemDualNet(feature_dim=2, memory_size=3)
        memory.primary_access_count.copy_(torch.tensor([2, 0, 0]))
        memory.primary_last_access.copy_(torch.tensor([3, 2, 1]))
        replaced = memory.replace_least_accessed("primary", torch.tensor([7.0, 8.0]))
        assert replaced == 2
        torch.testing.assert_close(memory.primary_memory[2], torch.tensor([7.0, 8.0]))
