"""COMMAND architecture; see the recreation assumptions in the video guide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pyclad.video.models.command.config import CommandArchitectureConfig


@dataclass
class DualMemoryOutput:
    """Euclidean distances produced by MemDualNet."""

    primary_distances: Tensor
    secondary_distances: Tensor
    primary_nearest_distance: Tensor
    secondary_nearest_distance: Tensor
    dual_memory_deviation: Tensor
    primary_nearest_slot: Tensor
    secondary_nearest_slot: Tensor


@dataclass
class CommandOutput:
    """Per-snippet outputs used by COMMAND's composite training loss."""

    logits: Tensor
    probabilities: Tensor
    projections: Tensor
    temporal_features: Tensor
    memory: DualMemoryOutput


class AugFuseNet(nn.Module):
    """Concatenate Kinetics-I3D RGB and optical-flow features (Eq. 1-2)."""

    def __init__(self, appearance_dim: int = 1024, motion_dim: int = 1024):
        super().__init__()
        if appearance_dim <= 0 or motion_dim <= 0:
            raise ValueError("appearance_dim and motion_dim must be positive")
        self.appearance_dim = int(appearance_dim)
        self.motion_dim = int(motion_dim)

    @property
    def output_dim(self) -> int:
        return self.appearance_dim + self.motion_dim

    def forward(self, appearance: Tensor, motion: Tensor) -> Tensor:
        self._validate_modality(appearance, self.appearance_dim, "appearance")
        self._validate_modality(motion, self.motion_dim, "motion")
        if appearance.shape[:-1] != motion.shape[:-1]:
            raise ValueError(
                "appearance and motion must share batch/time dimensions, got "
                f"{tuple(appearance.shape)} and {tuple(motion.shape)}"
            )
        return torch.cat((appearance, motion), dim=-1)

    @staticmethod
    def _validate_modality(features: Tensor, expected_dim: int, name: str) -> None:
        if features.ndim != 3 or features.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} features must have shape (batch, time, {expected_dim}), " f"got {tuple(features.shape)}"
            )


class MambaBlock(nn.Module):
    """TempMamba block implementing the paper's projected gated SSM equations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, state_dim, kernel_size) <= 0:
            raise ValueError("Mamba dimensions and kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.kernel_size = int(kernel_size)

        # Eq. 3: X W_in -> [Z_g, Z_m].
        self.input_projection = nn.Linear(input_dim, 2 * hidden_dim)
        self.activation = nn.SiLU()
        self.depthwise_convolution = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_dim,
        )

        # Eq. 4: diagonal state transition, input modulation, readout, and skip.
        self.state_input_projection = nn.Linear(hidden_dim, state_dim)
        self.state_transition = nn.Parameter(-torch.ones(state_dim))
        self.state_input_gain = nn.Parameter(torch.empty(state_dim))
        self.state_readout = nn.Parameter(torch.empty(state_dim))
        self.skip_gain = nn.Parameter(torch.ones(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        nn.init.normal_(self.state_input_gain, mean=0.0, std=0.01)
        nn.init.normal_(self.state_readout, mean=0.0, std=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"TempMamba input must have shape (batch, time, {self.input_dim}), " f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] == 0:
            raise ValueError("TempMamba requires at least one temporal step")

        projected = self.input_projection(inputs)
        gate_values, gate_modulators = projected.chunk(2, dim=-1)
        gated = gate_values * self.activation(gate_modulators)

        # Left padding followed by cropping is the causal depthwise Conv_d in
        # the paper.  It preserves the original temporal length.
        convolved = self.depthwise_convolution(gated.transpose(1, 2))
        convolved = convolved[..., : inputs.shape[1]].transpose(1, 2)
        convolved = self.activation(convolved)

        state_inputs = self.state_input_projection(convolved)
        state = torch.zeros(
            inputs.shape[0],
            self.state_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        outputs = []
        for step in range(inputs.shape[1]):
            state = self.state_transition * state + self.state_input_gain * state_inputs[:, step]
            readout = (state * self.state_readout).sum(dim=-1, keepdim=True)
            step_output = readout * convolved[:, step] + self.skip_gain * convolved[:, step]
            outputs.append(step_output)

        sequence = torch.stack(outputs, dim=1)
        return self.output_projection(self.dropout(sequence))


class ChannelAttention1D(nn.Module):
    """Channel attention branch of the one-dimensional CBAM."""

    def __init__(self, channels: int, reduction: int, dropout: float):
        super().__init__()
        reduced_channels = channels // reduction
        if reduced_channels <= 0:
            raise ValueError("channels must be at least reduction")
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.maximum_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_channels, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        average = self.average_pool(inputs).squeeze(-1)
        maximum = self.maximum_pool(inputs).squeeze(-1)
        return self.sigmoid(self.shared_mlp(average) + self.shared_mlp(maximum)).unsqueeze(-1)


class TemporalAttention1D(nn.Module):
    """Temporal attention branch of the one-dimensional CBAM."""

    def __init__(self, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("CBAM temporal kernel_size must be a positive odd integer")
        self.convolution = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        pooled = torch.cat((average, maximum), dim=1)
        return self.sigmoid(self.dropout(self.convolution(pooled)))


class CBAM1D(nn.Module):
    """Sequential channel and temporal attention used after TempMamba."""

    def __init__(self, channels: int, reduction: int, temporal_kernel_size: int, dropout: float):
        super().__init__()
        self.channel_attention = ChannelAttention1D(channels, reduction, dropout)
        self.temporal_attention = TemporalAttention1D(temporal_kernel_size, dropout)
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"CBAM input must be 3D (batch, time, channels), got {tuple(inputs.shape)}")
        channels_first = inputs.transpose(1, 2)
        channel_refined = channels_first * self.channel_attention(channels_first)
        temporal_refined = channel_refined * self.temporal_attention(channel_refined)
        return self.output_norm(temporal_refined.transpose(1, 2))


class TempMamba(nn.Module):
    """The paper's Mamba temporal unit followed by one-dimensional CBAM."""

    def __init__(self, config: CommandArchitectureConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    input_dim=config.fused_dim,
                    hidden_dim=config.hidden_dim,
                    state_dim=config.state_dim,
                    kernel_size=config.temporal_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.mamba_blocks)
            ]
        )
        self.cbam = CBAM1D(
            channels=config.fused_dim,
            reduction=config.cbam_reduction,
            temporal_kernel_size=config.cbam_temporal_kernel_size,
            dropout=config.dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return self.cbam(outputs)


class MemDualNet(nn.Module):
    """Two learnable normal-pattern memories with exact Euclidean DDM."""

    def __init__(self, feature_dim: int, memory_size: int = 256):
        super().__init__()
        if feature_dim <= 0 or memory_size <= 0:
            raise ValueError("feature_dim and memory_size must be positive")
        self.feature_dim = int(feature_dim)
        self.memory_size = int(memory_size)
        self.primary_memory = nn.Parameter(torch.empty(memory_size, feature_dim))
        self.secondary_memory = nn.Parameter(torch.empty(memory_size, feature_dim))
        nn.init.normal_(self.primary_memory, mean=0.0, std=0.01)
        nn.init.normal_(self.secondary_memory, mean=0.0, std=0.01)

        # These non-parameter buffers implement the frequency-aware FIFO-LRU
        # bookkeeping described by the paper. Replacement remains an explicit
        # training action because the novelty threshold is not published.
        self.register_buffer("primary_access_count", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("secondary_access_count", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("primary_last_access", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("secondary_last_access", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("access_clock", torch.zeros((), dtype=torch.long))

    def forward(self, features: Tensor, *, track_usage: bool = False) -> DualMemoryOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"MemDualNet input must have shape (batch, time, {self.feature_dim}), " f"got {tuple(features.shape)}"
            )
        primary_distances = self._distances(features, self.primary_memory)
        secondary_distances = self._distances(features, self.secondary_memory)
        primary_nearest_distance, primary_nearest_slot = primary_distances.min(dim=-1)
        secondary_nearest_distance, secondary_nearest_slot = secondary_distances.min(dim=-1)
        deviation = torch.minimum(primary_nearest_distance, secondary_nearest_distance)
        if track_usage:
            self._record_accesses("primary", primary_nearest_slot)
            self._record_accesses("secondary", secondary_nearest_slot)
        return DualMemoryOutput(
            primary_distances=primary_distances,
            secondary_distances=secondary_distances,
            primary_nearest_distance=primary_nearest_distance,
            secondary_nearest_distance=secondary_nearest_distance,
            dual_memory_deviation=deviation,
            primary_nearest_slot=primary_nearest_slot,
            secondary_nearest_slot=secondary_nearest_slot,
        )

    @staticmethod
    def _distances(features: Tensor, memory: Tensor) -> Tensor:
        # The matrix identity ||x||^2 + ||m||^2 - 2 x.m loses small distances
        # to cancellation, including exact matches inserted by novelty updates.
        if features.device.type != "mps":
            return torch.cdist(features, memory.unsqueeze(0), compute_mode="donot_use_mm_for_euclid_dist")

        # MPS lacks the direct cdist backward. Direct norms have its zero
        # subgradient, and checkpointing recomputes each chunk instead of
        # retaining the full (batch * time, slots, feature_dim) difference.
        flattened = features.reshape(-1, features.shape[-1])
        chunk_size = max(1, (64 * 1024 * 1024) // (memory.numel() * features.element_size()))
        distances = []
        for chunk in flattened.split(chunk_size):
            if torch.is_grad_enabled() and (chunk.requires_grad or memory.requires_grad):
                distances.append(
                    checkpoint(MemDualNet._direct_norm, chunk, memory, use_reentrant=False, preserve_rng_state=False)
                )
            else:
                distances.append(MemDualNet._direct_norm(chunk, memory))
        return torch.cat(distances, dim=0).reshape(*features.shape[:-1], memory.shape[0])

    @staticmethod
    def _direct_norm(chunk: Tensor, bank: Tensor) -> Tensor:
        """Compute exact Euclidean distances for one bounded feature chunk."""
        return torch.linalg.vector_norm(chunk.unsqueeze(-2) - bank, dim=-1)

    @torch.no_grad()
    def least_accessed_slot(self, bank: Literal["primary", "secondary"], *, exclude_slots: tuple[int, ...] = ()) -> int:
        counts, last_access = self._usage_buffers(bank)
        eligible = torch.ones_like(counts, dtype=torch.bool)
        eligible[list(exclude_slots)] = False
        if not eligible.any():
            raise ValueError("no eligible memory slots remain")
        minimum_count = counts[eligible].min()
        candidates = torch.nonzero(eligible & (counts == minimum_count), as_tuple=False).flatten()
        candidate_ages = last_access[candidates]
        return int(candidates[candidate_ages.argmin()].item())

    @torch.no_grad()
    def replace_least_accessed(
        self, bank: Literal["primary", "secondary"], feature: Tensor, *, exclude_slots: tuple[int, ...] = ()
    ) -> int:
        """Replace an eligible slot, optionally protecting earlier batch insertions."""
        if feature.shape != (self.feature_dim,):
            raise ValueError(f"replacement feature must have shape ({self.feature_dim},), got {tuple(feature.shape)}")
        slot = self.least_accessed_slot(bank, exclude_slots=exclude_slots)
        memory = self.primary_memory if bank == "primary" else self.secondary_memory
        memory[slot].copy_(feature.to(device=memory.device, dtype=memory.dtype))
        counts, last_access = self._usage_buffers(bank)
        self.access_clock.add_(1)
        counts[slot] = 1
        last_access[slot] = self.access_clock
        return slot

    @torch.no_grad()
    def _record_accesses(self, bank: Literal["primary", "secondary"], slots: Tensor) -> None:
        counts, last_access = self._usage_buffers(bank)
        flattened = slots.detach().flatten().to(device=counts.device)
        if not len(flattened):
            return
        self.access_clock.add_(1)
        unique_slots, frequencies = flattened.unique(return_counts=True)
        counts[unique_slots] += frequencies.to(dtype=counts.dtype)
        last_access[unique_slots] = self.access_clock

    def _usage_buffers(self, bank: Literal["primary", "secondary"]) -> tuple[Tensor, Tensor]:
        if bank == "primary":
            return self.primary_access_count, self.primary_last_access
        if bank == "secondary":
            return self.secondary_access_count, self.secondary_last_access
        raise ValueError(f"unknown memory bank {bank!r}")


class CommandNetwork(nn.Module):
    """AugFuseNet -> TempMamba+CBAM -> MemDualNet -> scoring heads."""

    def __init__(self, config: CommandArchitectureConfig | None = None):
        super().__init__()
        self.config = config or CommandArchitectureConfig()
        self.feature_fusion = AugFuseNet(self.config.appearance_dim, self.config.motion_dim)
        self.temporal = TempMamba(self.config)
        self.memory = MemDualNet(self.config.fused_dim, self.config.memory_size)
        self.classifier = nn.Linear(self.config.fused_dim, 1)
        self.projection = nn.Linear(self.config.fused_dim, self.config.projection_dim)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, appearance: Tensor, motion: Tensor, *, track_memory_usage: bool = False) -> CommandOutput:
        fused = self.feature_fusion(appearance, motion)
        temporal_features = self.temporal(fused)
        memory = self.memory(temporal_features, track_usage=track_memory_usage)
        logits = self.classifier(temporal_features).squeeze(-1)
        return CommandOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            projections=self.projection(temporal_features),
            temporal_features=temporal_features,
            memory=memory,
        )


class CommandVideoNetwork(nn.Module):
    """Apply COMMAND to a batch of complete RGB/flow video bags."""

    def __init__(self, architecture: CommandArchitectureConfig):
        super().__init__()
        self.architecture = architecture
        self.network = CommandNetwork(architecture)

    def forward(self, bags: Tensor, *, track_memory_usage: bool = False) -> CommandOutput:
        if bags.ndim != 3 or bags.shape[-1] != self.architecture.fused_dim:
            raise ValueError(f"COMMAND expects (batch, time, {self.architecture.fused_dim}) features")
        split = self.architecture.appearance_dim
        return self.network(bags[..., :split], bags[..., split:], track_memory_usage=track_memory_usage)
