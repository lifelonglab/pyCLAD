from random import randrange
from typing import Any

import torch

from pyclad.output.output_writer import InfoProvider


class ReservoirBuffer(InfoProvider):
    """Memory buffer filled using Vitter's Algorithm R.

    Stores (samples, outputs, targets) triples. Storage tensors are allocated
    lazily on the first update call so callers don't have to declare feature
    shapes upfront.
    """

    def __init__(self, *, max_capacity: int = 500, device: torch.device | str = "cpu") -> None:
        self._max_capacity = max_capacity
        self._device = torch.device(device)
        self._sample_buffer: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self._ground_truth_buffer: torch.Tensor | None = None
        self._size: int = 0  # number of all samples, at most `max_capacity`.
        self._seen: int = 0  # counts samples, used for reservoir sampling.

    def __len__(self) -> int:
        return self._size

    def _allocate(
        self,
        sample_template: torch.Tensor,
        output_template: torch.Tensor,
        target_template: torch.Tensor,
    ) -> None:
        self._sample_buffer = torch.empty(
            (self._max_capacity, *sample_template.shape[1:]),
            dtype=sample_template.dtype,
            device=self._device,
        )
        self._output_buffer = torch.empty(
            (self._max_capacity, *output_template.shape[1:]),
            dtype=output_template.dtype,
            device=self._device,
        )
        self._ground_truth_buffer = torch.empty(
            (self._max_capacity, *target_template.shape[1:]),
            dtype=target_template.dtype,
            device=self._device,
        )

    def _put(
        self,
        sample: torch.Tensor,
        output: torch.Tensor,
        target: torch.Tensor,
        idx: int,
    ) -> None:
        self._sample_buffer[idx] = sample
        self._output_buffer[idx] = output
        self._ground_truth_buffer[idx] = target

    def update(
        self,
        samples: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Updates the buffer with new samples.

        If buffer is filled, then each sample may be added with
        decreasing probability.
        If there is no sample in the buffer, memory for the storage
        tensors will be allocated.
        """
        if not (samples.shape[0] == outputs.shape[0] == targets.shape[0]):
            raise ValueError(
                f"Batch size mismatch: samples={samples.shape[0]}, "
                f"outputs={outputs.shape[0]}, targets={targets.shape[0]}"
            )

        if self._sample_buffer is None:
            self._allocate(samples, outputs, targets)

        for sample, output, target in zip(
            samples.to(self._device),
            outputs.to(self._device),
            targets.to(self._device),
        ):
            if self._size < self._max_capacity:
                self._put(sample, output, target, self._size)
                self._size += 1
            else:
                index_candidate = randrange(self._seen + 1)
                if index_candidate < self._max_capacity:
                    self._put(sample, output, target, index_candidate)
            self._seen += 1

    def sample(
        self, n: int, *, target_device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples the buffer.

        Also moves the samples to the desired device if requested.

        :param n: number of items to sample.
        :param target_device: sets device for output tensors. If not provided,
        uses the same device as buffer's storage tensors.
        """
        if self._size == 0 or self._sample_buffer is None:
            raise ValueError("Cannot sample from an empty buffer")

        device = self._device if target_device is None else torch.device(target_device)
        n = min(n, self._size)

        indices = torch.randperm(self._size, device=self._device)[:n]
        samples = self._sample_buffer[indices]
        outputs = self._output_buffer[indices]
        targets = self._ground_truth_buffer[indices]
        return samples.to(device), outputs.to(device), targets.to(device)

    def info(self) -> dict[str, Any]:
        return {"name": "Reservoir-sampled memory bank."}

    def additional_info(self) -> dict[str, Any]:
        return {
            "max_capacity": self._max_capacity,
            "current_size": self._size,
            "items_seen": self._seen,
            "device": str(self._device),
        }
