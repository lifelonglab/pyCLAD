from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extractors.

    Any callable nn.Module exposing an ``output_dim`` attribute and producing
    a ``(batch, output_dim)`` tensor from its forward call.
    """

    output_dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
