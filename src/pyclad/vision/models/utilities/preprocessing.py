from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ImagePreprocessor:
    def __init__(
        self,
        input_size: tuple[int, int],
        in_channels: int,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ):
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self._input_size = input_size
        self._in_channels = in_channels
        self._normalize_mean = tuple(normalize_mean) if normalize_mean is not None else None
        self._normalize_std = tuple(normalize_std) if normalize_std is not None else None

        if (self._normalize_mean is None) != (self._normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must be both set or both None")
        if self._normalize_mean is not None:
            if len(self._normalize_mean) != in_channels or len(self._normalize_std) != in_channels:
                raise ValueError(
                    f"normalize_mean/std length must match in_channels={in_channels}, got "
                    f"{len(self._normalize_mean)} and {len(self._normalize_std)}"
                )

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (NCHW or NHWC), got {x.shape}")

        if x.shape[1] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return np.transpose(x, (0, 3, 1, 2))

        raise ValueError(f"Cannot infer channel dimension from input shape {x.shape}")

    @classmethod
    def spatial_size(cls, data: np.ndarray) -> tuple[int, int]:
        x = cls._to_nchw(np.asarray(data))
        return int(x.shape[-2]), int(x.shape[-1])

    def _match_channels(self, x_t: torch.Tensor) -> torch.Tensor:
        channels = x_t.shape[1]
        if channels == self._in_channels:
            return x_t
        if channels == 1 and self._in_channels > 1:
            return x_t.repeat(1, self._in_channels, 1, 1)
        if channels > 1 and self._in_channels == 1:
            return x_t.mean(dim=1, keepdim=True)
        raise ValueError(f"Cannot convert from channels={channels} to in_channels={self._in_channels}")

    def transform(self, data: np.ndarray) -> torch.Tensor:
        x = np.asarray(data, dtype=np.float32)
        x = self._to_nchw(x)

        if x.size > 0 and float(x.max()) > 1.0:
            x = x / 255.0

        x_t = torch.from_numpy(x)
        x_t = self._match_channels(x_t)
        x_t = F.interpolate(x_t, size=self._input_size, mode="bilinear", align_corners=False)

        if self._normalize_mean is not None and self._normalize_std is not None:
            mean = torch.tensor(self._normalize_mean, dtype=x_t.dtype).view(1, -1, 1, 1)
            std = torch.tensor(self._normalize_std, dtype=x_t.dtype).view(1, -1, 1, 1)
            x_t = (x_t - mean) / std

        return x_t
