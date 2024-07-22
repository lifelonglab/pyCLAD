import math

import numpy as np

from pyclad.strategies.replay.buffers.buffer import ReplayBuffer
from pyclad.strategies.replay.selection.selection import SelectionMethod


class AdaptiveBalancedReplayBuffer(ReplayBuffer):
    def __init__(self, selection_method: SelectionMethod, buffer_size: int):
        self._selection_method = selection_method
        self._buffer_size = buffer_size
        self._buffers = []

    def update(self, data: np.ndarray) -> None:
        self._buffers.append(data)

        # resize buffers
        new_single_buffer_size = math.floor(self._buffer_size / len(self._buffers))
        self._buffers = [self._selection_method.select(buffer, new_single_buffer_size) for buffer in self._buffers]

    def data(self) -> np.ndarray:
        return np.concatenate(self._buffers)

    def name(self) -> str:
        return "AdaptiveBalancedReplayBuffer"
