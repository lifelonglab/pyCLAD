from collections import Counter
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from pyclad.strategies.replay.buffers.buffer import ReplayBuffer


class BalancedReplayBuffer(ReplayBuffer):
    """Balanced replay buffer with concept-aware rebalancing and optional auxiliary fields.

    Unlike ``AdaptiveBalancedReplayBuffer``:
    - it balances by explicit ``concept_indices`` rather than by update call;
    - it stores auxiliary per-sample fields in addition to raw examples;
    - it exposes ``arrays()`` and ``sample()`` for strategies such as A-GEM;
    - it rebalances by dropping samples from the currently largest concept
      instead of resizing every concept buffer to ``floor(max_size / n_buffers)``.
    """

    def __init__(self, max_size: int, seed: int = 0):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self._max_size = int(max_size)
        self._rng = np.random.default_rng(seed)
        self._fields: Dict[str, list] = {
            "examples": [],
            "concept_indices": [],
        }
        self._field_shapes: Dict[str, Tuple[int, ...]] = {}
        self._field_dtypes: Dict[str, np.dtype] = {
            "examples": np.dtype(np.float32),
            "concept_indices": np.dtype(np.int64),
        }
        self._next_concept_index = 0

    def __len__(self) -> int:
        return len(self._fields["examples"])

    def is_empty(self) -> bool:
        return len(self) == 0

    def update(self, data: np.ndarray) -> None:
        examples = np.asarray(data, dtype=np.float32)
        if len(examples) == 0:
            return

        concept_indices = np.full(len(examples), self._next_concept_index, dtype=np.int64)
        self.add(examples=examples, concept_indices=concept_indices)

    def add(self, examples: np.ndarray, concept_indices: np.ndarray, **auxiliary_fields) -> None:
        examples_array = np.asarray(examples, dtype=np.float32)
        concept_array = np.asarray(concept_indices, dtype=np.int64)
        if len(examples_array) != len(concept_array):
            raise ValueError("examples and concept_indices must have the same length")

        for field_name, values in auxiliary_fields.items():
            if len(values) != len(examples_array):
                raise ValueError(f"{field_name} must have the same length as examples")

        if len(examples_array) == 0:
            return

        self._validate_field_shape("examples", tuple(np.asarray(examples_array[0]).shape))
        for field_name, values in auxiliary_fields.items():
            array = np.asarray(values)
            self._validate_field_shape(field_name, tuple(np.asarray(array[0]).shape))
            self._field_dtypes.setdefault(field_name, array.dtype)
            self._fields.setdefault(field_name, [])

        for index in range(len(examples_array)):
            self._fields["examples"].append(np.array(examples_array[index], dtype=np.float32, copy=True))
            self._fields["concept_indices"].append(int(concept_array[index]))
            for field_name, values in auxiliary_fields.items():
                self._fields[field_name].append(np.array(values[index], copy=True))

        self._next_concept_index = max(self._next_concept_index, int(concept_array.max()) + 1)
        self._rebalance()

    def data(self) -> np.ndarray:
        return self.arrays()["examples"]

    def arrays(self) -> Dict[str, np.ndarray]:
        return {field_name: self._stack_field(field_name, values) for field_name, values in self._fields.items()}

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.is_empty():
            raise ValueError("Cannot sample from an empty replay buffer")

        size = min(int(batch_size), len(self))
        indices = self._rng.choice(len(self), size=size, replace=False)
        return {
            field_name: self._stack_selected(field_name, values, indices) for field_name, values in self._fields.items()
        }

    def name(self) -> str:
        return "BalancedReplayBuffer"

    def additional_info(self) -> Dict[str, Any]:
        auxiliary_fields = sorted(
            field_name for field_name in self._fields if field_name not in {"examples", "concept_indices"}
        )
        concept_indices = self._fields["concept_indices"]
        return {
            "max_size": self._max_size,
            "concept_count": len(set(concept_indices)),
            "auxiliary_fields": auxiliary_fields,
        }

    def _validate_field_shape(self, field_name: str, shape: Tuple[int, ...]) -> None:
        expected_shape = self._field_shapes.setdefault(field_name, shape)
        if expected_shape != shape:
            raise ValueError(f"{field_name} shape mismatch: expected {expected_shape}, got {shape}")

    def _rebalance(self) -> None:
        excess = len(self) - self._max_size
        if excess <= 0:
            return

        concept_indices = self._fields["concept_indices"]
        counts = Counter(concept_indices)
        candidates_by_concept: Dict[int, list[int]] = {}
        for index, concept_index in enumerate(concept_indices):
            candidates_by_concept.setdefault(concept_index, []).append(index)

        drop_indices = set()
        while excess > 0:
            largest_concept = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            candidates = candidates_by_concept[largest_concept]
            drop_offset = int(self._rng.integers(len(candidates)))
            drop_indices.add(candidates.pop(drop_offset))
            counts[largest_concept] -= 1
            if counts[largest_concept] == 0:
                del counts[largest_concept]
                del candidates_by_concept[largest_concept]
            excess -= 1

        kept_indices = [index for index in range(len(concept_indices)) if index not in drop_indices]
        for field_name, values in self._fields.items():
            self._fields[field_name] = [values[index] for index in kept_indices]

    def _stack_field(self, field_name: str, values: Sequence) -> np.ndarray:
        if not values:
            dtype = self._field_dtypes.get(field_name, np.dtype(np.float32))
            if field_name == "concept_indices":
                return np.empty((0,), dtype=np.int64)
            if field_name in self._field_shapes:
                return np.empty((0, *self._field_shapes[field_name]), dtype=dtype)
            return np.empty((0,), dtype=dtype)

        first_value = values[0]
        if np.isscalar(first_value) or np.asarray(first_value).ndim == 0:
            dtype = np.int64 if field_name == "concept_indices" else self._field_dtypes.get(field_name, None)
            return np.asarray(values, dtype=dtype)

        return np.stack(values)

    def _stack_selected(self, field_name: str, values: Sequence, indices: np.ndarray) -> np.ndarray:
        selected = [values[int(index)] for index in indices]
        return self._stack_field(field_name, selected)
