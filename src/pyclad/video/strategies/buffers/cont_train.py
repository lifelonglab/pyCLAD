"""Balanced complete-bag retention and rotating ContTrain++ replay."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from pyclad.video.data.sample import VideoBag


@dataclass(frozen=True)
class ReplayEntry:
    bag: VideoBag
    score_curve: np.ndarray

    def __post_init__(self) -> None:
        curve = np.asarray(self.score_curve, dtype=np.float32).reshape(-1)
        if curve.shape != (len(self.bag.features),) or not np.isfinite(curve).all():
            raise ValueError("replay score curve must be finite and match the bag length")
        object.__setattr__(self, "score_curve", curve)

    @property
    def maximum_score(self) -> float:
        return float(np.max(self.score_curve))

    @property
    def peak_quartile(self) -> int:
        length = len(self.score_curve)
        return min(3, int(np.argmax(self.score_curve)) * 4 // max(1, length))


class ContTrainReplayBuffer:
    """Balanced complete-bag retention with deterministic rotating replay."""

    def __init__(self, max_size: int = 1_000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._entries: Dict[str, ReplayEntry] = {}
        self._last_label: int | None = None
        self._last_cell: Dict[int, tuple[str, int]] = {}
        self._last_bag: Dict[tuple[int, str, int], str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Tuple[ReplayEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def add(self, entries: Iterable[ReplayEntry]) -> None:
        for entry in entries:
            self._entries[entry.bag.bag_id] = entry
        if len(self._entries) > self.max_size:
            retained = self._balanced_select(self.entries, self.max_size)
            self._entries = {entry.bag.bag_id: entry for entry in retained}
        cells = {(entry.bag.weak_label, entry.bag.task_id, entry.peak_quartile) for entry in self._entries.values()}
        self._last_bag = {cell: bag_id for cell, bag_id in self._last_bag.items() if cell in cells}

    def sample(self, count: int, *, exclude_bag_ids: Iterable[str] = ()) -> Tuple[VideoBag, ...]:
        """Rotate labels, task/quartile cells, and bag IDs without duplicates.

        ID successors keep rotation stable when scores change or an anchor is
        excluded or evicted. Capacity selection retains its score priorities.
        """
        if count <= 0:
            return ()
        excluded = set(exclude_bag_ids)
        cells: Dict[int, Dict[tuple[str, int], Dict[str, ReplayEntry]]] = {}
        available = 0
        for entry in self.entries:
            if entry.bag.bag_id not in excluded:
                cells.setdefault(entry.bag.weak_label, {}).setdefault((entry.bag.task_id, entry.peak_quartile), {})[
                    entry.bag.bag_id
                ] = entry
                available += 1
        selected = []
        for _ in range(min(count, available)):
            labels = sorted(cells)
            label_index = bisect_right(labels, self._last_label) % len(labels) if self._last_label is not None else 0
            label = labels[label_index]
            keys = sorted(cells[label])
            previous_cell = self._last_cell.get(label)
            cell_index = bisect_right(keys, previous_cell) % len(keys) if previous_cell is not None else 0
            cell = keys[cell_index]
            candidates = cells[label][cell]
            bag_ids = sorted(candidates)
            anchor = (label, *cell)
            previous_bag = self._last_bag.get(anchor)
            bag_index = bisect_right(bag_ids, previous_bag) % len(bag_ids) if previous_bag is not None else 0
            bag_id = bag_ids[bag_index]
            selected.append(candidates.pop(bag_id).bag)
            self._last_label = label
            self._last_cell[label] = cell
            self._last_bag[anchor] = bag_id
            if not candidates:
                del cells[label][cell]
                if not cells[label]:
                    del cells[label]
        return tuple(selected)

    @classmethod
    def _balanced_select(cls, entries: Sequence[ReplayEntry], count: int) -> Tuple[ReplayEntry, ...]:
        if count <= 0 or not entries:
            return ()
        labels = sorted({entry.bag.weak_label for entry in entries})
        selected: list[ReplayEntry] = []
        if len(labels) == 2:
            budgets = {labels[0]: count // 2, labels[1]: count - count // 2}
        else:
            budgets = {labels[0]: count}
        for label in labels:
            candidates = [entry for entry in entries if entry.bag.weak_label == label]
            selected.extend(cls._round_robin_cells(candidates, min(budgets[label], len(candidates))))

        if len(selected) < count:
            used = {entry.bag.bag_id for entry in selected}
            remaining = [entry for entry in entries if entry.bag.bag_id not in used]
            remaining.sort(key=lambda item: (-item.maximum_score, item.bag.task_id, item.bag.bag_id))
            selected.extend(remaining[: count - len(selected)])
        return tuple(selected[:count])

    @staticmethod
    def _round_robin_cells(entries: Sequence[ReplayEntry], count: int) -> list[ReplayEntry]:
        cells: Dict[tuple[str, int], list[ReplayEntry]] = {}
        for entry in entries:
            cells.setdefault((entry.bag.task_id, entry.peak_quartile), []).append(entry)
        for values in cells.values():
            values.sort(key=lambda item: (-item.maximum_score, item.bag.bag_id))
        keys = sorted(cells)
        selected: list[ReplayEntry] = []
        offset = 0
        while len(selected) < count:
            added = False
            for key in keys:
                values = cells[key]
                if offset < len(values):
                    selected.append(values[offset])
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                break
            offset += 1
        return selected

    def state_dict(self) -> Mapping[str, object]:
        return {
            "max_size": self.max_size,
            "sampling": {
                "last_label": self._last_label,
                "last_cell": dict(self._last_cell),
                "last_bag": dict(self._last_bag),
            },
            "entries": [
                {
                    "bag_id": entry.bag.bag_id,
                    "task_id": entry.bag.task_id,
                    "features": entry.bag.features,
                    "weak_label": entry.bag.weak_label,
                    "score_curve": entry.score_curve,
                }
                for entry in self.entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["max_size"]) != self.max_size:
            raise ValueError("checkpoint replay capacity does not match trainer configuration")
        self._entries.clear()
        for item in state["entries"]:
            bag = VideoBag(
                bag_id=str(item["bag_id"]),
                task_id=str(item["task_id"]),
                features=np.asarray(item["features"], dtype=np.float32),
                weak_label=int(item["weak_label"]),
            )
            entry = ReplayEntry(bag=bag, score_curve=np.asarray(item["score_curve"], dtype=np.float32))
            self._entries[bag.bag_id] = entry
        # Checkpoints predating rotating replay start a fresh deterministic cycle.
        sampling = state.get("sampling", {})
        self._last_label = sampling.get("last_label")
        self._last_cell = dict(sampling.get("last_cell", {}))
        self._last_bag = dict(sampling.get("last_bag", {}))
