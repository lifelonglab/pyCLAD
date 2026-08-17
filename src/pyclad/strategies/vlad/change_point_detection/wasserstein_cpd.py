from typing import Any, Dict, List, Optional

import numpy as np

from pyclad.strategies.vlad.change_point_detection.base import (
    ChangePoint,
    ChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory


class WassersteinChangePointDetector(ChangePointDetector):
    """Lifelong Change Point Detection (LCPD), VLAD paper Section 5.2 / Algorithm 2 — for the
    **concept-agnostic** scenario, where no concept boundaries are given.

    Incoming batches are re-chunked into non-overlapping `mini_batch_size`-sized mini-batches.
    While the active concept is still "forming" (uncalibrated — see `VladMemory`), every
    mini-batch is absorbed unconditionally. Once calibrated, each mini-batch is compared against
    the *active* concept first (hysteresis: only abandoned once it stops fitting, not whenever a
    better match exists elsewhere); only on failure is the full pool searched for the best match.
    No match anywhere spawns a brand-new concept.

    :param memory: shared hierarchical memory; every mini-batch's samples are written directly
        into it. Mini-batches are sized via `memory.mini_batch_size` — there is deliberately no
        separate copy of that parameter on this class, since the two must always agree.
    :param refresh_every: number of mini-batches between threshold recalibrations of the active
        concept/
    """

    def __init__(self, memory: VladMemory, refresh_every: int = 100):
        self._memory = memory
        self._refresh_every = refresh_every
        self._active_concept_id: Optional[str] = None
        self._mini_batches_processed = 0

    @property
    def memory(self) -> VladMemory:
        return self._memory

    def detect(self, batch: np.ndarray) -> List[ChangePoint]:
        batch = np.asarray(batch)
        mini_batch_size = self._memory.mini_batch_size
        change_points = []
        usable_length = len(batch) - len(batch) % mini_batch_size
        for start in range(0, usable_length, mini_batch_size):
            mini_batch = batch[start : start + mini_batch_size]
            self._mini_batches_processed += 1
            change_point = self._process_mini_batch(mini_batch, index=start)
            if change_point is not None:
                change_points.append(change_point)
        return change_points

    def _process_mini_batch(self, mini_batch: np.ndarray, index: int) -> Optional[ChangePoint]:
        if self._active_concept_id is None:
            concept_id = self._memory.create_concept(mini_batch)
            self._active_concept_id = concept_id
            return ChangePoint(index=index, is_new_concept=True, segment=mini_batch)

        active_threshold = self._memory.threshold(self._active_concept_id)
        if active_threshold is None:
            # The active concept is still forming (bootstrap phase): absorb unconditionally.
            self._memory.add_to_concept(self._active_concept_id, mini_batch)
            return None

        distance = self._memory.distance_to(self._active_concept_id, mini_batch)
        if distance < active_threshold:
            self._memory.add_to_concept(self._active_concept_id, mini_batch)
            self._maybe_refresh_threshold(self._active_concept_id)
            return None

        match = self._memory.find_best_match(mini_batch, exclude=frozenset({self._active_concept_id}))
        if match is not None:
            matched_id, _ = match
            self._active_concept_id = matched_id
            self._memory.add_to_concept(matched_id, mini_batch)
            self._maybe_refresh_threshold(matched_id)
            return ChangePoint(index=index, is_new_concept=False, segment=mini_batch, matched_concept_id=matched_id)

        # Neither the active concept nor any other known concept fits: start a brand-new one.
        self._active_concept_id = None
        return self._process_mini_batch(mini_batch, index)

    def _maybe_refresh_threshold(self, concept_id: str) -> None:
        if self._mini_batches_processed % self._refresh_every == 0:
            self._memory.recompute_threshold(concept_id)

    def name(self) -> str:
        return "WassersteinChangePointDetector"

    def additional_info(self) -> Dict[str, Any]:
        # mini_batch_size lives solely on `memory`, reported alongside this under VladStrategy.
        return {"refresh_every": self._refresh_every}
