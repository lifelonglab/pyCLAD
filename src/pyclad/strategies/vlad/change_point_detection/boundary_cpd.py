from typing import Any, Dict, List

import numpy as np

from pyclad.strategies.vlad.change_point_detection.base import (
    ChangePoint,
    ChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory


class BoundaryChangePointDetector(ChangePointDetector):
    """Change point handling for the **concept-incremental** scenario, where concept boundaries
    are given by the scenario driver (one `detect()` call per concept) but concept identity is
    not — i.e. it is unknown whether the current concept is new or a recurrence of one already
    in memory.

    Unlike :class:`~pyclad.strategies.vlad.change_point_detection.wasserstein_cpd.WassersteinChangePointDetector`,
    this does not scan for drift within a batch (the boundary is already known), but still runs
    Algorithm 2's new-vs-recurring matching once per call against every calibrated concept, so
    recurring concepts are still routed to their existing memory node rather than duplicated. The
    whole batch is treated as a single segment; an empty batch is a no-op.
    """

    def __init__(self, memory: VladMemory):
        self._memory = memory

    @property
    def memory(self) -> VladMemory:
        return self._memory

    def detect(self, batch: np.ndarray) -> List[ChangePoint]:
        batch = np.asarray(batch)
        if len(batch) == 0:
            return []

        match = self._memory.find_best_match(batch)
        if match is not None:
            matched_id, _ = match
            self._memory.add_to_concept(matched_id, batch)
            return [ChangePoint(index=0, is_new_concept=False, segment=batch, matched_concept_id=matched_id)]

        self._memory.create_concept(batch)
        return [ChangePoint(index=0, is_new_concept=True, segment=batch)]

    def name(self) -> str:
        return "BoundaryChangePointDetector"

    def additional_info(self) -> Dict[str, Any]:
        return {}
