import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pyclad.output.output_writer import InfoProvider
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory


@dataclass
class ChangePoint:
    """A detected concept boundary (VLAD paper, Section 5.2 - Lifelong Change Point Detection).

    :param index: offset, within the batch passed to `detect()`, where this segment begins.
    :param is_new_concept: `True` if `segment` did not match any known concept (a new
        distribution, `lambda_N` in the paper); `False` if it matched an existing one (a
        recurring distribution, `lambda_R`).
    :param segment: the raw samples that make up this segment.
    :param matched_concept_id: the memory concept id `segment` was routed to; set exactly when
        `is_new_concept` is `False`.
    """

    index: int
    is_new_concept: bool
    segment: np.ndarray
    matched_concept_id: Optional[str] = None


class ChangePointDetector(InfoProvider, abc.ABC):
    """Detects concept boundaries in a stream of batches, classifying each newly-identified
    segment as new or recurring and routing its samples into a shared
    :class:`~pyclad.strategies.vlad.memory.hierarchical_memory.VladMemory` as it goes. Distance
    computation lives entirely on that memory (`find_best_match`/`distance_to`); detectors never
    call a `WassersteinDistance` implementation directly.
    """

    @abc.abstractmethod
    def detect(self, batch: np.ndarray) -> List[ChangePoint]: ...

    @property
    @abc.abstractmethod
    def memory(self) -> VladMemory:
        """The `VladMemory` this detector reads from and writes into. `VladStrategy` checks this
        matches its own `memory` at construction time, so wiring the two to different instances
        by mistake fails loudly instead of silently leaving replay/summarization on empty data."""
        ...

    @abc.abstractmethod
    def name(self) -> str: ...

    def info(self) -> Dict[str, Any]:
        return {"name": self.name(), **self.additional_info()}

    def additional_info(self) -> Dict[str, Any]:
        return {}
