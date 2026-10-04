from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from pyclad.data.concept import Concept
from pyclad.vision.data.base import ImageLoadOptions, build_vision_concept
from pyclad.vision.data.sample import VisionSample


class LazyVisionConceptList(Sequence[Concept]):
    def __init__(
        self,
        samples_by_concept: Sequence[Tuple[str, List[VisionSample]]],
        options: ImageLoadOptions,
        with_labels: bool,
    ):
        self._samples_by_concept = list(samples_by_concept)
        self._options = options
        self._with_labels = with_labels

    def names(self) -> List[str]:
        return [name for name, _ in self._samples_by_concept]

    def __len__(self) -> int:
        return len(self._samples_by_concept)

    def __getitem__(self, index: int) -> Concept:
        name, samples = self._samples_by_concept[index]
        return build_vision_concept(
            name=name,
            samples=samples,
            options=self._options,
            with_labels=self._with_labels,
        )

    def __iter__(self) -> Iterator[Concept]:
        for index in range(len(self._samples_by_concept)):
            yield self[index]
