from typing import Any, Dict

import numpy as np

from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)
from pyclad.strategies.vlad.change_point_detection.base import ChangePointDetector
from pyclad.strategies.vlad.change_point_detection.boundary_cpd import (
    BoundaryChangePointDetector,
)
from pyclad.strategies.vlad.change_point_detection.wasserstein_cpd import (
    WassersteinChangePointDetector,
)
from pyclad.strategies.vlad.memory.hierarchical_memory import VladMemory


class VladStrategy(ConceptAgnosticStrategy, ConceptIncrementalStrategy):
    """VLAD: task-agnostic VAE-based lifelong anomaly detection (Faber, Corizzo, Śnieżyński &
    Japkowicz, *Neural Networks*, 2023) — Algorithm 1 of the paper.

    Combines a change point detector (which classifies each incoming segment as a new or
    recurring concept and stores it into `memory`), a hierarchical memory (Consolidation and
    Pyramidal Summarization), and an anomaly-scoring `model` updated via Experience Replay: the
    model is retrained on `concat(memory.replay_buffer(), data)` whenever a new concept is
    detected, a recurring concept is detected, the memory was just summarized, or too many
    samples have passed since the last update.

    Supports both of pyCLAD's task-boundary scenarios through the *same* orchestration logic —
    the only thing that differs between them is which `ChangePointDetector` is injected. Most
    users should reach for :class:`VladConceptAgnosticStrategy` or
    :class:`VladConceptIncrementalStrategy` below instead of constructing this directly: they
    pick and wire the right detector for you. Use this class yourself only if you need to inject
    a custom `ChangePointDetector`.

    :param model: anomaly-scoring model (e.g. a `VariationalAutoencoder` via
        `TorchModelAdapter`), updated via `fit()` with replay + current data (Eq. 10 for scoring
        is left entirely to the model — this strategy adds no scoring logic of its own).
    :param change_point_detector: see above. Must be constructed with the *same* `memory`
        instance passed here (checked at construction time — see `:raises` below).
    :param memory: hierarchical memory shared with `change_point_detector`.
    :param max_steps_between_updates: force a model update after this many samples have been
        processed without one, even with no detected change (`R_psi` in the paper).
    :raises ValueError: if `change_point_detector` was constructed with a different `VladMemory`
        instance than `memory` — otherwise the detector would populate one memory while
        replay/summarization silently read from another, empty one.

    When using, please cite:
      ```
          @article{faber2023vlad,
          title={VLAD: Task-agnostic VAE-based lifelong anomaly detection},
          author={Faber, Kamil and Corizzo, Roberto and Sniezynski, Bartlomiej and Japkowicz, Nathalie},
          journal={Neural Networks},
          volume={165},
          pages={248--273},
          year={2023},
          publisher={Elsevier}
        }
      ```
    """

    def __init__(
        self,
        model: Model,
        change_point_detector: ChangePointDetector,
        memory: VladMemory,
        max_steps_between_updates: int,
    ):
        if change_point_detector.memory is not memory:
            raise ValueError(
                "change_point_detector and memory must share the same VladMemory instance. "
                "Construct the detector with the same `memory` object passed here, e.g.: "
                "memory = VladMemory(...); VladStrategy(model, "
                "WassersteinChangePointDetector(memory), memory, ...)."
            )
        self._model = model
        self._change_point_detector = change_point_detector
        self._memory = memory
        self._max_steps_between_updates = max_steps_between_updates
        self._steps_since_update = 0
        self._ever_trained = False

    def learn(self, data: np.ndarray) -> None:
        change_points = self._change_point_detector.detect(data)
        new_concept_detected = any(cp.is_new_concept for cp in change_points)
        recurring_concept_detected = any(not cp.is_new_concept for cp in change_points)

        summarized = False
        if self._memory.should_summarize():
            self._memory.summarize()
            summarized = True

        self._steps_since_update += len(data)

        should_update_model = (
            not self._ever_trained
            or new_concept_detected
            or recurring_concept_detected
            or self._steps_since_update > self._max_steps_between_updates
            or summarized
        )
        if should_update_model:
            replay = self._memory.replay_buffer()
            train_data = np.concatenate([replay, data]) if len(replay) > 0 else data
            self._model.fit(train_data)
            self._ever_trained = True
            self._steps_since_update = 0

    def predict(self, data: np.ndarray) -> PredictionResults:
        return self._model.predict(data)

    def name(self) -> str:
        return "VLAD"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "change_point_detector": self._change_point_detector.info(),
            "memory": self._memory.info(),
            "max_steps_between_updates": self._max_steps_between_updates,
        }


class VladConceptAgnosticStrategy(VladStrategy):
    """VLAD for the **concept-agnostic** scenario — pair with `ConceptAgnosticScenario`.

    Constructs a `WassersteinChangePointDetector` for you, so there's no detector to import or
    wire up yourself; this is VLAD as published, for scenarios with no boundary information.

    :param model: see `VladStrategy`.
    :param memory: see `VladStrategy`. Build it with `VladMemory.for_concept_agnostic(...)`
        rather than the plain constructor, so it gets the `distance` default validated for this
        detector's `mini_batch_size`-sized comparisons.
    :param max_steps_between_updates: see `VladStrategy`.
    :param refresh_every: forwarded to `WassersteinChangePointDetector`.
    """

    def __init__(self, model: Model, memory: VladMemory, max_steps_between_updates: int, refresh_every: int = 100):
        super().__init__(
            model,
            WassersteinChangePointDetector(memory, refresh_every=refresh_every),
            memory,
            max_steps_between_updates,
        )


class VladConceptIncrementalStrategy(VladStrategy):
    """VLAD for the **concept-incremental** scenario — pair with `ConceptIncrementalScenario`.

    Constructs a `BoundaryChangePointDetector` for you, so there's no detector to import or wire
    up yourself; trusts the scenario driver's one-call-per-concept boundaries and skips online
    drift scanning, but still classifies each concept as new vs. recurring against memory.

    :param model: see `VladStrategy`.
    :param memory: see `VladStrategy`. Build it with `VladMemory.for_concept_incremental(...)`
        rather than the plain constructor, so it gets the faster `SlicedWassersteinDistance`
        default validated as reliable for this mode's whole-batch comparisons.
    :param max_steps_between_updates: see `VladStrategy`.
    """

    def __init__(self, model: Model, memory: VladMemory, max_steps_between_updates: int):
        super().__init__(model, BoundaryChangePointDetector(memory), memory, max_steps_between_updates)
