import logging
from typing import List, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.models.vision_model import VisionModel

logger = logging.getLogger(__name__)


class VisionConceptIncrementalScenario:
    """Concept-incremental scenario that surfaces per-pixel ``score_maps`` to callbacks.

    Mirrors :class:`pyclad.scenarios.concept_incremental.ConceptIncrementalScenario`
    except that on each evaluation it also asks the strategy's current model for
    pixel-level anomaly maps and passes them to callbacks as an extra
    ``score_maps`` keyword argument on ``after_evaluation``. This lets
    :class:`pyclad.vision.callbacks.vision_pixel_concept_metric_callback.VisionPixelConceptMetricCallback`
    consume pixel-level data without reaching into the strategy or model itself —
    matching the contract used by every other callback in pyclad.

    Non-vision callbacks ignore the new kwarg (they consume ``*args, **kwargs``).
    """

    def __init__(
        self,
        dataset: ConceptsDataset,
        strategy: ConceptIncrementalStrategy,
        callbacks: List[Callback],
    ):
        self._dataset = dataset
        self._strategy = strategy
        self._callbacks = callbacks

    def run(self) -> None:
        callback_composite = CallbackComposite(self._callbacks)
        callback_composite.before_scenario()

        for train_concept in self._dataset.train_concepts():
            logger.info(f"Starting training on concept {train_concept.name}")
            callback_composite.before_concept_processing(concept=train_concept)
            callback_composite.before_training()

            self._strategy.learn(data=train_concept.data)

            callback_composite.after_training(learned_concept=train_concept)

            for test_concept in self._dataset.test_concepts():
                logger.info(f"Starting evaluation of concept {test_concept.name}")
                callback_composite.before_evaluation()
                y_predicted, anomaly_scores = self._strategy.predict(data=test_concept.data)
                score_maps = self._compute_score_maps(test_concept)
                callback_composite.after_evaluation(
                    evaluated_concept=test_concept,
                    y_true=test_concept.labels,
                    y_pred=y_predicted,
                    anomaly_scores=anomaly_scores,
                    score_maps=score_maps,
                )

            callback_composite.after_concept_processing(concept=train_concept)

        callback_composite.after_scenario()

    def _compute_score_maps(self, evaluated_concept: Concept) -> Optional[np.ndarray]:
        if not isinstance(evaluated_concept, VisionConcept):
            return None
        model = self._strategy.model_for_concept(evaluated_concept.name)
        if not isinstance(model, VisionModel):
            return None
        return np.asarray(model.score_maps(evaluated_concept.data))
