import logging
from typing import List

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptAwareStrategy

logger = logging.getLogger(__name__)


class ConceptAwareScenario:
    def __init__(self, dataset: ConceptsDataset, strategy: ConceptAwareStrategy, callbacks: List[Callback]):
        self._dataset = dataset
        self._strategy = strategy
        self._callbacks = callbacks

    def run(self):
        callback_composite = CallbackComposite(self._callbacks)

        for train_concept in self._dataset.train_concepts():
            logger.info(f"Starting training on concept {train_concept.name}")
            callback_composite.before_training()
            self._strategy.learn(data=train_concept.data, concept_id=train_concept.name)
            callback_composite.after_training(learned_concept=train_concept)

            for test_concept in self._dataset.test_concepts():
                logger.info(f"Starting evaluation of concept {train_concept.name}")
                callback_composite.before_evaluation()
                anomaly_scores, y_predicted = self._strategy.predict(
                    data=test_concept.data, concept_id=test_concept.name
                )
                callback_composite.after_evaluation(
                    evaluated_concept=test_concept,
                    y_true=test_concept.labels,
                    y_pred=y_predicted,
                    anomaly_scores=anomaly_scores,
                )

        callback_composite.after_scenario()
