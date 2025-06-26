import logging
from typing import List

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptAgnosticStrategy

logger = logging.getLogger(__name__)


class ConceptAgnosticScenario:
    def __init__(self, dataset: ConceptsDataset, strategy: ConceptAgnosticStrategy, callbacks: List[Callback]):
        self._dataset = dataset
        self._strategy = strategy
        self._callbacks = callbacks

    def run(self, batch_size=256):
        callback_composite = CallbackComposite(self._callbacks)
        callback_composite.before_scenario()

        for train_concept in self._dataset.train_concepts():
            callback_composite.before_concept_processing(concept=train_concept)
            callback_composite.before_training()

            batch_id = 0
            while batch_size * batch_id < len(train_concept.data):
                batch = train_concept.data[batch_id * batch_size : (batch_id + 1) * batch_size]
                logger.info(
                    f"Starting training on concept {train_concept.name}, batch: {batch_id} with size {len(batch)}"
                )
                self._strategy.learn(data=batch)
                batch_id += 1
            callback_composite.after_training(learned_concept=train_concept)

            for test_concept in self._dataset.test_concepts():
                logger.info(f"Starting evaluation of concept {train_concept.name}")
                callback_composite.before_evaluation()
                y_predicted, anomaly_scores = self._strategy.predict(data=test_concept.data)
                callback_composite.after_evaluation(
                    evaluated_concept=test_concept,
                    y_true=test_concept.labels,
                    y_pred=y_predicted,
                    anomaly_scores=anomaly_scores,
                )

            callback_composite.after_concept_processing(concept=train_concept)

        callback_composite.after_scenario()
