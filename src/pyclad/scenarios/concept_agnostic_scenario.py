import logging
from typing import List

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptAgnosticStrategy

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def concept_agnostic_scenario(data_loader: ConceptsDataset, strategy: ConceptAgnosticStrategy,
                              callbacks: List[Callback], batch_size=256):
    callback_composite = CallbackComposite(callbacks)

    for train_concept in data_loader.train_concepts():
        batch_id = 0
        callback_composite.before_training()
        while batch_size * batch_id < len(train_concept.data):
            batch = train_concept.data[batch_id * batch_size: (batch_id + 1) * batch_size]
            logger.info(f'Starting training on concept {train_concept.name}, batch: {batch_id} with size {len(batch)}')
            strategy.learn(data=batch)
            batch_id += 1
        callback_composite.after_training(learned_concept=train_concept)

        for test_concept in data_loader.test_concepts():
            logger.info(f'Starting evaluation of concept {train_concept.name}')
            callback_composite.before_evaluation()
            anomaly_scores, y_predicted = strategy.predict(data=test_concept.data)
            callback_composite.after_evaluation(evaluated_concept=test_concept, y_true=test_concept.labels,
                                                y_pred=y_predicted, anomaly_scores=anomaly_scores)

    callback_composite.after_scenario()

