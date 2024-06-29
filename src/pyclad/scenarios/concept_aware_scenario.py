import logging
from typing import List

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.strategies.strategy import ConceptAwareStrategy

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def concept_aware_scenario(data_loader: ConceptsDataset, strategy: ConceptAwareStrategy,
                           callbacks: List[Callback]):
    callback_composite = CallbackComposite(callbacks)

    for train_concept in data_loader.train_concepts():
        logger.info(f'Starting training on concept {train_concept.name}')
        callback_composite.before_training()
        strategy.learn(data=train_concept.data, concept_id=train_concept.name)
        callback_composite.after_training(learned_concept=train_concept)

        for test_concept in data_loader.test_concepts():
            logger.info(f'Starting evaluation of concept {train_concept.name}')
            callback_composite.before_evaluation()
            anomaly_scores, y_predicted = strategy.predict(data=test_concept.data, concept_id=test_concept.name)
            callback_composite.after_evaluation(evaluated_concept=test_concept, y_true=test_concept.labels,
                                                y_pred=y_predicted, anomaly_scores=anomaly_scores)

    callback_composite.after_scenario()


