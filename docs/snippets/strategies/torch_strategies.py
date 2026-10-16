# requires: torch
import numpy as np
from torch import nn

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import BackwardTransfer, ContinualAverage, RocAuc
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import set_seed
from pyclad.strategies.regularization.der import DerPlusPlus
from pyclad.strategies.regularization.ewc import EWCStrategy
from pyclad.strategies.regularization.lwf import LwFStrategy
from pyclad.strategies.replay.agem import AGEMStrategy
from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer

set_seed(0)
train, test = [], []
for i, center in enumerate([0.2, 0.5, 0.8]):
    train.append(Concept(f"c{i}", data=np.random.normal(center, 0.05, size=(200, 8)).clip(0, 1)))
    test_data = np.concatenate([np.random.normal(center, 0.05, size=(50, 8)), np.random.uniform(0, 1, size=(5, 8))])
    test.append(Concept(f"c{i}", data=test_data.clip(0, 1), labels=np.array([0] * 50 + [1] * 5)))
dataset = ConceptsDataset(name="Synthetic", train_concepts=train, test_concepts=test)


# --8<-- [start:backbone]
def make_autoencoder(features: int = 8) -> Autoencoder:
    encoder = nn.Sequential(nn.Linear(features, 4), nn.ReLU())
    decoder = nn.Sequential(nn.Linear(4, features), nn.Sigmoid())
    return Autoencoder(encoder=encoder, decoder=decoder)


# --8<-- [end:backbone]

# --8<-- [start:strategies]
runner = StandardRunner(max_epochs=5)  # owns the training loop; see Models

strategies = [
    EWCStrategy(model=make_autoencoder(), runner=runner, lambda_ewc=1.0),
    LwFStrategy(model=make_autoencoder(), runner=runner, alpha=0.5),
    DerPlusPlus(model=make_autoencoder(), runner=runner, buffer=ReservoirBuffer(max_capacity=200), alpha=0.5, beta=0.5),
    AGEMStrategy(model=make_autoencoder(), runner=runner, buffer=ReservoirBuffer(max_capacity=200)),
]
# --8<-- [end:strategies]

# DER++ implements only the concept-agnostic contract; the others also run concept-incrementally.
for strategy in strategies[:2] + strategies[3:]:
    callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage(), BackwardTransfer()])
    ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[callback]).run()
    print(strategy.name(), callback.info()["concept_metric_callback_ROC-AUC"]["metrics"])
