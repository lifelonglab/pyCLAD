import json
import random

import numpy as np
import pytest
import torch

from pyclad.callbacks import ConceptMetricCallback
from pyclad.data import Concept, ConceptsDataset
from pyclad.metrics import ContinualAverage, RocAuc
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios import ConceptIncrementalScenario
from pyclad.seed import new_generator, set_seed
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.strategies.vlad.distance import SlicedWassersteinDistance


def _draws():
    return random.random(), float(np.random.rand()), float(torch.rand(1)), float(new_generator().random())


def test_same_seed_repeats_every_source_of_randomness():
    set_seed(7)
    first = _draws()
    set_seed(7)

    assert _draws() == first


def test_different_seeds_give_different_draws():
    set_seed(7)
    first = _draws()
    set_seed(8)

    assert all(a != b for a, b in zip(_draws(), first))


def test_generator_with_its_own_seed_ignores_the_global_seed():
    set_seed(7)
    first = new_generator(123).random()
    set_seed(8)

    assert new_generator(123).random() == first
    assert first == np.random.default_rng(123).random()


def test_component_given_no_generator_follows_the_seed():
    a, b = np.random.default_rng(0).random((20, 3)), np.random.default_rng(1).random((20, 3))

    set_seed(7)
    first = SlicedWassersteinDistance(n_projections=5)(a, b)
    set_seed(7)

    assert SlicedWassersteinDistance(n_projections=5)(a, b) == first


def test_seed_is_saved_with_the_results(tmp_path):
    path = tmp_path / "output.json"

    JsonOutputWriter(path).write([set_seed(7)])

    assert json.loads(path.read_text()) == {"seed": 7}


def _run_scenario():
    set_seed(7)
    rng = np.random.default_rng(0)  # the data is fixed; only the experiment's own randomness is under test
    train = [Concept(f"c{i}", data=rng.normal(i, 1, (80, 4))) for i in range(3)]
    test = [Concept(f"c{i}", data=rng.normal(i, 1, (40, 4)), labels=np.tile([0, 1], 20)) for i in range(3)]
    callback = ConceptMetricCallback(RocAuc(), [ContinualAverage()])
    strategy = ReplayEnhancedStrategy(
        IsolationForestAdapter(n_estimators=10),
        AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=30),
    )
    ConceptIncrementalScenario(ConceptsDataset("dataset", train, test), strategy, [callback]).run()
    return callback.info()["concept_metric_callback_ROC-AUC"]["metric_matrix"]


@pytest.fixture(scope="module")
def two_runs():
    return _run_scenario(), _run_scenario()


def test_seeded_scenario_gives_identical_results(two_runs):
    first, second = two_runs

    assert first == second
