import numpy as np
import pytest

from pyclad.data.concept import Concept
from pyclad.vision.data.benchmarks.continual_mega import ContinualMegaDataset


def _dataset(held_out_groups=()):
    groups = {"screw": "base", "pill": "base", "cable": "task_1", "hazelnut": "task_2"}
    for index, group in enumerate(held_out_groups):
        groups[f"heldout_{index}"] = group
    return ContinualMegaDataset(
        name="Continual-MEGA-test",
        train_concepts=[Concept(name=name, data=np.zeros((1, 2))) for name in ("base", "task_1", "task_2")],
        test_concepts=[Concept(name=name, data=np.zeros((1, 2))) for name in groups],
        group_by_concept=groups,
        training_groups=["base", "task_1", "task_2"],
        held_out_groups=list(held_out_groups),
        scenario=2,
        task_size=30,
    )


def test_first_seen_step_indexes_the_training_group_order():
    assert _dataset().first_seen_step() == {"screw": 0, "pill": 0, "cable": 1, "hazelnut": 2}


def test_first_seen_step_covers_every_test_concept():
    dataset = _dataset()
    assert set(dataset.first_seen_step()) == set(dataset.group_by_concept())


def test_first_seen_step_rejects_held_out_groups():
    with pytest.raises(ValueError, match="zero_shot=False"):
        _dataset(held_out_groups=["zeroshot_mvtec"]).first_seen_step()
