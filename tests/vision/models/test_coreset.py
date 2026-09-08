"""Shared greedy-coreset sampler: both sizings, and the invariant that ties them together."""

import numpy as np
import pytest
import torch

from pyclad.vision.models.utilities.coreset import ApproximateGreedyCoresetSampler

DEVICE = torch.device("cpu")
SAMPLER_KWARGS = dict(number_of_starting_points=3, dimension_to_project_features_to=8, random_seed=7)


def _features(rows: int, dim: int = 16) -> np.ndarray:
    return np.random.RandomState(rows + dim).rand(rows, dim).astype(np.float32)


def _sampler(**overrides):
    return ApproximateGreedyCoresetSampler(device=DEVICE, **{**SAMPLER_KWARGS, **overrides})


def test_target_size_returns_exactly_that_many_points():
    coreset = _sampler().run_with_target_size(_features(60), target_size=17)
    assert coreset.shape == (17, 16)


def test_target_size_keeps_everything_when_the_pool_is_smaller():
    features = _features(5)
    np.testing.assert_array_equal(_sampler().run_with_target_size(features, target_size=196), features)


@pytest.mark.parametrize("target_size", [0, -1])
def test_target_size_must_be_positive(target_size):
    with pytest.raises(ValueError, match="target_size must be positive"):
        _sampler().run_with_target_size(_features(20), target_size=target_size)


def test_percentage_keeps_that_fraction_of_the_pool():
    assert _sampler(percentage=0.3).run(_features(100)).shape == (30, 16)


def test_percentage_keeps_at_least_one_point():
    assert _sampler(percentage=0.01).run(_features(7)).shape == (1, 16)


def test_a_full_percentage_returns_the_pool_untouched():
    features = _features(40)
    assert _sampler(percentage=1.0).run(features) is features


def test_run_without_a_percentage_is_rejected():
    with pytest.raises(ValueError, match="needs a percentage"):
        _sampler().run(_features(20))


@pytest.mark.parametrize("percentage", [0.0, -0.2, 1.5])
def test_percentage_outside_the_unit_interval_is_rejected(percentage):
    with pytest.raises(ValueError, match=r"percentage must be in \(0, 1\]"):
        _sampler(percentage=percentage)


def test_the_two_entry_points_agree_when_they_ask_for_the_same_count():
    """`run(0.25)` on 80 points and `run_with_target_size(20)` must select the same 20."""
    features = _features(80)
    np.testing.assert_array_equal(
        _sampler(percentage=0.25).run(features),
        _sampler().run_with_target_size(features, target_size=20),
    )


def test_the_same_seed_selects_the_same_coreset():
    features = _features(50)
    first = _sampler(random_seed=3).run_with_target_size(features, target_size=11)
    second = _sampler(random_seed=3).run_with_target_size(features, target_size=11)
    np.testing.assert_array_equal(first, second)


def test_a_different_seed_selects_a_different_coreset():
    features = _features(50)
    assert not np.array_equal(
        _sampler(random_seed=3).run_with_target_size(features, target_size=11),
        _sampler(random_seed=99).run_with_target_size(features, target_size=11),
    )
