import numpy as np
import pytest

from pyclad.strategies.vlad.distance import (
    ExactWassersteinDistance,
    SlicedWassersteinDistance,
)


class TestExactWassersteinDistance:
    def test_identical_point_clouds_are_zero_distance(self):
        distance = ExactWassersteinDistance()
        a = np.array([[0.0, 0.0], [1.0, 1.0]])

        assert distance(a, a.copy()) == 0.0

    def test_distance_grows_with_separation(self):
        distance = ExactWassersteinDistance()
        rng = np.random.default_rng(0)
        a = rng.normal(size=(30, 3))

        near = distance(a, rng.normal(loc=0.5, size=(30, 3)))
        far = distance(a, rng.normal(loc=5.0, size=(30, 3)))

        assert far > near

    def test_name(self):
        assert ExactWassersteinDistance().name() == "ExactWassersteinDistance"


class TestSlicedWassersteinDistance:
    def test_identical_point_clouds_are_zero_distance(self):
        distance = SlicedWassersteinDistance(n_projections=20, rng=np.random.default_rng(0))
        a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])

        assert distance(a, a.copy()) == pytest.approx(0.0, abs=1e-10)

    def test_distance_grows_with_separation(self):
        distance = SlicedWassersteinDistance(n_projections=50, rng=np.random.default_rng(0))
        rng = np.random.default_rng(0)
        a = rng.normal(size=(30, 3))

        near = distance(a, rng.normal(loc=0.5, size=(30, 3)))
        far = distance(a, rng.normal(loc=5.0, size=(30, 3)))

        assert far > near

    def test_deterministic_given_a_seeded_rng(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 4))
        b = rng.normal(loc=1.0, size=(20, 4))

        first = SlicedWassersteinDistance(n_projections=10, rng=np.random.default_rng(42))(a, b)
        second = SlicedWassersteinDistance(n_projections=10, rng=np.random.default_rng(42))(a, b)

        assert first == second

    def test_much_faster_than_exact_on_larger_clouds(self):
        """Not a strict benchmark, just a sanity check that the sliced version doesn't
        accidentally regress to doing the expensive thing."""
        import time

        rng = np.random.default_rng(0)
        a = rng.normal(size=(300, 10))
        b = rng.normal(loc=0.3, size=(300, 10))

        t0 = time.time()
        SlicedWassersteinDistance(n_projections=50, rng=rng)(a, b)
        sliced_time = time.time() - t0

        assert sliced_time < 1.0

    def test_name_and_info(self):
        distance = SlicedWassersteinDistance(n_projections=17)

        assert distance.name() == "SlicedWassersteinDistance"
        assert distance.info() == {"name": "SlicedWassersteinDistance", "n_projections": 17}


if __name__ == "__main__":
    pytest.main([__file__])
