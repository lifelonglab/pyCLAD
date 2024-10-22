from typing import List
from unittest.mock import patch

import pytest

from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback


@pytest.mark.parametrize("expected", [[20, 40, 1124, 15124], [55, 22, 1124, 4111]])
def test_correct_current_size(expected: List[int]):
    with patch("pyclad.callbacks.evaluation.memory_usage.tracemalloc") as mock_tracemalloc:
        callback = MemoryUsageCallback()
        for i, value in enumerate(expected):
            mock_tracemalloc.get_traced_memory.return_value = (value, 0)
            callback.after_concept_processing(str(i))

        results = callback.info()["memory_usage_callback"]["current_memory_usage"]
        for i, value in enumerate(expected):
            assert results[str(i)] == value


@pytest.mark.parametrize("expected", [[20, 40, 1124, 15124], [55, 22, 1124, 4111]])
def test_correct_peak_size(expected: List[int]):
    with patch("pyclad.callbacks.evaluation.memory_usage.tracemalloc") as mock_tracemalloc:
        callback = MemoryUsageCallback()
        for i, value in enumerate(expected):
            mock_tracemalloc.get_traced_memory.return_value = (0, value)
            callback.after_concept_processing(str(i))

        results = callback.info()["memory_usage_callback"]["peak_memory_usage"]
        for i, value in enumerate(expected):
            assert results[str(i)] == value
