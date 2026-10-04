import torch

from pyclad.strategies.replay.buffers.reservoir import ReservoirBuffer


def test_info_reports_capacity_and_fill_level():
    buffer = ReservoirBuffer(max_capacity=4)
    samples = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    buffer.update(samples, samples, samples)

    info = buffer.info()

    assert info["max_capacity"] == 4
    assert info["current_size"] == 4
    assert info["items_seen"] == 6
    assert info["device"] == "cpu"
