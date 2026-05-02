import numpy as np
import torch
from torch import nn

from pyclad.models.autoencoder.autoencoder import (
    Autoencoder,
    TemporalAutoencoder,
    VariationalTemporalAutoencoder,
)


class _VariationalIdentityEncoder(nn.Module):
    def forward(self, x):
        return x, torch.zeros_like(x)


def test_autoencoder_predicts_label_and_score_vectors():
    model = Autoencoder(nn.Identity(), nn.Identity(), threshold=0.1, device="cpu")
    labels, scores = model.predict(np.ones((4, 3), dtype=np.float32))

    assert labels.shape == (4,)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def test_temporal_autoencoder_predicts_sequence_scores():
    model = TemporalAutoencoder(nn.Identity(), nn.Identity(), threshold=0.1, device="cpu")
    labels, scores = model.predict(np.ones((2, 3, 4), dtype=np.float32))

    assert labels.shape == (2, 3, 1)
    assert scores.shape == (2, 3, 1)
    assert np.isfinite(scores).all()


def test_variational_temporal_autoencoder_predicts_sequence_scores():
    model = VariationalTemporalAutoencoder(
        _VariationalIdentityEncoder(),
        nn.Identity(),
        threshold=0.1,
        device="cpu",
    )
    labels, scores = model.predict(np.ones((2, 3, 4), dtype=np.float32))

    assert labels.shape == (2, 3, 1)
    assert scores.shape == (2, 3, 1)
    assert np.isfinite(scores).all()
