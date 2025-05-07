import logging

import numpy as np
import torch

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

from pyclad.models.autoencoder.autoencoder import VariationalTemporalAutoencoder
from pyclad.models.autoencoder.config import (
    DecoderConfig,
    EncoderConfig,
    GRULayerConfig,
    AutoencoderConfig,
)
from pyclad.models.autoencoder.variational.gru import (
    GRUVariationalDecoder,
    GRUVariationalEncoder,
)

if __name__ == "__main__":
    batch_size, seq_len = 32, 10
    time_steps, n_features = 5000, 5
    dataset = np.random.rand(time_steps, n_features)

    config = AutoencoderConfig(
        seq_len=seq_len,
        encoder=EncoderConfig(
            layers=[
                GRULayerConfig(
                    input_size=n_features,
                    hidden_size=64,
                    num_layers=1,
                    activation="ReLU",
                    dropout=0.2,
                    bidirectional=False,
                ),
                GRULayerConfig(
                    input_size=64, hidden_size=32, num_layers=1, activation="ReLU", dropout=0.1, bidirectional=False
                ),
            ]
        ),
        decoder=DecoderConfig(
            layers=[
                GRULayerConfig(
                    input_size=32, hidden_size=64, num_layers=1, activation="ReLU", dropout=0.2, bidirectional=False
                ),
                GRULayerConfig(
                    input_size=64,
                    hidden_size=n_features,
                    num_layers=1,
                    activation="ReLU",
                    dropout=0.1,
                    bidirectional=False,
                ),
            ]
        ),
    )

    autoencoder = VariationalTemporalAutoencoder(
        GRUVariationalEncoder(config=config), GRUVariationalDecoder(config=config), epochs=5
    )

    autoencoder.fit(dataset)

    input_batch = torch.randn(batch_size, seq_len, n_features)
    true_labels_batch = torch.randn(batch_size, seq_len, 1)

    binary_predictions, rec_error = autoencoder.predict(input_batch.numpy())
    assert binary_predictions.shape == true_labels_batch.shape
