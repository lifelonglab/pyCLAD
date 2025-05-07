import logging

import numpy as np
import torch

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

from pyclad.models.autoencoder.autoencoder import TemporalAutoencoder
from pyclad.models.autoencoder.config import (
    DecoderConfig,
    EncoderConfig,
    AutoencoderConfig,
    TCNLayerConfig,
)
from pyclad.models.autoencoder.standard.tcn import TCNDecoder, TCNEncoder

if __name__ == "__main__":
    batch_size, seq_len = 32, 10
    time_steps, n_features = 5000, 5
    dataset = np.random.rand(time_steps, n_features)

    config = AutoencoderConfig(
        seq_len=seq_len,
        encoder=EncoderConfig(
            layers=[
                TCNLayerConfig(
                    in_channels=n_features,
                    out_channels=2 * 64,
                    kernel_size=3,
                    padding=1,
                    activation="Tanh",
                ),
                TCNLayerConfig(
                    in_channels=2 * 64,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    activation="Tanh",
                ),
            ]
        ),
        decoder=DecoderConfig(
            layers=[
                TCNLayerConfig(
                    in_channels=64,
                    out_channels=2 * 64,
                    kernel_size=3,
                    padding=1,
                    activation="Tanh",
                ),
                TCNLayerConfig(
                    in_channels=2 * 64,
                    out_channels=n_features,
                    kernel_size=3,
                    padding=1,
                    activation="Tanh",
                ),
            ]
        ),
    )

    autoencoder = TemporalAutoencoder(TCNEncoder(config=config), TCNDecoder(config=config), epochs=5)

    autoencoder.fit(dataset)

    input_batch = torch.randn(batch_size, seq_len, n_features)
    true_labels_batch = torch.randn(batch_size, seq_len, 1)

    binary_predictions, rec_error = autoencoder.predict(input_batch.numpy())
    assert binary_predictions.shape == true_labels_batch.shape
