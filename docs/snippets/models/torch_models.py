# requires: torch
import numpy as np

# --8<-- [start:autoencoder]
from torch import nn

from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.autoencoder.autoencoder import Autoencoder, VariationalAutoencoder
from pyclad.models.training.early_stopping import EarlyStopping
from pyclad.models.training.runners.standard import StandardRunner

features = 8
encoder = nn.Sequential(nn.Linear(features, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU())
decoder = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, features), nn.Sigmoid())
backbone = Autoencoder(encoder=encoder, decoder=decoder, lr=1e-2)

# The runner owns the training loop. Here it holds out 10% of each concept's training data for validation
# and stops when the validation loss has not improved for 3 epochs.
runner = StandardRunner(
    max_epochs=50,
    validation_fraction=0.1,
    early_stopping=EarlyStopping(patience=3),
    seed=0,
)

# The adapter turns the backbone into a Model, usable with any strategy.
model = TorchModelAdapter(backbone, runner, batch_size=32)
# --8<-- [end:autoencoder]

# --8<-- [start:vae]
vae_backbone = VariationalAutoencoder(
    encoder=nn.Sequential(nn.Linear(features, 6), nn.ReLU()),  # outputs hidden_dim features
    decoder=nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, features), nn.Sigmoid()),  # takes latent_dim
    hidden_dim=6,
    latent_dim=2,
)
vae = TorchModelAdapter(vae_backbone, StandardRunner(max_epochs=5), batch_size=32)
# --8<-- [end:vae]

train = np.random.default_rng(0).uniform(0.4, 0.6, size=(300, features))
for m in (model, vae):
    m.fit(train)
    print(m.name(), m.predict(train[:3]).anomaly_scores)
