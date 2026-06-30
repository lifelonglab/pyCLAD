import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from pyclad.models.feature_extractor import FeatureExtractor
from pyclad.models.model import Model
from pyclad.output.prediction_results import PredictionResults

logger = logging.getLogger(__name__)


ANOMALIES = ("seasonal", "trend", "global", "contextual", "shapelet")


def _inject_anomaly(window: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Injects a single anomaly type into a multivariate time series window.

    The implementation is meant to match the reference
    https://github.com/zamanzadeh/CARLA/blob/600d4f2af3e625e63e0204617f8b29d227ca2973/data/augment.py#L23
    the harcoded values come from there.

    Number of affected dims is ``randint(num_features // 10, num_features // 2)``.

    :param window: Time series window of ``(window_size, num_features)``.
    :param rng: numpy random Generator used for all random draws.
    :return: Anomaly-injected copy of the input window.
    """
    if rng is None:
        rng = np.random.default_rng()

    window_size, num_features = window.shape
    output_window = window.copy()

    min_dims = max(1, num_features // 10)
    max_dims = max(min_dims + 1, num_features // 2)
    num_anomalous_dims = int(rng.integers(min_dims, max_dims))
    anomaly_dims = rng.choice(num_features, size=num_anomalous_dims, replace=False)

    min_len = max(1, int(window_size * 0.1))
    max_len = max(min_len + 1, int(window_size * 0.9))
    subsequence_length = int(rng.integers(min_len, max_len))
    start = int(rng.integers(0, max(1, window_size - subsequence_length)))
    end = start + subsequence_length

    anomaly_type = rng.choice(ANOMALIES)
    match anomaly_type:
        case "seasonal":
            compression_factor = int(rng.integers(2, 5))
            offsets = (np.arange(subsequence_length) * compression_factor) % subsequence_length
            for dim in anomaly_dims:
                output_window[start:end, dim] = window[start + offsets, dim]
        case "trend":
            trend_factor = float(rng.normal(1.0, 0.5))
            coef = 1.0 if rng.uniform() >= 0.5 else -1.0
            for dim in anomaly_dims:
                output_window[start:, dim] = window[start:, dim] + coef * trend_factor
        case "global":
            g_len = 2
            g_start = int(rng.integers(0, max(1, window_size - g_len)))
            g_end = g_start + g_len
            for dim in anomaly_dims:
                output_window[g_start:g_end, dim] = window[g_start:g_end, dim] * 8.0
        case "contextual":
            c_len = 4
            c_start = int(rng.integers(0, max(1, window_size - c_len)))
            c_end = c_start + c_len
            for dim in anomaly_dims:
                output_window[c_start:c_end, dim] = window[c_start:c_end, dim] * 3.0
        case "shapelet":
            for dim in anomaly_dims:
                const = window[start, dim] + float(rng.random()) * 0.1
                output_window[start:end, dim] = const
        case _:
            err_msg = f"Unknown anomaly type: {anomaly_type}"
            raise ValueError(err_msg)

    return output_window


class _PretextDataset(Dataset):
    """Pre-built triplets (anchor, positive, negative).

    These triplets are used for triplet loss used at the pretext
    of training CARLA model.
    Each triplet consists of:
     * anchor: original time series window
     * positive: a window, up to ``positive_lookback`` in the past
     * negative: original window with anomaly injected.
    """

    def __init__(
        self,
        windows: np.ndarray,
        positive_lookback: int,
        rng: np.random.Generator,
    ) -> None:
        n_windows = len(windows)

        offsets = rng.integers(1, positive_lookback + 1, size=n_windows)
        p_idx = np.maximum(0, np.arange(n_windows) - offsets)

        negatives = np.stack([_inject_anomaly(windows[win_idx], rng=rng) for win_idx in range(n_windows)])
        self.anchors = torch.from_numpy(windows).float()
        self.negatives = torch.from_numpy(negatives).float()
        self.positives = self.anchors[p_idx]

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.anchors[index], self.positives[index], self.negatives[index]


class _L2Normalize(nn.Module):
    """Layer for L2-Normalization.

    It is used to normalise the outputs of the pretext stage head, as these are being
    used only in their normalised form.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)


def _pretext_loss(
    original: torch.Tensor,
    past: torch.Tensor,
    anomalous: torch.Tensor,
    margin: float,
    temperature: float,
) -> Tensor:
    """Triplet loss with hard-negative mining.

    The pretext module is expected to L2-normalise its output. For each anchor
    we compute the squared distance to every anomaly-injected window in the
    batch and use the minimum (hardest) as the effective negative distance.
    Distances are scaled by ``temperature`` before the relu-margin combination.

    :param original: Model output for original time window, shape ``(B, D)``.
    :param past: Model output for past time window, shape ``(B, D)``.
    :param anomalous: Model output for anomaly-injected time windows in the
        batch, shape ``(B, D)``. The hardest of these is selected per anchor.
    :param margin: margin to guarantee separation between positive and
        hardest-negative pairs.
    :param temperature: temperature scaling applied to squared distances.
    """
    d_pos = (original - past).pow(2).sum(dim=-1) / temperature
    pairwise_neg = (original.unsqueeze(1) - anomalous.unsqueeze(0)).pow(2).sum(dim=-1) / temperature
    d_neg_hardest = pairwise_neg.min(dim=1).values
    triplet = F.relu(d_pos - d_neg_hardest + margin).mean()

    return triplet


def _classification_loss(
    anchor: Tensor,
    nearest: Tensor,
    furthest: Tensor,
    entropy_loss_weight: float,
    inconsistency_loss_weight: float,
) -> Tensor:
    """Computes the self-supervised classification loss for CARLA.

    :param anchor: model softmax output for anchor windows, shape (B, C).
    :param nearest: model softmax output for the Q nearest neighbours per
        anchor, shape (B, Q, C).
    :param furthest: model softmax output for the Q furthest neighbours per
        anchor, shape (B, Q, C).
    :param entropy_loss_weight: weight for the loss_entropy term.
    :param inconsistency_loss_weight: weight for the furthest-neighbour
        (inconsistency) term.
    """
    sim_nearest = (anchor.unsqueeze(1) * nearest).sum(dim=-1)
    sim_furthest = (anchor.unsqueeze(1) * furthest).sum(dim=-1)
    eps = 1e-7
    loss_consistency = -torch.log(sim_nearest.clamp(min=eps)).mean()
    loss_inconsistency = -torch.log((1.0 - sim_furthest).clamp(min=eps)).mean()

    # clampign for numerical stability before taking log.
    p_bar = anchor.mean(dim=0).clamp(min=eps)
    loss_entropy = -(p_bar * torch.log(p_bar)).sum()

    return loss_consistency + inconsistency_loss_weight * loss_inconsistency - entropy_loss_weight * loss_entropy


def _compute_neighbours(
    module: nn.Module,
    anchors: torch.Tensor,
    negatives: torch.Tensor,
    num_neighbours: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Build the collection of all neighbours.

    Distances are computed in row-blocks of batch_size rather than as a
    single dense (2N, 2N) matrix, in order to minimize used memory.

    :returns: a triplet:
     * combined - tensor containing original and anomalous window pairs.
     * nearest_neighbour_indices - for each tensor combined[idx]
       nearest_neighbour_indices[idx] stores indices of nearest neighbours.
     * furthest_neighbour_indices - analogours to previous
    """
    combined = torch.cat([anchors, negatives], dim=0)
    n_total = len(combined)

    module.eval()
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index in range(0, n_total, batch_size):
            chunk = combined[batch_index : batch_index + batch_size].to(device)
            embeddings.append(module(chunk).cpu())
    embedded = torch.cat(embeddings, dim=0).to(device)

    nn_blocks = list[np.ndarray]()
    fn_blocks = list[np.ndarray]()
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            # Distances from this block of anchors to every window: (block, n_total).
            block = torch.cdist(embedded[start:end], embedded)

            rows = torch.arange(end - start, device=device)
            self_cols = torch.arange(start, end, device=device)
            block[rows, self_cols] = float("inf")
            nn_blocks.append(block.topk(num_neighbours, dim=1, largest=False).indices.cpu().numpy())
            block[rows, self_cols] = float("-inf")
            fn_blocks.append(block.topk(num_neighbours, dim=1, largest=True).indices.cpu().numpy())

    NN_idx = np.concatenate(nn_blocks, axis=0)
    FN_idx = np.concatenate(fn_blocks, axis=0)

    return combined, NN_idx, FN_idx


class Carla(Model):
    """CARLA self-supervised contrastive representation learning for TSAD.

    Based on the paper: https://arxiv.org/abs/2308.09296

    Training runs two stages internally:

    1. Pretext - triplet loss with anomaly-injected negatives
       trains a Feature Extractor Backbone + MLP projection head.
    2. Self-supervised classification: a classification head is
       attached on top of the pretext-trained backbone; the
       consistency / inconsistency / entropy loss is minimised over anchors
       and their Q nearest / furthest neighbours.

    Inference: a window is anomalous if its argmax class differs from the
    majority class observed at the end of training.
    """

    def __init__(
        self,
        backbone: FeatureExtractor,
        *,
        projection_dim: int = 128,
        n_classes: int = 10,
        num_neighbours: int = 10,
        margin: float = 1.0,
        temperature: float = 0.4,
        adjust_factor: float = 0.01,
        entropy_loss_weight: float = 2,
        inconsistency_loss_weight: float = 1.0,
        positive_lookback: int = 100,
        lr_pretext: float = 1e-2,
        lr_classification: float = 1e-3,
        batch_size: int = 64,
        pretext_epochs: int = 10,
        classification_epochs: int = 100,
        patience: int = 5,
        random_seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """:param backbone: Feature extractor producing ``(B, output_dim)`` from
            ``(B, window_size, num_features)``.
        :param projection_dim: Output dimensionality of the pretext MLP head.
        :param n_classes: Number of classes for the classification stage.
        :param num_neighbours: Number of nearest / furthest neighbours per anchor.
        :param margin: Initial margin for triplet loss in the pretext stage.
        :param temperature: Temperature scaling for pretext loss.
        :param adjust_factor: shrink applied to margin.
        :param entropy_loss_weight: Entropy regularizer weight in the classification loss.
        :param inconsistency_loss_weight: Weight on the furthest-neighbour (inconsistency)
            term of the classification loss.
        :param positive_lookback: Maximum look-back when sampling a positive
            pair in the pretext stage.
        :param lr_pretext: Adam learning rate, for the first stage.
        :param lr_classification: Adam learning rate, for the second stage.
        :param batch_size: Batch size for training and inference.
        :param pretext_epochs: Number of pretext-stage epochs.
        :param classification_epochs: Maximum number of classification-stage epochs.
        :param patience: Early-stopping patience (epochs without val-loss improvement).
        :param random_seed: Seed for the numpy Generator.
        :param device: PyTorch device for tensor storage.
        """
        self._num_neighbours = num_neighbours
        self._margin = margin
        self._temperature = temperature
        self._adjust_factor = adjust_factor
        self._entropy_loss_weight = entropy_loss_weight
        self._inconsistency_loss_weight = inconsistency_loss_weight
        self._positive_lookback = positive_lookback
        self._lr_pretext = lr_pretext
        self._lr_classification = lr_classification
        self._batch_size = batch_size
        self._pretext_epochs = pretext_epochs
        self._classification_epochs = classification_epochs
        self._patience = patience
        self._rng = np.random.default_rng(random_seed)
        self._device = torch.device(device)

        self._n_classes = n_classes

        output_dim = backbone.output_dim
        pretext_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, projection_dim),
        )
        self._pretext_module = nn.Sequential(backbone, pretext_head, _L2Normalize()).to(self._device)

        classification_head = nn.Sequential(
            nn.Linear(output_dim, n_classes),
            nn.Softmax(dim=-1),
        )
        self._classifier = nn.Sequential(backbone, classification_head).to(self._device)

        self._majority_class: int | None = None

        super().__init__()

    def fit(self, data: np.ndarray) -> None:
        """Run the two-stage CARLA training procedure on window-segmented data.

        Expected input shape: ``(num_windows, window_size, num_features)``.
        25% of windows are held out as a validation set (split before anomaly
        injection) for early stopping in the classification stage.
        """
        n_train = max(1, int(len(data) * 0.75))
        train_data, val_data = data[:n_train], data[n_train:]

        train_triplets = _PretextDataset(train_data, self._positive_lookback, self._rng)
        val_triplets = _PretextDataset(val_data, self._positive_lookback, self._rng)

        self._pretext_stage(self._pretext_module, train_triplets)

        train_windows, train_nn, train_fn = _compute_neighbours(
            self._pretext_module,
            train_triplets.anchors,
            train_triplets.negatives,
            self._num_neighbours,
            self._batch_size,
            self._device,
        )
        val_windows, val_nn, val_fn = _compute_neighbours(
            self._pretext_module,
            val_triplets.anchors,
            val_triplets.negatives,
            self._num_neighbours,
            self._batch_size,
            self._device,
        )

        self._classification_stage(
            self._classifier,
            train_windows,
            train_nn,
            train_fn,
            val_windows,
            val_nn,
            val_fn,
        )

        n_train_anchors = len(train_triplets.anchors)
        anchor_probs = self._predict_probabilities(self._classifier, train_windows[:n_train_anchors])
        self._majority_class = int(np.bincount(anchor_probs.argmax(axis=1), minlength=self._n_classes).argmax())

    def predict(self, data: np.ndarray) -> PredictionResults:
        """Return per-window ``PredictionResults``."""
        if self._majority_class is None:
            err_msg = "Carla.predict() called before fit()."
            raise RuntimeError(err_msg)

        x = torch.from_numpy(data).float()
        probs = self._predict_probabilities(self._classifier, x)
        labels = (probs.argmax(axis=1) != self._majority_class).astype(int)
        scores = 1.0 - probs[:, self._majority_class]
        return PredictionResults(y_pred=labels, anomaly_scores=scores)

    def _pretext_stage(self, module: nn.Module, triplets: _PretextDataset) -> None:
        optimizer = torch.optim.Adam(module.parameters(), lr=self._lr_pretext)
        module.train()
        loader = DataLoader(triplets, batch_size=self._batch_size, shuffle=True)
        margin = self._margin

        prev_loss: float | None = None
        for epoch in range(self._pretext_epochs):
            total_loss = 0.0
            n_samples = 0
            for batch in loader:
                anchors, positives, negatives = (t.to(self._device) for t in batch)
                b = anchors.shape[0]
                combined = torch.cat([anchors, positives, negatives], dim=0)
                z_all = module(combined)
                z_anchors, z_positives, z_negatives = torch.split(z_all, b, dim=0)
                if prev_loss is not None:
                    margin = max(0.01, margin - self._adjust_factor * prev_loss)
                loss = _pretext_loss(
                    z_anchors,
                    z_positives,
                    z_negatives,
                    margin=margin,
                    temperature=self._temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prev_loss = loss.item()
                total_loss += loss.item() * anchors.size(0)
                n_samples += anchors.size(0)
            logger.info(
                "[CARLA pretext] epoch %d/%d  mean_loss=%.4f  margin=%.4f",
                epoch + 1,
                self._pretext_epochs,
                total_loss / max(1, n_samples),
                margin,
            )

    def _classification_stage(
        self,
        module: nn.Module,
        train_windows: torch.Tensor,
        train_nn_indices: np.ndarray,
        train_fn_indices: np.ndarray,
        val_windows: torch.Tensor,
        val_nn_indices: np.ndarray,
        val_fn_indices: np.ndarray,
    ) -> None:
        """Runs classification stage of training with val-loss early stopping."""
        train_windows = train_windows.to(self._device)
        train_nn_idx = torch.from_numpy(train_nn_indices).to(self._device)
        train_fn_idx = torch.from_numpy(train_fn_indices).to(self._device)
        n_train = train_windows.shape[0]

        val_windows = val_windows.to(self._device)
        val_nn_idx = torch.from_numpy(val_nn_indices).to(self._device)
        val_fn_idx = torch.from_numpy(val_fn_indices).to(self._device)
        n_val = val_windows.shape[0]

        optimizer = torch.optim.Adam(module.parameters(), lr=self._lr_classification)

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        for epoch in range(self._classification_epochs):
            module.train()
            perm = torch.randperm(n_train, device=self._device)
            total_loss = 0.0
            n_samples = 0
            for start in range(0, n_train, self._batch_size):
                batch_idx = perm[start : start + self._batch_size]
                anchors = train_windows[batch_idx]
                nearest = train_windows[train_nn_idx[batch_idx]]
                furthest = train_windows[train_fn_idx[batch_idx]]
                batch_size, n_neighbours, window_size, feature_dimension = nearest.shape
                anchor_probs = module(anchors)
                nearest_probs = module(
                    nearest.reshape(batch_size * n_neighbours, window_size, feature_dimension)
                ).reshape(batch_size, n_neighbours, -1)
                furthest_probs = module(
                    furthest.reshape(batch_size * n_neighbours, window_size, feature_dimension)
                ).reshape(batch_size, n_neighbours, -1)
                loss = _classification_loss(
                    anchor_probs,
                    nearest_probs,
                    furthest_probs,
                    entropy_loss_weight=self._entropy_loss_weight,
                    inconsistency_loss_weight=self._inconsistency_loss_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size
                n_samples += batch_size
            train_mean_loss = total_loss / max(1, n_samples)

            module.eval()
            val_total = 0.0
            val_samples = 0
            with torch.no_grad():
                for start in range(0, n_val, self._batch_size):
                    end = min(start + self._batch_size, n_val)
                    batch_idx = torch.arange(start, end, device=self._device)
                    anchors = val_windows[batch_idx]
                    nearest = val_windows[val_nn_idx[batch_idx]]
                    furthest = val_windows[val_fn_idx[batch_idx]]
                    batch_size, n_neighbours, window_size, feature_dimension = nearest.shape
                    anchor_probs = module(anchors)
                    nearest_probs = module(
                        nearest.reshape(batch_size * n_neighbours, window_size, feature_dimension)
                    ).reshape(batch_size, n_neighbours, -1)
                    furthest_probs = module(
                        furthest.reshape(batch_size * n_neighbours, window_size, feature_dimension)
                    ).reshape(batch_size, n_neighbours, -1)
                    val_loss = _classification_loss(
                        anchor_probs,
                        nearest_probs,
                        furthest_probs,
                        entropy_loss_weight=self._entropy_loss_weight,
                        inconsistency_loss_weight=self._inconsistency_loss_weight,
                    )
                    val_total += val_loss.item() * batch_size
                    val_samples += batch_size
            val_mean_loss = val_total / max(1, val_samples)

            logger.info(
                "[CARLA classification] epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch + 1,
                self._classification_epochs,
                train_mean_loss,
                val_mean_loss,
            )

            if val_mean_loss < best_val_loss:
                best_val_loss = val_mean_loss
                best_state = copy.deepcopy(module.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info(
                        "[CARLA classification] early stop at epoch %d (patience=%d)",
                        epoch + 1,
                        self._patience,
                    )
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
            logger.info("[CARLA classification] restored best checkpoint (val_loss=%.4f)", best_val_loss)

    def _predict_probabilities(
        self,
        module: nn.Module,
        x: torch.Tensor,
    ) -> np.ndarray:
        """Run inference and return softmax probabilities as numpy."""
        module.eval()
        out: list[torch.Tensor] = []
        with torch.no_grad():
            # batching to save on memory
            for batch_index in range(0, len(x), self._batch_size):
                chunk = x[batch_index : batch_index + self._batch_size].to(self._device)
                out.append(module(chunk).cpu())
        return torch.cat(out, dim=0).numpy()

    def name(self) -> str:
        return "CARLA"

    def additional_info(self) -> dict:
        backbone = self._pretext_module[0]
        return {
            "backbone": str(backbone),
            "output_dim": backbone.output_dim,
            "n_classes": self._n_classes,
            "num_neighbours": self._num_neighbours,
            "margin": self._margin,
            "temperature": self._temperature,
            "adjust_factor": self._adjust_factor,
            "entropy_loss_weight": self._entropy_loss_weight,
            "inconsistency_loss_weight": self._inconsistency_loss_weight,
            "positive_lookback": self._positive_lookback,
            "lr_pretext": self._lr_pretext,
            "lr_classification": self._lr_classification,
            "batch_size": self._batch_size,
            "pretext_epochs": self._pretext_epochs,
            "classification_epochs": self._classification_epochs,
            "patience": self._patience,
        }
