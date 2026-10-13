"""COMMAND model with complete-bag ContTrain++ optimization and replay."""

from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pyclad.models.model import Model
from pyclad.video.data.sample import VideoBag
from pyclad.video.models.command.architecture import CommandOutput, CommandVideoNetwork
from pyclad.video.models.command.config import CommandLossConfig, CommandTrainerConfig
from pyclad.video.prediction_results import VideoPredictionResults


@dataclass(frozen=True)
class ReplayEntry:
    bag: VideoBag
    score_curve: np.ndarray

    def __post_init__(self) -> None:
        curve = np.asarray(self.score_curve, dtype=np.float32).reshape(-1)
        if curve.shape != (len(self.bag.features),) or not np.isfinite(curve).all():
            raise ValueError("replay score curve must be finite and match the bag length")
        object.__setattr__(self, "score_curve", curve)

    @property
    def maximum_score(self) -> float:
        return float(np.max(self.score_curve))

    @property
    def peak_quartile(self) -> int:
        length = len(self.score_curve)
        return min(3, int(np.argmax(self.score_curve)) * 4 // max(1, length))


class ContTrainReplayBuffer:
    """Balanced complete-bag retention with deterministic rotating replay."""

    def __init__(self, max_size: int = 1_000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = int(max_size)
        self._entries: Dict[str, ReplayEntry] = {}
        self._last_label: int | None = None
        self._last_cell: Dict[int, tuple[str, int]] = {}
        self._last_bag: Dict[tuple[int, str, int], str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Tuple[ReplayEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def add(self, entries: Iterable[ReplayEntry]) -> None:
        for entry in entries:
            self._entries[entry.bag.bag_id] = entry
        if len(self._entries) > self.max_size:
            retained = self._balanced_select(self.entries, self.max_size)
            self._entries = {entry.bag.bag_id: entry for entry in retained}
        cells = {(entry.bag.weak_label, entry.bag.task_id, entry.peak_quartile) for entry in self._entries.values()}
        self._last_bag = {cell: bag_id for cell, bag_id in self._last_bag.items() if cell in cells}

    def sample(self, count: int, *, exclude_bag_ids: Iterable[str] = ()) -> Tuple[VideoBag, ...]:
        """Rotate labels, task/quartile cells, and bag IDs without duplicates.

        ID successors keep rotation stable when scores change or an anchor is
        excluded or evicted. Capacity selection retains its score priorities.
        """
        if count <= 0:
            return ()
        excluded = set(exclude_bag_ids)
        cells: Dict[int, Dict[tuple[str, int], Dict[str, ReplayEntry]]] = {}
        available = 0
        for entry in self.entries:
            if entry.bag.bag_id not in excluded:
                cells.setdefault(entry.bag.weak_label, {}).setdefault((entry.bag.task_id, entry.peak_quartile), {})[
                    entry.bag.bag_id
                ] = entry
                available += 1
        selected = []
        for _ in range(min(count, available)):
            labels = sorted(cells)
            label_index = bisect_right(labels, self._last_label) % len(labels) if self._last_label is not None else 0
            label = labels[label_index]
            keys = sorted(cells[label])
            previous_cell = self._last_cell.get(label)
            cell_index = bisect_right(keys, previous_cell) % len(keys) if previous_cell is not None else 0
            cell = keys[cell_index]
            candidates = cells[label][cell]
            bag_ids = sorted(candidates)
            anchor = (label, *cell)
            previous_bag = self._last_bag.get(anchor)
            bag_index = bisect_right(bag_ids, previous_bag) % len(bag_ids) if previous_bag is not None else 0
            bag_id = bag_ids[bag_index]
            selected.append(candidates.pop(bag_id).bag)
            self._last_label = label
            self._last_cell[label] = cell
            self._last_bag[anchor] = bag_id
            if not candidates:
                del cells[label][cell]
                if not cells[label]:
                    del cells[label]
        return tuple(selected)

    @classmethod
    def _balanced_select(cls, entries: Sequence[ReplayEntry], count: int) -> Tuple[ReplayEntry, ...]:
        if count <= 0 or not entries:
            return ()
        labels = sorted({entry.bag.weak_label for entry in entries})
        selected: list[ReplayEntry] = []
        if len(labels) == 2:
            budgets = {labels[0]: count // 2, labels[1]: count - count // 2}
        else:
            budgets = {labels[0]: count}
        for label in labels:
            candidates = [entry for entry in entries if entry.bag.weak_label == label]
            selected.extend(cls._round_robin_cells(candidates, min(budgets[label], len(candidates))))

        if len(selected) < count:
            used = {entry.bag.bag_id for entry in selected}
            remaining = [entry for entry in entries if entry.bag.bag_id not in used]
            remaining.sort(key=lambda item: (-item.maximum_score, item.bag.task_id, item.bag.bag_id))
            selected.extend(remaining[: count - len(selected)])
        return tuple(selected[:count])

    @staticmethod
    def _round_robin_cells(entries: Sequence[ReplayEntry], count: int) -> list[ReplayEntry]:
        cells: Dict[tuple[str, int], list[ReplayEntry]] = {}
        for entry in entries:
            cells.setdefault((entry.bag.task_id, entry.peak_quartile), []).append(entry)
        for values in cells.values():
            values.sort(key=lambda item: (-item.maximum_score, item.bag.bag_id))
        keys = sorted(cells)
        selected: list[ReplayEntry] = []
        offset = 0
        while len(selected) < count:
            added = False
            for key in keys:
                values = cells[key]
                if offset < len(values):
                    selected.append(values[offset])
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                break
            offset += 1
        return selected

    def state_dict(self) -> Mapping[str, object]:
        return {
            "max_size": self.max_size,
            "sampling": {
                "last_label": self._last_label,
                "last_cell": dict(self._last_cell),
                "last_bag": dict(self._last_bag),
            },
            "entries": [
                {
                    "bag_id": entry.bag.bag_id,
                    "task_id": entry.bag.task_id,
                    "features": entry.bag.features,
                    "weak_label": entry.bag.weak_label,
                    "score_curve": entry.score_curve,
                }
                for entry in self.entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["max_size"]) != self.max_size:
            raise ValueError("checkpoint replay capacity does not match trainer configuration")
        self._entries.clear()
        for item in state["entries"]:
            bag = VideoBag(
                bag_id=str(item["bag_id"]),
                task_id=str(item["task_id"]),
                features=np.asarray(item["features"], dtype=np.float32),
                weak_label=int(item["weak_label"]),
            )
            entry = ReplayEntry(bag=bag, score_curve=np.asarray(item["score_curve"], dtype=np.float32))
            self._entries[bag.bag_id] = entry
        # Checkpoints predating rotating replay start a fresh deterministic cycle.
        sampling = state.get("sampling", {})
        self._last_label = sampling.get("last_label")
        self._last_cell = dict(sampling.get("last_cell", {}))
        self._last_bag = dict(sampling.get("last_bag", {}))


@dataclass(frozen=True)
class CommandLossBreakdown:
    total: Tensor
    mil: Tensor
    contrastive: Tensor
    focal: Tensor
    anomaly_separation: Tensor
    sparsity: Tensor
    smoothness: Tensor
    score_l2: Tensor

    def detached(self) -> Dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.__dict__.items()}


def command_composite_loss(
    output: CommandOutput,
    weak_labels: Tensor,
    config: CommandLossConfig | None = None,
) -> CommandLossBreakdown:
    """Compute COMMAND's MIL, contrastive, focal, and separation objective."""

    config = config or CommandLossConfig()
    labels = weak_labels.to(device=output.logits.device, dtype=output.logits.dtype).reshape(-1)
    if output.logits.ndim != 2 or len(labels) != output.logits.shape[0]:
        raise ValueError("weak_labels must provide one value per video bag")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("weak_labels must contain only zero and one")

    scores = output.memory.dual_memory_deviation
    bag_scores = scores.amax(dim=1)
    positive = labels > 0.5
    normal = ~positive
    zero = scores.sum() * 0.0

    if positive.any() and normal.any():
        ranking = F.relu(config.mil_margin - bag_scores[positive, None] + bag_scores[None, normal]).mean()
        separation = F.relu(
            config.anomaly_margin - scores[positive].mean(dim=1, keepdim=True) + scores[normal].mean(dim=1).unsqueeze(0)
        ).mean()
    else:
        ranking = zero
        separation = zero

    sparsity = scores[positive].mean() if positive.any() else zero
    smoothness = (scores[:, 1:] - scores[:, :-1]).square().mean() if scores.shape[1] > 1 else zero
    mil = ranking + config.sparsity_weight * sparsity + config.smoothness_weight * smoothness

    projections = F.normalize(output.projections.mean(dim=1), dim=-1)
    contrastive = (
        _supervised_info_nce(projections, labels, config.contrastive_temperature)
        if positive.any() and normal.any()
        else zero
    )

    cross_entropy = F.binary_cross_entropy_with_logits(output.logits.amax(dim=1), labels, reduction="none")
    positive_probability = torch.exp(-cross_entropy)
    alpha = torch.where(labels > 0.5, config.focal_alpha, 1.0 - config.focal_alpha)
    focal = (alpha * (1.0 - positive_probability).pow(config.focal_gamma) * cross_entropy).mean()
    score_l2 = scores.square().mean()

    total = (
        config.mil_weight * mil
        + config.contrastive_weight * contrastive
        + config.focal_weight * focal
        + config.anomaly_separation_weight * separation
        + config.score_l2_weight * score_l2
    )
    return CommandLossBreakdown(
        total=total,
        mil=mil,
        contrastive=contrastive,
        focal=focal,
        anomaly_separation=separation,
        sparsity=sparsity,
        smoothness=smoothness,
        score_l2=score_l2,
    )


def _supervised_info_nce(projections: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if len(projections) < 2:
        return projections.sum() * 0.0
    logits = projections @ projections.transpose(0, 1) / temperature
    identity = torch.eye(len(projections), dtype=torch.bool, device=projections.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return projections.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1, keepdim=True)
    mean_positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_positive_log_prob[valid].mean()


class CommandModel(Model):
    """Dedicated complete-bag trainer implementing the ContTrain++ protocol."""

    def __init__(
        self,
        model: CommandVideoNetwork | None = None,
        config: CommandTrainerConfig | None = None,
    ):
        self.config = config or CommandTrainerConfig()
        torch.manual_seed(self.config.seed)
        self.model = model or CommandVideoNetwork(self.config.architecture)
        if self.model.architecture != self.config.architecture:
            raise ValueError("model architecture and trainer configuration must match")
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.replay = ContTrainReplayBuffer(self.config.buffer_size)
        self._rng = np.random.default_rng(self.config.seed)
        self._global_epoch = 0
        self._calibration_median = 0.0
        self._calibration_scale = 1.0
        self._threshold = self.config.threshold
        self.optimizer = self._optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )

    def fit(self, data: np.ndarray) -> None:
        """Learn one complete-bag task, retaining ContTrain++ replay state."""
        bags = self._bags(data)
        task_ids = {bag.task_id for bag in bags}
        if len(task_ids) != 1:
            raise ValueError("training data must contain exactly one task")
        self.fit_task(bags, task_id=next(iter(task_ids)))

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        """Return bag maxima and aligned temporal scores for frame evaluation."""
        bags = self._bags(data)
        result = self.predict_bags(bags)
        scores = result["anomaly_scores"]
        maxima = scores.max(axis=1) if bags else np.empty(0)
        if bags and self._threshold is None:
            raise RuntimeError(
                "COMMAND: video threshold is not calibrated. Fit on normal bags or set config.threshold explicitly."
            )
        raw_maxima = result["raw_scores"].max(axis=1) if bags else np.empty(0)
        return VideoPredictionResults(
            y_pred=(raw_maxima > self._threshold).astype(np.int64) if bags else np.empty(0, dtype=np.int64),
            anomaly_scores=maxima,
            window_scores=scores.reshape(-1),
            classifier_scores=result["classifier_scores"].reshape(-1),
            windows=tuple(window for bag in bags for window in bag.windows),
        )

    @staticmethod
    def _bags(data: np.ndarray) -> tuple[VideoBag, ...]:
        values = np.asarray(data, dtype=object)
        if values.ndim != 1 or any(not isinstance(bag, VideoBag) for bag in values):
            raise ValueError("COMMAND requires a one-dimensional array of VideoBag objects")
        return tuple(values)

    def name(self) -> str:
        return "COMMAND"

    def _optimizer(self) -> torch.optim.Optimizer:
        secondary = []
        primary = []
        for name, parameter in self.model.named_parameters():
            if "secondary_memory" in name:
                secondary.append(parameter)
            else:
                primary.append(parameter)
        groups = [{"params": primary, "lr": self.config.learning_rate, "name": "primary"}]
        if secondary:
            groups.append(
                {
                    "params": secondary,
                    "lr": self.config.secondary_memory_learning_rate,
                    "name": "secondary_memory",
                }
            )
        return torch.optim.Adam(groups)

    def fit_task(self, bags: Sequence[VideoBag], *, task_id: str | None = None) -> Dict[str, object]:
        if not bags:
            raise ValueError("fit_task requires at least one video bag")
        if task_id is None and len({bag.task_id for bag in bags}) != 1:
            raise ValueError("training data must contain exactly one task")
        prepared = tuple(
            VideoBag(
                bag_id=bag.bag_id,
                task_id=task_id if task_id is not None else bag.task_id,
                features=bag.features,
                weak_label=bag.weak_label,
                windows=bag.windows,
            )
            for bag in bags
        )
        temporal_lengths = {len(bag.features) for bag in prepared}
        if len(temporal_lengths) != 1:
            raise ValueError("all video bags in a task must have the same temporal length")
        if self.config.replay_batch_size and any(
            len(entry.bag.features) not in temporal_lengths for entry in self.replay.entries
        ):
            raise ValueError("task bags and replay bags must have the same temporal length")
        feature_dims = {bag.features.shape[1] for bag in prepared}
        if feature_dims != {self.config.architecture.fused_dim}:
            raise ValueError(
                f"task bags must have feature dimension {self.config.architecture.fused_dim}, got {sorted(feature_dims)}"
            )
        # Restart the per-task schedule while retaining Adam moments and replay.
        if self._global_epoch:
            self._reset_task_scheduler()
        started = time.perf_counter()
        totals: Dict[str, float] = {}
        batches = 0
        novelty_replacements = 0
        latest_scores: Dict[str, np.ndarray] = {}
        self.model.train()
        for _ in range(self.config.epochs):
            permutation = self._rng.permutation(len(prepared))
            offset = 0
            while offset < len(prepared):
                # Replay can fill during the first epoch, so reserve space per batch.
                replay_count = min(self.config.replay_batch_size, len(self.replay))
                current_batch_size = self.config.batch_size - replay_count
                current = tuple(prepared[index] for index in permutation[offset : offset + current_batch_size])
                replay = self.replay.sample(
                    self.config.replay_batch_size,
                    exclude_bag_ids=(bag.bag_id for bag in current),
                )
                offset += len(current)
                combined = (*current, *replay)
                features = torch.as_tensor(
                    np.stack([bag.features for bag in combined]),
                    dtype=torch.float32,
                    device=self.device,
                )
                labels = torch.as_tensor(
                    [bag.weak_label for bag in combined],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(features, track_memory_usage=True)
                loss = command_composite_loss(output, labels, self.config.loss)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                novelty_replacements += self._replace_novel_normal_features(output, labels)

                curves = output.memory.dual_memory_deviation.detach().cpu().numpy().astype(np.float32)
                batch_entries = []
                for index, bag in enumerate(current):
                    latest_scores[bag.bag_id] = curves[index]
                    batch_entries.append(ReplayEntry(bag=bag, score_curve=curves[index]))
                self.replay.add(batch_entries)
                detached = loss.detached()
                for name, value in detached.items():
                    totals[name] = totals.get(name, 0.0) + value
                batches += 1
            self.scheduler.step()
            self._global_epoch += 1

        replay_entries = []
        for bag in prepared:
            curve = latest_scores[bag.bag_id]
            replay_entries.append(ReplayEntry(bag=bag, score_curve=curve))
        self.replay.add(replay_entries)
        self._update_calibration(prepared)
        return {
            "task_id": task_id or prepared[0].task_id,
            "bags": len(prepared),
            "epochs": self.config.epochs,
            "batches": batches,
            "loss": {name: value / max(1, batches) for name, value in totals.items()},
            "buffer_bags": len(self.replay),
            "novelty_replacements": novelty_replacements,
            "learning_rates": {
                group.get("name", str(index)): group["lr"] for index, group in enumerate(self.optimizer.param_groups)
            },
            "wall_seconds": time.perf_counter() - started,
        }

    def _reset_task_scheduler(self) -> None:
        for group in self.optimizer.param_groups:
            if group.get("name") == "secondary_memory":
                group["lr"] = self.config.secondary_memory_learning_rate
            else:
                group["lr"] = self.config.learning_rate
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma,
        )

    def _replace_novel_normal_features(
        self,
        output: CommandOutput,
        labels: Tensor,
    ) -> int:
        if self._global_epoch < self.config.novelty_warmup_epochs:
            return 0
        normal = labels < 0.5
        if not normal.any():
            return 0
        scores = output.memory.dual_memory_deviation[normal].detach()
        flattened = scores.reshape(-1)
        median = flattened.median()
        mad = (flattened - median).abs().median()
        threshold = median + self.config.novelty_mad_scale * mad
        candidates = torch.nonzero(scores > threshold, as_tuple=False)
        if not len(candidates):
            return 0
        candidate_scores = scores[candidates[:, 0], candidates[:, 1]]
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        temporal = output.temporal_features[normal].detach()
        memory = self.model.network.memory
        replaced = []
        # Preserve the highest-novelty candidates when the batch exceeds capacity.
        for candidate in candidates[order[: len(memory.primary_memory)]]:
            slot = memory.replace_least_accessed(
                "primary", temporal[candidate[0], candidate[1]], exclude_slots=tuple(replaced)
            )
            replaced.append(slot)
        return len(replaced)

    def _update_calibration(self, current_bags: Sequence[VideoBag] = ()) -> None:
        """Calibrate fresh normal snippet scores and a separate raw video threshold.

        Include current normals even if replay evicts them. Historical replay
        curves were scored by older training-mode networks and are not reused.
        """
        normal_bags = {
            bag.bag_id: bag
            for bag in (*[entry.bag for entry in self.replay.entries], *current_bags)
            if bag.weak_label == 0
        }
        self._threshold = self.config.threshold
        if not normal_bags:
            self._calibration_median = 0.0
            self._calibration_scale = 1.0
            return

        # Replay-disabled tasks may have different temporal lengths.
        by_length: Dict[int, list[VideoBag]] = {}
        for bag in normal_bags.values():
            by_length.setdefault(len(bag.features), []).append(bag)
        curves = []
        maxima = []
        for bags in by_length.values():
            raw = self.predict_bags(bags, batch_size=self.config.batch_size)["raw_scores"].astype(np.float64)
            curves.append(raw.reshape(-1))
            maxima.append(raw.max(axis=1))
        values = np.concatenate(curves)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        self._calibration_median = median
        self._calibration_scale = max(1e-8, 1.4826 * mad)
        if self._threshold is None:
            self._threshold = float(np.quantile(np.concatenate(maxima), self.config.threshold_quantile))

    def predict_bags(self, bags: Sequence[VideoBag], *, batch_size: int = 8) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not bags:
            return {
                "anomaly_scores": np.empty((0, 0), dtype=np.float32),
                "raw_scores": np.empty((0, 0), dtype=np.float32),
                "classifier_scores": np.empty((0, 0), dtype=np.float32),
            }
        was_training = self.model.training
        self.model.eval()
        try:
            raw_blocks = []
            classifier_blocks = []
            with torch.no_grad():
                for offset in range(0, len(bags), batch_size):
                    batch = bags[offset : offset + batch_size]
                    features = torch.as_tensor(
                        np.stack([bag.features for bag in batch]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    output = self.model(features)
                    raw_blocks.append(output.memory.dual_memory_deviation.cpu().numpy())
                    classifier_blocks.append(output.probabilities.cpu().numpy())
        finally:
            self.model.train(was_training)
        raw = np.concatenate(raw_blocks).astype(np.float32, copy=False)
        # A bounded monotonic map that preserves rank in extreme score tails.
        z = (raw.astype(np.float64) - self._calibration_median) / self._calibration_scale
        calibrated = 0.5 + np.arctan(z) / np.pi
        classifier = np.concatenate(classifier_blocks).astype(np.float32, copy=False)
        return {"anomaly_scores": calibrated, "raw_scores": raw, "classifier_scores": classifier}

    def state_dict(self) -> Mapping[str, object]:
        return {
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "replay": self.replay.state_dict(),
            "global_epoch": self._global_epoch,
            "calibration_median": self._calibration_median,
            "calibration_scale": self._calibration_scale,
            "threshold": self._threshold,
            "distance_mode": "direct_euclidean",
            "rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        saved_config = {key: value for key, value in state["config"].items() if key != "device"}
        current_config = {key: value for key, value in asdict(self.config).items() if key != "device"}
        # Legacy checkpoints recorded no video decision policy; allow its new controls.
        saved_config.setdefault("threshold", current_config["threshold"])
        saved_config.setdefault("threshold_quantile", current_config["threshold_quantile"])
        if saved_config != current_config:
            raise ValueError("checkpoint configuration does not match trainer configuration")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.replay.load_state_dict(state["replay"])
        self._global_epoch = int(state["global_epoch"])
        self._calibration_median = float(state["calibration_median"])
        self._calibration_scale = float(state["calibration_scale"])
        self._threshold = state.get("threshold", self.config.threshold)
        # Older distance math can change score ranks near memory prototypes.
        # Recompute its calibration rather than retain a stale inferred threshold.
        if "threshold" not in state or state.get("distance_mode") != "direct_euclidean":
            self._update_calibration()
        self._rng.bit_generator.state = state["rng_state"]
        # Older checkpoints remain loadable, but lack exact dropout continuation.
        if "torch_rng_state" in state:
            torch.random.set_rng_state(state["torch_rng_state"].cpu())
        cuda_state = state.get("torch_cuda_rng_state")
        if cuda_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            torch.save(dict(self.state_dict()), temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        # Load RNG byte tensors on CPU; model/optimizer loading moves parameters
        # and optimizer state to the configured destination device.
        state = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
        self.load_state_dict(state)

    def additional_info(self) -> Dict[str, object]:
        return {
            "architecture": asdict(self.config.architecture),
            "training": {
                key: value for key, value in asdict(self.config).items() if key not in {"architecture", "loss"}
            },
            "loss": asdict(self.config.loss),
            "buffer_bags": len(self.replay),
            "global_epochs": self._global_epoch,
            "calibration": {
                "median": self._calibration_median,
                "mad_scale": self._calibration_scale,
                "threshold": self._threshold,
            },
        }
