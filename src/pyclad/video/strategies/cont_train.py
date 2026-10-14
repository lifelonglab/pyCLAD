"""ContTrain++: continual optimization, replay, and task/checkpoint state."""

from __future__ import annotations

import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from pyclad.strategies.strategy import ConceptIncrementalStrategy
from pyclad.video.data.sample import VideoBag, as_video_bags
from pyclad.video.models.command.architecture import CommandOutput
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import CommandConfig
from pyclad.video.prediction_results import VideoPredictionResults
from pyclad.video.strategies.buffers.cont_train import (
    ContTrainReplayBuffer,
    ReplayEntry,
)
from pyclad.video.strategies.config import ContTrainPlusPlusConfig


class ContTrainPlusPlusStrategy(ConceptIncrementalStrategy):
    """Train a COMMAND backbone on labelled video tasks with rotating replay."""

    def __init__(self, model: CommandModel, config: ContTrainPlusPlusConfig | None = None):
        self._model = model
        self.config = config or ContTrainPlusPlusConfig()
        self.replay = ContTrainReplayBuffer(self.config.buffer_size)
        self._rng = np.random.default_rng(self.config.seed)
        self._global_epoch = 0
        self.optimizer = self._model.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma
        )

    @property
    def device(self) -> torch.device:
        """Use the backbone's device for complete-video training batches."""
        return self._model.device()

    def learn(self, data: np.ndarray) -> None:
        """Learn one video task while retaining this strategy's replay and optimizer."""
        self.fit_task(as_video_bags(data))

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        """Delegate inference to the trained model, without task identity."""
        return self._model.predict(data)

    def name(self) -> str:
        """Return the strategy's name."""
        return "ContTrain++"

    def fit_task(self, bags: Sequence[VideoBag], *, task_id: str | None = None) -> Dict[str, object]:
        """Learn one complete-video task and return its training diagnostics."""
        bags = as_video_bags(bags)
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
        if feature_dims != {self._model.config.architecture.fused_dim}:
            raise ValueError(
                f"task bags must have feature dimension {self._model.config.architecture.fused_dim}, got {sorted(feature_dims)}"
            )
        # Restart the per-task schedule while retaining Adam moments and replay.
        if self._global_epoch:
            self._reset_task_scheduler()
        started = time.perf_counter()
        totals: Dict[str, float] = {}
        batches = 0
        novelty_replacements = 0
        latest_scores: Dict[str, np.ndarray] = {}
        self._model.get_module().train()
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
                output = self._model.forward_output(features, track_memory_usage=True)
                loss = self._model.loss_breakdown(output, labels)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self._model.get_module().parameters(), self.config.gradient_clip)
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
        """Restart the task schedule while preserving optimizer moments."""
        for group in self.optimizer.param_groups:
            if group.get("name") == "secondary_memory":
                group["lr"] = self._model.config.secondary_memory_learning_rate
            else:
                group["lr"] = self._model.config.learning_rate
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
        """Apply this strategy's novelty rule to the primary model memory."""
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
        memory = self._model.get_module().network.memory
        replaced = []
        # Preserve the highest-novelty candidates when the batch exceeds capacity.
        for candidate in candidates[order[: len(memory.primary_memory)]]:
            slot = memory.replace_least_accessed(
                "primary", temporal[candidate[0], candidate[1]], exclude_slots=tuple(replaced)
            )
            replaced.append(slot)
        return len(replaced)

    def _update_calibration(self, current_bags: Sequence[VideoBag] = ()) -> None:
        """Choose retained/current normals and ask the model to calibrate predictions."""
        normal_bags = {
            bag.bag_id: bag
            for bag in (*[entry.bag for entry in self.replay.entries], *current_bags)
            if bag.weak_label == 0
        }
        self._model.calibrate(tuple(normal_bags.values()), batch_size=self.config.batch_size)

    def state_dict(self) -> Mapping[str, object]:
        """Save the model together with all continual-training state."""
        return {
            "version": 2,
            "config": asdict(self.config),
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "replay": self.replay.state_dict(),
            "global_epoch": self._global_epoch,
            "rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "torch_mps_rng_state": torch.mps.get_rng_state() if self.device.type == "mps" else None,
        }

    def _checkpoint_parts(self, state: Mapping[str, object]) -> tuple[dict, Mapping[str, object]]:
        """Read current checkpoints or split the former combined trainer format."""
        if "version" in state:
            if state["version"] != 2:
                raise ValueError("unsupported ContTrain++ checkpoint version")
            return dict(state["config"]), state["model"]
        legacy_config = state["config"]
        strategy_config = {field.name: legacy_config[field.name] for field in fields(ContTrainPlusPlusConfig)}
        model_config = {
            field.name: legacy_config[field.name] for field in fields(CommandConfig) if field.name in legacy_config
        }
        model_state = {
            "config": model_config,
            "module": state["model"],
            "calibration_median": state["calibration_median"],
            "calibration_scale": state["calibration_scale"],
            "distance_mode": state.get("distance_mode"),
        }
        if "threshold" in state:
            model_state["threshold"] = state["threshold"]
        return strategy_config, model_state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore training state, allowing the destination model device to differ."""
        strategy_config, model_state = self._checkpoint_parts(state)
        if strategy_config != asdict(self.config):
            raise ValueError("checkpoint configuration does not match strategy configuration")
        self._model.validate_state_dict(model_state)
        if int(state["replay"]["max_size"]) != self.replay.max_size:
            raise ValueError("checkpoint replay capacity does not match strategy configuration")
        self._model.load_state_dict(model_state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.replay.load_state_dict(state["replay"])
        self._global_epoch = int(state["global_epoch"])
        if "threshold" not in model_state or model_state.get("distance_mode") != "direct_euclidean":
            self._update_calibration()
        self._rng.bit_generator.state = state["rng_state"]
        if "torch_rng_state" in state:
            torch.random.set_rng_state(state["torch_rng_state"].cpu())
        cuda_state = state.get("torch_cuda_rng_state")
        if cuda_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)
        mps_state = state.get("torch_mps_rng_state")
        if mps_state is not None and self.device.type == "mps":
            torch.mps.set_rng_state(mps_state.cpu())

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save the model and this strategy's resumable state."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            torch.save(dict(self.state_dict()), temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a current or legacy checkpoint onto the model's device."""
        # Load RNG byte tensors on CPU; model/optimizer loading moves parameters
        # and optimizer state to the configured destination device.
        state = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
        self.load_state_dict(state)

    def additional_info(self) -> Dict[str, object]:
        """Report training configuration and history under the strategy metadata."""
        return {
            "model": self._model.name(),
            "training": asdict(self.config),
            "buffer_bags": len(self.replay),
            "global_epochs": self._global_epoch,
        }
