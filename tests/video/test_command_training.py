"""Composite-loss and complete-bag ContTrain++ tests."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch


class TestCommandTraining:

    @staticmethod
    def _architecture():
        from pyclad.video.models.command.config import CommandArchitectureConfig

        return CommandArchitectureConfig(
            appearance_dim=4,
            motion_dim=4,
            hidden_dim=8,
            state_dim=3,
            temporal_kernel_size=5,
            mamba_blocks=1,
            memory_size=5,
            projection_dim=6,
            cbam_reduction=2,
            cbam_temporal_kernel_size=7,
            dropout=0.0,
        )

    def test_composite_loss_is_finite_and_zeros_positive_dependent_terms_without_positives(self):
        from pyclad.video.models.command.architecture import CommandNetwork
        from pyclad.video.models.command.command import command_composite_loss

        model = CommandNetwork(self._architecture())
        appearance = torch.randn(4, 8, 4)
        motion = torch.randn(4, 8, 4)
        mixed = command_composite_loss(model(appearance, motion), torch.tensor([0.0, 0.0, 1.0, 1.0]))
        normal_only = command_composite_loss(model(appearance, motion), torch.zeros(4))
        assert torch.isfinite(mixed.total)
        assert float(mixed.mil.detach()) >= 0.0
        assert float(mixed.contrastive.detach()) >= 0.0
        assert float(normal_only.contrastive.detach()) == 0.0
        assert float(normal_only.anomaly_separation.detach()) == 0.0

    def test_contrastive_loss_remains_stable_at_small_temperature(self):
        from dataclasses import replace

        from pyclad.video.models.command.architecture import CommandNetwork
        from pyclad.video.models.command.command import command_composite_loss
        from pyclad.video.models.command.config import CommandLossConfig

        output = CommandNetwork(self._architecture())(torch.randn(4, 8, 4), torch.randn(4, 8, 4))
        projections = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [-0.8, -0.6]])
        projections.requires_grad_()
        output = replace(output, projections=projections[:, None, :].expand(-1, 8, -1))
        loss = command_composite_loss(
            output, torch.tensor([0.0, 0.0, 1.0, 1.0]), CommandLossConfig(contrastive_temperature=1e-4)
        )
        assert loss.contrastive.item() == pytest.approx(0.0, abs=1e-6)
        loss.contrastive.backward()
        assert torch.isfinite(projections.grad).all()

    def test_focal_loss_trains_confidently_wrong_predictions(self):
        from dataclasses import replace

        from pyclad.video.models.command.architecture import CommandNetwork
        from pyclad.video.models.command.command import command_composite_loss

        output = CommandNetwork(self._architecture())(torch.randn(2, 8, 4), torch.randn(2, 8, 4))
        logits = torch.tensor([[100.0] * 8, [-100.0] * 8], requires_grad=True)
        output = replace(output, logits=logits, probabilities=torch.sigmoid(logits))
        loss = command_composite_loss(output, torch.tensor([0.0, 1.0]))
        loss.focal.backward()
        assert torch.isfinite(loss.focal)
        assert logits.grad[0].sum() > 0
        assert logits.grad[1].sum() < 0

    def test_seed_controls_network_initialization_and_training(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=2, seed=17
        )
        first = CommandModel(config=config)
        first.fit_task(self._bags("T1"))
        torch.manual_seed(1234)
        second = CommandModel(config=config)
        second.fit_task(self._bags("T1"))
        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)

    def test_replay_selection_balances_labels_tasks_and_keeps_complete_bags(self):
        from pyclad.video.data.sample import VideoBag
        from pyclad.video.models.command.command import (
            ContTrainReplayBuffer,
            ReplayEntry,
        )

        buffer = ContTrainReplayBuffer(max_size=8)
        entries = []
        for index in range(12):
            bag = VideoBag(
                bag_id=f"bag-{index:02d}",
                task_id=f"T{index % 3 + 1}",
                features=np.full((8, 8), index, dtype=np.float32),
                weak_label=index % 2,
            )
            curve = np.zeros(8, dtype=np.float32)
            curve[index % 8] = index + 1
            entries.append(ReplayEntry(bag, curve))
        buffer.add(entries)
        sample = buffer.sample(6)
        assert len(buffer) == 8
        assert len(sample) == 6
        assert sum((bag.weak_label == 0 for bag in sample)) == 3
        assert sum((bag.weak_label == 1 for bag in sample)) == 3
        assert all((bag.features.shape == (8, 8) for bag in sample))
        assert len({bag.task_id for bag in sample}) >= 2

    def test_optimizer_uses_slow_secondary_memory_group_and_checkpoint_resumes(self):
        from pyclad.video.data.sample import VideoBag
        from pyclad.video.models.command.architecture import CommandVideoNetwork
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=self._architecture(),
            epochs=1,
            batch_size=4,
            replay_batch_size=2,
            buffer_size=8,
            learning_rate=0.001,
            secondary_memory_learning_rate=0.0001,
            novelty_warmup_epochs=10,
            device="cpu",
        )
        trainer = CommandModel(CommandVideoNetwork(config.architecture), config)
        bags = tuple(
            (
                VideoBag(
                    bag_id=f"video-{index}",
                    task_id="T1",
                    features=np.random.default_rng(index).normal(size=(8, 8)).astype(np.float32),
                    weak_label=index % 2,
                )
                for index in range(4)
            )
        )
        result = trainer.fit_task(bags, task_id="T1")
        before = trainer.predict_bags(bags, batch_size=2)["raw_scores"]
        assert [group["name"] for group in trainer.optimizer.param_groups] == ["primary", "secondary_memory"]
        assert result["buffer_bags"] == 4
        assert np.isfinite(before).all()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint)
            restored = CommandModel(CommandVideoNetwork(config.architecture), config)
            restored.load_checkpoint(checkpoint)
            after = restored.predict_bags(bags, batch_size=2)["raw_scores"]
        np.testing.assert_array_equal(before, after)
        assert len(restored.replay) == 4

    def test_calibration_is_bounded_monotonic_and_retains_extreme_tail_rank(self):
        from dataclasses import replace

        from pyclad.video.data.sample import VideoBag
        from pyclad.video.models.command.architecture import CommandVideoNetwork
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=self._architecture(), epochs=1, batch_size=2, replay_batch_size=1, device="cpu"
        )
        trainer = CommandModel(CommandVideoNetwork(config.architecture), config)
        bags = tuple(
            (
                VideoBag(
                    bag_id=f"tail-{index}",
                    task_id="test",
                    features=np.full((8, 8), -1000.0 + index, dtype=np.float32),
                    weak_label=0,
                )
                for index in range(2)
            )
        )
        trainer._calibration_median = 1000.0
        trainer._calibration_scale = 0.01
        # Isolate calibration from random network outputs, which may be identical.
        raw = torch.arange(16, dtype=torch.float32).reshape(2, 8) - 1000.0
        handle = trainer.model.register_forward_hook(
            lambda module, args, output: replace(output, memory=replace(output.memory, dual_memory_deviation=raw))
        )
        try:
            scores = trainer.predict_bags(bags)["anomaly_scores"]
        finally:
            handle.remove()
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert np.all(np.diff(scores.reshape(-1)) > 0)

    def test_learning_rate_schedule_restarts_at_each_task_boundary(self):
        from pyclad.video.data.sample import VideoBag
        from pyclad.video.models.command.architecture import CommandVideoNetwork
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=self._architecture(),
            epochs=2,
            batch_size=2,
            replay_batch_size=1,
            buffer_size=8,
            learning_rate=0.001,
            secondary_memory_learning_rate=0.0001,
            lr_step_size=1,
            lr_gamma=0.1,
            novelty_warmup_epochs=10,
            device="cpu",
        )
        trainer = CommandModel(CommandVideoNetwork(config.architecture), config)

        def bags(task):
            return tuple(
                (
                    VideoBag(
                        bag_id=f"{task}-{index}",
                        task_id=task,
                        features=np.random.default_rng(index).normal(size=(8, 8)).astype(np.float32),
                        weak_label=index % 2,
                    )
                    for index in range(2)
                )
            )

        first = trainer.fit_task(bags("T1"), task_id="T1")
        second = trainer.fit_task(bags("T2"), task_id="T2")
        assert first["learning_rates"]["primary"] == pytest.approx(1e-05)
        assert second["learning_rates"]["primary"] == pytest.approx(1e-05)
        assert second["learning_rates"]["secondary_memory"] == pytest.approx(1e-06)

    @staticmethod
    def _bags(task, count=8):
        from pyclad.video.data.sample import VideoBag

        return tuple(
            VideoBag(str(index) + task, task, np.random.default_rng(index).normal(size=(8, 8)), index % 2)
            for index in range(count)
        )

    def test_replay_never_exceeds_total_batch_size(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(architecture=self._architecture(), epochs=2, batch_size=4, replay_batch_size=2)
        )
        sizes = []
        handle = model.model.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
        try:
            model.fit_task(self._bags("T1"))
            model.fit_task(self._bags("T2"))
        finally:
            handle.remove()
        assert sizes and max(sizes) <= 4

    def test_incompatible_replay_length_is_rejected_before_resetting_scheduler(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=2, lr_step_size=1
            )
        )
        model.fit_task(self._bags("T1"))
        scheduler = model.scheduler.state_dict()
        learning_rates = [group["lr"] for group in model.optimizer.param_groups]
        shorter_bags = tuple(replace(bag, features=bag.features[:4]) for bag in self._bags("T2"))
        with pytest.raises(ValueError, match="task bags and replay bags.*temporal length"):
            model.fit_task(shorter_bags)
        assert model.scheduler.state_dict() == scheduler
        assert [group["lr"] for group in model.optimizer.param_groups] == learning_rates

    def test_mixed_task_ids_are_rejected_before_any_training_state_changes(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=2)
        )
        model.fit_task(self._bags("T0"))
        model.model.eval()
        scheduler = deepcopy(model.scheduler.state_dict())
        learning_rates = [group["lr"] for group in model.optimizer.param_groups]
        rng = deepcopy(model._rng.bit_generator.state)
        torch_rng = torch.random.get_rng_state().clone()
        replay_ids = [entry.bag.bag_id for entry in model.replay.entries]
        epochs = model._global_epoch
        calibration = dict(model.info()["model"]["calibration"])
        weights = {key: value.clone() for key, value in model.model.state_dict().items()}
        mixed = (*self._bags("T1", 2), *self._bags("T2", 2))
        with pytest.raises(ValueError, match="exactly one task"):
            model.fit_task(mixed)
        assert model.scheduler.state_dict() == scheduler
        assert [group["lr"] for group in model.optimizer.param_groups] == learning_rates
        assert model._rng.bit_generator.state == rng
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)
        assert [entry.bag.bag_id for entry in model.replay.entries] == replay_ids
        assert model._global_epoch == epochs
        assert model.info()["model"]["calibration"] == calibration
        assert not model.model.training
        for key, value in weights.items():
            torch.testing.assert_close(model.model.state_dict()[key], value, rtol=0, atol=0)

    def test_explicit_task_override_renames_every_bag(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=2)
        )
        result = model.fit_task((*self._bags("T1", 2), *self._bags("T2", 2)), task_id="joined")
        assert result["task_id"] == "joined"
        assert {entry.bag.task_id for entry in model.replay.entries} == {"joined"}

    def test_empty_task_override_is_rejected_without_training(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(config=CommandTrainerConfig(architecture=self._architecture(), epochs=1))
        with pytest.raises(ValueError, match="task_id must be non-empty"):
            model.fit_task(self._bags("T1"), task_id="")
        assert model._global_epoch == 0
        assert len(model.replay) == 0

    def test_novelty_replacements_preserve_highest_scoring_candidates(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=replace(self._architecture(), memory_size=2), novelty_warmup_epochs=0
            )
        )
        output = model.model(torch.randn(1, 8, 8))
        features = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        output = replace(
            output,
            temporal_features=features,
            memory=replace(
                output.memory, dual_memory_deviation=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 10.0]])
            ),
        )
        memory = model.model.network.memory
        memory.primary_access_count.fill_(2)
        replaced = model._replace_novel_normal_features(output, torch.zeros(1))
        assert replaced == 2
        torch.testing.assert_close(memory.primary_memory, features[0, [7, 6]])

    def test_checkpoint_allows_device_change_but_rejects_training_change(self, tmp_path):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1)
        model = CommandModel(config=config)
        checkpoint = tmp_path / "model.pt"
        model.save_checkpoint(checkpoint)
        state = model.state_dict()
        state["config"]["device"] = "cuda:0"
        torch.save(state, checkpoint)
        restored = CommandModel(config=config)
        restored.load_checkpoint(checkpoint)
        np.testing.assert_array_equal(
            model.predict_bags(self._bags("test"))["raw_scores"],
            restored.predict_bags(self._bags("test"))["raw_scores"],
        )
        incompatible = CommandModel(config=replace(config, learning_rate=0.02))
        with pytest.raises(ValueError, match="configuration"):
            incompatible.load_checkpoint(checkpoint)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
        ],
    )
    def test_checkpoint_resumes_training_with_dropout(self, tmp_path, device):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=replace(self._architecture(), dropout=0.25),
            device=device,
            epochs=1,
            batch_size=4,
            replay_batch_size=2,
            novelty_warmup_epochs=10,
        )
        torch.manual_seed(42)
        model = CommandModel(config=config)
        model.fit_task(self._bags("T1"))
        model.replay.sample(3)
        checkpoint = tmp_path / "model.pt"
        model.save_checkpoint(checkpoint)
        model.fit_task(self._bags("T2"))
        restored = CommandModel(config=config)
        restored.load_checkpoint(checkpoint)
        restored.fit_task(self._bags("T2"))
        for name, value in model.model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)

    @pytest.mark.parametrize("quantile, positives", [(0.99, 1), (0.5, 16)])
    def test_normal_video_decisions_use_bag_maxima_instead_of_snippet_median(self, quantile, positives):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(),
                epochs=1,
                batch_size=8,
                replay_batch_size=0,
                learning_rate=1e-12,
                secondary_memory_learning_rate=1e-12,
                novelty_warmup_epochs=100,
                threshold_quantile=quantile,
            )
        )
        bags = tuple(
            replace(bag, weak_label=0, features=np.random.default_rng(index).normal(size=(32, 8)))
            for index, bag in enumerate(self._bags("T1", count=32))
        )
        model.fit_task(bags)
        prediction = model.predict(np.asarray(bags, dtype=object))
        assert prediction.y_pred.sum() == positives

    def test_calibration_uses_final_eval_scores_after_each_task(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=replace(self._architecture(), dropout=0.25),
                epochs=1,
                batch_size=4,
                replay_batch_size=2,
                buffer_size=16,
                learning_rate=0.01,
                secondary_memory_learning_rate=0.005,
            )
        )
        for task in ("T1", "T2"):
            model.fit_task(self._bags(task))
            normals = tuple(entry.bag for entry in model.replay.entries if entry.bag.weak_label == 0)
            raw = model.predict_bags(normals, batch_size=4)["raw_scores"].astype(np.float64)
            calibration = model.info()["model"]["calibration"]
            assert calibration["median"] == float(np.median(raw))
            assert calibration["threshold"] == float(np.quantile(raw.max(axis=1), 0.99))
            assert model.model.training

    def test_single_normal_bag_remains_normal_when_replay_evicts_it(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(),
                epochs=1,
                batch_size=4,
                replay_batch_size=0,
                buffer_size=1,
                learning_rate=1e-12,
                secondary_memory_learning_rate=1e-12,
            )
        )
        bags = self._bags("T1", count=2)
        model.fit_task(bags)
        assert all(entry.bag.weak_label == 1 for entry in model.replay.entries)
        prediction = model.predict(np.asarray([bags[0]], dtype=object))
        np.testing.assert_array_equal(prediction.y_pred, [0])

    @pytest.mark.parametrize("threshold, expected", [(0.0, 1), (1000.0, 0)])
    def test_explicit_raw_threshold_is_used_for_video_decisions(self, threshold, expected):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(), epochs=1, threshold=threshold, batch_size=4, replay_batch_size=2
            )
        )
        bags = self._bags("T1", count=2)
        model.fit_task(bags)
        np.testing.assert_array_equal(model.predict(np.asarray(bags, dtype=object)).y_pred, [expected, expected])

    def test_anomaly_only_task_invalidates_calibration_without_nominal_replay(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=0, buffer_size=1
            )
        )
        model.fit_task(self._bags("T1", count=2))
        anomalies = tuple(replace(bag, weak_label=1) for bag in self._bags("T2", count=2))
        model.fit_task(anomalies)
        with pytest.raises(RuntimeError, match="threshold.*calibrat"):
            model.predict(np.asarray(anomalies, dtype=object))
        assert model.predict(np.empty(0, dtype=object)).y_pred.size == 0

    def test_calibration_supports_changed_temporal_lengths_without_replay(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        model = CommandModel(
            config=CommandTrainerConfig(
                architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=0, buffer_size=16
            )
        )
        model.fit_task(self._bags("T1"))
        shorter = tuple(replace(bag, features=bag.features[:4]) for bag in self._bags("T2"))
        model.fit_task(shorter)
        assert np.isfinite(model.info()["model"]["calibration"]["threshold"])
        assert model.predict(np.asarray(shorter, dtype=object)).y_pred.shape == (8,)

    def test_legacy_checkpoint_recalibrates_from_nominal_replay(self, tmp_path):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=2)
        model = CommandModel(config=config)
        model.fit_task(self._bags("T1"))
        checkpoint = tmp_path / "legacy.pt"
        state = dict(model.state_dict())
        state.pop("threshold", None)
        state["config"].pop("threshold", None)
        state["config"].pop("threshold_quantile", None)
        state["normal_score_history"] = [np.full(8, 1000.0)]
        state["calibration_median"] = 1000.0
        torch.save(state, checkpoint)
        restored = CommandModel(config=config)
        restored.load_checkpoint(checkpoint)
        normals = tuple(entry.bag for entry in restored.replay.entries if entry.bag.weak_label == 0)
        raw = restored.predict_bags(normals)["raw_scores"]
        assert restored.info()["model"]["calibration"]["median"] == np.median(raw.astype(np.float64))
        np.testing.assert_array_equal(
            model.predict(np.asarray(self._bags("test"), dtype=object)).y_pred,
            restored.predict(np.asarray(self._bags("test"), dtype=object)).y_pred,
        )

    def test_checkpoint_preserves_threshold_when_nominal_bags_are_not_in_replay(self, tmp_path):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(
            architecture=self._architecture(), epochs=1, batch_size=4, replay_batch_size=0, buffer_size=1
        )
        model = CommandModel(config=config)
        bags = self._bags("T1", count=2)
        model.fit_task(bags)
        assert all(entry.bag.weak_label == 1 for entry in model.replay.entries)
        checkpoint = tmp_path / "model.pt"
        model.save_checkpoint(checkpoint)
        restored = CommandModel(config=config)
        restored.load_checkpoint(checkpoint)
        assert restored.info()["model"]["calibration"]["threshold"] is not None
        assert restored.info()["model"]["calibration"] == model.info()["model"]["calibration"]
        np.testing.assert_array_equal(
            model.predict(np.asarray(bags, dtype=object)).y_pred,
            restored.predict(np.asarray(bags, dtype=object)).y_pred,
        )

    def test_legacy_checkpoint_without_nominal_replay_requires_a_threshold(self, tmp_path):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1)
        model = CommandModel(config=config)
        state = dict(model.state_dict())
        state.pop("threshold", None)
        state["config"].pop("threshold", None)
        state["config"].pop("threshold_quantile", None)
        state["normal_score_history"] = [np.full(8, 1000.0)]
        checkpoint = tmp_path / "legacy.pt"
        torch.save(state, checkpoint)
        restored = CommandModel(config=config)
        restored.load_checkpoint(checkpoint)
        with pytest.raises(RuntimeError, match="threshold.*calibrat"):
            restored.predict(np.asarray(self._bags("test"), dtype=object))

    def test_legacy_checkpoint_accepts_new_explicit_threshold_without_nominal_replay(self, tmp_path):
        from dataclasses import replace

        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1)
        state = dict(CommandModel(config=config).state_dict())
        state.pop("threshold", None)
        state["config"].pop("threshold", None)
        state["config"].pop("threshold_quantile", None)
        state["normal_score_history"] = []
        checkpoint = tmp_path / "legacy.pt"
        torch.save(state, checkpoint)
        restored = CommandModel(config=replace(config, threshold=1000.0, threshold_quantile=0.95))
        restored.load_checkpoint(checkpoint)
        np.testing.assert_array_equal(
            restored.predict(np.asarray(self._bags("test"), dtype=object)).y_pred, np.zeros(8)
        )

    def test_old_distance_checkpoint_recalibrates_even_if_it_saved_a_threshold(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1)
        model = CommandModel(config=config)
        model.fit_task(self._bags("T1"))
        state = deepcopy(model.state_dict())
        state.pop("distance_mode", None)
        state["calibration_median"] = 1000.0
        state["calibration_scale"] = 100.0
        state["threshold"] = 1001.0
        restored = CommandModel(config=config)
        restored.load_state_dict(state)
        normals = [entry.bag for entry in restored.replay.entries if entry.bag.weak_label == 0]
        raw = restored.predict_bags(normals)["raw_scores"].astype(np.float64)
        calibration = restored.info()["model"]["calibration"]
        assert calibration["median"] == float(np.median(raw))
        assert calibration["threshold"] == float(np.quantile(raw.max(axis=1), config.threshold_quantile))

    def test_old_distance_checkpoint_without_normals_requires_fresh_calibration(self):
        from pyclad.video.models.command.command import CommandModel
        from pyclad.video.models.command.config import CommandTrainerConfig

        config = CommandTrainerConfig(architecture=self._architecture(), epochs=1, buffer_size=1)
        model = CommandModel(config=config)
        model.fit_task(self._bags("T1"))
        assert all(entry.bag.weak_label == 1 for entry in model.replay.entries)
        state = deepcopy(model.state_dict())
        state.pop("distance_mode", None)
        restored = CommandModel(config=config)
        restored.load_state_dict(state)
        with pytest.raises(RuntimeError, match="threshold.*calibrat"):
            restored.predict(np.asarray(self._bags("test"), dtype=object))

    @pytest.mark.parametrize("threshold_quantile", [0.0, 1.0, float("nan")])
    def test_invalid_threshold_quantile_is_rejected(self, threshold_quantile):
        from pyclad.video.models.command.config import CommandTrainerConfig

        with pytest.raises(ValueError, match="threshold_quantile"):
            CommandTrainerConfig(threshold_quantile=threshold_quantile)

    @pytest.mark.parametrize("threshold", [-1.0, float("inf"), float("nan")])
    def test_invalid_raw_threshold_is_rejected(self, threshold):
        from pyclad.video.models.command.config import CommandTrainerConfig

        with pytest.raises(ValueError, match="threshold"):
            CommandTrainerConfig(threshold=threshold)


class TestCommandReplayRotation:
    @staticmethod
    def _entries(tasks=3, per_cell=4, labels=(0, 1)):
        from pyclad.video.data.sample import VideoBag
        from pyclad.video.models.command.command import ReplayEntry

        entries = []
        for label in labels:
            for task in range(tasks):
                for quartile in range(4):
                    for index in range(per_cell):
                        bag = VideoBag(
                            f"{label}-T{task}-{quartile}-{index}",
                            f"T{task}",
                            np.full((8, 8), index, dtype=np.float32),
                            label,
                        )
                        curve = np.zeros(8, dtype=np.float32)
                        curve[quartile * 2] = 100 * (index + 1)
                        entries.append(ReplayEntry(bag, curve))
        return entries

    def test_small_odd_draws_cover_every_label_cell_and_bag(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        entries = self._entries()
        buffer.add(entries)
        seen = set()
        for _ in range(20):
            sample = buffer.sample(5)
            assert len(sample) == len({bag.bag_id for bag in sample}) == 5
            normal_count = sum(bag.weak_label == 0 for bag in sample)
            assert normal_count in {2, 3}
            seen.update(bag.bag_id for bag in sample)
        assert seen == {entry.bag.bag_id for entry in entries}

    def test_single_draws_alternate_labels(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        buffer.add(self._entries(tasks=1, per_cell=1))
        labels = [buffer.sample(1)[0].weak_label for _ in range(8)]
        assert labels == [0, 1] * 4

    def test_scarce_label_and_exclusions_still_fill_unique_samples(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        entries = self._entries(tasks=1, per_cell=2, labels=(1,))
        nominal = self._entries(tasks=1, per_cell=1, labels=(0,))[:2]
        buffer.add([*entries, *nominal])
        excluded = {entries[0].bag.bag_id, nominal[1].bag.bag_id}
        seen = set()
        for _ in range(4):
            sample = buffer.sample(4, exclude_bag_ids=excluded)
            ids = {bag.bag_id for bag in sample}
            assert len(sample) == len(ids) == 4
            assert not ids.intersection(excluded)
            assert nominal[0].bag.bag_id in ids
            seen.update(ids)
        assert seen == {entry.bag.bag_id for entry in [*entries, *nominal]} - excluded

    def test_checkpoint_restores_the_next_replay_draws(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        buffer.add(self._entries())
        for count in (1, 3, 5):
            buffer.sample(count)
        state = deepcopy(buffer.state_dict())
        restored = ContTrainReplayBuffer()
        restored.load_state_dict(state)
        for count in (5, 1, 8, 3):
            assert [bag.bag_id for bag in buffer.sample(count)] == [bag.bag_id for bag in restored.sample(count)]

    def test_legacy_checkpoint_resets_sampling_state(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        buffer.add(self._entries())
        buffer.sample(7)
        state = deepcopy(buffer.state_dict())
        state.pop("sampling", None)
        expected = ContTrainReplayBuffer()
        expected.load_state_dict(state)
        buffer.load_state_dict(state)
        for count in (1, 5, 3):
            assert [bag.bag_id for bag in buffer.sample(count)] == [bag.bag_id for bag in expected.sample(count)]

    def test_eviction_rebucketing_and_new_ids_preserve_replay_coverage(self):
        from dataclasses import replace

        from pyclad.video.models.command.command import (
            ContTrainReplayBuffer,
            ReplayEntry,
        )

        buffer = ContTrainReplayBuffer(max_size=12)
        entries = self._entries(tasks=1, per_cell=2)
        buffer.add(entries)
        buffer.sample(7)
        survivor = buffer.entries[0]
        curve = np.zeros(8, dtype=np.float32)
        curve[-1] = 1000
        additions = [
            ReplayEntry(replace(survivor.bag, bag_id=prefix + survivor.bag.bag_id), curve)
            for prefix in ("!new-", "znew-")
        ]
        buffer.add([ReplayEntry(survivor.bag, curve), *additions])
        eligible = {entry.bag.bag_id for entry in buffer.entries}
        seen = set()
        for _ in range(24):
            sample = buffer.sample(2)
            ids = {bag.bag_id for bag in sample}
            assert len(ids) == len(sample) == 2
            assert ids <= eligible
            seen.update(ids)
        assert seen == eligible

    def test_sampling_snapshot_does_not_change_when_draws_advance(self):
        from pyclad.video.models.command.command import ContTrainReplayBuffer

        buffer = ContTrainReplayBuffer()
        buffer.add(self._entries())
        buffer.sample(3)
        state = buffer.state_dict()
        before = deepcopy(state["sampling"])
        buffer.sample(7)
        assert state["sampling"] == before
