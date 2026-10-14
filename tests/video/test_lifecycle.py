"""Full archive-to-scenario coverage for the single COMMAND workflow."""

import json

import numpy as np
import pytest
import torch

from pyclad.video.data.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    CommandUcfCrimeDataset,
)
from pyclad.video.models.command.command import CommandModel
from pyclad.video.models.command.config import (
    CommandArchitectureConfig,
    CommandConfig,
)
from pyclad.video.strategies.config import ContTrainPlusPlusConfig
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy


@pytest.fixture
def archive(tmp_path):
    """Make disjoint 32-window RGB/flow train and test splits."""
    anomalies = [f"{category}/train.mp4" for category in COMMAND_UCF_CRIME_CONCEPT_ORDER]
    normals = [f"Normal_Videos_event/train{index}.mp4" for index in range(13)]
    test = ["Normal_Videos_event/test.mp4", "Abuse/test.mp4"]
    rng = np.random.default_rng(19)
    for stream in ("all_rgbs", "all_flows"):
        for name in anomalies + normals + test:
            path = tmp_path / stream / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, rng.normal(size=(32, 1024)).astype(np.float32))
    (tmp_path / "train_normal.txt").write_text("\n".join(normals) + "\n")
    (tmp_path / "train_anomaly.txt").write_text("\n".join(anomalies) + "\n")
    (tmp_path / "test_normalv2.txt").write_text(f"{test[0]} 64 -1\n")
    (tmp_path / "test_anomalyv2.txt").write_text(f"{test[1]}|64|[33, 48]\n")
    return tmp_path


def test_three_tasks_use_core_model_scenario_metrics_and_writer(archive):
    from pyclad.callbacks.evaluation.concept_metric_evaluation import (
        ConceptMetricCallback,
    )
    from pyclad.metrics.base.roc_auc import RocAuc
    from pyclad.models.torch_backbone import TorchBackbone
    from pyclad.output.json_writer import JsonOutputWriter
    from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
    from pyclad.video.callbacks.video_frame_metric_callback import (
        VideoFrameMetricCallback,
    )
    from pyclad.video.metrics.frame_average_precision import FrameAveragePrecision
    from pyclad.video.metrics.frame_roc_auc import FrameRocAuc
    from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy

    torch.manual_seed(11)
    architecture = CommandArchitectureConfig(
        hidden_dim=8, state_dim=3, mamba_blocks=1, memory_size=5, projection_dim=6, dropout=0.0
    )
    config = CommandConfig(architecture=architecture, device="cpu")
    strategy_config = ContTrainPlusPlusConfig(epochs=1, batch_size=4, replay_batch_size=2, buffer_size=6)
    model = CommandModel(config=config)
    assert isinstance(model, TorchBackbone)
    dataset = CommandUcfCrimeDataset(archive)
    assert [c.name for c in dataset.train_concepts()] == ["T1", "T2", "T3"]
    assert [len(c.data) for c in dataset.train_concepts()] == [8, 8, 10]
    strategy = ContTrainPlusPlusStrategy(model, config=strategy_config)
    callbacks = [
        VideoFrameMetricCallback(FrameRocAuc()),
        VideoFrameMetricCallback(FrameAveragePrecision()),
        ConceptMetricCallback(RocAuc(), []),
    ]
    ConceptIncrementalScenario(dataset, strategy, callbacks).run()
    assert len(strategy.replay) == 6
    for callback in callbacks:
        info = next(iter(callback.info().values()))
        assert list(info["metric_matrix"]) == ["T1", "T2", "T3"]
        assert all(0 <= row["test"] <= 1 for row in info["metric_matrix"].values())
    output = archive / "result.json"
    JsonOutputWriter(output).write([dataset, model, strategy, *callbacks])
    saved = json.loads(output.read_text())
    assert saved["model"]["name"] == "COMMAND"
    assert saved["concept_metric_callback_Frame-AP"]["test_order"] == ["test"]
    strategy.save_checkpoint(archive / "strategy.pt")
    restored = CommandModel(config=config)
    restored_strategy = ContTrainPlusPlusStrategy(restored, config=strategy_config)
    restored_strategy.load_checkpoint(archive / "strategy.pt")
    test_data = dataset.test_concepts()[0].data
    for training in (True, False):
        model.get_module().train(training)
        prediction = model.predict(test_data)
        assert model.get_module().training is training
        np.testing.assert_array_equal(prediction.window_scores, restored.predict(test_data).window_scores)


def test_example_runs_core_callbacks_and_writes_summaries(archive, monkeypatch):
    """Execute the example with real training and output, substituting only the data source and run size."""
    import runpy
    import sys
    from dataclasses import replace
    from pathlib import Path

    from pyclad.video.data import command_ucf_crime
    from pyclad.video.models.command import config
    from pyclad.video.strategies import config as strategy_config
    from tests.video.test_data import write_feature_zip

    model_config = config.CommandConfig
    trainer_config = strategy_config.ContTrainPlusPlusConfig

    def small_model_config(**kwargs):
        return replace(
            model_config(**kwargs),
            architecture=CommandArchitectureConfig(
                hidden_dim=8, state_dim=3, mamba_blocks=1, memory_size=5, projection_dim=6, dropout=0.0
            ),
            device="cpu",
        )

    def small_config(**kwargs):
        return replace(
            trainer_config(**kwargs),
            epochs=1,
            batch_size=4,
            replay_batch_size=2,
            buffer_size=6,
        )

    example = Path(__file__).resolve().parents[2] / "examples/models/video/command_example.py"
    feature_zip = write_feature_zip(archive, archive / "snapshot/ucf-crime-rgb-flow.zip")
    monkeypatch.setattr(command_ucf_crime, "hf_hub_download", lambda **kwargs: str(feature_zip))
    monkeypatch.setattr(config, "CommandConfig", small_model_config)
    monkeypatch.setattr(strategy_config, "ContTrainPlusPlusConfig", small_config)
    monkeypatch.setattr(sys, "argv", [str(example)])
    monkeypatch.chdir(archive)
    runpy.run_path(str(example), run_name="__main__")

    saved = json.loads((archive / "output.json").read_text())
    for metric in ("ROC-AUC", "AP", "Frame-ROC-AUC", "Frame-AP"):
        payload = saved[f"concept_metric_callback_{metric}"]
        assert payload["concepts_order"] == ["T1", "T2", "T3"]
        assert payload["test_order"] == ["test"]
        final = payload["metric_matrix"]["T3"]["test"]
        assert 0 <= final <= 1
        assert payload["metrics"]["FinalStepAverage"] == final
    timing = saved["time_evaluation_callback"]
    assert timing["train_time_total"] > 0 and timing["eval_time_total"] > 0
    assert all(timing["time_by_concept"][task]["train_time"] > 0 for task in ("T1", "T2", "T3"))
    assert saved["strategy"]["global_epochs"] == 3
    assert saved["strategy"]["buffer_bags"] == 6
    assert "training" not in saved["model"]


def test_duplicate_training_rows_keep_distinct_bags(archive):
    path = archive / "train_normal.txt"
    lines = path.read_text().splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n")
    dataset = CommandUcfCrimeDataset(archive).read_dataset()
    bags = [bag for c in dataset.train_concepts() for bag in c.data]
    assert len(bags) == len({b.bag_id for b in bags}) == 26
    assert all(b.features.shape == (32, 2048) for b in bags)
    labels = dataset.test_concepts()[0].frame_labels
    assert labels["Abuse/test.mp4"].sum() == 16


@pytest.mark.parametrize("fault", ["overlap", "shape", "nonfinite", "balance"])
def test_reader_rejects_invalid_archive(archive, fault):
    if fault == "overlap":
        (archive / "test_normalv2.txt").write_text("Normal_Videos_event/train0.mp4 64 -1\n")
    elif fault == "balance":
        (archive / "train_normal.txt").write_text("")
    else:
        values = np.zeros((2, 1024)) if fault == "shape" else np.full((32, 1024), np.nan)
        np.save(archive / "all_rgbs/Abuse/train.mp4.npy", values)
    with pytest.raises(ValueError):
        CommandUcfCrimeDataset(archive).read_dataset()


def test_training_rejects_mixed_tasks_and_row_matrices(archive):
    dataset = CommandUcfCrimeDataset(archive).read_dataset()
    model = CommandModel(config=CommandConfig(architecture=CommandArchitectureConfig(hidden_dim=8, memory_size=5)))
    strategy = ContTrainPlusPlusStrategy(model, config=ContTrainPlusPlusConfig(epochs=1))
    with pytest.raises(ValueError, match="exactly one task"):
        strategy.learn(np.concatenate([c.data for c in dataset.train_concepts()]))
    with pytest.raises(ValueError, match="VideoBag"):
        strategy.learn(np.zeros((3, 2048)))
