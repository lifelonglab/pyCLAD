from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pyclad.vision.models.fastflow.config import FastFlowConfig
from pyclad.vision.models.fastflow.fastflow import FastFlow
from pyclad.vision.prediction_results import VisionPredictionResults


def _config(**overrides) -> FastFlowConfig:
    defaults = dict(
        input_size=(64, 64),
        backbone_name="resnet18",
        pretrained_backbone=False,
        flow_steps=2,
        input_range="float01",
        batch_size=1,
        epochs=0,
        show_training_progress=False,
    )
    defaults.update(overrides)
    return FastFlowConfig(**defaults)


@pytest.mark.parametrize("backbone_name", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_fastflow_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(backbone_name=backbone_name, threshold=0.0))

    result = model.predict(data)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (1,)
    assert result.anomaly_scores.shape == (1,)
    assert result.score_maps.shape == (1, 64, 64)


def test_fastflow_rejects_empty_return_nodes():
    with pytest.raises(ValueError, match="at least one feature node"):
        FastFlow(_config(backbone_return_nodes=()))


def test_fastflow_predict_handles_empty_input():
    model = FastFlow(_config(threshold=0.0))

    result = model.predict(np.empty((0, 64, 64, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)
    assert result.anomaly_scores.shape == (0,)
    assert result.score_maps.shape == (0,)


def test_fastflow_seed_makes_results_reproducible():
    data = np.random.default_rng(7).random((2, 64, 64, 3), dtype=np.float32)

    def score_maps(seed: int) -> np.ndarray:
        # pretrained_backbone=False -> backbone init is random; combined with the random
        # flow weights and channel permutations, the whole model is seed-dependent.
        model = FastFlow(_config(seed=seed, threshold=0.0))
        return model.predict(data).score_maps

    assert np.array_equal(score_maps(123), score_maps(123))  # same seed -> identical
    assert not np.array_equal(score_maps(123), score_maps(456))  # different seed -> different


def test_fastflow_inference_restores_training_mode():
    # _inference_maps must not leave the module stuck in eval(): pyCLAD reuses the same
    # instance across the concept stream and Lightning (>=2.2) no longer resets train mode at
    # fit start, so a leaked eval() would silently train BatchNorm/Dropout in eval mode on
    # later fits. predict() must round-trip the module's training flag.
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(threshold=0.0))

    model.module.train()
    model.predict(data)
    assert model.module.training is True

    model.module.eval()
    model.predict(data)
    assert model.module.training is False


def test_fastflow_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(batch_size=2, epochs=1))

    model.fit(data)
    result = model.predict(data)

    assert result.y_pred.shape == (2,)
    assert result.anomaly_scores.shape == (2,)
    assert result.score_maps.shape == (2, 64, 64)
    assert model.additional_info()["last_loss"] is not None


def test_fastflow_second_fit_still_trains_in_train_mode():
    # The eval-mode restore in _run_inference must let a 2nd fit on a reused instance keep
    # training in TRAIN mode after a predict() forced eval(). A trainable BatchNorm's
    # num_batches_tracked only advances in train mode, so a leaked eval() (training BN in eval
    # mode) would freeze it -- this is the regression to guard, not mere weight movement (the
    # optimizer steps in eval mode too). freeze_backbone=False puts the backbone BN in the
    # trainable path (and also covers that otherwise-untested branch).
    rng = np.random.default_rng(2)
    data_a = rng.random((2, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((2, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(batch_size=2, epochs=1, freeze_backbone=False))

    model.fit(data_a)
    model.predict(data_a)  # forces eval(); must be restored before the next fit

    bn = next(m for m in model.module.network.modules() if isinstance(m, torch.nn.BatchNorm2d))
    tracked_before = bn.num_batches_tracked.clone()
    model.fit(data_b)

    assert bn.num_batches_tracked.item() > tracked_before.item()  # 2nd fit ran in train mode
    assert model.module.training is True  # left in train mode, not stuck in eval


def test_fastflow_fit_calibrates_threshold_from_quantile():
    # With threshold=None, fit() must calibrate a finite threshold from the configured quantile
    # of the training scores. Assert against an independent recomputation of that quantile (not
    # predict()'s own decision rule, which would be tautological).
    data = np.random.default_rng(3).random((4, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(batch_size=2, epochs=1))  # threshold defaults to None

    model.fit(data)

    threshold = model.additional_info()["threshold"]
    assert threshold is not None and np.isfinite(threshold)

    expected = float(np.quantile(model._score_data(data), model.config.threshold_quantile))
    assert threshold == pytest.approx(expected)


def test_fastflow_early_stopping_restores_best_weights():
    # Exercises the early-stopping + best-weights-restore branch in LightningVisionModel.fit,
    # which is on by default (early_stopping_restore_best=True) but otherwise untested.
    data = np.random.default_rng(4).random((2, 64, 64, 3), dtype=np.float32)
    model = FastFlow(_config(batch_size=2, epochs=2, early_stopping_patience=0))

    with patch.object(
        model.module.network,
        "load_state_dict",
        wraps=model.module.network.load_state_dict,
    ) as restore_spy:
        model.fit(data)

    assert restore_spy.called  # best weights were restored after training
    assert model.module.training is True


@pytest.mark.parametrize("input_size", [(0, 64), (64, 0), (-1, 64)])
def test_fastflow_config_rejects_nonpositive_input_size(input_size):
    with pytest.raises(ValidationError):
        FastFlowConfig(input_size=input_size)


@pytest.mark.parametrize("normalize_std", [(0.0, 0.5, 0.5), (-0.1, 0.5, 0.5)])
def test_fastflow_config_rejects_nonpositive_normalize_std(normalize_std):
    with pytest.raises(ValidationError):
        FastFlowConfig(normalize_std=normalize_std)


def test_fastflow_config_rejects_half_specified_normalization():
    with pytest.raises(ValidationError):
        FastFlowConfig(normalize_std=None)  # mean keeps its default 3-tuple -> mismatch
    with pytest.raises(ValidationError):
        FastFlowConfig(normalize_mean=None)  # std keeps its default 3-tuple -> mismatch


def test_fastflow_config_accepts_valid_normalization():
    FastFlowConfig(normalize_mean=None, normalize_std=None)  # both off
    FastFlowConfig(normalize_std=(0.2, 0.2, 0.2))  # both set (mean defaults)
