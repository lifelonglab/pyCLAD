from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pyclad.vision.models.rd4ad.config import RD4ADConfig
from pyclad.vision.models.rd4ad.rd4ad import RD4AD
from pyclad.vision.prediction_results import VisionPredictionResults


def _config(**overrides) -> RD4ADConfig:
    defaults = dict(
        input_size=(32, 32),
        backbone_name="resnet18",
        pretrained_encoder=False,
        input_range="float01",
        batch_size=1,
        epochs=0,
        show_training_progress=False,
    )
    defaults.update(overrides)
    return RD4ADConfig(**defaults)


@pytest.mark.parametrize("backbone_name", ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])
def test_rd4ad_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(backbone_name=backbone_name, threshold=0.0))

    result = model.predict(data)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (1,)
    assert result.anomaly_scores.shape == (1,)
    assert result.score_maps.shape == (1, 32, 32)


def test_rd4ad_predict_handles_empty_input():
    model = RD4AD(_config(threshold=0.0))

    result = model.predict(np.empty((0, 32, 32, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)
    assert result.anomaly_scores.shape == (0,)
    assert result.score_maps.shape == (0,)


def test_rd4ad_seed_makes_results_reproducible():
    data = np.random.default_rng(7).random((2, 32, 32, 3), dtype=np.float32)

    def score_maps(seed: int) -> np.ndarray:
        # pretrained_encoder=False -> teacher init is random; combined with the random
        # OCBE / decoder weights, the whole network is seed-dependent.
        model = RD4AD(_config(seed=seed, threshold=0.0))
        return model.predict(data).score_maps

    assert np.array_equal(score_maps(123), score_maps(123))  # same seed -> identical
    assert not np.array_equal(score_maps(123), score_maps(456))  # different seed -> different


def test_rd4ad_inference_restores_training_mode():
    # _inference_maps must not leave the module stuck in eval(): pyCLAD reuses the same instance
    # across the concept stream and Lightning (>=2.2) no longer resets train mode at fit start,
    # so a leaked eval() would silently train BatchNorm/Dropout in eval mode on later fits.
    data = np.random.default_rng(0).random((1, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(threshold=0.0))

    model.module.train()
    model.predict(data)
    assert model.module.training is True

    model.module.eval()
    model.predict(data)
    assert model.module.training is False


def test_rd4ad_frozen_teacher_stays_in_eval_during_training():
    # The teacher encoder is frozen by default and must remain in eval() even when the module is
    # switched to train mode, so its BatchNorm running stats never drift.
    model = RD4AD(_config())

    model.module.train()

    assert model.module.network.encoder.training is False
    assert model.module.network.decoder.training is True
    assert model.module.network.bn.training is True
    assert all(not p.requires_grad for p in model.module.network.encoder.parameters())


def test_rd4ad_pretrained_weights_prefer_imagenet_v1():
    # RD4AD's local weights resolver pins the official ImageNet V1 teacher checkpoints;
    # torchvision's .DEFAULT switched resnet50 / wide_resnet50_2 to the V2 "improved recipe",
    # which would change the frozen teacher's feature distribution. Resolving the enum member
    # does not download weights.
    import torchvision.models as tv_models

    from pyclad.vision.models.rd4ad.standard.resnet import _resolve_torchvision_weights

    for backbone_name in ("resnet18", "resnet34", "resnet50", "wide_resnet50_2"):
        weights = _resolve_torchvision_weights(tv_models, getattr(tv_models, backbone_name))
        assert weights is not None
        assert weights.name == "IMAGENET1K_V1"


def test_rd4ad_score_smoothing_changes_score_maps():
    # The Gaussian-smoothing path (paper's sigma=4) is RD4AD-specific. With identical weights
    # (same seed) a positive sigma must actually alter the fused anomaly map vs the unsmoothed
    # one, proving the gaussian_blur branch in _anomaly_map is reached.
    data = np.random.default_rng(5).random((1, 32, 32, 3), dtype=np.float32)
    unsmoothed = RD4AD(_config(seed=11, threshold=0.0)).predict(data).score_maps
    smoothed = RD4AD(_config(seed=11, threshold=0.0, score_smoothing_sigma=2.0)).predict(data).score_maps

    assert unsmoothed.shape == smoothed.shape == (1, 32, 32)
    assert not np.array_equal(unsmoothed, smoothed)


def test_rd4ad_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(batch_size=2, epochs=1))

    model.fit(data)
    result = model.predict(data)

    assert result.y_pred.shape == (2,)
    assert result.anomaly_scores.shape == (2,)
    assert result.score_maps.shape == (2, 32, 32)
    assert model.additional_info()["last_loss"] is not None


def test_rd4ad_fit_calibrates_threshold_from_quantile():
    # With threshold=None, fit() must calibrate a finite threshold from the configured quantile
    # of the training scores. Assert against an independent recomputation of that quantile.
    data = np.random.default_rng(3).random((4, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(batch_size=2, epochs=1))  # threshold defaults to None

    model.fit(data)

    threshold = model.additional_info()["threshold"]
    assert threshold is not None and np.isfinite(threshold)

    expected = float(np.quantile(model._score_data(data), model.config.threshold_quantile))
    assert threshold == pytest.approx(expected)


def test_rd4ad_fit_reduces_training_loss():
    # Guards the gradient path through the OCBE bottleneck + student decoder: more training on a
    # fixed batch must reduce the distillation loss (same seed -> identical init and shuffling).
    data = np.random.default_rng(6).random((4, 32, 32, 3), dtype=np.float32)

    def final_loss(epochs: int) -> float:
        model = RD4AD(_config(seed=0, batch_size=4, epochs=epochs))
        model.fit(data)
        return model.additional_info()["last_loss"]

    loss_short = final_loss(1)
    loss_long = final_loss(20)

    assert np.isfinite(loss_short) and np.isfinite(loss_long)
    assert loss_long < loss_short


def test_rd4ad_second_fit_still_trains_in_train_mode():
    # The eval-mode restore in the shared base _run_inference must let a 2nd fit on a reused
    # instance keep training in TRAIN mode after a predict() forced eval(). freeze_encoder=False
    # puts the teacher BatchNorm in the trainable path; its num_batches_tracked only advances in
    # train mode, so a leaked eval() (training BN in eval mode) would freeze it.
    rng = np.random.default_rng(2)
    data_a = rng.random((2, 32, 32, 3), dtype=np.float32)
    data_b = rng.random((2, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(batch_size=2, epochs=1, freeze_encoder=False))

    model.fit(data_a)
    model.predict(data_a)  # forces eval(); must be restored before the next fit

    bn = next(m for m in model.module.network.modules() if isinstance(m, torch.nn.BatchNorm2d))
    tracked_before = bn.num_batches_tracked.clone()
    model.fit(data_b)

    assert bn.num_batches_tracked.item() > tracked_before.item()  # 2nd fit ran in train mode
    assert model.module.training is True


def test_rd4ad_early_stopping_restores_best_weights():
    # Exercises the early-stopping + best-weights-restore branch in LightningVisionModel.fit,
    # which is on by default (early_stopping_restore_best=True) but otherwise untested here.
    data = np.random.default_rng(4).random((2, 32, 32, 3), dtype=np.float32)
    model = RD4AD(_config(batch_size=2, epochs=2, early_stopping_patience=0))

    with patch.object(
        model.module.network,
        "load_state_dict",
        wraps=model.module.network.load_state_dict,
    ) as restore_spy:
        model.fit(data)

    assert restore_spy.called  # best weights were restored after training
    assert model.module.training is True


def test_rd4ad_unfrozen_encoder_is_optimized():
    # freeze_encoder=False must put the teacher encoder parameters into the optimizer and the
    # encoder into train mode (covering the otherwise-untested branch).
    model = RD4AD(_config(freeze_encoder=False))

    model.module.train()
    assert model.module.network.encoder.training is True
    assert all(p.requires_grad for p in model.module.network.encoder.parameters())

    optimizer = model.module.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in model.module.network.encoder.parameters())


def test_rd4ad_config_rejects_negative_smoothing_sigma():
    with pytest.raises(ValidationError):
        RD4ADConfig(score_smoothing_sigma=-1.0)


@pytest.mark.parametrize("input_size", [(0, 32), (32, 0), (-1, 32)])
def test_rd4ad_config_rejects_nonpositive_input_size(input_size):
    with pytest.raises(ValidationError):
        RD4ADConfig(input_size=input_size)


@pytest.mark.parametrize("normalize_std", [(0.0, 0.5, 0.5), (-0.1, 0.5, 0.5)])
def test_rd4ad_config_rejects_nonpositive_normalize_std(normalize_std):
    with pytest.raises(ValidationError):
        RD4ADConfig(normalize_std=normalize_std)


def test_rd4ad_config_rejects_half_specified_normalization():
    with pytest.raises(ValidationError):
        RD4ADConfig(normalize_std=None)  # mean keeps its default 3-tuple -> mismatch
    with pytest.raises(ValidationError):
        RD4ADConfig(normalize_mean=None)  # std keeps its default 3-tuple -> mismatch


def test_rd4ad_config_accepts_valid_normalization():
    RD4ADConfig(normalize_mean=None, normalize_std=None)  # both off
    RD4ADConfig(normalize_std=(0.2, 0.2, 0.2))  # both set (mean defaults)


def test_rd4ad_resolve_backbone_spec_rejects_unknown():
    # The RD4ADConfig Literal shadows this guard on the public path, so exercise it directly to
    # keep the shipped "unsupported backbone" error from rotting.
    from pyclad.vision.models.rd4ad.standard.resnet import resolve_backbone_spec

    with pytest.raises(ValueError, match="Unsupported RD4AD backbone"):
        resolve_backbone_spec("bogus")
