from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pyclad.vision.models.stfpm.backbones import (
    default_stfpm_return_nodes,
    supported_backbone_names,
)
from pyclad.vision.models.stfpm.config import STFPMConfig
from pyclad.vision.models.stfpm.stfpm import STFPM
from pyclad.vision.models.utilities.backbones import (
    EFFICIENTNET_BACKBONES,
    MOBILENET_BACKBONES,
    RESNET_BACKBONES,
)
from pyclad.vision.prediction_results import VisionPredictionResults


def _config(**overrides) -> STFPMConfig:
    # pretrained_teacher=False keeps the tests offline (no ImageNet download) and, combined with
    # the always-random student, makes the whole model seed-dependent.
    defaults = dict(
        input_size=(64, 64),
        backbone_name="resnet18",
        pretrained_teacher=False,
        pretrained_student=False,
        input_range="float01",
        batch_size=1,
        epochs=0,
        show_training_progress=False,
    )
    defaults.update(overrides)
    return STFPMConfig(**defaults)


@pytest.mark.parametrize("backbone_name", ["resnet18", "mobilenet_v2", "efficientnet_b0"])
def test_stfpm_predict_shapes(backbone_name: str):
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(backbone_name=backbone_name, threshold=0.0))

    result = model.predict(data)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (1,)
    assert result.anomaly_scores.shape == (1,)
    assert result.score_maps.shape == (1, 64, 64)


def test_stfpm_rejects_empty_return_nodes():
    with pytest.raises(ValueError, match="at least one feature node"):
        STFPM(_config(backbone_return_nodes=()))


def test_stfpm_predict_handles_empty_input():
    model = STFPM(_config(threshold=0.0))

    result = model.predict(np.empty((0, 64, 64, 3), dtype=np.float32))

    assert result.y_pred.shape == (0,)
    assert result.anomaly_scores.shape == (0,)
    assert result.score_maps.shape == (0,)


def test_stfpm_seed_makes_results_reproducible():
    data = np.random.default_rng(7).random((2, 64, 64, 3), dtype=np.float32)

    def score_maps(seed: int) -> np.ndarray:
        # Both teacher (pretrained_teacher=False) and student are randomly initialized, so the
        # whole teacher-student feature discrepancy is seed-dependent.
        model = STFPM(_config(seed=seed, threshold=0.0))
        return model.predict(data).score_maps

    assert np.array_equal(score_maps(123), score_maps(123))  # same seed -> identical
    assert not np.array_equal(score_maps(123), score_maps(456))  # different seed -> different


def test_stfpm_inference_restores_training_mode():
    # _inference_maps must not leave the module stuck in eval(): pyCLAD reuses the same instance
    # across the concept stream and Lightning (>=2.2) no longer resets train mode at fit start,
    # so a leaked eval() would silently train the student's BatchNorm in eval mode on later fits.
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(threshold=0.0))

    model.module.train()
    model.predict(data)
    assert model.module.training is True

    model.module.eval()
    model.predict(data)
    assert model.module.training is False


def test_stfpm_teacher_pinned_to_eval_in_train_mode():
    # Canonical STFPM keeps the teacher frozen and in eval() throughout, even while the module is
    # in train mode, so the teacher's BatchNorm running stats never drift during student training.
    model = STFPM(_config())

    model.module.train()
    network = model.module.network

    assert network.training is True
    assert network.student.training is True
    assert network.teacher.training is False  # pinned to eval despite train(True)
    assert all(not p.requires_grad for p in network.teacher.parameters())  # frozen teacher
    assert any(p.requires_grad for p in network.student.parameters())  # trainable student


def test_stfpm_optimizes_student_only():
    # The optimizer must update ONLY the student; the teacher is frozen throughout training.
    model = STFPM(_config())
    optimizer = model.module.configure_optimizers()

    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    student_params = {id(p) for p in model.module.network.student.parameters()}
    teacher_params = {id(p) for p in model.module.network.teacher.parameters()}

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimized == student_params
    assert optimized.isdisjoint(teacher_params)


def test_stfpm_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(batch_size=2, epochs=1))

    model.fit(data)
    result = model.predict(data)

    assert result.y_pred.shape == (2,)
    assert result.anomaly_scores.shape == (2,)
    assert result.score_maps.shape == (2, 64, 64)
    assert model.additional_info()["last_loss"] is not None


def test_stfpm_second_fit_still_trains_in_train_mode():
    # The eval-mode restore in the base _run_inference must let a 2nd fit on a reused instance keep
    # training in TRAIN mode after a predict() forced eval(). A trainable student BatchNorm's
    # num_batches_tracked only advances in train mode, so a leaked eval() (training BN in eval mode)
    # would freeze it -- this is the regression to guard, not mere weight movement (the optimizer
    # steps in eval mode too). This is exactly the lifecycle bug the original standalone STFPM had.
    rng = np.random.default_rng(2)
    data_a = rng.random((2, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((2, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(batch_size=2, epochs=1))

    model.fit(data_a)
    model.predict(data_a)  # forces eval(); must be restored before the next fit

    bn = next(m for m in model.module.network.student.modules() if isinstance(m, torch.nn.BatchNorm2d))
    tracked_before = bn.num_batches_tracked.clone()
    model.fit(data_b)

    assert bn.num_batches_tracked.item() > tracked_before.item()  # 2nd fit ran in train mode
    assert model.module.training is True  # left in train mode, not stuck in eval


def test_stfpm_fit_calibrates_threshold_from_quantile():
    # With threshold=None, fit() must calibrate a finite threshold from the configured quantile of
    # the training scores. Assert against an independent recomputation of that quantile (not
    # predict()'s own decision rule, which would be tautological).
    data = np.random.default_rng(3).random((4, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(batch_size=2, epochs=1))  # threshold defaults to None

    model.fit(data)

    threshold = model.additional_info()["threshold"]
    assert threshold is not None and np.isfinite(threshold)

    expected = float(np.quantile(model._score_data(data), model.config.threshold_quantile))
    assert threshold == pytest.approx(expected)


def test_stfpm_early_stopping_restores_best_weights():
    # Exercises the early-stopping + best-weights-restore branch in LightningVisionModel.fit,
    # which is on by default (early_stopping_restore_best=True) but otherwise untested.
    data = np.random.default_rng(4).random((2, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(batch_size=2, epochs=2, early_stopping_patience=0))

    with patch.object(
        model.module.network,
        "load_state_dict",
        wraps=model.module.network.load_state_dict,
    ) as restore_spy:
        model.fit(data)

    assert restore_spy.called  # best weights were restored after training
    assert model.module.training is True


def test_stfpm_config_paper_defaults():
    # The defaults must reproduce the original STFPM paper (BMVC 2021): SGD lr=0.4, momentum=0.9,
    # weight_decay=1e-4, 100 epochs, batch 32, 256x256 input, resnet18, max scoring.
    config = STFPMConfig()

    assert config.backbone_name == "resnet18"
    assert config.input_size == (256, 256)
    assert config.batch_size == 32
    assert config.epochs == 100
    assert config.learning_rate == pytest.approx(0.4)
    assert config.momentum == pytest.approx(0.9)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.score_mode == "max"
    assert config.pretrained_teacher is True
    assert config.pretrained_student is False
    assert config.freeze_teacher is True


@pytest.mark.parametrize("input_size", [(0, 64), (64, 0), (-1, 64)])
def test_stfpm_config_rejects_nonpositive_input_size(input_size):
    with pytest.raises(ValidationError):
        STFPMConfig(input_size=input_size)


@pytest.mark.parametrize("momentum", [-0.1, 1.0, 1.5])
def test_stfpm_config_rejects_out_of_range_momentum(momentum):
    with pytest.raises(ValidationError):
        STFPMConfig(momentum=momentum)


@pytest.mark.parametrize("normalize_std", [(0.0, 0.5, 0.5), (-0.1, 0.5, 0.5)])
def test_stfpm_config_rejects_nonpositive_normalize_std(normalize_std):
    with pytest.raises(ValidationError):
        STFPMConfig(normalize_std=normalize_std)


def test_stfpm_config_rejects_half_specified_normalization():
    with pytest.raises(ValidationError):
        STFPMConfig(normalize_std=None)  # mean keeps its default 3-tuple -> mismatch
    with pytest.raises(ValidationError):
        STFPMConfig(normalize_mean=None)  # std keeps its default 3-tuple -> mismatch


def test_stfpm_config_accepts_valid_normalization():
    STFPMConfig(normalize_mean=None, normalize_std=None)  # both off
    STFPMConfig(normalize_std=(0.2, 0.2, 0.2))  # both set (mean defaults)


# --- non-frozen teacher (freeze_teacher=False) path -------------------------------------------
# Non-canonical for STFPM (the paper always freezes the teacher) but a fully wired, supported knob,
# so it gets the same contract-level coverage as the frozen path.


def test_stfpm_non_frozen_teacher_is_trainable_and_follows_train_mode():
    model = STFPM(_config(freeze_teacher=False))
    network = model.module.network

    model.module.train()
    assert network.teacher.training is True  # follows train(True), not pinned to eval
    model.module.eval()
    assert network.teacher.training is False

    assert any(p.requires_grad for p in network.teacher.parameters())  # teacher is trainable


def test_stfpm_non_frozen_teacher_optimizes_all_parameters():
    # When the teacher is not frozen the optimizer must cover BOTH networks, unlike the canonical
    # frozen case (test_stfpm_optimizes_student_only) which optimizes the student only.
    model = STFPM(_config(freeze_teacher=False))
    optimizer = model.module.configure_optimizers()

    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    all_params = {id(p) for p in model.module.network.parameters()}

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimized == all_params


def test_stfpm_non_frozen_teacher_fit_and_predict_smoke():
    # End-to-end smoke for the non-frozen branch including early-stopping best-weights restore,
    # which load_state_dicts the WHOLE network (teacher included) -- a wider scope than the frozen
    # case. Confirms fit+predict succeed and the module is left in train mode.
    data = np.random.default_rng(8).random((2, 64, 64, 3), dtype=np.float32)
    model = STFPM(_config(freeze_teacher=False, batch_size=2, epochs=2, early_stopping_patience=0))

    model.fit(data)
    result = model.predict(data)

    assert result.score_maps.shape == (2, 64, 64)
    assert model.module.training is True  # train mode restored after fit/predict


# --- default backbone return-node presets ------------------------------------------------------


@pytest.mark.parametrize(
    "backbone_name, expected",
    [
        ("resnet18", ("layer1", "layer2", "layer3")),
        ("wide_resnet50_2", ("layer1", "layer2", "layer3")),
        ("mobilenet_v2", ("features.3", "features.6", "features.13")),
        ("efficientnet_b0", ("features.2", "features.3", "features.5")),
    ],
)
def test_stfpm_default_return_nodes_values(backbone_name, expected):
    # Locks the canonical per-family node presets so the shared-tuple composition cannot change them.
    assert tuple(default_stfpm_return_nodes(backbone_name)) == expected


def test_stfpm_default_return_nodes_cover_shared_backbone_families():
    # Drift guard: the preset table is composed from the shared backbone tuples, so every backbone
    # in those families must have a preset. Catches a future family member silently missing here.
    supported = set(supported_backbone_names())
    shared = set(RESNET_BACKBONES) | set(MOBILENET_BACKBONES) | set(EFFICIENTNET_BACKBONES)
    assert shared <= supported


def test_stfpm_default_return_nodes_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="No default return nodes"):
        default_stfpm_return_nodes("not_a_backbone")
