from typing import Optional

import numpy as np
import pytest
import torch

from pyclad.vision.models.paste.config import PaSTeConfig
from pyclad.vision.models.paste.paste import PaSTe
from pyclad.vision.prediction_results import VisionPredictionResults


@pytest.mark.parametrize(
    ("backbone_name", "student_bootstrap_layer"),
    [
        ("resnet18", 0),
        ("mobilenet_v2", 1),
        ("efficientnet_b0", 1),
        ("resnet18", None),
    ],
)
def test_paste_predict_shapes(backbone_name: str, student_bootstrap_layer: Optional[int]):
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name=backbone_name,
            student_bootstrap_layer=student_bootstrap_layer,
            pretrained_teacher=False,
            pretrained_student=False,
            input_range="float01",
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
        )
    )

    result = model.predict(data)

    assert isinstance(result, VisionPredictionResults)
    assert result.y_pred.shape == (1,)
    assert result.anomaly_scores.shape == (1,)
    assert result.score_maps.shape == (1, 64, 64)


def test_paste_rejects_invalid_bootstrap_layer():
    with pytest.raises(ValueError, match="student_bootstrap_layer"):
        PaSTeConfig(backbone_name="resnet18", ad_layers=(1, 2, 3), student_bootstrap_layer=1)


def test_paste_seed_makes_results_reproducible():
    data = np.random.default_rng(7).random((2, 64, 64, 3), dtype=np.float32)

    def score_maps(seed: int) -> np.ndarray:
        model = PaSTe(
            PaSTeConfig(
                input_size=(64, 64),
                backbone_name="resnet18",
                student_bootstrap_layer=0,
                pretrained_teacher=False,
                pretrained_student=False,  # random student init -> seed-dependent
                input_range="float01",
                batch_size=1,
                epochs=0,
                threshold=0.5,
                show_training_progress=False,
                seed=seed,
            )
        )
        return model.predict(data).score_maps

    assert np.array_equal(score_maps(123), score_maps(123))  # same seed -> identical
    assert not np.array_equal(score_maps(123), score_maps(456))  # different seed -> different


def test_paste_inference_restores_training_mode():
    # _inference_maps must not leave the module stuck in eval(): the same instance is fitted
    # repeatedly across the concept stream and Lightning (>=2.2) no longer resets train mode at
    # fit start, so a leaked eval() would silently train the (always-trainable) student
    # BatchNorm in eval mode on later fits. predict() must round-trip the module's training flag.
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            student_bootstrap_layer=0,
            pretrained_teacher=False,
            pretrained_student=False,
            input_range="float01",
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
        )
    )

    model.module.train()
    model.predict(data)
    assert model.module.training is True

    model.module.eval()
    model.predict(data)
    assert model.module.training is False


def test_paste_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            student_bootstrap_layer=0,
            pretrained_teacher=False,
            pretrained_student=False,
            input_range="float01",
            batch_size=2,
            epochs=1,
            show_training_progress=False,
        )
    )

    model.fit(data)
    result = model.predict(data)

    assert result.y_pred.shape == (2,)
    assert result.anomaly_scores.shape == (2,)
    assert result.score_maps.shape == (2, 64, 64)


def test_paste_second_fit_still_trains_student_in_train_mode():
    # The eval-mode restore must let a 2nd fit on the reused instance keep training the
    # (always-trainable) student in TRAIN mode after a predict() forced eval(). A student
    # BatchNorm's num_batches_tracked only advances in train mode, so a leaked eval() would
    # freeze it -- that is the regression to guard (the optimizer steps in eval mode too).
    rng = np.random.default_rng(2)
    data_a = rng.random((2, 64, 64, 3), dtype=np.float32)
    data_b = rng.random((2, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            student_bootstrap_layer=0,
            pretrained_teacher=False,
            pretrained_student=False,
            input_range="float01",
            batch_size=2,
            epochs=1,
            show_training_progress=False,
        )
    )

    model.fit(data_a)
    model.predict(data_a)  # forces eval(); must be restored before the next fit

    bn = next(m for m in model.module.network.student.modules() if isinstance(m, torch.nn.BatchNorm2d))
    tracked_before = bn.num_batches_tracked.clone()
    model.fit(data_b)

    assert bn.num_batches_tracked.item() > tracked_before.item()  # 2nd fit trained student in train mode
    assert model.module.training is True
