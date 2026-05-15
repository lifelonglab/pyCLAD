import numpy as np
import pytest

from pyclad.models.vision.paste.config import PaSTeConfig
from pyclad.models.vision.paste.paste import PaSTe


@pytest.mark.parametrize(
    ("backbone_name", "student_bootstrap_layer"),
    [
        ("resnet18", 0),
        ("mobilenet_v2", 1),
        ("efficientnet_b0", 1),
        ("resnet18", None),
    ],
)
def test_paste_predict_shapes(backbone_name: str, student_bootstrap_layer: int | None):
    data = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name=backbone_name,
            student_bootstrap_layer=student_bootstrap_layer,
            pretrained_teacher=False,
            pretrained_student=False,
            batch_size=1,
            epochs=0,
            threshold=0.5,
            show_training_progress=False,
        )
    )

    y_pred, scores = model.predict(data)
    score_maps = model.score_maps(data)

    assert y_pred.shape == (1,)
    assert scores.shape == (1,)
    assert score_maps.shape == (1, 64, 64)


def test_paste_rejects_invalid_bootstrap_layer():
    with pytest.raises(ValueError, match="student_bootstrap_layer"):
        PaSTeConfig(backbone_name="resnet18", ad_layers=(1, 2, 3), student_bootstrap_layer=1)


def test_paste_fit_smoke_run():
    data = np.random.default_rng(1).random((2, 64, 64, 3), dtype=np.float32)
    model = PaSTe(
        PaSTeConfig(
            input_size=(64, 64),
            backbone_name="resnet18",
            student_bootstrap_layer=0,
            pretrained_teacher=False,
            pretrained_student=False,
            batch_size=2,
            epochs=1,
            show_training_progress=False,
        )
    )

    model.fit(data)
    y_pred, scores = model.predict(data)

    assert y_pred.shape == (2,)
    assert scores.shape == (2,)
