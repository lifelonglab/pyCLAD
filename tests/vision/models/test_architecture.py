import numpy as np
import pytest
import torch

from pyclad.vision.models.paste.architecture import PaSTeArchitecture


def _make_feature(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random(shape, dtype=np.float32))


def test_feature_loss_normalizes_first_layer_so_train_and_inference_objectives_match():
    """Scaling a normalized layer's teacher features must not change the loss."""
    teacher_features = [_make_feature((2, 8, 4, 4), seed=0), _make_feature((2, 16, 2, 2), seed=1)]
    student_features = [_make_feature((2, 8, 4, 4), seed=2), _make_feature((2, 16, 2, 2), seed=3)]

    base_loss = PaSTeArchitecture.feature_loss(teacher_features, student_features)
    scaled_loss = PaSTeArchitecture.feature_loss(
        teacher_features=[teacher_features[0] * 3.7, teacher_features[1]],
        student_features=student_features,
    )

    assert torch.allclose(base_loss, scaled_loss, atol=1e-5)


def test_feature_loss_is_zero_when_teacher_and_student_have_same_direction():
    base_0 = _make_feature((2, 8, 4, 4), seed=10)
    base_1 = _make_feature((2, 16, 2, 2), seed=11)

    loss = PaSTeArchitecture.feature_loss(
        teacher_features=[base_0 * 1.0, base_1 * 1.0],
        student_features=[base_0 * 5.0, base_1 * 0.2],
    )
    assert loss.item() < 1e-5


def test_feature_loss_rejects_empty_feature_lists():
    with pytest.raises(ValueError, match="non-empty"):
        PaSTeArchitecture.feature_loss(teacher_features=[], student_features=[])
