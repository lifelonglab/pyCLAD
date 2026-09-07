"""Tests for the shared torchvision backbone helpers in ``utilities/backbones.py``.

None of these download weights: weight *resolution* only inspects torchvision's enums, and the
models that are actually built are constructed with ``pretrained=False``.
"""

from enum import Enum
from unittest.mock import create_autospec, patch

import pytest
import torch
import torchvision.models as tv_models

from pyclad.vision.models.utilities.backbones import (
    SUPPORTED_BACKBONES,
    TorchvisionFeatureExtractor,
    create_torchvision_model,
    resolve_pretrained_weights,
)


def test_default_weights_pin_imagenet_v1_for_every_supported_backbone():
    # torchvision's ``.DEFAULT`` is a moving pointer and differs per backbone (V2 for resnet50,
    # wide_resnet50_2, mobilenet_v2 and efficientnet_b1; V1 for the other nine). Pinning V1 is the
    # only choice that means the same thing across the whole supported matrix.
    for backbone_name in SUPPORTED_BACKBONES:
        weights = resolve_pretrained_weights(backbone_name)
        assert weights is not None, backbone_name
        assert weights.name == "IMAGENET1K_V1", backbone_name


def test_explicit_weights_variant_is_honoured():
    weights = resolve_pretrained_weights("resnet50", "IMAGENET1K_V2")

    assert weights.name == "IMAGENET1K_V2"


def test_default_keyword_opts_back_into_torchvisions_own_choice():
    weights = resolve_pretrained_weights("resnet50", "DEFAULT")

    assert weights is tv_models.get_model_weights(tv_models.resnet50).DEFAULT


def test_unknown_weights_name_reports_the_variants_available_for_that_backbone():
    with pytest.raises(ValueError, match="IMAGENET1K_V2"):
        resolve_pretrained_weights("resnet50", "IMAGENET1K_V9")


def test_unknown_weights_name_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="NOT_A_VARIANT"):
        create_torchvision_model("resnet18", pretrained=True, weights="NOT_A_VARIANT")


def test_weights_are_ignored_for_randomly_initialised_backbones():
    # STFPM and PaSTe build a pretrained teacher and a random student from one config, so the
    # weights setting must not blow up on the ``pretrained=False`` half.
    model = create_torchvision_model("resnet18", pretrained=False, weights="IMAGENET1K_V2")

    assert isinstance(model, torch.nn.Module)


def test_feature_extractor_returns_requested_nodes_in_order():
    extractor = TorchvisionFeatureExtractor(
        backbone_name="resnet18",
        return_nodes=("layer1", "layer3"),
        pretrained=False,
        freeze=False,
    )

    features = extractor(torch.rand(1, 3, 64, 64))

    assert extractor.return_nodes == ("layer1", "layer3")
    assert [feature.shape[1] for feature in features] == [64, 256]


def test_feature_extractor_rejects_empty_return_nodes():
    with pytest.raises(ValueError, match="at least one feature node"):
        TorchvisionFeatureExtractor(backbone_name="resnet18", return_nodes=(), pretrained=False, freeze=False)


def test_feature_extractor_freezes_parameters_when_asked():
    extractor = TorchvisionFeatureExtractor(
        backbone_name="resnet18",
        return_nodes=("layer1",),
        pretrained=False,
        freeze=True,
    )

    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_resolved_weights_are_handed_to_the_torchvision_builder():
    # Guards the last hop: resolution can be correct while the resolved value never reaches
    # torchvision. autospec preserves the real signature, so the modern ``weights=`` branch is
    # taken; the builder is a mock, so nothing is downloaded.
    with patch.object(tv_models, "resnet18", create_autospec(tv_models.resnet18)) as builder:
        create_torchvision_model("resnet18", pretrained=True)

    builder.assert_called_once_with(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)


def test_backbone_without_the_pinned_variant_falls_back_to_torchvision_default():
    # No supported backbone hits this today; the branch exists so a future torchvision addition
    # that skips IMAGENET1K_V1 stays usable instead of failing on a naming detail.
    class _FakeWeights(Enum):
        IMAGENET1K_V2 = "v2"
        DEFAULT = "v2"

    with patch("pyclad.vision.models.utilities.backbones._weights_enum", return_value=_FakeWeights):
        assert resolve_pretrained_weights("resnet18") is _FakeWeights.DEFAULT
