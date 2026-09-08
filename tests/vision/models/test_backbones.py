import pytest
import torch

from pyclad.vision.models.utilities.backbones import (
    TorchvisionFeatureExtractor,
    create_torchvision_model,
)


def test_named_weights_are_requested_verbatim():
    import torchvision.models as tv_models

    captured = {}

    def fake_wide_resnet50_2(weights=None):
        captured["weights"] = weights
        return torch.nn.Identity()

    original = tv_models.wide_resnet50_2
    tv_models.wide_resnet50_2 = fake_wide_resnet50_2
    try:
        create_torchvision_model("wide_resnet50_2", pretrained=True, weights_name="IMAGENET1K_V1")
    finally:
        tv_models.wide_resnet50_2 = original

    assert captured["weights"] is tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1


def test_unknown_weights_name_lists_available_members():
    with pytest.raises(ValueError, match="IMAGENET1K_V1"):
        create_torchvision_model("wide_resnet50_2", pretrained=True, weights_name="NOPE")


def test_weights_name_is_ignored_when_not_pretrained():
    model = create_torchvision_model("resnet18", pretrained=False, weights_name="NOPE")
    assert isinstance(model, torch.nn.Module)


def test_feature_extractor_reports_channels_and_shapes():
    extractor = TorchvisionFeatureExtractor(
        backbone_name="resnet18",
        return_nodes=["layer2", "layer3"],
        pretrained=False,
        freeze=True,
    )

    assert extractor.infer_out_channels((32, 32)) == (128, 256)

    features = extractor(torch.zeros(1, 3, 32, 32))
    assert [f.shape[1] for f in features] == [128, 256]
