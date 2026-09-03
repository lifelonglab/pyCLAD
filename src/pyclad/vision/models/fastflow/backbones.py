from __future__ import annotations

from pyclad.vision.models.utilities.backbones import (  # noqa: F401  (re-exported for FastFlow callers)
    TorchvisionFeatureExtractor,
    default_backbone_return_nodes,
    supported_backbone_names,
)


def default_fastflow_return_nodes(backbone_name: str) -> tuple[str, ...]:
    try:
        default_nodes = tuple(default_backbone_return_nodes(backbone_name))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported FastFlow backbone '{backbone_name}'. Supported default presets: "
            f"{', '.join(supported_backbone_names())}. You can still use a custom torchvision backbone by "
            "setting backbone_return_nodes explicitly."
        ) from exc

    # FastFlow is typically applied to a small set of intermediate 2D feature maps.
    if len(default_nodes) >= 4:
        return default_nodes[1:4]
    if len(default_nodes) >= 3:
        return default_nodes[:3]
    if len(default_nodes) >= 1:
        return default_nodes
    raise ValueError(f"Backbone '{backbone_name}' did not expose any default feature nodes")
