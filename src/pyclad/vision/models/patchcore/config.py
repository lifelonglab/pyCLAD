from typing import Optional

from pydantic import Field

from pyclad.vision.models.utilities.config import ImageSize, VisionConfig


class PatchCoreConfig(VisionConfig):
    """Configuration for the PatchCore memory-bank detector.

    Defaults follow the reference implementation used by ReplayCAD's benchmark
    (ADer ``model/patchcore.py``): 1024-dimensional patch embeddings, 3x3 patches,
    a single nearest neighbour and a 10% greedy coreset.
    """

    input_size: ImageSize = (256, 256)
    batch_size: int = Field(default=32, gt=0)

    backbone_name: str = "wide_resnet50_2"
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    pretrained_weights: Optional[str] = "IMAGENET1K_V1"

    pretrain_embed_dimension: int = Field(default=1024, gt=0)
    target_embed_dimension: int = Field(default=1024, gt=0)
    patchsize: int = Field(default=3, gt=0)
    patchstride: int = Field(default=1, gt=0)
    coreset_sampling_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    coreset_projection_dimension: int = Field(default=128, gt=0)
    coreset_starting_points: int = Field(default=10, gt=0)
    n_neighbors: int = Field(default=1, gt=0)

    smoothing_sigma: float = Field(default=4.0, ge=0.0)
