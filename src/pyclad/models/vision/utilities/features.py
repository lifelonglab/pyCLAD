from typing import Sequence

import torch
import torch.nn.functional as F


def align_feature_maps(features: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("Feature extractor returned no feature maps")

    target_size = features[0].shape[-2:]
    aligned = []
    for feature in features:
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        aligned.append(feature)
    return torch.cat(aligned, dim=1)
