import torch

FOCAL_GAMMA = 2.0
FOCAL_SMOOTH = 1e-5
DICE_SMOOTH = 1.0


def focal_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    classes = probabilities.shape[1]
    flattened = probabilities.reshape(probabilities.shape[0], classes, -1).permute(0, 2, 1).reshape(-1, classes)
    indices = targets.reshape(-1, 1).long()

    one_hot = torch.zeros_like(flattened).scatter_(1, indices, 1.0)
    one_hot = one_hot.clamp(FOCAL_SMOOTH / (classes - 1), 1.0 - FOCAL_SMOOTH)

    confidence = (one_hot * flattened).sum(dim=1) + FOCAL_SMOOTH
    return (-((1.0 - confidence) ** FOCAL_GAMMA) * confidence.log()).mean()


def binary_dice_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    samples = targets.shape[0]
    flat_predictions = predictions.reshape(samples, -1)
    flat_targets = targets.reshape(samples, -1)

    intersection = (flat_predictions * flat_targets).sum(dim=1)
    overlap = (2.0 * intersection + DICE_SMOOTH) / (flat_predictions.sum(dim=1) + flat_targets.sum(dim=1) + DICE_SMOOTH)
    return 1.0 - overlap.mean()
