"""UCAD: task-agnostic continual visual anomaly detection (Liu et al., AAAI 2024)."""

from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad import (
    UCAD,
    UCADTaskMemory,
    structure_contrastive_loss,
)

__all__ = ["UCAD", "UCADConfig", "UCADTaskMemory", "structure_contrastive_loss"]
