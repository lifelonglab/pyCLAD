from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VisionSample:
    category: str
    split: str
    image_path: Path
    image_label: int
    mask_path: Optional[Path] = None
    defect_type: Optional[str] = None
