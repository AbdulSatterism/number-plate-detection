"""Runtime configuration.

Values are read from environment variables (or a .env you export) with safe
defaults, so the app runs with zero setup. See .env.example for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class Settings:
    # --- Detection ---
    detector_weights: str = field(
        default_factory=lambda: os.getenv("DETECTOR_WEIGHTS", "models/plate.pt")
    )
    base_weights: str = field(
        default_factory=lambda: os.getenv("BASE_WEIGHTS", "yolov8n.pt")
    )
    conf_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONF_THRESHOLD", "0.25"))
    )
    iou_threshold: float = field(
        default_factory=lambda: float(os.getenv("IOU_THRESHOLD", "0.45"))
    )
    crop_padding: float = field(
        default_factory=lambda: float(os.getenv("CROP_PADDING", "0.02"))
    )

    # --- OCR ---
    ocr_languages: List[str] = field(
        default_factory=lambda: [
            s.strip() for s in os.getenv("OCR_LANGUAGES", "bn,en").split(",") if s.strip()
        ]
    )
    ocr_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("OCR_MIN_CONFIDENCE", "0.1"))
    )

    # --- Runtime ---
    device: Optional[str] = field(default_factory=lambda: os.getenv("DEVICE") or None)
    use_gpu: bool = field(default_factory=lambda: _get_bool("USE_GPU", _detect_gpu()))
    full_frame_fallback: bool = field(
        default_factory=lambda: _get_bool("FULL_FRAME_FALLBACK", True)
    )
    max_upload_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "15"))
    )


settings = Settings()
