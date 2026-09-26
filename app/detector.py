"""License-plate detection using Ultralytics YOLO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from .config import settings
from .schemas import Detection

logger = logging.getLogger(__name__)


class PlateDetector:
    """Thin wrapper around a YOLO model that returns plate bounding boxes.

    If the configured plate weights are missing, we still load a base YOLO model
    (so imports/UX don't break), but `is_trained` stays False and `detect()`
    returns nothing — the pipeline then falls back to full-frame OCR so the app
    is demoable end-to-end before you train the detector.
    """

    def __init__(self, weights: Optional[str] = None, device: Optional[str] = None):
        weights_path = Path(weights or settings.detector_weights)
        self.is_trained = weights_path.exists()

        if self.is_trained:
            logger.info("Loading trained plate detector: %s", weights_path)
            self.model = YOLO(str(weights_path))
        else:
            logger.warning(
                "Plate weights not found at '%s'. Falling back to base '%s'. "
                "The detector will NOT locate plates until you train it "
                "(see scripts/train.py). Full-frame OCR fallback is active meanwhile.",
                weights_path,
                settings.base_weights,
            )
            self.model = YOLO(settings.base_weights)

        self.device = device if device is not None else settings.device
        self.conf = settings.conf_threshold
        self.iou = settings.iou_threshold

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a BGR image (as read by OpenCV)."""
        if not self.is_trained:
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                cls = int(b.cls[0])
                detections.append(
                    Detection(
                        box=[x1, y1, x2, y2],
                        confidence=round(float(b.conf[0]), 4),
                        class_id=cls,
                        class_name=self.model.names.get(cls, str(cls)),
                    )
                )
        return detections
