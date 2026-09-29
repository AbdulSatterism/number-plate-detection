"""End-to-end ALPR pipeline: detect plates -> crop -> recognize text."""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from .config import settings
from .detector import PlateDetector
from .recognizer import PlateRecognizer
from .schemas import PlateResult, PredictResponse

logger = logging.getLogger(__name__)


class ALPRPipeline:
    def __init__(
        self,
        detector: Optional[PlateDetector] = None,
        recognizer: Optional[PlateRecognizer] = None,
    ):
        self.detector = detector or PlateDetector()
        self.recognizer = recognizer or PlateRecognizer()
        self.full_frame_fallback = settings.full_frame_fallback

    def _crop(self, image: np.ndarray, box: List[int], pad: float = 0.0):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        if pad:
            bw, bh = x2 - x1, y2 - y1
            x1, x2 = x1 - bw * pad, x2 + bw * pad
            y1, y2 = y1 - bh * pad, y2 + bh * pad
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def predict(self, image: np.ndarray) -> PredictResponse:
        # Step 1: Detect with configured threshold
        detections = self.detector.detect(image)

        # Step 2: If no plates found with primary threshold, try sensitive secondary pass
        if not detections and self.detector.is_trained:
            sensitive_conf = max(0.08, settings.conf_threshold * 0.5)
            logger.info("Retrying plate detection with sensitive threshold (conf=%.2f)...", sensitive_conf)
            detections = self.detector.detect(image, conf=sensitive_conf)

        plates: List[PlateResult] = []

        if detections:
            for det in detections:
                crop = self._crop(image, det.box, pad=settings.crop_padding)
                rec = self.recognizer.recognize(crop) if crop is not None else None
                plates.append(
                    PlateResult(
                        box=det.box,
                        detection_confidence=det.confidence,
                        text=rec.text if rec else "",
                        ocr_confidence=rec.confidence if rec else 0.0,
                        tokens=rec.tokens if rec else [],
                    )
                )
        elif self.full_frame_fallback:
            logger.info("No plate detected: running full-frame OCR fallback.")
            rec = self.recognizer.recognize(image)
            h, w = image.shape[:2]
            if rec.text:
                plates.append(
                    PlateResult(
                        box=[0, 0, w, h],
                        detection_confidence=0.0,
                        text=rec.text,
                        ocr_confidence=rec.confidence,
                        tokens=rec.tokens,
                        note="full_frame_fallback",
                    )
                )

        return PredictResponse(count=len(plates), plates=plates)

    def annotate(self, image: np.ndarray, response: PredictResponse) -> np.ndarray:
        """Draw boxes + a detection-confidence label.

        Note: OpenCV cannot render Bangla glyphs, so recognized text is returned
        in the JSON / shown in the web UI (which renders Bangla natively) rather
        than burned into the image.
        """
        out = image.copy()
        for p in response.plates:
            x1, y1, x2, y2 = p.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"plate {p.detection_confidence:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        return out
