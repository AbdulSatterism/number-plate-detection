"""Text recognition (OCR) for Bangla + English using EasyOCR.

Isolated behind a small interface so it can be swapped for Tesseract, PaddleOCR,
or a custom-trained recognition model without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

from .config import settings
from .schemas import OcrToken, RecognitionResult

logger = logging.getLogger(__name__)


def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance plate crop contrast and resolution for superior OCR accuracy."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    # 1. Intelligent Upscaling: EasyOCR struggles if plate is less than ~300px wide
    min_w = 320
    if w < min_w:
        scale = min_w / float(w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 2. Contrast Enhancement (CLAHE on L-channel in LAB space)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. Bilateral smoothing to remove noise while preserving sharp font edges
    denoised = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=50, sigmaSpace=50)
    return denoised


class PlateRecognizer:
    def __init__(self, languages: Optional[List[str]] = None, gpu: Optional[bool] = None):
        # Imported locally: EasyOCR pulls in torch and is heavy to import.
        import easyocr

        langs = languages or settings.ocr_languages
        use_gpu = settings.use_gpu if gpu is None else gpu
        logger.info("Initializing EasyOCR (languages=%s, gpu=%s)", langs, use_gpu)
        # First run downloads the detection + recognition models (~100 MB).
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        self.min_conf = settings.ocr_min_confidence

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        """OCR a cropped plate (BGR image). Returns joined text + per-token detail."""
        if crop is None or crop.size == 0:
            return RecognitionResult()

        processed_crop = _preprocess_plate_crop(crop)

        # Pass 1: Enhanced crop
        raw = self.reader.readtext(
            processed_crop,
            detail=1,
            paragraph=False,
            mag_ratio=1.5,
        )

        # Pass 2: If pass 1 returned nothing, try raw crop with high magnification
        if not raw:
            raw = self.reader.readtext(
                crop,
                detail=1,
                paragraph=False,
                mag_ratio=2.0,
            )

        tokens: List[OcrToken] = []
        for _bbox, text, conf in raw:
            text = (text or "").strip()
            if not text or float(conf) < self.min_conf:
                continue
            tokens.append(OcrToken(text=text, confidence=round(float(conf), 4)))

        full_text = " ".join(t.text for t in tokens)
        avg_conf = (
            round(sum(t.confidence for t in tokens) / len(tokens), 4) if tokens else 0.0
        )
        return RecognitionResult(text=full_text, confidence=avg_conf, tokens=tokens)
