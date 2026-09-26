"""Text recognition (OCR) for Bangla + English using EasyOCR.

Isolated behind a small interface so it can be swapped for Tesseract, PaddleOCR,
or a custom-trained recognition model without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import settings
from .schemas import OcrToken, RecognitionResult

logger = logging.getLogger(__name__)


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

        # detail=1 -> [(bbox, text, confidence), ...]; paragraph=False keeps tokens.
        raw = self.reader.readtext(crop, detail=1, paragraph=False)

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
