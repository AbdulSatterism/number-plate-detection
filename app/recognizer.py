"""Text recognition (OCR) with Dual-Script Isolation (Pure Bangla + Pure English).

Uses separate language models to prevent cross-language corruption (e.g. English letters
turning into Bengali characters or vice versa), followed by 2D spatial reading-order sorting.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .config import settings
from .schemas import OcrToken, RecognitionResult

logger = logging.getLogger(__name__)


def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance plate contrast and normalize resolution for accurate character extraction."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    # Normalize height to ~140px while maintaining aspect ratio
    target_h = 140
    scale = target_h / float(h)
    new_w = max(int(w * scale), 280)
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement in LAB color space
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return bgr_enhanced


def _sort_reading_order(detections: List[Dict[str, Any]]) -> str:
    """Sort text boxes strictly in natural reading order (Top-to-Bottom, Left-to-Right)."""
    if not detections:
        return ""

    # Sort boxes primarily by vertical center
    detections.sort(key=lambda d: d["y_center"])

    # Cluster into horizontal lines
    rows: List[List[Dict[str, Any]]] = []
    avg_h = sum(d["height"] for d in detections) / len(detections)
    row_tolerance = max(10.0, avg_h * 0.45)

    current_row = [detections[0]]
    for d in detections[1:]:
        row_y_avg = sum(item["y_center"] for item in current_row) / len(current_row)
        if abs(d["y_center"] - row_y_avg) <= row_tolerance:
            current_row.append(d)
        else:
            current_row.sort(key=lambda item: item["x_min"])
            rows.append(current_row)
            current_row = [d]

    if current_row:
        current_row.sort(key=lambda item: item["x_min"])
        rows.append(current_row)

    # Join lines
    line_texts = []
    for row in rows:
        line_str = " ".join(item["text"] for item in row if item["text"].strip())
        if line_str:
            line_texts.append(line_str)

    return " \n ".join(line_texts) if len(line_texts) > 1 else (line_texts[0] if line_texts else "")


class PlateRecognizer:
    def __init__(self, languages: Optional[List[str]] = None, gpu: Optional[bool] = None):
        import easyocr

        use_gpu = settings.use_gpu if gpu is None else gpu
        logger.info("Initializing Script-Isolated OCR Engines (Bangla & English, gpu=%s)", use_gpu)
        
        # Dedicated unpolluted English engine (prevents English letters turning into Bengali)
        self.reader_en = easyocr.Reader(['en'], gpu=use_gpu)
        # Dedicated unpolluted Bangla engine (prevents Bengali characters turning into English)
        self.reader_bn = easyocr.Reader(['bn'], gpu=use_gpu)
        
        self.min_conf = settings.ocr_min_confidence

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        """Run script-isolated OCR and select the most accurate transcription per region."""
        if crop is None or crop.size == 0:
            return RecognitionResult()

        processed = _preprocess_plate_crop(crop)

        # 1. Run pure English OCR pass
        raw_en = self.reader_en.readtext(
            processed,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
        )

        # 2. Run pure Bangla OCR pass
        raw_bn = self.reader_bn.readtext(
            processed,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
        )

        # 3. Disambiguate and select the best candidate per text box
        def parse_items(raw_list, script_name):
            items = []
            for bbox, text, conf in raw_list:
                cleaned = (text or "").strip()
                if not cleaned or float(conf) < self.min_conf:
                    continue
                ys = [p[1] for p in bbox]
                xs = [p[0] for p in bbox]
                items.append({
                    "text": cleaned,
                    "confidence": float(conf),
                    "script": script_name,
                    "bbox": bbox,
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "y_min": min(ys),
                    "y_max": max(ys),
                    "y_center": sum(ys) / len(ys),
                    "height": max(ys) - min(ys),
                })
            return items

        items_en = parse_items(raw_en, "en")
        items_bn = parse_items(raw_bn, "bn")

        # 3. Spatial box matching: for each detected text region, choose the best script candidate
        selected_items: List[Dict[str, Any]] = []

        if not items_en and not items_bn:
            # Fallback on raw crop
            fallback_en = self.reader_en.readtext(crop, detail=1, paragraph=False)
            fallback_bn = self.reader_bn.readtext(crop, detail=1, paragraph=False)
            items_en = parse_items(fallback_en, "en")
            items_bn = parse_items(fallback_bn, "bn")

        # If one engine found nothing, use the other directly
        if not items_bn:
            selected_items = items_en
        elif not items_en:
            selected_items = items_bn
        else:
            # Both engines produced results: for each region, select the most confident, script-accurate candidate
            has_bangla_chars = any(
                any('\u0980' <= ch <= '\u09FF' for ch in i["text"]) for i in items_bn
            )
            
            # If purely English (no genuine Bangla characters found in Bangla pass)
            if not has_bangla_chars:
                selected_items = items_en
            else:
                # Merge per-region: take Bangla candidate if it has genuine Bengali glyphs, else take English candidate
                matched_en_indices = set()
                for b_item in items_bn:
                    is_bangla = any('\u0980' <= ch <= '\u09FF' for ch in b_item["text"])
                    if is_bangla:
                        selected_items.append(b_item)
                    else:
                        # Find overlapping English item
                        best_overlap_en = None
                        best_overlap_idx = -1
                        for idx, e_item in enumerate(items_en):
                            if idx in matched_en_indices:
                                continue
                            # Check vertical and horizontal proximity
                            y_diff = abs(b_item["y_center"] - e_item["y_center"])
                            x_diff = abs(b_item["x_min"] - e_item["x_min"])
                            if y_diff < max(b_item["height"], e_item["height"]) and x_diff < 80:
                                best_overlap_en = e_item
                                best_overlap_idx = idx
                                break
                        if best_overlap_en and best_overlap_en["confidence"] >= b_item["confidence"]:
                            selected_items.append(best_overlap_en)
                            matched_en_indices.add(best_overlap_idx)
                        else:
                            selected_items.append(b_item)

                # Add any non-overlapping English items
                for idx, e_item in enumerate(items_en):
                    if idx not in matched_en_indices:
                        # Check if it overlaps with any already selected item
                        overlaps = any(
                            abs(e_item["y_center"] - s["y_center"]) < max(e_item["height"], s["height"]) * 0.7
                            and abs(e_item["x_min"] - s["x_min"]) < 50
                            for s in selected_items
                        )
                        if not overlaps:
                            selected_items.append(e_item)

        # 4. Natural 2D reading order assembly (Top-to-Bottom, Left-to-Right)
        full_text = _sort_reading_order(selected_items)
        
        tokens: List[OcrToken] = [
            OcrToken(text=item["text"], confidence=round(item["confidence"], 4))
            for item in selected_items
        ]
        avg_conf = (
            round(sum(t.confidence for t in tokens) / len(tokens), 4) if tokens else 0.0
        )

        return RecognitionResult(text=full_text, confidence=avg_conf, tokens=tokens)
