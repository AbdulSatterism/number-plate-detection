"""Pydantic models describing API requests/responses."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    confidence: float
    class_id: int
    class_name: str


class OcrToken(BaseModel):
    text: str
    confidence: float


class RecognitionResult(BaseModel):
    text: str = Field("", description="Joined recognized text (Bangla and/or English)")
    confidence: float = 0.0
    tokens: List[OcrToken] = Field(default_factory=list)


class PlateResult(BaseModel):
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    detection_confidence: float
    text: str = ""
    ocr_confidence: float = 0.0
    tokens: List[OcrToken] = Field(default_factory=list)
    note: Optional[str] = None


class PredictResponse(BaseModel):
    count: int
    plates: List[PlateResult] = Field(default_factory=list)
