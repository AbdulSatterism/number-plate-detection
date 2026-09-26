"""FastAPI service for Bangla + English license-plate recognition."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from .config import settings
from .pipeline import ALPRPipeline
from .schemas import PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("alpr")

STATE: Dict[str, ALPRPipeline] = {}
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ALPR pipeline (first run downloads models, please wait)...")
    STATE["pipeline"] = ALPRPipeline()
    logger.info("Pipeline ready.")
    yield
    STATE.clear()


app = FastAPI(title="Bangla + English ALPR", version="0.1.0", lifespan=lifespan)


def _read_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


async def _load_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (>{settings.max_upload_mb} MB)."
        )
    return data


@app.get("/health")
def health():
    pipe = STATE.get("pipeline")
    return {
        "status": "ok" if pipe else "loading",
        "detector_trained": bool(pipe and pipe.detector.is_trained),
        "ocr_languages": settings.ocr_languages,
        "gpu": settings.use_gpu,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    data = await _load_upload(file)
    img = _read_image(data)
    pipe = STATE["pipeline"]
    return await run_in_threadpool(pipe.predict, img)


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    data = await _load_upload(file)
    img = _read_image(data)
    pipe = STATE["pipeline"]
    resp = await run_in_threadpool(pipe.predict, img)
    annotated = pipe.annotate(img, resp)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
