# Bangla + English License Plate Recognition (ALPR)

Two-stage Automatic License Plate Recognition:

1. **Detect** the plate with an Ultralytics **YOLO** model.
2. **Read** the characters with **EasyOCR** — supports **Bangla (`bn`)** and **English (`en`)** together.

Served via **FastAPI** with a small web UI, plus a CLI and a training script.

```
Python ├ PyTorch ├ Ultralytics YOLO ├ OpenCV ├ NumPy ├ Pillow ├ FastAPI ├ EasyOCR
```

---

## Project layout

```
app/
  config.py        # settings (env-configurable, sensible defaults)
  detector.py      # YOLO wrapper -> bounding boxes
  recognizer.py    # EasyOCR wrapper -> Bangla/English text (swappable)
  pipeline.py      # detect -> crop -> recognize (+ annotate)
  schemas.py       # pydantic response models
  main.py          # FastAPI app + endpoints
  static/index.html# upload UI (renders Bangla natively)
scripts/
  train.py         # fine-tune YOLO on your dataset
  predict_cli.py   # run ALPR on a local image
data/
  data.yaml        # dataset config template (YOLO format)
models/            # put trained weights here as plate.pt
requirements.txt
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

> **GPU (optional):** the default `pip install torch` gives a CPU build. For CUDA,
> install torch from the official index first, e.g.
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`,
> then `pip install -r requirements.txt`.

Run the API + UI:

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> and upload an image.

> **First run downloads models:** the base YOLO (`yolov8n.pt`, ~6 MB) and the
> EasyOCR Bangla+English models (~100 MB). This is a one-time download.

### Works before you train
Until you provide trained plate weights, the detector can't locate plates, so the
pipeline falls back to **OCR on the whole image** (flagged `full_frame_fallback`
in the response). This lets you verify the Bangla/English OCR end-to-end
immediately. Once `models/plate.pt` exists, detection + per-plate OCR kicks in
automatically.

---

## API

| Method | Path                 | Description                                  |
|--------|----------------------|----------------------------------------------|
| GET    | `/`                  | Web UI                                        |
| GET    | `/health`            | Status, whether a trained detector is loaded  |
| POST   | `/predict`           | `multipart/form-data` `file=<image>` → JSON   |
| POST   | `/predict/annotated` | Same input → annotated PNG                     |

Example:

```bash
curl -F "file=@car.jpg" http://127.0.0.1:8000/predict
```

```json
{
  "count": 1,
  "plates": [
    {
      "box": [412, 233, 690, 350],
      "detection_confidence": 0.91,
      "text": "ঢাকা মেট্রো গ ১২৩৪",
      "ocr_confidence": 0.78,
      "tokens": [ { "text": "ঢাকা মেট্রো", "confidence": 0.81 } ]
    }
  ]
}
```

---

## CLI

```bash
python scripts/predict_cli.py car.jpg            # prints JSON, saves car_annotated.png
python scripts/predict_cli.py car.jpg -o out.png
```

---

## Training the detector

1. Label plates in YOLO format and arrange them as described in `data/data.yaml`:
   ```
   data/images/train/*.jpg   data/labels/train/*.txt
   data/images/val/*.jpg     data/labels/val/*.txt
   ```
2. Train:
   ```bash
   python scripts/train.py --data data/data.yaml --epochs 100 --model yolov8n.pt
   # add --device 0 for GPU, or --device cpu
   ```
3. Serve the best weights:
   ```bash
   copy runs\train\plate_yolo\weights\best.pt models\plate.pt   # Windows
   ```
   Restart the API — `/health` will show `"detector_trained": true`.

**Datasets:** public Bangladeshi plate sets exist on Roboflow Universe and Kaggle
(search "Bangla license plate"). Export in **YOLOv8** format to match `data.yaml`.

---

## Configuration

All optional — copy `.env.example` to `.env` (or set env vars). Highlights:

| Variable            | Default          | Meaning                                   |
|---------------------|------------------|-------------------------------------------|
| `DETECTOR_WEIGHTS`  | `models/plate.pt`| Trained YOLO weights (auto-used if present)|
| `OCR_LANGUAGES`     | `bn,en`          | EasyOCR languages                          |
| `CONF_THRESHOLD`    | `0.25`           | Detection confidence cutoff                |
| `DEVICE` / `USE_GPU`| auto             | Force `cpu` / GPU index, toggle CUDA       |

---

## Notes & tips

- **Bangla in images:** OpenCV can't render Bangla glyphs, so annotated PNGs draw
  only the box + detection score; the recognized Bangla/English text comes back in
  the JSON and renders correctly in the web UI and CLI output.
- **Accuracy:** general OCR reads Bangla plates reasonably but not perfectly. For
  production, fine-tune the detector on plate data and, if needed, swap
  `recognizer.py` for a recognition model trained on plate crops — the interface
  (`recognize(crop) -> RecognitionResult`) stays the same.
- **Preprocessing:** `crop_padding` pads plate crops before OCR; tune it if
  characters get clipped.
