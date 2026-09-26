"""Run the ALPR pipeline on a local image from the command line.

Usage (from repo root):
    python scripts/predict_cli.py path/to/car.jpg
    python scripts/predict_cli.py path/to/car.jpg -o out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import ALPRPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ALPR on a local image.")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("-o", "--output", default=None, help="where to save the annotated image")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    pipe = ALPRPipeline()
    resp = pipe.predict(img)

    # ensure_ascii=False so Bangla text prints readably.
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))

    src = Path(args.image)
    out = args.output or str(src.with_name(src.stem + "_annotated.png"))
    cv2.imwrite(out, pipe.annotate(img, resp))
    print(f"\nAnnotated image saved to: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
