"""Fine-tune a YOLO detector for license plates.

Usage (from repo root):
    python scripts/train.py --data data/data.yaml --epochs 100 --model yolov8n.pt

After training, copy the best weights to models/plate.pt so the app serves them:
    Windows:  copy runs\\train\\plate_yolo\\weights\\best.pt models\\plate.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO for license-plate detection.")
    ap.add_argument("--data", default="data/data.yaml", help="dataset config (Ultralytics YAML)")
    ap.add_argument("--model", default="yolov8n.pt", help="base model to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help='GPU index like "0", or "cpu"')
    ap.add_argument("--name", default="plate_yolo", help="run name")
    ap.add_argument("--project", default="runs/train", help="output directory")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
    )

    best = f"{args.project}/{args.name}/weights/best.pt"
    print("\nTraining complete.")
    print(f"Best weights: {best}")
    print("Serve them by copying to models/plate.pt, e.g. (Windows):")
    print(f"  copy {best.replace('/', chr(92))} models\\plate.pt")


if __name__ == "__main__":
    main()
