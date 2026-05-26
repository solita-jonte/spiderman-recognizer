#!/usr/bin/env python

"""
Train a Spider-Man image classifier using YOLOv8 classification.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "spiderman.pt"
NUM_EPOCHS = 10
IMAGE_SIZE = 224


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # Load a pretrained YOLOv8 classification model
    model = YOLO("yolov8n-cls.pt")

    # Train on dataset (expects train/ and val/ subdirs with class folders)
    results = model.train(
        data=str(DATA_DIR),
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(MODEL_DIR),
        name="train",
        exist_ok=True,
    )

    # Copy the best weights to our standard model path
    best_weights = Path("runs") / "classify" / "model" / "train" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, MODEL_PATH)
        print(f"\nModel saved to: {MODEL_PATH}")
    else:
        print("Warning: best.pt not found in training output.")


if __name__ == "__main__":
    main()
