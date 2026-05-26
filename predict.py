#!/usr/bin/env python

"""
Determine if Spider-Man is in a given image.
Uses the trained YOLOv8 classification model.

Usage:
    python predict.py <image_path>
"""

import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("model", "spiderman.pt")


def predict(image_path, model):
    """Predict whether an image contains Spider-Man."""
    results = model(image_path, verbose=False)
    result = results[0]

    # Get the top-1 predicted class
    probs = result.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    class_name = result.names[top1_idx]

    is_spiderman = class_name == "spiderman"
    confidence = top1_conf if is_spiderman else 1 - top1_conf

    return is_spiderman, confidence


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))
    is_spiderman, confidence = predict(str(image_path), model)

    print(f"Image: {image_path}")
    print(f"Spider-Man: {is_spiderman}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
