#!/usr/bin/env python

"""
Determine if Spider-Man is in a given image.
Uses the trained model to classify a single image.

Usage:
    python predict.py <image_path>
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path("model", "spiderman.pth")
IMAGE_SIZE = 224


def load_model(model_path):
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["classes"], checkpoint["class_to_idx"]


def predict(image_path, model, class_to_idx):
    """Predict whether an image contains Spider-Man."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        probability = torch.sigmoid(output).item()

    # The spiderman class index determines the positive class
    spiderman_idx = class_to_idx["spiderman"]
    if spiderman_idx == 1:
        is_spiderman = probability > 0.5
        confidence = probability
    else:
        is_spiderman = probability < 0.5
        confidence = 1 - probability

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

    model, classes, class_to_idx = load_model(MODEL_PATH)
    is_spiderman, confidence = predict(image_path, model, class_to_idx)

    print(f"Image: {image_path}")
    print(f"Spider-Man: {is_spiderman}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
