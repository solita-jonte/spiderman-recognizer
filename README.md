# Spider-Man Recognizer

A PyTorch implementation using transfer learning (ResNet-18) to determine if Spider-Man is in an image or not.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Download Training Images

Uses Google Custom Search API to download images in a 2:1 ratio (not-Spider-Man : Spider-Man).

Set your API credentials:
```bash
export GOOGLE_API_KEY="your-api-key"
export GOOGLE_CSE_ID="your-cse-id"
```

Then run:
```bash
python download_images.py
```

## 2. Train the Model

```bash
python train.py
```

This fine-tunes a pretrained ResNet-18 for binary classification. The best model is saved to `model/spiderman.pth`.

## 3. Predict

```bash
python predict.py path/to/image.jpg
```

Output:
```
Image: path/to/image.jpg
Spider-Man: True
Confidence: 94.32%
```

## Project Structure

```
├── download_images.py   # Download training images via Google CSE
├── train.py             # Train the classifier
├── predict.py           # Run inference on a single image
├── requirements.txt     # Python dependencies
├── model/
    └── spiderman.pth    # Trained model (generated)
└── dataset/
    ├── train/
    │   ├── spiderman/
    │   └── not-spiderman/
    └── val/
        ├── spiderman/
        └── not-spiderman/
```
