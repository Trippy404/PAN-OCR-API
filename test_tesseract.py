import pytesseract
import cv2
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "F:/project_yolo")

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from src.detection.detector import PANDetector

# Load model
detector = PANDetector("best.pt")

# Load image
img = cv2.imread(r"C:\Users\suman\Downloads\original.jpg")

# Detect fields
detections = detector.detect(img)

print(f"Found {len(detections)} fields\n")

# Loop through each detection
for det in detections:
    field = det["label"]
    conf  = det["confidence"]

    # Crop the field
    crop = detector.crop_field(img, det, padding=10)

    # Save crop so you can see it
    cv2.imwrite(f"outputs/images/crop_{field}.jpg", crop)

    # Try to read text
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale for better reading
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config="--psm 7 --oem 3"
    )

    print(f"Field    : {field}")
    print(f"Confidence: {conf:.0%}")
    print(f"Text found: {text.strip()}")
    print("-" * 40)