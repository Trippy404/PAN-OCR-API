# image_utils.py
# Handles all image processing tasks

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load image from file path"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def preprocess_for_ocr(crop: np.ndarray, field: str = "text") -> np.ndarray:
    """Clean up image for better OCR reading"""

    h, w = crop.shape[:2]

    # Scale to minimum 400px height
    if h < 400:
        scale = max(5, 400 // max(h, 1))
        crop = cv2.resize(
            crop,
            (w * scale, h * scale),
            interpolation=cv2.INTER_LANCZOS4
        )

    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Fix gray background — make it white
    # This is the key fix for your images
    gray = cv2.normalize(
        gray, None,
        alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Make text pure black on pure white
    _, processed = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return processed

def deskew(image: np.ndarray) -> np.ndarray:
    """Fix tilted PAN card images"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw colored boxes around detected fields"""
    COLOR_MAP = {
        "name":        (52, 168, 83),
        "father_name": (66, 133, 244),
        "dob":         (251, 188, 4),
        "pan_number":  (234, 67, 53),
        "photo":       (155, 50, 200),
    }
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = COLOR_MAP.get(det["label"], (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['label']} {det['confidence']:.0%}"
        cv2.rectangle(
            vis,
            (x1, y1 - 20),
            (x1 + len(label_text) * 9, y1),
            color, -1
        )
        cv2.putText(
            vis, label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
    return vis
