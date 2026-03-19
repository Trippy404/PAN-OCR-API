# detector.py
# Loads YOLOv8 model and detects fields on PAN card

from pathlib import Path
import numpy as np
from ultralytics import YOLO


# Class names must match your data.yaml exactly
FIELD_NAMES = {
    0: "dob",
    1: "father_name",
    2: "name",
    3: "pan_number"
}

FIELD_PADDING = {
    "name":        2,
    "dob":         4,
    "pan_number":  4,
    "photo":       2,
}


class PANDetector:
    """
    Detects PAN card fields using trained YOLOv8 model

    Usage:
        detector = PANDetector("best.pt")
        detections = detector.detect(image)
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        """
        Load the trained model

        Args:
            weights_path: path to your best.pt file
            conf_threshold: minimum confidence to accept detection
        """

        # Check if weights file exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model not found at: {weights_path}\n"
                f"Please copy best.pt to your project folder"
            )

        # Load YOLOv8 model
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        print(f"Model loaded from: {weights_path}")

    def detect(self, image: np.ndarray) -> list:
        """
        Detect all fields on PAN card image

        Args:
            image: BGR numpy array from OpenCV

        Returns:
            List of detections like:
            [
                {
                    "label": "name",
                    "confidence": 0.99,
                    "bbox": (x1, y1, x2, y2)
                }
            ]
        """

        # Run YOLOv8 detection
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        # Convert results to simple list
        detections = []
        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label":      FIELD_NAMES.get(class_id, f"class_{class_id}"),
                "class_id":   class_id,
                "confidence": round(confidence, 3),
                "bbox":       (int(x1), int(y1), int(x2), int(y2))
            })

        # Sort detections top to bottom
        detections.sort(key=lambda d: d["bbox"][1])

        return detections

    def crop_field(
        self,
        image: np.ndarray,
        detection: dict,
        padding: int = 6
    ) -> np.ndarray:
        """
        Crop detected field from image

        Args:
            image: full PAN card image
            detection: single detection from detect()
            padding: extra pixels around crop

        Returns:
            Cropped field image
        """

        x1, y1, x2, y2 = detection["bbox"]
        h, w = image.shape[:2]

        # Add padding but stay within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return image[y1:y2, x1:x2]
