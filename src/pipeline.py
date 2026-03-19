# pipeline.py
import numpy as np
from PIL import Image

from src.detection.detector import PANDetector
from src.ocr.tesseract_ocr import TesseractOCR
from src.utils.image_utiles import (
    pil_to_cv2,
    cv2_to_pil,
    preprocess_for_ocr,
    deskew,
    draw_detections
)
from src.utils.json_builder import build_output, save_json


class PANPipeline:

    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        print("Loading YOLOv8 model...")
        self.detector = PANDetector(weights_path, conf_threshold)
        print("Loading Tesseract OCR...")
        self.ocr = TesseractOCR()
        print("Pipeline ready!")

    # def _extract_father_name(
    #     self,
    #     cv_image,
    #     name_detection: dict
    # ) -> str:
    #     """
    #     Extract father name from area just below name field
    #     Father name is always directly below name on PAN card
    #     """
    #     x1, y1, x2, y2 = name_detection["bbox"]
    #     h, w = cv_image.shape[:2]

    #     # Calculate field height
    #     field_height = y2 - y1

    #     # Crop area just below name field
    #     new_y1 = y2 + 2
    #     new_y2 = y2 + field_height + 10

    #     # Stay within image boundaries
    #     new_y1 = max(0, new_y1)
    #     new_y2 = min(h, new_y2)

    #     # Crop the father name area
    #     father_crop = cv_image[new_y1:new_y2, x1:x2]

    #     if father_crop.size == 0:
    #         return ""

    #     # Preprocess and read text
    #     processed = preprocess_for_ocr(father_crop, field="name")
    #     text = self.ocr.read_field(processed, field="name")
    #     return text

    def run(
        self,
        image: Image.Image,
        source_filename: str = "",
        return_annotated: bool = True
    ) -> dict:

        # Step 1 — Convert PIL to OpenCV
        print("Step 1: Converting image...")
        cv_image = pil_to_cv2(image)

        # Step 2 — Fix tilted image
        print("Step 2: Fixing tilt...")
        cv_image = deskew(cv_image)

        # Step 3 — Detect fields using YOLOv8
        print("Step 3: Detecting fields...")
        detections = self.detector.detect(cv_image)

        if not detections:
            print("No fields detected!")
            return {
                "output": {
                    "error": "No fields detected. Check image quality."
                },
                "detections": [],
                "annotated": image
            }

        print(f"Found {len(detections)} fields!")

        # Step 4 — Crop and read each field
        print("Step 4: Reading text from fields...")
        ocr_results = {}
        # name_detection = None

        for detection in detections:
            field = detection["label"]

            # Skip photo field — no text to read
            if field == "photo":
                continue

            # Remember name detection for father name extraction
            if field == "name":
                name_detection = detection

            # Tight crop for name, more padding for others
            if field in ["name", "father_name"]:
                pad = 1
            else:
                pad = 8

            crop = self.detector.crop_field(
                cv_image,
                detection,
                padding=pad
            )

            processed = preprocess_for_ocr(crop, field=field)
            text = self.ocr.read_field(processed, field=field)
            ocr_results[field] = text
            print(f"  {field}: {text}")
# 
        # Extract father name from position below name field
        # if name_detection:
        #     print("  Extracting father name...")
        #     father_name = self._extract_father_name(
        #         cv_image,
        #         name_detection
        #     )
        #     ocr_results["father_name"] = father_name
        #     print(f"  father_name: {father_name}")

        # Step 5 — Build JSON output
        print("Step 5: Building JSON...")
        output = build_output(
            ocr_results,
            detections,
            source_filename
        )

        # Step 6 — Draw boxes on image
        annotated = None
        if return_annotated:
            print("Step 6: Drawing boxes...")
            annotated_cv = draw_detections(cv_image, detections)
            annotated = cv2_to_pil(annotated_cv)

        print("Pipeline complete!")

        return {
            "output":     output,
            "detections": detections,
            "annotated":  annotated
        }