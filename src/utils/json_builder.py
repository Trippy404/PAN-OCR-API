# json_builder.py
# Builds the final JSON output from OCR results

import json
from datetime import datetime
from pathlib import Path


def build_output(
    ocr_results: dict,
    detections: list,
    source_filename: str = ""
) -> dict:
    """
    Build final JSON from detected fields and OCR text

    Example output:
    {
        "pan_data": {
            "name": "RAHUL SHARMA",
            "pan_number": "ABCDE1234F"
        }
    }
    """

    # Get confidence score for each field
    confidence_map = {
        d["label"]: d["confidence"]
        for d in detections
    }

    # Build the output dictionary
    output = {
        "metadata": {
            "source_file":   source_filename,
            "processed_at":  datetime.utcnow().isoformat() + "Z",
            "fields_found":  len(ocr_results),
            "model_version": "yolov8n-pan-v1"
        },
        "pan_data": {
            "name":          ocr_results.get("name", ""),
            "father_name":   ocr_results.get("father_name", ""),
            "date_of_birth": ocr_results.get("dob", ""),
            "pan_number":    ocr_results.get("pan_number", ""),
        },
        "detection_confidence": {
            field: confidence_map.get(field, 0.0)
            for field in ["name", "father_name", "dob", "pan_number"]
        }
    }

    return output


def save_json(output: dict, save_path: str) -> None:
    """Save output dictionary as a JSON file"""

    # Create folder if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as pretty printed JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"JSON saved → {save_path}")




