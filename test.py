# test.py
# Test your complete pipeline on a single PAN card image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
from pathlib import Path

# Add project to Python path
sys.path.insert(0, "F:/project_yolo")

from PIL import Image
from src.pipeline import PANPipeline


# ── CHANGE THESE TWO LINES ────────────────────────────────────
WEIGHTS = "best.pt"              # path to your trained model
IMAGE   = "test_pan.jpg"         # path to your PAN card image
# ─────────────────────────────────────────────────────────────


def main():
    # Check image exists
    if not Path(IMAGE).exists():
        print(f"ERROR: Image not found: {IMAGE}")
        print("Please put a PAN card image in F:/project_yolo/")
        print("and rename it to test_pan.jpg")
        return

    print("\n" + "="*50)
    print("   PAN CARD OCR SYSTEM")
    print("="*50)

    # Load pipeline
    pipeline = PANPipeline(WEIGHTS)

    # Open image
    print(f"\nOpening image: {IMAGE}")
    image = Image.open(IMAGE).convert("RGB")

    # Run pipeline
    result = pipeline.run(
        image,
        source_filename=IMAGE
    )

    # Check for errors
    if "error" in result["output"]:
        print(f"\nERROR: {result['output']['error']}")
        return

    # Print results
    pan_data = result["output"]["pan_data"]
    confs    = result["output"]["detection_confidence"]

    print("\n" + "="*50)
    print("   EXTRACTED DATA")
    print("="*50)
    print(f"  Name         : {pan_data['name']}")
    print(f"  Father Name  : {pan_data['father_name']}")
    print(f"  Date of Birth: {pan_data['date_of_birth']}")
    print(f"  PAN Number   : {pan_data['pan_number']}")

    print("\n" + "="*50)
    print("   CONFIDENCE SCORES")
    print("="*50)
    for field, score in confs.items():
        bar = "█" * int(score * 10)
        print(f"  {field:<16} [{bar}] {score:.0%}")

    # Print full JSON
    print("\n" + "="*50)
    print("   FULL JSON OUTPUT")
    print("="*50)
    print(json.dumps(result["output"], indent=2))

    # Save annotated image
    if result["annotated"]:
        output_path = "outputs/images/result.jpg"
        Path("outputs/images").mkdir(parents=True, exist_ok=True)
        result["annotated"].save(output_path)
        print(f"\nAnnotated image saved → {output_path}")

    # Save JSON file
    json_path = "outputs/json/result.json"
    Path("outputs/json").mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result["output"], f, indent=2)
    print(f"JSON saved → {json_path}")

    print("\n" + "="*50)
    print("   DONE!")
    print("="*50)


if __name__ == "__main__":
    main()