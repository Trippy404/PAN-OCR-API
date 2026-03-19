
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import sys
from pathlib import Path

sys.path.insert(0, "F:/project_yolo")

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.pipeline import PANPipeline

# Create FastAPI app
app = FastAPI(
    title="PAN Card OCR API",
    description="Extract PAN card data using YOLOv8 and Tesseract",
    version="1.0.0"
)

# Allow requests from any website or app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline once when server starts
print("Loading pipeline...")
pipeline = PANPipeline("best.pt")
print("Pipeline ready!")


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def home():
    """Check if API is running"""
    return {
        "status":  "running",
        "message": "PAN Card OCR API is ready",
        "routes": {
            "extract": "POST /extract — send image get JSON",
            "health":  "GET /health — check server status",
            "docs":    "GET /docs — API documentation"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy"
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Extract PAN card data from image

    How to use:
    Send POST request with image file
    Get back JSON with name, dob, pan_number
    """
    try:
        # Check file type
        if not file.filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return {
                "success": False,
                "error": "Only JPG PNG WEBP images allowed"
            }

        # Read image
        contents = await file.read()
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Run pipeline
        result = pipeline.run(
            image,
            source_filename=file.filename,
            return_annotated=False
        )

        # Return result
        return {
            "success": True,
            "filename": file.filename,
            "data": result["output"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }