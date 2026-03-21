# streamlit_app.py
# Full web UI for PAN Card OCR System

import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import PANPipeline



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project to Python path
sys.path.insert(0, "F:/project_yolo")

from src.pipeline import PANPipeline

# ── Page settings ─────────────────────────────────────────────
st.set_page_config(
    page_title="PAN Card OCR",
    page_icon="🪪",
    layout="wide"
)

# ── Custom styling ────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 {
    color: #63b3ed;
    font-size: 2.5rem;
    margin: 0;
}
.main-header p {
    color: #a0aec0;
    margin: 0.5rem 0 0;
}
.field-box {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.field-label {
    color: #718096;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
.field-value {
    color: #e2e8f0;
    font-size: 1.2rem;
    font-weight: bold;
    font-family: monospace;
}
.pan-value {
    color: #f6ad55;
    font-size: 1.4rem;
    letter-spacing: 0.2em;
}
.dob-value {
    color: #68d391;
}
</style>
""", unsafe_allow_html=True)


# ── Load pipeline once ────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return PANPipeline("best.pt")


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🪪 PAN Card OCR System</h1>
    <p>Upload a PAN card image and get structured JSON output instantly</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher = stricter detection"
    )

    st.divider()

    st.markdown("### 📋 How it works")
    st.markdown("""
1. 📤 Upload PAN card image
2. 🎯 YOLOv8 detects fields
3. 📝 Tesseract reads text
4. 📦 Get JSON output
    """)

    st.divider()

    st.markdown("### 🏷️ Detected Fields")
    st.markdown("""
🟢 Name
🟡 Date of Birth
🔴 PAN Number
🟣 Photo
    """)


# ── Main tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Image", "📂 Batch Process"])


# ── Tab 1: Single image ───────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    # Left column — upload
    with col1:
        st.markdown("#### 📤 Upload PAN Card")
        uploaded = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", width='stretch')

            run = st.button(
                "🚀 Extract Data",
                type="primary",
                width='stretch'
            )

    # Right column — results
    with col2:
        st.markdown("#### 📊 Extracted Data")

        if not uploaded:
            st.info("👈 Upload a PAN card image to begin")

        elif uploaded and run:
            with st.spinner("Processing..."):
                try:
                    # Load pipeline
                    pipeline = load_pipeline()

                    # Run pipeline
                    result = pipeline.run(
                        image,
                        source_filename=uploaded.name
                    )

                    output     = result["output"]
                    detections = result["detections"]
                    annotated  = result["annotated"]

                    # Show error if any
                    if "error" in output:
                        st.error(f"❌ {output['error']}")

                    else:
                        pan   = output["pan_data"]
                        confs = output["detection_confidence"]

                        # Success message
                        st.success(
                            f"✓ {len(detections)} fields detected!"
                        )

                        # Field boxes
                        st.markdown(f"""
                        <div class="field-box">
                            <div class="field-label">Name</div>
                            <div class="field-value">{pan['name'] or '—'}</div>
                        </div>
                        <div class="field-box">
                            <div class="field-label">Father / Spouse Name</div>
                            <div class="field-value">{pan['father_name'] or '—'}</div>
                        </div>
                        <div class="field-box">
                            <div class="field-label">Date of Birth</div>
                            <div class="field-value dob-value">{pan['date_of_birth'] or '—'}</div>
                        </div>
                        <div class="field-box">
                            <div class="field-label">PAN Number</div>
                            <div class="field-value pan-value">{pan['pan_number'] or '—'}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence scores
                        st.markdown("#### 📈 Confidence Scores")
                        for field, score in confs.items():
                            st.progress(
                                score,
                                text=f"{field}: {score:.0%}"
                            )

                        # Annotated image
                        if annotated:
                            st.markdown("#### 🎯 Detected Regions")
                            st.image(
                                annotated,
                                width='stretch'
                            )

                        # JSON output
                        st.markdown("#### 📦 JSON Output")
                        st.json(output)

                        # Download button
                        json_str = json.dumps(output, indent=2)
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json_str,
                            file_name=f"{Path(uploaded.name).stem}_result.json",
                            mime="application/json",
                            width='stretch'
                        )

                        # Save outputs
                        Path("outputs/images").mkdir(parents=True, exist_ok=True)
                        Path("outputs/json").mkdir(parents=True, exist_ok=True)

                        if annotated:
                            annotated.save(f"outputs/images/{Path(uploaded.name).stem}_result.jpg")

                        with open(f"outputs/json/{Path(uploaded.name).stem}_result.json", "w") as f:
                            json.dump(output, f, indent=2)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)


# ── Tab 2: Batch process ──────────────────────────────────────
with tab2:
    st.markdown("#### 📂 Upload Multiple PAN Cards")

    batch_files = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files:
        st.info(f"{len(batch_files)} images uploaded")

        if st.button("🚀 Process All", type="primary"):
            pipeline = load_pipeline()
            results  = []
            progress = st.progress(0)

            for i, f in enumerate(batch_files):
                img = Image.open(f).convert("RGB")
                res = pipeline.run(
                    img,
                    source_filename=f.name,
                    return_annotated=False
                )
                results.append(res["output"])
                progress.progress((i + 1) / len(batch_files))

            st.success(f"✓ {len(results)} images processed!")

            # Show results table
            import pandas as pd
            rows = []
            for r in results:
                pd_data = r.get("pan_data", {})
                rows.append({
                    "File":       r.get("metadata", {}).get("source_file", ""),
                    "Name":       pd_data.get("name", ""),
                    "DOB":        pd_data.get("date_of_birth", ""),
                    "PAN Number": pd_data.get("pan_number", ""),
                })

            st.dataframe(
                pd.DataFrame(rows),
                width='stretch'
            )

            # Download all results
            all_json = json.dumps(results, indent=2)
            st.download_button(
                "⬇️ Download All Results",
                data=all_json,
                file_name="batch_results.json",
                mime="application/json",
                width='stretch'
            )