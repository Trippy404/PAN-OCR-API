# 🪪 PAN Card OCR System

**LIVE URL**=https://pan-ocr-api.onrender.com/

Extract structured data from PAN card images using YOLOv8 and Tesseract OCR.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

---

## 📌 What it does
```
Upload PAN card image
        ↓
YOLOv8 detects fields
        ↓
Tesseract OCR reads text
        ↓
Returns structured JSON
```

---

## 📦 Output JSON
```json
{
  "pan_data": {
    "name":          "Kocherla Srikanth",
    "father_name":   "Mukkanti Kocherla",
    "date_of_birth": "04/05/1997",
    "pan_number":    "GQBPK8700C"
  },
  "detection_confidence": {
    "name":        0.833,
    "father_name": 0.791,
    "dob":         0.818,
    "pan_number":  0.818
  }
}
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| YOLOv8 | Detect PAN card fields |
| Tesseract OCR | Read text from fields |
| OpenCV | Image preprocessing |
| FastAPI | REST API server |
| Streamlit | Web UI |
| Python | Programming language |

---

## 📁 Project Structure
```
pan_ocr_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── src/
│   ├── detection/
│   │   └── detector.py
│   ├── ocr/
│   │   └── tesseract_ocr.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── json_builder.py
│   └── pipeline.py
├── app/
│   └── streamlit_app.py
├── outputs/
│   ├── images/
│   └── json/
├── api.py
├── train.py
├── test.py
├── test_api.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Step 1 — Clone repository
```bash
git clone https://github.com/yourusername/pan-ocr-api.git
cd pan-ocr-api
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install libraries
```bash
pip install -r requirements.txt
```

### Step 4 — Install Tesseract OCR
Download from:
```
https://github.com/UB-Mannheim/tesseract/wiki
```
Install to default path:
```
C:\Program Files\Tesseract-OCR\
```

### Step 5 — Add trained model
Copy your trained model to project root:
```
best.pt
```

---

## 🚀 How to run

### Run Streamlit UI
```bash
streamlit run app/streamlit_app.py
```
Open browser at:
```
http://localhost:8501
```

### Run FastAPI server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
Open browser at:
```
http://localhost:8000
http://localhost:8000/docs
```

---

## 🌐 API Usage

### Extract PAN card data

**Endpoint:**
```
POST /extract
```

**Request:**
```python
import requests

response = requests.post(
    "https://pan-ocr-api.onrender.com/extract",
    files={"file": open("pan_card.jpg", "rb")}
)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "filename": "pan_card.jpg",
  "data": {
    "pan_data": {
      "name":          "Kocherla Srikanth",
      "father_name":   "Mukkanti Kocherla",
      "date_of_birth": "04/05/1997",
      "pan_number":    "GQBPK8700C"
    }
  }
}
```

### Check API status
```
GET /
GET /health
```

---

## 🏋️ Training your own model

### Step 1 — Annotate images
Use LabelImg to annotate PAN card images with 4 classes:
```
0: name
1: dob
2: pan_number
3: father_name
```

### Step 2 — Train model
```bash
python train.py
```

### Step 3 — Copy best weights
```bash
copy runs\pan_card_model\weights\best.pt best.pt
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| mAP@50 | 0.991 |
| Precision | 0.989 |
| Recall | 0.989 |
| Training images | 1458 |
| Validation images | 268 |
| Epochs | 30 |

---

## 🔧 Troubleshooting

### Tesseract not found
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### OMP Error on Windows
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### Model not found
```
Make sure best.pt is in project root folder
F:\project_yolo\best.pt
```

---

## 🌍 Real World Use Cases
```
✓ Bank KYC verification
✓ Loan application auto fill
✓ HR employee onboarding
✓ Tax filing automation
✓ Hospital patient verification
✓ E-commerce seller verification
```

---



## 👨‍💻 Author

Built with ❤️ using YOLOv8 and Tesseract OCR

---

## 📄 License

MIT License — free to use for personal and commercial projects
