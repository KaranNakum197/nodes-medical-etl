# Sprint 1 MVP - Medical ETL Pipeline Setup & Usage Guide

## 📋 Overview

This document provides complete instructions for setting up, running, and testing the Sprint 1 Medical ETL Pipeline. The pipeline consists of four core components:

1. **PDF Processor** – Converts multi-page PDFs to 300 DPI JPEGs
2. **VLM Extractor** – Qwen2-VL inference for medical data extraction
3. **FastAPI Server** – REST API for image upload and inference
4. **Agent Orchestration** – CrewAI agents for extraction and validation

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** 
- **ROCm 5.7+** (for AMD MI300X GPU support)
- **pip** (Python package manager)
- **Git**

### Installation

#### 1. Backend Setup (VLM + API Server)

```bash
cd backend
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7
```

**Note on ROCm:** The `--extra-index-url` flag points pip to AMD's PyTorch distribution. Adjust the ROCm version if needed (e.g., `rocm5.6` for older versions).

#### 2. Agent Orchestration Setup

```bash
cd agents
pip install -r requirements.txt
```

#### 3. Environment Configuration

Create a `.env` file in the **root project directory**:

```bash
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database (for future use)
DATABASE_URL=postgresql://user:password@localhost/medical_etl

# Orchestration LLM (optional, for advanced CrewAI features)
OPENAI_API_KEY=your_key_here
ORCHESTRATOR_MODEL=gpt-4o-mini
```

---

## 🏃 Running the Pipeline

### Step 1: Start the FastAPI Server

```bash
cd backend
python main.py --host 0.0.0.0 --port 8000
```

Expected output:
```
2024-05-05 10:00:00 - root - INFO - Starting Medical ETL API on 0.0.0.0:8000
2024-05-05 10:00:15 - root - INFO - ✓ VLM model loaded and ready
```

The API will be available at: `http://localhost:8000`

**Interactive Docs:** Visit `http://localhost:8000/docs` for Swagger UI

### Step 2: Test the /extract Endpoint

**Option A: Using curl**

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@medical_report.jpg"
```

**Option B: Using Python**

```python
import requests

with open("medical_report.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/extract",
        files={"file": f}
    )

print(response.json())
```

**Response:**
```json
{
  "success": true,
  "data": {
    "patient_details": {
      "name": "John Doe",
      "patient_id": "PT-2024-001",
      "age": "45 Years",
      "gender": "Male"
    },
    "lab_details": {
      "lab_name": "PathLab Medical Center",
      "lab_code": "PLC-001"
    },
    "sample_details": {...},
    "report_results": [...]
  },
  "metadata": {
    "filename": "medical_report.jpg",
    "size_bytes": 245632,
    "processing_time_ms": 3240
  }
}
```

### Step 3: Run the Agent Orchestration Pipeline

**Option A: Process a single image**

```bash
cd agents
python crew.py /path/to/medical_report.jpg
```

**Option B: Process a PDF (multi-page)**

```bash
cd agents
python crew.py /path/to/medical_report.pdf
```

The pipeline will:
1. Convert PDF pages to JPEGs (if PDF)
2. Call mock_api_extract() for each page (Spring 1 MVP)
3. Validate data against Pydantic schema
4. Output formatted JSON

---

## 🧪 Testing the Components

### Test 1: PDF Processor

```python
from agents.tools.pdf_processor import PDFProcessor

# Process a PDF
with PDFProcessor(dpi=300) as processor:
    image_paths, temp_dir = processor.process_pdf("report.pdf")
    print(f"Extracted {len(image_paths)} pages")
    
    # Validate generated images
    if processor.validate_images(image_paths):
        print("✓ All images valid")
```

### Test 2: VLM Extractor

```python
from backend.vlm_engine.extractor import QwenVLMExtractor
import json

# Initialize extractor
extractor = QwenVLMExtractor()

# Extract JSON from image
result = extractor.extract_json("medical_report.jpg")
print(json.dumps(result, indent=2))
```

### Test 3: Pydantic Schema Validation

```python
from agents.tasks import ExtractedReport, validate_extracted_data
import json

# Raw JSON from VLM
raw_json = '{"patient_details": {...}, "lab_details": {...}, ...}'

# Validate and clean
validated = validate_extracted_data(raw_json)

if "error" not in validated:
    # Convert to Pydantic model
    report = ExtractedReport(**validated)
    print(f"✓ Validation passed for patient: {report.patient_details.name}")
else:
    print(f"✗ Validation failed: {validated['error']}")
```

### Test 4: Full Pipeline Integration

```python
from agents.crew import MedicalETLPipeline
import json

# Initialize pipeline
pipeline = MedicalETLPipeline()

# Process an image
result = pipeline.process_image("medical_report.jpg")

if result["success"]:
    print(json.dumps(result["data"], indent=2, default=str))
else:
    print(f"Error: {result['error']}")
```

---

## 📊 Data Schema Reference

The pipeline enforces strict Pydantic validation. All extracted data must match:

```python
ExtractedReport(
    patient_details: PatientDetailsModel,
    lab_details: LabDetailsModel,
    sample_details: SampleDetailsModel,
    report_results: List[ReportResultModel],
    result_details: Optional[str] = None,
    global_remarks: Optional[str] = None,
    global_notes: Optional[str] = None,
    extra_details: Optional[Dict] = None,
)
```

### Key Validation Rules

| Field | Type | Validation |
|-------|------|-----------|
| `patient_details.name` | `str` | Required, defaults to "Unknown" |
| `lab_details.lab_name` | `str` | Required |
| `report_results[].value` | `float` | Coerced from "95 mg/dL" → 95.0 |
| `sample_details.collected_at` | `datetime` | ISO8601 UTC |
| `phone` | `str` | Normalized to string |
| Null values | Excluded | Omitted from output |

---

## 🔍 Debugging

### Enable Verbose Logging

```bash
# Backend
LOG_LEVEL=DEBUG python backend/main.py

# Agents
LOG_LEVEL=DEBUG python agents/crew.py image.jpg
```

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "vlm_loaded": true
}
```

### Inspect VLM Model Info

```python
from backend.vlm_engine.extractor import QwenVLMExtractor

extractor = QwenVLMExtractor()
print(f"Device: {extractor.device}")
print(f"Dtype: {extractor.torch_dtype}")
print(f"Model: {extractor.model_id}")
```

---

## 📝 File Structure

```
nodes-medical-etl/
├── agents/
│   ├── crew.py                 # Agent definitions & orchestration
│   ├── tasks.py                # Task definitions & Pydantic schemas
│   ├── tools/
│   │   ├── pdf_processor.py    # PDF to image converter
│   │   └── __init__.py
│   └── requirements.txt
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── vlm_engine/
│   │   ├── extractor.py        # VLM inference engine
│   │   └── __init__.py
│   ├── database/
│   │   ├── models.py
│   │   ├── connection.py
│   │   └── __init__.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
└── README.md
```

---

## ⚙️ Configuration & Customization

### Adjust PDF Resolution

```python
# Higher DPI for better quality (slower)
processor = PDFProcessor(dpi=600)
```

### Custom VLM Prompt

```python
from backend.vlm_engine.extractor import QwenVLMExtractor

extractor = QwenVLMExtractor()

custom_prompt = """
Your custom extraction prompt here...
"""

result = extractor.extract_json(
    "image.jpg",
    custom_prompt=custom_prompt,
    max_tokens=4096  # Increase for longer outputs
)
```

### Change API Port

```bash
python backend/main.py --host 127.0.0.1 --port 9000
```

---

## 🐛 Common Issues

### Issue 1: "CUDA not available" or ROCm not detected

**Solution:** Ensure ROCm is installed and PyTorch is built for your GPU:

```bash
# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Verify ROCm installation
rocm-smi
```

### Issue 2: VLM Model Download Fails

**Solution:** Set HuggingFace cache directory:

```bash
export HF_HOME=/path/to/cache
python backend/main.py
```

### Issue 3: Validation Errors with VLM Output

**Solution:** Check the raw VLM output:

```python
extractor = QwenVLMExtractor()
raw = extractor.extract_from_image("image.jpg")  # Raw text output
print(raw)
```

The `validate_extracted_data()` function includes recovery logic for common issues.

---

## 🚢 Production Deployment (Future Sprints)

### Docker Containerization

```dockerfile
FROM rocm/pytorch:latest
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7
CMD ["python", "backend/main.py", "--host", "0.0.0.0"]
```

### Database Integration

```python
from backend.database.models import ExtractionRecord
from sqlalchemy import create_engine, Session

engine = create_engine(DATABASE_URL)
session = Session(engine)

# Store validated data
record = ExtractionRecord(
    patient_name=data["patient_details"]["name"],
    lab_name=data["lab_details"]["lab_name"],
    raw_output=json.dumps(data)
)
session.add(record)
session.commit()
```

---

## 📞 Support & Troubleshooting

For issues or questions:

1. **Check logs** – Enable DEBUG logging for detailed traces
2. **Review docstrings** – Each function has usage examples
3. **Test components independently** – Isolate issues
4. **Inspect intermediate outputs** – Print VLM raw output, parsed JSON, validation results

---

## 🎯 Sprint 1 MVP Checklist

- [x] PDF processor with 300 DPI conversion
- [x] Qwen2-VL-7B-Instruct inference engine
- [x] FastAPI /extract endpoint
- [x] Pydantic schema validation
- [x] CrewAI agent orchestration
- [x] Mock API integration (ready for real endpoint in Sprint 2)
- [x] Error recovery and logging
- [x] Command-line interface

---

**Version:** 1.0.0  
**Date:** 2024-05-05  
**Status:** Production Ready (Sprint 1)
