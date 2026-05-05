# Sprint 1 MVP - Quick API Reference

## 🔌 FastAPI Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

**Purpose:** Check if VLM model is loaded and server is healthy

**Response:**
```json
{
  "status": "healthy" | "degraded",
  "vlm_loaded": true | false
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Medical Data Extraction
**Endpoint:** `POST /extract`

**Purpose:** Upload medical document image and extract structured JSON

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image file (JPEG, PNG, BMP, GIF, WebP)
  - Max size: 50MB

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "patient_details": {
      "name": "string",
      "patient_id": "string",
      "age": "string (Years)",
      "gender": "string",
      "address": "string"
    },
    "lab_details": {
      "lab_name": "string",
      "lab_code": "string",
      "phone": "string",
      "email": "string",
      "referred_by": "string"
    },
    "sample_details": {
      "sample_id": "string",
      "specimen": "string",
      "collected_at": "ISO8601_DateTime",
      "reported_at": "ISO8601_DateTime"
    },
    "report_results": [
      {
        "is_panel": boolean,
        "test_name": "string",
        "value": number,
        "unit": "string",
        "reference_range": "string",
        "interpretation": "string",
        "members": [...]
      }
    ]
  },
  "metadata": {
    "filename": "string",
    "size_bytes": number,
    "processing_time_ms": number
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "error message",
  "metadata": {...}
}
```

**Examples:**

```bash
# Using curl
curl -X POST http://localhost:8000/extract \
  -F "file=@medical_report.jpg"

# Using Python
import requests

with open("medical_report.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/extract",
        files={"file": f}
    )

data = r.json()
print(data["data"]["patient_details"]["name"])
```

**HTTP Status Codes:**
- `200` – Extraction successful
- `400` – Invalid file format or upload error
- `503` – VLM model not loaded
- `500` – Inference or parsing error

---

## 🤖 CrewAI Pipeline API

### MedicalETLPipeline Class

**Import:**
```python
from agents.crew import MedicalETLPipeline
```

**Methods:**

#### `process_image(image_path: str) -> dict`

Process a single medical document image.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
```python
{
    "success": bool,
    "data": {...validated_json...},
    "error": "error message if failed",
}
```

**Example:**
```python
pipeline = MedicalETLPipeline()
result = pipeline.process_image("medical_report.jpg")

if result["success"]:
    patient_name = result["data"]["patient_details"]["name"]
    print(f"Patient: {patient_name}")
else:
    print(f"Error: {result['error']}")
```

#### `process_pdf(pdf_path: str) -> dict`

Process a multi-page PDF by converting to images first.

**Parameters:**
- `pdf_path` (str): Path to PDF file

**Returns:**
```python
{
    "success": bool,
    "results": [
        {
            "page": int,
            "success": bool,
            "data": {...},
            "error": "..."
        }
    ],
    "summary": {
        "patient_details": {...},
        "lab_details": {...},
        "report_results": [...]
    }
}
```

**Example:**
```python
pipeline = MedicalETLPipeline()
result = pipeline.process_pdf("medical_report.pdf")

for page_result in result["results"]:
    print(f"Page {page_result['page']}: {'✓' if page_result['success'] else '✗'}")

print("\nAggregated Results:")
print(result["summary"]["patient_details"]["name"])
```

---

## 🔧 VLM Extractor API

**Import:**
```python
from backend.vlm_engine.extractor import QwenVLMExtractor
```

### Class: QwenVLMExtractor

#### `__init__(model_id: str = None)`

Initialize VLM with model loading.

**Parameters:**
- `model_id` (optional): Override default "Qwen/Qwen2-VL-7B-Instruct"

**Example:**
```python
# Default model
extractor = QwenVLMExtractor()

# Custom model
extractor = QwenVLMExtractor(model_id="Qwen/Qwen2-VL-7B")
```

#### `extract_from_image(image_path: str, custom_prompt: str = None, max_tokens: int = 2048) -> str`

Extract raw text from image using VLM.

**Parameters:**
- `image_path` (str): Path to image file
- `custom_prompt` (optional): Override default extraction prompt
- `max_tokens` (optional): Max tokens in output (default: 2048)

**Returns:**
- Raw VLM output text (should be JSON)

**Example:**
```python
extractor = QwenVLMExtractor()
raw_output = extractor.extract_from_image("page_0001.jpg")
print(raw_output[:500])
```

#### `extract_json(image_path: str, custom_prompt: str = None, max_tokens: int = 2048) -> dict`

Extract and parse JSON from image.

**Parameters:**
- Same as `extract_from_image()`

**Returns:**
- Parsed JSON dictionary

**Example:**
```python
extractor = QwenVLMExtractor()
data = extractor.extract_json("medical_report.jpg")

print(data["patient_details"]["name"])
print(data["report_results"][0]["test_name"])
```

---

## 📦 PDF Processor API

**Import:**
```python
from agents.tools.pdf_processor import PDFProcessor
```

### Class: PDFProcessor

#### `__init__(dpi: int = 300, temp_dir: str = None)`

Initialize PDF processor.

**Parameters:**
- `dpi` (optional): Resolution in DPI (default: 300)
- `temp_dir` (optional): Custom temp directory (default: system temp)

**Example:**
```python
# Default: 300 DPI, system temp
processor = PDFProcessor()

# Custom: 600 DPI, specific directory
processor = PDFProcessor(dpi=600, temp_dir="/tmp/medical_docs")
```

#### `process_pdf(pdf_path: str) -> (List[str], str)`

Convert PDF to JPEG images.

**Parameters:**
- `pdf_path` (str): Path to PDF file

**Returns:**
- Tuple of (image_paths list, temp_directory path)

**Example:**
```python
processor = PDFProcessor(dpi=300)
image_paths, temp_dir = processor.process_pdf("report.pdf")

print(f"Extracted {len(image_paths)} pages")
for img in image_paths:
    print(f"  - {img}")
```

#### `cleanup() -> None`

Remove temporary directory and images.

**Example:**
```python
processor = PDFProcessor()
image_paths, _ = processor.process_pdf("report.pdf")

# ... use images ...

processor.cleanup()  # Delete all temp files
```

#### Context Manager Support

```python
# Automatically cleanup on exit
with PDFProcessor(dpi=300) as processor:
    image_paths, temp_dir = processor.process_pdf("report.pdf")
    # ... process images ...
# Cleanup happens automatically here
```

---

## ✅ Pydantic Schema API

**Import:**
```python
from agents.tasks import ExtractedReport, validate_extracted_data
```

### Validation Function

#### `validate_extracted_data(raw_json_str: str) -> dict`

Validate raw VLM output against Pydantic schema.

**Parameters:**
- `raw_json_str` (str): Raw JSON string from VLM

**Returns:**
- Validated data dict or error dict

**Example:**
```python
from agents.tasks import validate_extracted_data
import json

raw = '{"patient_details": {...}, "lab_details": {...}, ...}'
result = validate_extracted_data(raw)

if "error" not in result:
    print(f"Patient: {result['patient_details']['name']}")
else:
    print(f"Validation failed: {result['error']}")
```

### ExtractedReport Model

**Fields:**
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

**Methods:**
- `to_dict()` – Convert to dictionary, excluding None values

**Example:**
```python
from agents.tasks import ExtractedReport

data = {...validated_data...}
report = ExtractedReport(**data)

# Convert to dict without None values
output = report.to_dict()
```

---

## 🛠️ Utility Functions

### parse_number(val: Any) -> Optional[float]

Convert numeric strings to float.

**Example:**
```python
from agents.tasks import parse_number

parse_number("95 mg/dL")      # → 95.0
parse_number("1,234.56")       # → 1234.56
parse_number(95)               # → 95.0
parse_number("abc")            # → None
```

### mock_api_extract(image_path: str) -> str

Mock API call for Sprint 1 testing.

**Example:**
```python
from agents.tasks import mock_api_extract
import json

json_str = mock_api_extract("test_image.jpg")
data = json.loads(json_str)
print(data["patient_details"]["name"])
```

---

## 📊 Common Response Patterns

### Successful Extraction
```python
{
    "success": True,
    "data": {
        "patient_details": {
            "name": "John Doe",
            "patient_id": "PT-2024-001",
            "age": "45 Years",
            "gender": "Male"
        },
        "lab_details": {
            "lab_name": "PathLab Medical Center",
            "lab_code": "PLC-001",
            "phone": "555-0123",
            "email": "contact@pathlab.com"
        },
        "sample_details": {
            "sample_id": "SMP-2024-05-001",
            "specimen": "Whole Blood",
            "collected_at": "2024-05-05T09:30:00+00:00"
        },
        "report_results": [
            {
                "is_panel": False,
                "test_name": "Glucose",
                "value": 95.0,
                "unit": "mg/dL",
                "reference_range": "70-100",
                "interpretation": "Normal"
            }
        ]
    },
    "metadata": {
        "filename": "medical_report.jpg",
        "size_bytes": 245632,
        "processing_time_ms": 3240
    }
}
```

### Validation Error with Recovery
```python
{
    "error": "Validation and recovery failed: ...",
    "raw_output": "{incomplete json...}"
}
```

### HTTP Error Response
```json
{
    "detail": "Invalid file format: .doc. Supported: {'.jpg', '.jpeg', '.png', ...}"
}
```

---

## ⚡ Performance Tips

1. **GPU Memory:** Use `device_map="auto"` (default) for multi-GPU
2. **DPI Trade-off:** 300 DPI is standard; 600 DPI for high-detail, 150 DPI for speed
3. **Batch Processing:** Process multiple images in sequence for better throughput
4. **Temperature:** Set `temperature=0.2` for deterministic extraction
5. **Max Tokens:** Use 2048 for typical reports; increase for long documents

---

## 🔐 Security Notes

1. **File Upload:** Max 50MB enforced in FastAPI
2. **CORS:** Currently allows all origins; restrict in production
3. **VLM Model:** Runs locally (no data sent to external API)
4. **Environment:** Use `.env` for sensitive config

---

**Version:** 1.0.0  
**Last Updated:** 2024-05-05
