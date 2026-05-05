# Medical ETL Pipeline - Setup & Test Execution Plan (uv + venv)

**Status:** Step-by-step guide for fast setup using `uv` (modern pip replacement) and `venv`  
**OS:** Windows PowerShell  
**Duration:** ~30 minutes (depending on internet speed)

---

## 🎯 **Executive Overview**

This guide sets up your Medical ETL pipeline using:
- **uv** – Ultra-fast package installer (50x faster than pip)
- **venv** – Python's built-in virtual environment tool
- **Separate environments** – Backend (ROCm PyTorch) and Agents (CrewAI) isolation
- **Automated testing** – Integration test suite verification

---

## 📋 **Prerequisites Check**

Before starting, verify you have:

```powershell
# Check Python 3.10+
python --version

# If Python not installed, get it from:
# https://www.python.org/downloads/
```

**Expected Output:**
```
Python 3.10.x (or higher)
```

---

## 🚀 **Option 1: Automated Setup (Recommended)**

### **Step 1: Run Automated Setup Script**

```powershell
# Navigate to project root
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"

# Run the setup script
.\setup_and_test.ps1
```

**What it does:**
1. ✅ Checks prerequisites (Python, uv)
2. ✅ Creates two separate venvs (backend, agents)
3. ✅ Installs backend deps with ROCm PyTorch
4. ✅ Installs agents deps (CrewAI, Pydantic)
5. ✅ Creates `.env` configuration
6. ✅ Runs integration tests
7. ✅ Shows next steps

**Expected Duration:** 15-20 minutes (first time)

**If you want to skip tests:**
```powershell
.\setup_and_test.ps1 -SkipTests
```

**If you want verbose output:**
```powershell
.\setup_and_test.ps1 -VerboseLogging
```

---

## 🔧 **Option 2: Manual Step-by-Step Setup**

If you prefer manual control or the script fails:

### **Step 1: Prepare Project Directory**

```powershell
# Navigate to project root
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"

# Verify structure
dir .\backend\main.py
dir .\agents\crew.py
```

**Expected Output:**
```
    Directory: d:\Projects Only\AMD Hackathon\nodes-medical-etl\backend

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---           5/5/2024  3:45 PM              7K main.py

    Directory: d:\Projects Only\AMD Hackathon\nodes-medical-etl\agents

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---           5/5/2024  3:45 PM              9K crew.py
```

### **Step 2: Install uv (if not installed)**

```powershell
# Check if uv is installed
uv --version

# If not found, install via pip
pip install uv

# Verify installation
uv --version
```

**Expected Output:**
```
uv 0.x.x
```

### **Step 3: Create Backend Virtual Environment**

```powershell
# Create venv for backend
python -m venv backend\.venv

# Activate it
backend\.venv\Scripts\Activate.ps1

# You should see (backend\.venv) in your prompt
```

**Expected Prompt:**
```
(backend\.venv) PS d:\Projects Only\AMD Hackathon\nodes-medical-etl>
```

### **Step 4: Install Backend Dependencies (Fast with uv)**

```powershell
# Make sure you're in the project root
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"

# Upgrade pip first
python -m pip install --upgrade pip

# Install with uv (FAST!)
uv pip install -r backend\requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/rocm5.7
```

**What's being installed:**
- PyTorch 2.6.0 (ROCm) – ~2 GB
- Transformers 4.48.0 (HuggingFace models)
- FastAPI + Uvicorn
- Pydantic (data validation)
- SQLAlchemy + PostgreSQL driver

**Expected Duration:** 5-10 minutes

**Verify Installation:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.6.0
Transformers: 4.48.0
FastAPI: 0.111.0
```

### **Step 5: Deactivate Backend venv & Create Agents venv**

```powershell
# Deactivate backend environment
deactivate

# Create agents venv
python -m venv agents\.venv

# Activate it
agents\.venv\Scripts\Activate.ps1
```

**Expected Prompt:**
```
(agents\.venv) PS d:\Projects Only\AMD Hackathon\nodes-medical-etl>
```

### **Step 6: Install Agents Dependencies (Fast with uv)**

```powershell
# Make sure you're in project root
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"

# Upgrade pip
python -m pip install --upgrade pip

# Install with uv (FAST!)
uv pip install -r agents\requirements.txt
```

**What's being installed:**
- CrewAI (agent orchestration)
- Pydantic (data validation)
- pdf2image (PDF processing)
- Pillow (image handling)
- Additional utilities

**Expected Duration:** 2-3 minutes

**Verify Installation:**
```powershell
python -c "import crewai; print('CrewAI installed ✓')"
python -c "import pydantic; print('Pydantic installed ✓')"
python -c "import pdf2image; print('pdf2image installed ✓')"
```

### **Step 7: Create `.env` Configuration File**

```powershell
# Create .env file with default values
@"
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database (for future use)
DATABASE_URL=postgresql://user:password@localhost/medical_etl

# Orchestration LLM (optional)
OPENAI_API_KEY=
ORCHESTRATOR_MODEL=gpt-4o-mini

# Model Cache
HF_HOME=./model_cache
"@ | Set-Content .env

# Verify file created
Get-Content .env
```

**Expected Output:**
```
API_HOST=0.0.0.0
API_PORT=8000
...
```

### **Step 8: Run Integration Tests**

```powershell
# Make sure agents venv is active
# agents\.venv\Scripts\Activate.ps1

# Run tests
python integration_test.py
```

**Expected Output:**
```
========================================================================
            MEDICAL ETL PIPELINE - INTEGRATION TEST SUITE
========================================================================

Test 1: Pydantic Schema Validation
────────────────────────────────────────────────────────────
✓ Schema validation passed
✓ Phone coercion: 5551234567 → 5551234567
✓ Numeric coercion: '95 mg/dL' → 95.0
...

TEST SUMMARY
========================================================================
✓ Passed:  6
✗ Failed:  0
⊘ Skipped: 0

✓ All tests passed!
```

---

## ✅ **Verification Checklist**

After setup, verify each component:

### **✓ Backend Environment**

```powershell
# Activate backend venv
backend\.venv\Scripts\Activate.ps1

# Test imports
python -c "
import torch
import transformers
import fastapi
import pydantic
print('✓ All backend packages imported successfully')
"

# Check GPU/ROCm
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Device Count: {torch.cuda.device_count()}')
"
```

### **✓ Agents Environment**

```powershell
# Deactivate backend
deactivate

# Activate agents venv
agents\.venv\Scripts\Activate.ps1

# Test imports
python -c "
from agents.tasks import ExtractedReport, validate_extracted_data
from agents.crew import MedicalETLPipeline
print('✓ All agents packages imported successfully')
"
```

### **✓ Test Schemas**

```powershell
# Make sure agents venv is active
python agents/tasks.py
```

**Expected Output:**
```
Testing Pydantic schemas...

✓ Schema validation passed!
{
  "patient_details": {
    "name": "Test Patient",
    ...
  },
  ...
}
```

### **✓ Test PDF Processor**

```powershell
# Create a simple test
python -c "
from agents.tools.pdf_processor import PDFProcessor
print('✓ PDF Processor imported successfully')
print('✓ Ready to process PDFs')
"
```

---

## 🧪 **Test Scenarios**

### **Test 1: Individual Component Tests**

```powershell
# Activate agents environment
agents\.venv\Scripts\Activate.ps1

# Test Pydantic schemas only
python integration_test.py --test pydantic

# Test PDF processor only
python integration_test.py --test pdf_processor

# Test VLM import only
python integration_test.py --test vlm

# Test agents creation only
python integration_test.py --test agents
```

### **Test 2: Full Integration Test**

```powershell
# Run all tests with verbose output
python integration_test.py --verbose

# Run all tests quietly
python integration_test.py
```

### **Test 3: Manual Component Testing**

```powershell
# Test 1: Pydantic schema validation
python -c "
from agents.tasks import ExtractedReport

test_data = {
    'patient_details': {'name': 'Test'},
    'lab_details': {'lab_name': 'Lab'},
    'sample_details': {},
    'report_results': []
}

report = ExtractedReport(**test_data)
print('✓ Pydantic validation works')
print(f'  Patient: {report.patient_details.name}')
"

# Test 2: Mock API extract
python -c "
from agents.tasks import mock_api_extract, validate_extracted_data
import json

raw_json = mock_api_extract('test.jpg')
validated = validate_extracted_data(raw_json)
print('✓ Mock API and validation works')
print(f'  Patient: {validated[\"patient_details\"][\"name\"]}')
"

# Test 3: Agent creation
python -c "
from agents.crew import create_extractor_agent, create_validator_agent

extractor = create_extractor_agent()
validator = create_validator_agent()
print('✓ Agents created successfully')
print(f'  Extractor: {extractor.role}')
print(f'  Validator: {validator.role}')
"
```

---

## 🚀 **Running the Complete Pipeline**

Once everything is set up and tested:

### **Terminal 1: Start FastAPI Server**

```powershell
# Activate backend environment
backend\.venv\Scripts\Activate.ps1

# Start server
python backend/main.py
```

**Expected Output:**
```
2024-05-05 15:30:00 - root - INFO - Loading VLM model...
2024-05-05 15:30:15 - root - INFO - ✓ VLM model loaded and ready
2024-05-05 15:30:20 - uvicorn - INFO - Uvicorn running on http://0.0.0.0:8000
```

### **Terminal 2: Test API Health**

```powershell
# Test health check
curl http://localhost:8000/health

# Or use Python
python -c "
import requests
r = requests.get('http://localhost:8000/health')
print(r.json())
"
```

**Expected Output:**
```json
{
  "status": "healthy",
  "vlm_loaded": true
}
```

### **Terminal 3: Run Agent Pipeline**

```powershell
# Activate agents environment
agents\.venv\Scripts\Activate.ps1

# Process an image (when you have one)
python agents/crew.py "C:\path\to\medical_report.jpg"

# Or process a PDF
python agents/crew.py "C:\path\to\medical_report.pdf"
```

---

## 📊 **Troubleshooting**

### **Issue 1: uv command not found**

```powershell
# Install uv
pip install uv

# Verify
uv --version
```

### **Issue 2: PyTorch/CUDA/ROCm errors**

```powershell
# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# If CUDA not available, it's OK for now (uses CPU)
# For GPU support, ensure ROCm 5.7+ is installed
```

### **Issue 3: Module not found after installation**

```powershell
# Verify you're in the correct venv
# Check prompt - should show (backend\.venv) or (agents\.venv)

# If not, activate:
backend\.venv\Scripts\Activate.ps1
# or
agents\.venv\Scripts\Activate.ps1

# Then try import again
python -c "import fastapi"
```

### **Issue 4: Permission denied on script**

```powershell
# If setup_and_test.ps1 fails to run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\setup_and_test.ps1
```

### **Issue 5: Long paths in Windows**

```powershell
# If you get path errors, use shorter paths
# Avoid: d:\Projects Only\AMD Hackathon\...

# Create a junction link instead:
# New-Item -ItemType Junction -Path C:\etl -Target "d:\Projects Only\AMD Hackathon\nodes-medical-etl"
```

---

## 📈 **Performance Tips**

### **Faster Downloads**

```powershell
# uv is already ~50x faster than pip
# But you can further optimize:

# 1. Use a fast mirror (optional)
uv pip install -r backend/requirements.txt `
    --index-url https://pypi.org/simple/

# 2. Cache wheels locally
uv pip install --cache-dir ./pip-cache -r requirements.txt
```

### **Reusing Environments**

```powershell
# Recreate environments if needed:
rm -r backend\.venv
python -m venv backend\.venv

# Or clear and reinstall:
backend\.venv\Scripts\Activate.ps1
pip uninstall -r backend/requirements.txt -y
uv pip install -r backend/requirements.txt --upgrade
```

---

## 📝 **Virtual Environment Management**

### **List Installed Packages**

```powershell
# In active venv
pip list

# Show only critical packages
pip list | grep -E "(torch|fastapi|crewai|pydantic)"
```

### **Update Packages (if needed)**

```powershell
# Activate venv
backend\.venv\Scripts\Activate.ps1

# Update all
uv pip install --upgrade -r backend/requirements.txt

# Or specific package
uv pip install --upgrade torch
```

### **Remove Virtual Environments**

```powershell
# Clean up (when starting over)
rm -Recurse backend\.venv
rm -Recurse agents\.venv

# Create fresh ones
python -m venv backend\.venv
python -m venv agents\.venv
```

---

## ✨ **Summary of Commands**

| Task | Command |
|------|---------|
| **Automated Setup** | `.\setup_and_test.ps1` |
| **Activate Backend** | `backend\.venv\Scripts\Activate.ps1` |
| **Activate Agents** | `agents\.venv\Scripts\Activate.ps1` |
| **Install Backend Deps** | `uv pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7` |
| **Install Agents Deps** | `uv pip install -r agents/requirements.txt` |
| **Run All Tests** | `python integration_test.py` |
| **Run Specific Test** | `python integration_test.py --test pydantic` |
| **Start API Server** | `python backend/main.py` |
| **Run Pipeline** | `python agents/crew.py report.jpg` |
| **Check Health** | `curl http://localhost:8000/health` |

---

## 🎯 **Success Criteria**

Your setup is complete when:

- ✅ Both venvs created (`backend\.venv`, `agents\.venv`)
- ✅ All dependencies installed (backend + agents)
- ✅ Integration tests pass (6/6 or at least 4/6)
- ✅ `.env` file created with defaults
- ✅ Can import all key modules
- ✅ API server starts without errors

---

## 🔗 **Next Steps**

After successful setup:

1. **Create test documents** – Get sample medical PDFs/images
2. **Start API server** – `python backend/main.py`
3. **Test extraction** – Upload images via `/extract` endpoint
4. **Run full pipeline** – `python agents/crew.py report.jpg`
5. **Check results** – Validate JSON output against schema
6. **Scale to Sprint 2** – Add database integration

---

## 📞 **Getting Help**

If setup fails:

1. **Check logs** – Look for error messages above
2. **Run specific test** – `python integration_test.py --test pydantic`
3. **Verify Python** – `python --version` (should be 3.10+)
4. **Check internet** – Large packages like PyTorch need bandwidth
5. **Review prerequisites** – Ensure Python and uv installed

---

**Version:** 1.0.0  
**Last Updated:** May 5, 2024  
**Status:** Ready for execution
