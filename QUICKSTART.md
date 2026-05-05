# Quick Start - 5 Minute Setup

**TL;DR:** Copy-paste commands below to get running in 5 minutes with `uv` + `venv`

---

## 🚀 **Ultra-Fast Setup (Automated)**

```powershell
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"
.\setup_and_test.ps1
```

**That's it!** Script handles everything. Go grab coffee ☕

---

## 🔧 **Manual Setup (Step-by-Step)**

### Step 1: Create Virtual Environments (30 seconds)

```powershell
cd "d:\Projects Only\AMD Hackathon\nodes-medical-etl"

# Backend environment
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1

# Install uv if needed
pip install uv

# Backend dependencies (5-10 min)
uv pip install -r backend/requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7

# Done with backend
deactivate
```

### Step 2: Agents Environment (30 seconds)

```powershell
# Agents environment
python -m venv agents\.venv
agents\.venv\Scripts\Activate.ps1

# Agents dependencies (2-3 min)
uv pip install -r agents/requirements.txt

# Done
deactivate
```

### Step 3: Test Everything (2 minutes)

```powershell
# Activate agents and run tests
agents\.venv\Scripts\Activate.ps1
python integration_test.py
```

---

## ✅ **Verify Installation**

```powershell
# Backend check
backend\.venv\Scripts\Activate.ps1
python -c "import torch, fastapi; print('✓ Backend OK')"
deactivate

# Agents check
agents\.venv\Scripts\Activate.ps1
python -c "import crewai, pydantic; print('✓ Agents OK')"
deactivate
```

---

## 🎮 **Run the System**

### Terminal 1: Start API Server

```powershell
backend\.venv\Scripts\Activate.ps1
python backend/main.py
```

### Terminal 2: Test API

```powershell
curl http://localhost:8000/health
```

### Terminal 3: Run Pipeline

```powershell
agents\.venv\Scripts\Activate.ps1
python agents/crew.py "path/to/medical_report.jpg"
```

---

## 📊 **Environment Sizes**

| Environment | Size | Time |
|-------------|------|------|
| backend\.venv | ~3 GB | 5-10 min |
| agents\.venv | ~500 MB | 2-3 min |
| **Total** | **~3.5 GB** | **10-15 min** |

---

## 🛠️ **Troubleshooting Quick Fixes**

| Issue | Fix |
|-------|-----|
| `uv: command not found` | `pip install uv` |
| `Permission denied` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `CUDA not available` | OK – uses CPU fallback (slow but works) |
| Module import fails | Check you're in correct venv |
| Script won't run | Enable execution: `Set-ExecutionPolicy RemoteSigned` |

---

## 📝 **Key Files**

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI server |
| `backend/vlm_engine/extractor.py` | Qwen2-VL inference |
| `agents/crew.py` | Agent orchestration |
| `agents/tasks.py` | Pydantic schemas |
| `agents/tools/pdf_processor.py` | PDF → JPEG converter |

---

## 🎯 **Success = These Commands Work**

```powershell
# 1. Create venvs
python -m venv backend\.venv
python -m venv agents\.venv

# 2. Install fast with uv
uv pip install -r backend/requirements.txt
uv pip install -r agents/requirements.txt

# 3. Run tests
python integration_test.py
# Expected: ✓ All tests passed

# 4. Start server
python backend/main.py
# Expected: Uvicorn running on http://0.0.0.0:8000

# 5. Health check
curl http://localhost:8000/health
# Expected: {"status": "healthy", "vlm_loaded": true}
```

---

## 💡 **Why uv?**

- **50x faster** than pip for large installs
- **Parallel downloads** – doesn't wait for packages
- **Same CLI** as pip (drop-in replacement)
- **Perfect** for 2GB PyTorch downloads

---

## ⏱️ **Estimated Timeline**

| Step | Duration |
|------|----------|
| Install uv | 1 min |
| Create venvs | 1 min |
| Backend install | 8 min (with uv) |
| Agents install | 2 min |
| Tests | 2 min |
| **Total** | **~15 min** |

---

## 📞 **Need Help?**

See **EXECUTION_PLAN.md** for detailed troubleshooting and manual steps.

---

**Ready? Run:** `.\setup_and_test.ps1` 🚀
