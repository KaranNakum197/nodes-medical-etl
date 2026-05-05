# Architecture Diagram

Place the system architecture diagram (`architecture.png`) in this directory.

The diagram should illustrate:
- **Frontend** (React Dashboard) → REST calls → **Backend** (FastAPI)
- **Backend** → GPU inference → **VLM Engine** (Qwen2-VL / ROCm)
- **Backend** → persistence → **PostgreSQL**
- **Agents** (CrewAI) → orchestrates extraction & validation tasks
- **Agents** → REST calls → **Backend** `/extract` endpoint
