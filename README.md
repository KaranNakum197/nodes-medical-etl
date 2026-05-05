# nodes-medical-etl

High-throughput medical document extraction pipeline powered by **Qwen2-VL** on **AMD MI300X** and **CrewAI**.

## Project Structure

```
nodes-medical-etl/
│
├── backend/                  # Karan's Domain: VLM & Infrastructure
│   ├── main.py               # FastAPI entry point
│   ├── vlm_engine/           # Qwen2-VL inference scripts (ROCm optimized)
│   ├── database/             # PostgreSQL schemas and connection logic
│   └── requirements.txt      # PyTorch (ROCm), transformers, etc.
│
├── agents/                   # Shivam's Domain: Orchestration
│   ├── crew.py               # CrewAI setup and agent definitions
│   ├── tasks.py              # Task prompts (Extraction, Validation)
│   ├── tools/                # Custom tools (e.g., PDF to Image converter)
│   └── requirements.txt      # crewai, langchain, pdf2image
│
├── frontend/                 # React Dashboard (To be built later)
│   ├── src/
│   ├── public/
│   └── package.json
│
├── docs/                     # For the "Build in Public" challenge
│   └── architecture.png      # Drop the system diagram here
│
├── .env.example              # Template for API keys and DB credentials
└── README.md                 # The face of your project
```

## Quick Start

### 1 – Clone & configure

```bash
git clone https://github.com/KaranNakum197/nodes-medical-etl.git
cd nodes-medical-etl
cp .env.example .env
# Edit .env with your credentials
```

### 2 – Backend

```bash
cd backend
# Install ROCm-enabled PyTorch + other deps
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/rocm5.7

# Initialise the database tables
python -m database.schema

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3 – Agents

```bash
cd agents
pip install -r requirements.txt

# Run the full ETL crew on a document
python crew.py /path/to/document.pdf
```

### 4 – Frontend

```bash
cd frontend
npm install
npm start
```

The dashboard will be available at `http://localhost:3000`.

## Architecture

See [`docs/`](docs/) for the system diagram.

## Tech Stack

| Layer | Technology |
|---|---|
| VLM Inference | Qwen2-VL-7B-Instruct, PyTorch + ROCm |
| GPU | AMD MI300X |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy |
| Orchestration | CrewAI + LangChain |
| Frontend | React 18 |

