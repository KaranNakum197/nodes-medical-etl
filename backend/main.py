"""
FastAPI entry point for the nodes-medical-etl backend.

Exposes endpoints for document upload, VLM inference, and ETL status.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from vlm_engine.inference import run_inference
from database.connection import get_db_session
from database.models import ExtractionRecord

app = FastAPI(
    title="Medical ETL API",
    description="High-throughput medical document extraction powered by Qwen2-VL on AMD MI300X",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Return service health status."""
    return {"status": "ok"}


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """
    Accept a medical document (PDF or image) and run VLM extraction.

    Returns the extracted structured data as JSON.
    """
    if file.content_type not in ("application/pdf", "image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    contents = await file.read()
    result = run_inference(contents, filename=file.filename)
    return {"filename": file.filename, "extracted_data": result}


@app.get("/records")
async def list_records():
    """Return all previously extracted records from the database."""
    with get_db_session() as session:
        records = session.query(ExtractionRecord).all()
        return [r.to_dict() for r in records]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
