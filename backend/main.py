"""
FastAPI Medical ETL Server - Image Upload and VLM Inference Endpoint.

This module provides a minimal, production-ready FastAPI application
with an /extract endpoint that accepts medical document images and
returns JSON-formatted extracted data via the VLM inference engine.

Dependencies:
    - fastapi
    - uvicorn
    - python-multipart
    - pillow
"""

import os
import logging
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import VLM extractor from sibling module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from vlm_engine.extractor import QwenVLMExtractor, VLMInferenceError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI application
app = FastAPI(
    title="Medical ETL Extraction API",
    description="Vision Language Model powered medical document extraction",
    version="1.0.0",
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global VLM extractor instance (loaded once on startup)
vlm_extractor: Optional[QwenVLMExtractor] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize VLM model on server startup.
    
    Loads the Qwen2-VL model once to avoid reload overhead.
    """
    global vlm_extractor
    try:
        logger.info("Loading VLM model...")
        vlm_extractor = QwenVLMExtractor()
        logger.info("✓ VLM model loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load VLM model: {e}", exc_info=True)
        # Don't fail startup - allow graceful error handling per request
        vlm_extractor = None


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on server shutdown.
    """
    global vlm_extractor
    if vlm_extractor:
        del vlm_extractor
        logger.info("VLM model unloaded")


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        status: "healthy" or "degraded"
        vlm_loaded: Whether VLM is available
    """
    return {
        "status": "healthy" if vlm_extractor else "degraded",
        "vlm_loaded": vlm_extractor is not None,
    }


@app.post("/extract", tags=["Extraction"])
async def extract_medical_data(
    file: UploadFile = File(..., description="Medical document image (JPEG/PNG)"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Extract structured medical data from an uploaded image.
    
    This endpoint accepts a medical document image, passes it to the
    Vision Language Model, and returns extracted data as JSON.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        {
            "success": true/false,
            "data": {...extracted JSON...},
            "error": "error message if failed",
            "metadata": {
                "filename": "...",
                "size_bytes": ...,
                "processing_time_ms": ...
            }
        }
        
    Raises:
        HTTPException 400: Invalid file format or upload error
        HTTPException 503: VLM model not available
        HTTPException 500: Inference error
        
    Example:
        curl -X POST http://localhost:8000/extract \\
             -F "file=@medical_report.jpg"
    """
    import time
    start_time = time.time()
    
    # Validate VLM availability
    if not vlm_extractor:
        logger.error("VLM extractor not initialized")
        raise HTTPException(
            status_code=503,
            detail="VLM model not available. Server may still be initializing."
        )
    
    # Validate file upload
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format: {file_ext}. "
                   f"Supported: {', '.join(valid_extensions)}"
        )
    
    # Validate file size (max 50MB for safety)
    max_size_bytes = 50 * 1024 * 1024
    
    temp_path = None
    try:
        # Read file into temporary location
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        if len(contents) > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {len(contents)} bytes "
                       f"(max: {max_size_bytes} bytes)"
            )
        
        # Write to temporary file for inference
        with tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
            dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        
        logger.info(f"Processing upload: {file.filename} ({len(contents)} bytes)")
        
        # Run VLM inference
        try:
            extracted_data = vlm_extractor.extract_json(temp_path)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"✓ Extraction successful: {file.filename} "
                f"({processing_time_ms}ms)"
            )
            
            # Schedule temp file cleanup
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": extracted_data,
                    "metadata": {
                        "filename": file.filename,
                        "size_bytes": len(contents),
                        "processing_time_ms": processing_time_ms,
                    }
                }
            )
        
        except VLMInferenceError as e:
            logger.error(f"VLM inference failed: {e}")
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {str(e)[:100]}"
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
            raise HTTPException(
                status_code=500,
                detail="VLM output was not valid JSON"
            )
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
            raise HTTPException(
                status_code=500,
                detail="Internal server error during processing"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        raise
    
    except Exception as e:
        logger.error(f"Unexpected top-level error: {e}", exc_info=True)
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error"
        )


@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API documentation.
    """
    return {
        "service": "Medical ETL Extraction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "extract": "POST /extract (multipart/form-data with 'file')"
        }
    }


def cleanup_temp_file(filepath: str):
    """
    Background task to cleanup temporary files.
    
    Args:
        filepath: Path to file to delete.
    """
    try:
        if Path(filepath).exists():
            Path(filepath).unlink()
            logger.debug(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {filepath}: {e}")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """
    Start the FastAPI server.
    
    Args:
        host: Bind address (default: 0.0.0.0 for all interfaces)
        port: Port number (default: 8000)
        reload: Auto-reload on file change (development only)
        
    Example:
        >>> run_server(host="localhost", port=8000, reload=True)
    """
    logger.info(f"Starting Medical ETL API on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical ETL FastAPI Server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", 8000)),
        help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes (dev only)"
    )
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)


@app.get("/records")
async def list_records():
    """Return all previously extracted records from the database."""
    with get_db_session() as session:
        records = session.query(ExtractionRecord).all()
        return [r.to_dict() for r in records]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
