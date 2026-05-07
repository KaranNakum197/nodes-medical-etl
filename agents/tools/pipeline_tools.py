from typing import Dict, Any, Optional
import json
import httpx
from crewai.tools import tool
import sys
import os
from pathlib import Path

# Add backend to path to import database models
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from backend.database.connection import get_db_session
from backend.database.models import MedicalRecord
from datetime import datetime


def vlm_api_client(image_paths: list[str]) -> str:
    """
    Sends medical document images to the FastAPI VLM engine and returns the extracted JSON data.
    Input MUST be a list of absolute paths to the image files.
    """
    try:
        # FastAPI backend URL
        api_url = os.getenv("API_URL", "http://localhost:8000/extract")
        
        # Prepare the multipart form data
        files_data = []
        file_handles = []
        
        try:
            for path in image_paths:
                f = open(path, "rb")
                file_handles.append(f)
                filename = os.path.basename(path)
                # Form field name 'files' must match FastAPI endpoint parameter name
                files_data.append(("files", (filename, f, "image/jpeg")))
            
            with httpx.Client(timeout=300.0) as client:
                response = client.post(api_url, files=files_data)
                
                if response.status_code == 200:
                    return json.dumps(response.json().get("data", {}))
                else:
                    return f"Error from VLM API: {response.text}"
        finally:
            # Ensure all file handles are closed
            for f in file_handles:
                f.close()
                
    except Exception as e:
        return f"Failed to call VLM API: {str(e)}"


def postgres_insert_tool(validated_json_str: str = None, **kwargs) -> str:
    """
    Inserts the validated medical record JSON into the PostgreSQL database.
    Input MUST be the cleaned, strictly formatted JSON string from the validator.
    """
    if not validated_json_str and kwargs:
        if "raw_json" in kwargs:
            validated_json_str = kwargs["raw_json"]
        elif "function_params" in kwargs and "raw_json" in kwargs["function_params"]:
            validated_json_str = kwargs["function_params"]["raw_json"]
        else:
            validated_json_str = json.dumps(kwargs)

    try:
        if validated_json_str:
            # Strip markdown formatting
            clean_str = validated_json_str.strip()
            if clean_str.startswith('```json'):
                clean_str = clean_str[7:]
            elif clean_str.startswith('```'):
                clean_str = clean_str[3:]
            if clean_str.endswith('```'):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()
            data = json.loads(clean_str)
        else:
            data = {}
        
        patient_name = data.get("patient_details", {}).get("name")
        
        # Extract report date from sample details
        sample_details = data.get("sample_details", {})
        report_date_str = sample_details.get("reported_at") or sample_details.get("collected_at")
        
        report_date = None
        if report_date_str:
            try:
                # Expecting ISO format like 2024-05-05T14:00:00Z
                dt = datetime.fromisoformat(report_date_str.replace('Z', '+00:00'))
                report_date = dt.date()
            except ValueError:
                pass

        # Check for anomalies in top-level results AND nested panel members
        def _has_anomaly(items):
            for res in (items or []):
                if str(res.get("interpretation", "")).lower() in ["high", "low", "abnormal"]:
                    return True
                if _has_anomaly(res.get("members")):
                    return True
            return False

        anomalies_flagged = _has_anomaly(data.get("report_results", []))

        with get_db_session() as session:
            record = MedicalRecord(
                patient_name=patient_name,
                report_date=report_date,
                test_results=data,
                anomalies_flagged=anomalies_flagged
            )
            session.add(record)
        
        return "SUCCESS: Record successfully inserted into PostgreSQL (medical_records table)."
    
    except Exception as e:
        return f"Database Error: Failed to insert record: {str(e)}"
