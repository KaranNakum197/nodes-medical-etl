from typing import Dict, Any, Optional
import json
import httpx
from crewai.tools import tool
import sys
from pathlib import Path

# Add backend to path to import database models
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from database.connection import get_db_session
from database.models import MedicalRecord
from datetime import datetime


@tool("VLM_API_Client")
def vlm_api_client(image_path: str) -> str:
    """
    Sends a medical document image to the FastAPI VLM engine and returns the extracted JSON data.
    Input MUST be the absolute path to the image file.
    """
    try:
        # Use sync httpx or requests for tool simplicity
        with httpx.Client(timeout=60.0) as client:
            with open(image_path, 'rb') as f:
                response = client.post(
                    "http://localhost:8000/extract",
                    files={"file": f}
                )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                return json.dumps(data)
            else:
                return f"Error from VLM API: {response.text}"
    except Exception as e:
        return f"Failed to call VLM API: {str(e)}"


@tool("Postgres_Insert_Tool")
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
        data = json.loads(validated_json_str)
        
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

        # Check for anomalies loosely based on 'global_remarks' or 'High'/'Low' interpretations
        anomalies_flagged = False
        results = data.get("report_results", [])
        for res in results:
            if str(res.get("interpretation")).lower() in ["high", "low", "abnormal"]:
                anomalies_flagged = True
                break

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
