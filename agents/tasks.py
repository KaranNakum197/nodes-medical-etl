"""
CrewAI Tasks & Pydantic Schemas for Medical ETL Agent Orchestration.

This module defines:
1. Pydantic schemas for strict medical data validation
2. CrewAI Task definitions for extraction and validation
3. Integration points with VLM inference and external APIs

Dependencies:
    - crewai
    - pydantic
    - python-httpx (for async API calls)
"""

import logging
import re
import json
from typing import Optional, List, Any, Dict, Union
from datetime import datetime, timezone

from pydantic import BaseModel, field_validator, ConfigDict
from crewai import Task


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC SCHEMAS FOR MEDICAL DATA VALIDATION
# ============================================================================

def parse_number(val: Any) -> Optional[float]:
    """
    Parse incoming unformatted string or numeric input into precise float.

    Handles strings like "123.45 mg/dL", "1,234.56", integers, and floats.

    Args:
        val (Any): String, float, int, or None representing numeric value.

    Returns:
        float | None: Parsed floating point number, or None if unparseable.
    """
    if val is None:
        return None
    
    if isinstance(val, (int, float)):
        return float(val)
    
    # Convert to string and strip whitespace
    s = str(val).strip()
    
    # Extract first number (handles "123.45 mg/dL" -> 123.45)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    
    return None


class MemberModel(BaseModel):
    """
    Nested test result for panel members or sub-panels.
    
    Represents child observations within a panel test (e.g., WBC Differential).
    Supports recursive nesting for multi-level panels.
    """
    is_panel: bool = False
    test_name: str
    value: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    sample_type: Optional[str] = None
    method: Optional[str] = None
    test_remarks: Optional[str] = None
    extra_details: Optional[Dict[str, Any]] = None
    members: Optional[List['MemberModel']] = None

    @field_validator("value", mode='before')
    @classmethod
    def coerce_value(cls, v):
        """Coerce numeric value using parse_number."""
        return parse_number(v)
    
    class Config:
        # Allow extra fields but ignore them
        extra = "ignore"


class ReportResultModel(BaseModel):
    """
    Individual test result or panel from medical report.
    
    Represents a single test observation or grouped panel of tests.
    May contain nested members for hierarchical test panels.
    """
    is_panel: bool
    test_name: str
    sample_type: Optional[str] = None
    method: Optional[str] = None
    technology: Optional[str] = None
    value: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    test_remarks: Optional[str] = None
    test_notes: Optional[str] = None
    test_interpretations: Optional[str] = None
    extra_details: Optional[Dict[str, Any]] = None
    members: Optional[List[MemberModel]] = None

    @field_validator("value", mode='before')
    @classmethod
    def coerce_value(cls, v):
        """Coerce numeric value using parse_number."""
        return parse_number(v)
    
    class Config:
        extra = "ignore"


class SampleDetailsModel(BaseModel):
    """Sample collection and handling metadata."""
    sample_id: Optional[str] = None
    specimen: Optional[str] = None
    collected_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    reported_at: Optional[datetime] = None

    @field_validator("collected_at", "received_at", "reported_at", mode='before')
    @classmethod
    def parse_dt(cls, v):
        """
        Parse ISO8601 datetime strings and ensure UTC timezone.

        Args:
            v: String or datetime.datetime object, or None.

        Returns:
            Timezone-aware UTC datetime, or None if unparseable.
        """
        if v is None:
            return None
        
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        
        try:
            # Handle both ISO formats and 'Z' suffix
            dt_str = str(v).replace('Z', '+00:00')
            dt = datetime.fromisoformat(dt_str)
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            return dt
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{v}': {e}")
            return None
    
    class Config:
        extra = "ignore"


class PatientDetailsModel(BaseModel):
    """Patient demographic information."""
    patient_id: Optional[str] = None
    name: str = "Unknown"  # Default for missing names
    age: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None

    class Config:
        extra = "ignore"


class LabDetailsModel(BaseModel):
    """Laboratory facility information."""
    lab_name: str
    lab_address: Optional[str] = None
    lab_website: Optional[str] = None
    lab_code: Optional[Union[str, int]] = None
    email: Optional[str] = None
    phone: Optional[Union[str, int]] = None
    referred_by: Optional[str] = None

    @field_validator("phone", mode='before')
    @classmethod
    def normalize_phone(cls, v):
        """Normalize phone numbers to string."""
        if v is None:
            return None
        return str(v)
    
    class Config:
        extra = "ignore"


class ExtractedReport(BaseModel):
    """
    Complete extracted medical report matching Pydantic schema.
    
    This is the primary output schema for VLM extraction and validation.
    All fields except patient_details are optional to support partial reports.
    
    Example:
        >>> data = {
        ...     "patient_details": {"name": "John Doe"},
        ...     "lab_details": {"lab_name": "PathLab"},
        ...     "sample_details": {},
        ...     "report_results": [...]
        ... }
        >>> report = ExtractedReport(**data)
    """
    patient_details: PatientDetailsModel
    lab_details: LabDetailsModel
    sample_details: SampleDetailsModel
    report_results: List[ReportResultModel]
    result_details: Optional[str] = None
    global_remarks: Optional[str] = None
    global_notes: Optional[str] = None
    extra_details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary, excluding None values.
        
        Returns:
            Dict with non-null values only.
        """
        return self.model_dump(exclude_none=True)


# ============================================================================
# CREWAI TASK DEFINITIONS
# ============================================================================

def create_extractor_task(agent) -> Task:
    """
    Create the Extractor agent task.
    
    Calls the FastAPI /extract endpoint (or mock API in Sprint 1) to get
    raw JSON from VLM. This task is responsible for communicating with the
    inference layer.
    
    Args:
        agent: The Extractor Agent instance from crew.py.
        
    Returns:
        CrewAI Task configured for extraction.
    """
    return Task(
        description="""
        Extract structured medical data from a document image using the Vision Language Model.
        
        Steps:
        1. Read the provided image at {image_path}.
        2. You MUST use the `VLM_API_Client` tool to send the image to the FastAPI server and extract the data. DO NOT output a tool-call JSON as your final answer.
        3. Extract the raw JSON response returned by the tool.
        4. Handle HTTP errors gracefully and retry if necessary.
        
        CRITICAL: Use the tool provided. Your Final Answer MUST be the actual extracted medical data JSON, NOT a function call JSON.
        """,
        expected_output="""
        Raw JSON string from VLM containing the required keys: patient_details, lab_details, sample_details, and report_results.
        
        If extraction fails, return a JSON object with an 'error' key containing the detailed error message.
        """,
        agent=agent,
        async_execution=False,
    )


def create_validator_task(agent) -> Task:
    """
    Create the Validator agent task.
    
    Takes raw JSON from Extractor, validates against Pydantic schema,
    performs data quality checks, and returns cleaned data.
    
    Args:
        agent: The Validator Agent instance from crew.py.
        
    Returns:
        CrewAI Task configured for validation.
    """
    return Task(
        description="""
        Validate and clean extracted medical data against strict Pydantic schema.
        
        Steps:
        1. Accept raw JSON from Extractor task
        2. Validate against the ExtractedReport schema rules.
        3. If validation fails, attempt recovery (e.g. coerce numbers, fix dates).
        4. ONCE VALIDATED, you MUST use the `Postgres_Insert_Tool` to insert the clean JSON into the database. Pass the clean JSON string to the `validated_json_str` parameter.
        5. Return the clean validated data along with the database insertion success message.
        
        CRITICAL RULES:
        - Omit fields with None/null values
        - Enforce required fields: patient_details, lab_details, sample_details, report_results
        - Ensure all numeric values are floats
        - Ensure all datetimes are ISO8601 UTC
        - Reject if patient_details.name is missing
        - Log all validation errors for debugging
        - YOU MUST actually invoke the `Postgres_Insert_Tool` using the tool mechanism. Do NOT output a function call JSON as your final answer.
        """,
        expected_output="""
        Clean, validated JSON conforming to the ExtractedReport Pydantic schema.
        The output must be a valid JSON string with the root keys: patient_details, lab_details, sample_details, and report_results.
        
        On validation failure, return a JSON object with an 'error' key detailing the failure and a 'raw_output' key containing the original string.
        """,
        agent=agent,
        async_execution=False,
    )


# ============================================================================
# UTILITY FUNCTIONS FOR AGENTS
# ============================================================================

def mock_api_extract(image_path: str) -> str:
    """
    Mock API call to FastAPI /extract endpoint.
    
    For Sprint 1 MVP, this returns a pre-defined medical report JSON.
    In production, replace with actual HTTP request to backend/main.py.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        JSON string with extracted data.
        
    Example (replace with real API call):
        >>> import httpx
        >>> async def real_api_extract(image_path: str) -> str:
        ...     async with httpx.AsyncClient() as client:
        ...         with open(image_path, 'rb') as f:
        ...             r = await client.post(
        ...                 "http://localhost:8000/extract",
        ...                 files={"file": f}
        ...             )
        ...         return r.json()["data"]
    """
    logger.info(f"Mock API extract called for: {image_path}")
    
    # Mock response for testing
    mock_data = {
        "patient_details": {
            "name": "John Doe",
            "patient_id": "PT-2024-001",
            "age": "45 Years",
            "gender": "Male",
        },
        "lab_details": {
            "lab_name": "PathLab Medical Center",
            "lab_code": "PLC-001",
            "phone": "555-0123",
            "email": "contact@pathlab.com",
        },
        "sample_details": {
            "sample_id": "SMP-2024-05-001",
            "specimen": "Whole Blood",
            "collected_at": "2024-05-05T09:30:00Z",
            "reported_at": "2024-05-05T14:00:00Z",
        },
        "report_results": [
            {
                "is_panel": True,
                "test_name": "Complete Blood Count (CBC)",
                "sample_type": "Whole Blood",
                "method": "Automated Hematology Analyzer",
                "value": None,
                "members": [
                    {
                        "is_panel": False,
                        "test_name": "Hemoglobin",
                        "value": 14.5,
                        "unit": "g/dL",
                        "reference_range": "13.0-17.0",
                        "interpretation": "Normal",
                    },
                    {
                        "is_panel": False,
                        "test_name": "White Blood Cell Count",
                        "value": 7.2,
                        "unit": "K/uL",
                        "reference_range": "4.5-11.0",
                        "interpretation": "Normal",
                    },
                ]
            },
            {
                "is_panel": False,
                "test_name": "Glucose, Fasting",
                "value": 95.0,
                "unit": "mg/dL",
                "reference_range": "70-100",
                "interpretation": "Normal",
                "sample_type": "Plasma",
            }
        ],
        "global_remarks": "All results are within normal limits. No abnormalities detected.",
    }
    
    return json.dumps(mock_data)


def validate_extracted_data(raw_json_str: str) -> Dict[str, Any]:
    """
    Validate raw VLM output against Pydantic schema with error recovery.
    
    Args:
        raw_json_str: Raw JSON string from VLM/Extractor.
        
    Returns:
        Validated data dictionary or error dict.
        
    Example:
        >>> json_str = '{"patient_details": {...}, ...}'
        >>> result = validate_extracted_data(json_str)
        >>> if "error" not in result:
        ...     print(result["patient_details"]["name"])
    """
    try:
        # Parse JSON
        data = json.loads(raw_json_str)
        
        # Attempt Pydantic validation
        report = ExtractedReport(**data)
        
        logger.info(f"✓ Validation successful for patient: {report.patient_details.name}")
        return report.to_dict()
    
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "raw_output": raw_json_str[:200]}
    
    except Exception as e:
        # Try to extract what we can
        error_msg = f"Validation error: {str(e)}"
        logger.warning(f"{error_msg}. Attempting recovery...")
        
        try:
            data = json.loads(raw_json_str)
            
            # Ensure required fields exist
            if "patient_details" not in data:
                data["patient_details"] = {"name": "Unknown"}
            if "lab_details" not in data:
                data["lab_details"] = {"lab_name": "Unknown"}
            if "sample_details" not in data:
                data["sample_details"] = {}
            if "report_results" not in data:
                data["report_results"] = []
            
            # Retry validation
            report = ExtractedReport(**data)
            logger.info(f"✓ Recovery successful for patient: {report.patient_details.name}")
            return report.to_dict()
        
        except Exception as recovery_error:
            error_msg = f"Validation and recovery failed: {str(recovery_error)}"
            logger.error(error_msg)
            return {"error": error_msg, "raw_output": raw_json_str[:500]}


if __name__ == "__main__":
    # Test Pydantic schemas
    print("Testing Pydantic schemas...\n")
    
    test_data = {
        "patient_details": {"name": "Test Patient", "age": "50 Years"},
        "lab_details": {"lab_name": "Test Lab"},
        "sample_details": {"sample_id": "S001", "collected_at": "2024-05-05T10:00:00Z"},
        "report_results": [
            {
                "is_panel": False,
                "test_name": "Glucose",
                "value": "95 mg/dL",
                "unit": "mg/dL",
            }
        ]
    }
    
    try:
        report = ExtractedReport(**test_data)
        print("✓ Schema validation passed!")
        print(json.dumps(report.to_dict(), indent=2, default=str))
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
