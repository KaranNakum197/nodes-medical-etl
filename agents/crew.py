"""
CrewAI Agent Definitions & Crew Orchestration for Medical ETL Pipeline.

This module defines:
1. Extractor Agent - VLM inference and raw JSON extraction
2. Validator Agent - Pydantic schema validation and data cleaning
3. Medical ETL Crew - Orchestrates agent collaboration

The Crew runs as a sequential process: Extraction -> Validation -> Output

Dependencies:
    - crewai
    - python-dotenv
"""

import logging
import os
import sys
import time
from pathlib import Path

from crewai import Agent, Crew, LLM, Process

# Import task builders and validation utilities
from tasks import (
    create_extractor_task,
    create_validator_task,
    mock_api_extract,
    validate_extracted_data,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration — reads from env with sensible default
ORCHESTRATOR_MODEL = os.getenv(
    "ORCHESTRATOR_MODEL",
    "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
)


def _get_fireworks_llm() -> LLM:
    """
    Create a configured Fireworks LLM with proper timeout and retry settings.
    
    Uses ORCHESTRATOR_MODEL env var for model selection and FIREWORKS_API_KEY
    for authentication.  Timeout is set high (300s) to handle serverless
    cold-starts on the Fireworks shared-GPU tier.
    """
    model_name = ORCHESTRATOR_MODEL
    # LiteLLM requires the provider prefix. If the user copied the string directly
    # from the Fireworks UI (which starts with 'accounts/'), prepend the provider.
    if model_name.startswith("accounts/fireworks/"):
        model_name = f"fireworks_ai/{model_name}"

    return LLM(
        model=model_name,
        api_key=os.getenv("FIREWORKS_API_KEY"),
        timeout=300,      # 5 minutes — prevents premature client-side timeout
        max_retries=3,    # auto-retry on transient 408/503 errors
    )


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_extractor_agent() -> Agent:
    """
    Create the Extractor Agent.
    
    Responsible for:
    - Accepting medical document image paths
    - Calling FastAPI /extract endpoint (or mock API)
    - Returning raw JSON from VLM inference
    - Handling API errors with retry logic
    
    Returns:
        CrewAI Agent configured for extraction tasks.
    """
    from tools.pipeline_tools import vlm_api_client
    return Agent(
        role="Medical Data Extractor",
        goal=(
            "Extract structured medical data from document images using "
            "the `VLM_API_Client` tool exclusively. You MUST call this tool. "
            "Return the exact JSON produced by the tool, do not make up data."
        ),
        backstory=(
            "You are an expert medical document processor with deep knowledge "
            "of laboratory reports, diagnostic summaries, and clinical data "
            "formats. Your role is to take raw medical report strings and "
            "ensure they are properly structured."
        ),
        verbose=True,
        allow_delegation=False,
        llm=_get_fireworks_llm(),
    )


def create_validator_agent() -> Agent:
    """
    Create the Validator Agent.
    
    Responsible for:
    - Accepting raw JSON from Extractor
    - Validating against strict Pydantic schema
    - Cleaning and normalizing data
    - Detecting and handling validation errors
    - Returning production-ready JSON
    
    Returns:
        CrewAI Agent configured for validation tasks.
    """
    from tools.pipeline_tools import postgres_insert_tool
    return Agent(
        role="Medical Data Validator",
        goal=(
            "Ensure extracted medical data conforms to strict schema requirements. "
            "Validate, clean, and then YOU MUST insert data using `Postgres_Insert_Tool`."
        ),
        backstory=(
            "You are a data quality expert specializing in healthcare ETL pipelines. "
            "Your expertise includes medical data standards (HL7, FHIR), data "
            "normalization, and validation rule enforcement. You ensure that every "
            "data point meets clinical and regulatory compliance standards before "
            "entering the database."
        ),
        verbose=True,
        allow_delegation=False,
        llm=_get_fireworks_llm(),
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def create_medical_etl_crew() -> Crew:
    """
    Create the complete Medical ETL Crew.
    
    Orchestrates two agents in sequential process:
    1. Extractor: Get raw JSON from VLM
    2. Validator: Validate and clean the data
    
    Returns:
        CrewAI Crew instance ready to process documents.
        
    Example:
        >>> crew = create_medical_etl_crew()
        >>> result = crew.kickoff(inputs={"image_path": "page_0001.jpg"})
    """
    
    # Create agents
    extractor = create_extractor_agent()
    validator = create_validator_agent()
    
    # Create tasks
    extraction_task = create_extractor_task(extractor)
    validation_task = create_validator_task(validator)
    
    # Create crew with just the validator
    # Extraction is handled natively via API to improve reliability
    crew = Crew(
        agents=[validator],
        tasks=[validation_task],
        process=Process.sequential,
        verbose=True,
    )
    
    logger.info("✓ Medical ETL Crew created and ready for processing")
    return crew


# ============================================================================
# HIGH-LEVEL API FOR SPRINT 1 MVP
# ============================================================================

class MedicalETLPipeline:
    """
    High-level interface for medical document ETL processing.
    
    Encapsulates the CrewAI crew and provides a simple API for
    processing documents end-to-end.
    
    Example:
        >>> pipeline = MedicalETLPipeline()
        >>> result = pipeline.process_image("medical_report.jpg")
        >>> if result["success"]:
        ...     print(f"Patient: {result['data']['patient_details']['name']}")
        ... else:
        ...     print(f"Error: {result['error']}")
    """
    
    def __init__(self):
        """Initialize pipeline with CrewAI crew."""
        self.crew = create_medical_etl_crew()
        logger.info("MedicalETLPipeline initialized")
    
    def process_images(self, image_paths: list[str]) -> dict:
        """
        Process multiple medical document images in a single batch.
        
        Args:
            image_paths: List of paths to image files (JPEG, PNG, etc.)
            
        Returns:
            {
                "success": bool,
                "data": {...validated JSON...},
                "error": "error message if failed",
            }
        """
        for path in image_paths:
            if not Path(path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {path}",
                    "data": None,
                }
        
        logger.info(f"Processing {len(image_paths)} images in a single batch...")
        
        try:
            # 1. Direct Tool Invocation (Bypassing LLM orchestration)
            from tools.pipeline_tools import vlm_api_client, postgres_insert_tool
            
            logger.info("Invoking VLM API Client...")
            raw_json_str = vlm_api_client(image_paths)
            
            if "Error from VLM API" in raw_json_str or "Failed to call VLM API" in raw_json_str:
                 return {
                     "success": False,
                     "error": raw_json_str,
                     "data": None
                 }
                 
            logger.info("VLM API returned raw data. Handing off to Validator Agent...")
            
            # 2. Run validator crew with raw JSON input
            result = self.crew.kickoff(
                inputs={
                    "raw_json": raw_json_str,
                }
            )
            
            # Extract the actual text from the Crew output
            if hasattr(result, "raw"):
                clean_json_str = result.raw
            else:
                clean_json_str = str(result)
            
            logger.info("Validator Agent returned clean JSON. Inserting into Postgres...")
            
            # 3. Direct Postgres Insertion
            db_response = postgres_insert_tool(validated_json_str=clean_json_str)
            logger.info(f"Database tool response: {db_response}")
            
            logger.info(f"✓ Processing complete for batch of {len(image_paths)} images")
            
            return {
                "success": True,
                "data": clean_json_str,
                "error": None,
            }
        
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "data": None,
            }
    
    def process_pdf(self, pdf_path: str) -> dict:
        """
        Process a PDF by converting to images first, then extracting all at once.
        """
        from tools.pdf_processor import PDFProcessor, PDFProcessingError
        
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
            }
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            with PDFProcessor(dpi=300) as processor:
                image_paths, temp_dir = processor.process_pdf(str(pdf_path))
                
                logger.info(f"Converted PDF to {len(image_paths)} pages. Sending all at once to GPU...")
                
                return self.process_images(image_paths)
                
        except PDFProcessingError as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
            }
        
        except Exception as e:
            error_msg = f"Unexpected error during PDF processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
            }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Command-line entry point for testing the pipeline.
    
    Usage:
        python crew.py <image_or_pdf_path>
        python crew.py image.jpg
        python crew.py report.pdf
    """
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python crew.py <image_or_pdf_path>")
        print("  Example: python crew.py medical_report.jpg")
        print("  Example: python crew.py medical_report.pdf")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        pipeline = MedicalETLPipeline()
        
        if file_path.lower().endswith('.pdf'):
            result = pipeline.process_pdf(file_path)
        else:
            result = pipeline.process_images([file_path])
        
        # Print results
        if result["success"]:
            print("\n✓ Processing successful!")
            if "data" in result:
                print("\nExtracted Data:")
                print(json.dumps(result["data"], indent=2, default=str))
            elif "summary" in result:
                print("\nAggregated Data:")
                print(json.dumps(result["summary"], indent=2, default=str))
        else:
            print(f"\n✗ Processing failed: {result['error']}")
            sys.exit(1)
    
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
