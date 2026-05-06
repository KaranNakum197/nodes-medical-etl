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
from pathlib import Path

from crewai import Agent, Crew, Process

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
            "Vision Language Model inference. Return clean, parseable JSON."
        ),
        backstory=(
            "You are an expert medical document processor with deep knowledge "
            "of laboratory reports, diagnostic summaries, and clinical data "
            "formats. Your role is to interface with the VLM inference engine "
            "and reliably extract data from medical documents, handling edge "
            "cases and image quality issues gracefully."
        ),
        tools=[vlm_api_client],
        verbose=True,
        allow_delegation=False,
        llm="fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "Validate, clean, and prepare data for database ingestion."
        ),
        backstory=(
            "You are a data quality expert specializing in healthcare ETL pipelines. "
            "Your expertise includes medical data standards (HL7, FHIR), data "
            "normalization, and validation rule enforcement. You ensure that every "
            "data point meets clinical and regulatory compliance standards before "
            "entering the database."
        ),
        tools=[postgres_insert_tool],
        verbose=True,
        allow_delegation=False,
        llm="fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
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
    
    # Create crew with sequential process
    crew = Crew(
        agents=[extractor, validator],
        tasks=[extraction_task, validation_task],
        process=Process.sequential,  # Tasks run in order
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
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single medical document image.
        
        Args:
            image_path: Path to image file (JPEG, PNG, etc.)
            
        Returns:
            {
                "success": bool,
                "data": {...validated JSON...},
                "error": "error message if failed",
                "logs": "crew execution logs"
            }
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "data": None,
            }
        
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Run crew with image path input
            result = self.crew.kickoff(
                inputs={
                    "image_path": str(image_path),
                }
            )
            
            logger.info(f"✓ Processing complete for {image_path.name}")
            
            return {
                "success": True,
                "data": result,
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
        Process a PDF by converting to images first.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            {
                "success": bool,
                "results": [
                    {page_num: {...data...}, "error": "...", ...}
                ],
                "summary": {...aggregated results...}
            }
        """
        from tools.pdf_processor import PDFProcessor, PDFProcessingError
        
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
            }
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        results = []
        aggregated_data = {
            "patient_details": {},
            "lab_details": {},
            "sample_details": {},
            "report_results": [],
        }
        
        try:
            # Convert PDF to images
            with PDFProcessor(dpi=300) as processor:
                image_paths, temp_dir = processor.process_pdf(str(pdf_path))
                
                # Process each page
                for page_num, image_path in enumerate(image_paths, start=1):
                    logger.info(f"Processing page {page_num}/{len(image_paths)}")
                    
                    page_result = self.process_image(image_path)
                    results.append({
                        "page": page_num,
                        **page_result
                    })
                    
                    # Aggregate results
                    if page_result["success"] and page_result["data"]:
                        self._aggregate_results(
                            aggregated_data,
                            page_result["data"]
                        )
            
            logger.info(f"✓ PDF processing complete: {len(image_paths)} pages")
            
            return {
                "success": True,
                "results": results,
                "summary": aggregated_data,
            }
        
        except PDFProcessingError as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "results": results,
            }
        
        except Exception as e:
            error_msg = f"Unexpected error during PDF processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "results": results,
            }
    
    @staticmethod
    def _aggregate_results(aggregated: dict, page_data: dict) -> None:
        """
        Aggregate results from multiple pages.
        
        Merges patient/lab/sample details and concatenates test results.
        
        Args:
            aggregated: Running aggregation dictionary.
            page_data: Data from current page.
        """
        # Merge patient details (use first non-empty)
        if page_data.get("patient_details"):
            if not aggregated["patient_details"]:
                aggregated["patient_details"] = page_data["patient_details"]
        
        # Merge lab details
        if page_data.get("lab_details"):
            if not aggregated["lab_details"]:
                aggregated["lab_details"] = page_data["lab_details"]
        
        # Merge sample details
        if page_data.get("sample_details"):
            aggregated["sample_details"].update(page_data["sample_details"])
        
        # Concatenate test results
        if page_data.get("report_results"):
            aggregated["report_results"].extend(page_data["report_results"])


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
            result = pipeline.process_image(file_path)
        
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
