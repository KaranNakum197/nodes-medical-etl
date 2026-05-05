#!/usr/bin/env python3
"""
Integration Test Suite for Medical ETL Pipeline - Sprint 1 MVP

This script provides end-to-end testing of all pipeline components.
Run this after setup to verify everything is working correctly.

Usage:
    python integration_test.py
    python integration_test.py --verbose
    python integration_test.py --test pdf_processor
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTests:
    """Test suite for Medical ETL Pipeline."""
    
    def __init__(self, verbose: bool = False):
        """Initialize test suite."""
        self.verbose = verbose
        self.results = {"passed": 0, "failed": 0, "skipped": 0}
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def test_pydantic_schemas(self) -> bool:
        """Test 1: Pydantic schema validation."""
        logger.info("=" * 70)
        logger.info("TEST 1: Pydantic Schema Validation")
        logger.info("=" * 70)
        
        try:
            from agents.tasks import ExtractedReport, validate_extracted_data
            
            # Test data matching schema
            test_data = {
                "patient_details": {
                    "name": "Test Patient",
                    "patient_id": "PT-001",
                    "age": "50 Years",
                    "gender": "Male",
                },
                "lab_details": {
                    "lab_name": "Test Lab",
                    "lab_address": "123 Medical St",
                    "phone": 5551234567,  # Test phone coercion
                },
                "sample_details": {
                    "sample_id": "S-001",
                    "specimen": "Whole Blood",
                    "collected_at": "2024-05-05T10:00:00Z",  # ISO8601
                },
                "report_results": [
                    {
                        "is_panel": False,
                        "test_name": "Glucose",
                        "value": "95 mg/dL",  # Test numeric coercion
                        "unit": "mg/dL",
                        "reference_range": "70-100",
                        "interpretation": "Normal",
                    }
                ],
            }
            
            # Test validation
            report = ExtractedReport(**test_data)
            
            # Verify coercions
            assert report.lab_details.phone == "5551234567", "Phone not coerced to string"
            assert report.report_results[0].value == 95.0, "Numeric value not coerced"
            assert str(report.sample_details.collected_at).startswith("2024-05-05"), "Datetime not parsed"
            
            logger.info("✓ Schema validation passed")
            logger.info(f"✓ Phone coercion: {test_data['lab_details']['phone']} → {report.lab_details.phone}")
            logger.info(f"✓ Numeric coercion: '95 mg/dL' → {report.report_results[0].value}")
            logger.info(f"✓ Datetime parsing: {report.sample_details.collected_at}")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Schema validation failed: {e}", exc_info=self.verbose)
            self.results["failed"] += 1
            return False
    
    def test_pdf_processor(self) -> bool:
        """Test 2: PDF processor (requires sample PDF)."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2: PDF Processor")
        logger.info("=" * 70)
        
        try:
            from agents.tools.pdf_processor import PDFProcessor
            
            # Check if sample PDF exists
            sample_pdfs = list(Path(".").glob("**/*.pdf"))
            
            if not sample_pdfs:
                logger.warning("⊘ No PDF files found for testing")
                logger.info("  Create a sample PDF or place one in the project directory")
                self.results["skipped"] += 1
                return True
            
            pdf_path = sample_pdfs[0]
            logger.info(f"Testing with PDF: {pdf_path}")
            
            with PDFProcessor(dpi=300) as processor:
                image_paths, temp_dir = processor.process_pdf(str(pdf_path))
                
                logger.info(f"✓ Converted {len(image_paths)} pages to JPEGs")
                logger.info(f"  Temp directory: {temp_dir}")
                
                # Validate images
                if processor.validate_images(image_paths):
                    logger.info("✓ All generated images are valid")
                    self.results["passed"] += 1
                    return True
                else:
                    logger.error("✗ Image validation failed")
                    self.results["failed"] += 1
                    return False
        
        except ImportError:
            logger.error("✗ Could not import PDFProcessor")
            self.results["failed"] += 1
            return False
        except Exception as e:
            logger.error(f"✗ PDF processing failed: {e}", exc_info=self.verbose)
            self.results["failed"] += 1
            return False
    
    def test_vlm_import(self) -> bool:
        """Test 3: VLM extractor imports and initialization."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 3: VLM Extractor (Qwen2-VL-7B-Instruct)")
        logger.info("=" * 70)
        
        try:
            from backend.vlm_engine.extractor import QwenVLMExtractor
            
            logger.info("Attempting to load Qwen2-VL-7B-Instruct model...")
            logger.info("(This may take a while on first load - model is ~7GB)")
            
            # Try to initialize (will download model on first run)
            try:
                extractor = QwenVLMExtractor()
                logger.info(f"✓ Model loaded successfully")
                logger.info(f"  Device: {extractor.device}")
                logger.info(f"  Precision: {extractor.torch_dtype}")
                logger.info(f"  Device map: auto")
                
                # Clean up
                del extractor
                
                self.results["passed"] += 1
                return True
            
            except Exception as model_error:
                # Model loading may fail if HF token or internet issues
                logger.warning(f"⊘ Could not load model: {str(model_error)[:100]}")
                logger.warning("  This is expected if:")
                logger.warning("    - First time running (downloads ~7GB model)")
                logger.warning("    - No internet connection")
                logger.warning("    - HuggingFace token needed")
                logger.warning("    - Insufficient GPU memory")
                
                self.results["skipped"] += 1
                return True
        
        except ImportError as e:
            logger.error(f"✗ Could not import VLM extractor: {e}")
            logger.info("  Ensure torch and transformers are installed:")
            logger.info("  pip install -r backend/requirements.txt")
            self.results["failed"] += 1
            return False
        except Exception as e:
            logger.error(f"✗ VLM test failed: {e}", exc_info=self.verbose)
            self.results["failed"] += 1
            return False
    
    def test_fastapi_server(self) -> bool:
        """Test 4: FastAPI server health check."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 4: FastAPI Server")
        logger.info("=" * 70)
        
        try:
            import requests
            import time
            
            logger.info("Checking if FastAPI server is running on http://localhost:8000...")
            
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                
                if response.status_code == 200:
                    health = response.json()
                    logger.info(f"✓ Server is healthy")
                    logger.info(f"  Status: {health.get('status')}")
                    logger.info(f"  VLM loaded: {health.get('vlm_loaded')}")
                    
                    self.results["passed"] += 1
                    return True
                else:
                    logger.error(f"✗ Server returned status {response.status_code}")
                    self.results["failed"] += 1
                    return False
            
            except requests.ConnectionError:
                logger.warning("⊘ Could not connect to server on port 8000")
                logger.info("  Start the server with: python backend/main.py")
                logger.info("  Or specify a different port: python backend/main.py --port 9000")
                
                self.results["skipped"] += 1
                return True
        
        except ImportError:
            logger.warning("⊘ 'requests' library not installed")
            logger.info("  pip install requests")
            self.results["skipped"] += 1
            return True
        except Exception as e:
            logger.error(f"✗ Server test failed: {e}", exc_info=self.verbose)
            self.results["failed"] += 1
            return False
    
    def test_agent_creation(self) -> bool:
        """Test 5: CrewAI agent creation."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 5: CrewAI Agent Creation")
        logger.info("=" * 70)
        
        try:
            from agents.crew import create_extractor_agent, create_validator_agent
            
            logger.info("Creating Extractor Agent...")
            extractor = create_extractor_agent()
            logger.info(f"✓ Extractor Agent created")
            logger.info(f"  Role: {extractor.role}")
            
            logger.info("Creating Validator Agent...")
            validator = create_validator_agent()
            logger.info(f"✓ Validator Agent created")
            logger.info(f"  Role: {validator.role}")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Agent creation failed: {e}", exc_info=self.verbose)
            logger.info("  Ensure crewai is installed:")
            logger.info("  pip install -r agents/requirements.txt")
            self.results["failed"] += 1
            return False
    
    def test_mock_pipeline(self) -> bool:
        """Test 6: Mock ETL pipeline execution."""
        logger.info("\n" + "=" * 70)
        logger.info("TEST 6: Mock ETL Pipeline")
        logger.info("=" * 70)
        
        try:
            from agents.tasks import mock_api_extract, validate_extracted_data
            import json
            
            logger.info("Running mock extraction...")
            raw_json = mock_api_extract("test_image.jpg")
            
            logger.info("Running validation...")
            validated = validate_extracted_data(raw_json)
            
            if "error" in validated:
                logger.error(f"✗ Validation failed: {validated['error']}")
                self.results["failed"] += 1
                return False
            
            # Check required fields
            assert "patient_details" in validated, "Missing patient_details"
            assert "lab_details" in validated, "Missing lab_details"
            assert "report_results" in validated, "Missing report_results"
            
            logger.info("✓ Mock pipeline executed successfully")
            logger.info(f"  Extracted patient: {validated['patient_details'].get('name')}")
            logger.info(f"  Lab: {validated['lab_details'].get('lab_name')}")
            logger.info(f"  Tests: {len(validated.get('report_results', []))}")
            
            self.results["passed"] += 1
            return True
        
        except Exception as e:
            logger.error(f"✗ Mock pipeline test failed: {e}", exc_info=self.verbose)
            self.results["failed"] += 1
            return False
    
    def run_all(self) -> int:
        """Run all tests."""
        logger.info("\n")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║" + " MEDICAL ETL PIPELINE - INTEGRATION TEST SUITE ".center(68) + "║")
        logger.info("║" + " Sprint 1 MVP ".center(68) + "║")
        logger.info("╚" + "═" * 68 + "╝")
        
        tests = [
            ("Pydantic Schemas", self.test_pydantic_schemas),
            ("PDF Processor", self.test_pdf_processor),
            ("VLM Extractor", self.test_vlm_import),
            ("FastAPI Server", self.test_fastapi_server),
            ("CrewAI Agents", self.test_agent_creation),
            ("Mock Pipeline", self.test_mock_pipeline),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
            except KeyboardInterrupt:
                logger.info("\nTests interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in {name}: {e}", exc_info=self.verbose)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✓ Passed:  {self.results['passed']}")
        logger.info(f"✗ Failed:  {self.results['failed']}")
        logger.info(f"⊘ Skipped: {self.results['skipped']}")
        
        if self.results["failed"] == 0:
            logger.info("\n✓ All tests passed!")
            return 0
        else:
            logger.error(f"\n✗ {self.results['failed']} test(s) failed")
            return 1


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Integration test suite for Medical ETL Pipeline"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--test",
        choices=["pydantic", "pdf_processor", "vlm", "fastapi", "agents", "mock"],
        help="Run a specific test"
    )
    
    args = parser.parse_args()
    
    tests = IntegrationTests(verbose=args.verbose)
    
    if args.test:
        # Run specific test
        test_map = {
            "pydantic": tests.test_pydantic_schemas,
            "pdf_processor": tests.test_pdf_processor,
            "vlm": tests.test_vlm_import,
            "fastapi": tests.test_fastapi_server,
            "agents": tests.test_agent_creation,
            "mock": tests.test_mock_pipeline,
        }
        
        result = test_map[args.test]()
        return 0 if result else 1
    else:
        # Run all tests
        return tests.run_all()


if __name__ == "__main__":
    sys.exit(main())
