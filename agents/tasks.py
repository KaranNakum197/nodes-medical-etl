"""
Task definitions for the medical ETL CrewAI pipeline.

Two tasks are defined:
1. extraction_task  – calls the VLM backend and returns raw JSON.
2. validation_task  – validates and enriches the extracted JSON.
"""

from __future__ import annotations

from crewai import Agent, Task

from tools.pdf_to_image import PDFToImageTool

# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_tasks(
    extractor: Agent,
    validator: Agent,
    document_path: str,
) -> list[Task]:
    """
    Build and return the ordered list of ETL tasks.

    Parameters
    ----------
    extractor:
        The extractor agent instance.
    validator:
        The validator agent instance.
    document_path:
        Path to the medical document being processed.
    """
    pdf_tool = PDFToImageTool()

    extraction_task = Task(
        description=(
            f"Process the medical document at '{document_path}'.\n"
            "1. If the file is a PDF, convert each page to an image using the "
            "   PDFToImageTool.\n"
            "2. Send each image to the backend /extract endpoint.\n"
            "3. Aggregate the per-page results into a single JSON object.\n"
            "Return ONLY the final JSON – no extra commentary."
        ),
        expected_output=(
            "A JSON object containing: patient_name, date_of_birth, diagnosis, "
            "medications (list), lab_results (dict), physician_name, visit_date."
        ),
        agent=extractor,
        tools=[pdf_tool],
    )

    validation_task = Task(
        description=(
            "You will receive the raw extraction JSON from the previous task.\n"
            "Perform the following validations and corrections:\n"
            "  • Ensure all date fields follow ISO-8601 (YYYY-MM-DD).\n"
            "  • Verify that medications is a list (even if only one item).\n"
            "  • Replace any empty string values with null.\n"
            "  • Add a 'validation_status' field: 'passed' or 'needs_review'.\n"
            "  • Add a 'validation_notes' list describing any issues found.\n"
            "Return the corrected, annotated JSON."
        ),
        expected_output=(
            "The corrected JSON object with 'validation_status' and "
            "'validation_notes' fields added."
        ),
        agent=validator,
        context=[extraction_task],
    )

    return [extraction_task, validation_task]
