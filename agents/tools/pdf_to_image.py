"""
PDFToImageTool – CrewAI custom tool that converts a PDF file to a list of
image file paths (one PNG per page) so they can be sent to the VLM backend.
"""

from __future__ import annotations

import os
import tempfile
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class PDFToImageInput(BaseModel):
    """Input schema for PDFToImageTool."""

    pdf_path: str = Field(..., description="Absolute path to the PDF file.")
    output_dir: str = Field(
        default="",
        description=(
            "Directory where JPEG images will be saved. "
            "Defaults to a temporary directory."
        ),
    )
    dpi: int = Field(default=300, description="Resolution for PDF rendering (DPI). Medical docs recommend 300.")


class PDFToImageTool(BaseTool):
    """
    Converts a PDF document to a list of JPEG images.

    Each page of the PDF becomes a separate JPEG file.
    Returns a newline-separated list of absolute image paths.
    """

    name: str = "PDFToImageConverter"
    description: str = (
        "Converts a PDF file to JPEG images (one per page). "
        "Returns a newline-separated list of absolute paths to the generated images."
    )
    args_schema: Type[BaseModel] = PDFToImageInput

    def _run(
        self,
        pdf_path: str,
        output_dir: str = "",
        dpi: int = 300,
    ) -> str:
        """Execute the PDF-to-image conversion."""
        from pdf2image import convert_from_path

        if not os.path.isfile(pdf_path):
            return f"ERROR: File not found – {pdf_path}"

        save_dir = output_dir if output_dir else tempfile.mkdtemp(prefix="pdf_pages_")
        os.makedirs(save_dir, exist_ok=True)

        pages = convert_from_path(pdf_path, dpi=dpi)
        image_paths: list[str] = []

        for idx, page in enumerate(pages, start=1):
            img_path = os.path.join(save_dir, f"page_{idx:04d}.jpg")
            page.save(img_path, "JPEG", quality=95, optimize=True)
            image_paths.append(img_path)

        return "\n".join(image_paths)
