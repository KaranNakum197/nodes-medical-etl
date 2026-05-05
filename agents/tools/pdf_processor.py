"""
PDF to Image Processor - Converts multi-page medical PDFs to high-resolution JPEGs.

This module handles robust PDF-to-image conversion at 300 DPI for downstream
VLM processing. Includes error handling, temporary directory management,
and validation for medical document compliance.

Dependencies:
    - pdf2image
    - Pillow
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from pdf2image import convert_from_path
from PIL import Image


# Configure logging for debugging and production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing failures."""
    pass


class PDFProcessor:
    """
    Robust PDF-to-image converter optimized for medical document extraction.
    
    Attributes:
        dpi (int): Resolution in dots per inch (default: 300 for medical docs).
        temp_dir (str): Temporary directory for image storage.
        fmt (str): Output image format (default: 'jpeg').
    """
    
    def __init__(self, dpi: int = 300, temp_dir: Optional[str] = None):
        """
        Initialize PDF processor with configurable DPI and temp storage.
        
        Args:
            dpi: Resolution in DPI (default: 300).
            temp_dir: Custom temp directory (default: system temp).
            
        Raises:
            ValueError: If DPI < 72 or temp_dir is invalid.
        """
        if dpi < 72:
            raise ValueError(f"DPI must be >= 72, got {dpi}")
        
        self.dpi = dpi
        self.fmt = "jpeg"
        
        # Use custom or system temp directory
        if temp_dir:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            self.temp_dir = temp_dir
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="pdf_processing_")
        
        logger.info(f"PDFProcessor initialized: DPI={dpi}, temp_dir={self.temp_dir}")
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[str], str]:
        """
        Convert a multi-page PDF to high-resolution JPEGs.
        
        Args:
            pdf_path: Path to the input PDF file.
            
        Returns:
            Tuple of:
                - List of full paths to generated JPEG images (ordered by page).
                - Path to temporary directory containing images.
                
        Raises:
            PDFProcessingError: If PDF is invalid, unreadable, or conversion fails.
            FileNotFoundError: If PDF file does not exist.
            
        Example:
            >>> processor = PDFProcessor(dpi=300)
            >>> images, temp_dir = processor.process_pdf("report.pdf")
            >>> for img_path in images:
            ...     print(f"Processed: {img_path}")
        """
        pdf_path = Path(pdf_path)
        
        # Validate input file
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != ".pdf":
            raise PDFProcessingError(f"File must be PDF format, got: {pdf_path.suffix}")
        
        if pdf_path.stat().st_size == 0:
            raise PDFProcessingError("PDF file is empty")
        
        try:
            logger.info(f"Processing PDF: {pdf_path} at {self.dpi} DPI")
            
            # Convert PDF pages to PIL Image objects
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt=self.fmt
            )
            
            if not images:
                raise PDFProcessingError("No pages extracted from PDF")
            
            logger.info(f"Extracted {len(images)} page(s) from PDF")
            
            # Save images to temp directory with zero-padded names
            saved_paths = []
            for page_num, image in enumerate(images, start=1):
                # Ensure consistent naming for multi-page documents
                filename = f"page_{page_num:04d}.{self.fmt}"
                filepath = os.path.join(self.temp_dir, filename)
                
                # Optimize for storage without quality loss at 300 DPI
                image.save(filepath, format='JPEG', quality=95, optimize=True)
                saved_paths.append(filepath)
                logger.debug(f"Saved: {filepath} ({image.size} pixels)")
            
            logger.info(f"Successfully processed {len(saved_paths)} images to {self.temp_dir}")
            return saved_paths, self.temp_dir
        
        except Exception as e:
            # Wrap non-custom exceptions
            if isinstance(e, PDFProcessingError):
                raise
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PDFProcessingError(error_msg) from e
    
    def cleanup(self) -> None:
        """
        Remove temporary directory and all generated images.
        
        WARNING: This permanently deletes all images in temp_dir.
        Only call when extraction and VLM processing is complete.
        """
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def validate_images(self, image_paths: List[str]) -> bool:
        """
        Validate generated images for corruption or incomplete processing.
        
        Args:
            image_paths: List of paths to validate.
            
        Returns:
            True if all images are valid, False otherwise.
        """
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                logger.debug(f"Validated: {path}")
            except Exception as e:
                logger.error(f"Image validation failed for {path}: {e}")
                return False
        return True
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup on scope exit."""
        self.cleanup()
        return False


def process_medical_pdf(pdf_path: str, dpi: int = 300) -> Tuple[List[str], str]:
    """
    Convenience function for one-shot PDF processing with automatic cleanup.
    
    Args:
        pdf_path: Path to input PDF.
        dpi: Resolution (default: 300 DPI).
        
    Returns:
        Tuple of (image_paths, temp_directory).
        
    Note:
        Directory is NOT auto-cleaned. Call cleanup() when done.
    """
    processor = PDFProcessor(dpi=dpi)
    return processor.process_pdf(pdf_path)


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file>")
        sys.exit(1)
    
    try:
        with PDFProcessor(dpi=300) as processor:
            images, temp_dir = processor.process_pdf(sys.argv[1])
            print(f"✓ Processed {len(images)} pages to {temp_dir}")
            for img in images:
                print(f"  - {img}")
    except PDFProcessingError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
