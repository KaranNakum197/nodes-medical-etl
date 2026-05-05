"""
Qwen2-VL inference module – ROCm optimised for AMD MI300X.

Usage:
    from vlm_engine.inference import run_inference
    result = run_inference(file_bytes, filename="report.pdf")
"""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image

# ---------------------------------------------------------------------------
# Model loading (lazy – only once)
# ---------------------------------------------------------------------------
_model = None
_processor = None


def _load_model():
    """Load Qwen2-VL model and processor (called once on first request)."""
    global _model, _processor  # noqa: PLW0603

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    import torch

    model_id = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
    device = os.getenv("VLM_DEVICE", "cuda")  # ROCm exposes GPUs as "cuda"

    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    _model.eval()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "You are a medical document parser. "
    "Extract ALL structured information from this document page and return it as JSON. "
    "Include: patient_name, date_of_birth, diagnosis, medications, lab_results, "
    "physician_name, visit_date. Use null for missing fields."
)


def run_inference(file_bytes: bytes, filename: str = "") -> dict[str, Any]:
    """
    Run Qwen2-VL inference on a medical document.

    Parameters
    ----------
    file_bytes:
        Raw bytes of a PDF or image file.
    filename:
        Original filename (used to detect PDF vs. image).

    Returns
    -------
    dict
        Parsed structured data extracted from the document.
    """
    global _model, _processor  # noqa: PLW0603

    if _model is None:
        _load_model()

    images = _bytes_to_images(file_bytes, filename)

    import json
    import torch

    results: list[dict[str, Any]] = []
    for page_image in images:
        inputs = _processor(
            text=EXTRACTION_PROMPT,
            images=page_image,
            return_tensors="pt",
        ).to(_model.device)

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        output_text = _processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        try:
            page_data = json.loads(output_text)
        except json.JSONDecodeError:
            page_data = {"raw_text": output_text}

        results.append(page_data)

    return results[0] if len(results) == 1 else {"pages": results}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_images(file_bytes: bytes, filename: str) -> list[Image.Image]:
    """Convert raw bytes to a list of PIL Images (one per page for PDFs)."""
    if filename.lower().endswith(".pdf"):
        from pdf2image import convert_from_bytes

        return convert_from_bytes(file_bytes)

    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
