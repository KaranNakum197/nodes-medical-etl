"""
VLM Engine - Qwen2-VL-7B-Instruct Inference for Medical Document Extraction.

This module loads and executes inference on a Vision Language Model optimized
for ROCm (AMD GPU compute). Uses bfloat16 precision and auto device mapping
for efficient GPU memory utilization.

Dependencies:
    - torch
    - transformers (Hugging Face)
    - Pillow
    - qwen-vl (model download from HuggingFace)

Model: Qwen/Qwen2-VL-7B-Instruct
License: Model license agreement from Alibaba Qwen team
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image


# Configure logging for inference monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLMInferenceError(Exception):
    """Custom exception for VLM inference failures."""
    pass


class QwenVLMExtractor:
    """
    Vision Language Model for medical document data extraction.
    
    Uses Qwen2-VL-7B-Instruct with bfloat16 precision and auto device mapping
    optimized for ROCm (AMD GPU). Handles image loading, prompt construction,
    and JSON response parsing with validation.
    
    Attributes:
        model_id: HuggingFace model identifier.
        device_map: PyTorch device allocation strategy ('auto' for ROCm).
        torch_dtype: Precision (torch.bfloat16 for ROCm efficiency).
    """
    
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Medical data extraction prompt template
    EXTRACTION_PROMPT = """
    You are an expert medical data extractor. 
    Analyze the provided medical lab report image and extract all visible data.
    
    CRITICAL INSTRUCTIONS:
    1. Extract the REAL DATA from the image.
    2. Do NOT hallucinate or guess data. Never use fake names like "John Doe" or fake IDs like "PAT123".
    3. If a value is missing from the image or illegible, COMPLETELY OMIT that field from your JSON. Do NOT output null, do NOT output "string" or "alphanumeric".
    4. Return EXACTLY one JSON object using the structure below, populated with the ACTUAL text found in the image.
    
    JSON SCHEMA STRUCTURE (Use this structure, but replace the right-hand values with REAL data from the image):
    {
    "patient_details": {
        "patient_id": "alphanumeric",
        "name": "string",
        "age": "string (Years)",
        "gender": "string (Male/Female)",
        "address": "string"
        },
    "lab_details": {
        "lab_name": "string",
        "lab_address": "string",
        "lab_website": "string",
        "lab_code": "string",
        "email": "string",
        "phone": "numeric",
        "referred_by": "string"
        },
    "sample_details": {
        "sample_id": "alphanumeric",
        "specimen": "string",
        "collected_at": "ISO8601_DateTime",
        "received_at": "ISO8601_DateTime",
        "reported_at": "ISO8601_DateTime"
        },
    "report_results": [
        {
        "is_panel": "boolean",
        "test_name": "string",
        "sample_type": "string",
        "method": "string",
        "technology": "string",
        "value": "numeric",
        "value_text": "string",
        "unit": "string",
        "reference_range": "string",
        "interpretation": "string | null (High/Low/Normal)",
        "test_remarks": "string", 
        "test_notes": "string",
        "test_interpretations": "string",
        "extra_details": "object", 
        "members": [
            {
                "is_panel": "boolean",
                "test_name": "string",
                "value": "numeric",
                "value_text": "string",
                "unit": "string",
                "reference_range": "string",
                "interpretation": "string | null (High/Low/Normal)",
                "sample_type": "string",
                "method": "string",
                "test_remarks": "string",
                "extra_details": "object",
                "members": []
            }
        ]
        }
    ],
    "result_details": "string",
    "global_remarks": "string",
    "global_notes": "string",
    "extra_details": "object" 
    }

    Return ONLY valid JSON. Do not include trailing commas.
    """
    
    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize VLM with model loading and device mapping.
        
        Args:
            model_id: Override default model ID if needed.
            
        Raises:
            VLMInferenceError: If model loading fails or GPU unavailable.
        """
        self.model_id = model_id or self.MODEL_ID
        
        # Verify ROCm/CUDA availability
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA not available. Model will load on CPU (slow). "
                "For production, use ROCm or NVIDIA GPU."
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.info(
            f"Loading VLM: {self.model_id} | "
            f"Device: {self.device} | Dtype: {self.torch_dtype}"
        )
        
        try:
            # Force device_map="cuda" instead of "auto" 
            # Accelerate sometimes misreads AMD ROCm memory and silently falls back to CPU
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map="cuda",
                trust_remote_code=True,  # Required for Qwen models
                low_cpu_mem_usage=True,  # Minimize CPU memory during loading
                attn_implementation="eager",  # Critical fix for ROCm PyTorch SDPA
            )
            
            # Load processor for image/text tokenization
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            logger.info(f"✓ VLM loaded successfully on {self.device}")
            
            # Log model info for debugging
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.debug(f"Model parameters: {total_params:,}")
        
        except Exception as e:
            error_msg = f"Failed to load VLM {self.model_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VLMInferenceError(error_msg) from e
    
    def extract_from_images(
        self,
        image_paths: list[str],
        custom_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        """
        Extract medical data from multiple images in chronological order using VLM inference.
        
        Args:
            image_paths: List of paths to image files (JPEG, PNG, etc.).
            custom_prompt: Override default extraction prompt.
            max_tokens: Maximum tokens in generation (default: 4096).
            temperature: Sampling temperature for generation (default: 0.2, deterministic).
            
        Returns:
            Raw VLM output text (should be JSON).
            
        Raises:
            VLMInferenceError: If inference fails or image invalid.
            FileNotFoundError: If image file not found.
            
        Example:
            >>> extractor = QwenVLMExtractor()
            >>> result = extractor.extract_from_images(["page_1.jpg", "page_2.jpg"])
            >>> data = json.loads(result)
        """
        image_path = Path(image_path)
        
        
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        processed_images = []
        
        try:
            logger.info(f"Extracting data from {len(image_paths)} images")
            
            for path_str in image_paths:
                img_path = Path(path_str)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                if img_path.suffix.lower() not in valid_formats:
                    raise VLMInferenceError(f"Unsupported image format: {img_path.suffix}")
                
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                # Prevent OOM by capping maximum resolution per image
                max_size = 1536
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
                    image = image.resize(new_size, resample_filter)
                    
                processed_images.append(image)
            
            logger.debug(f"Loaded {len(processed_images)} images.")
            
            # Use custom prompt or default
            prompt = custom_prompt or self.EXTRACTION_PROMPT
            
            # Prepare inputs for model
            content = [{"type": "image", "image": img} for img in processed_images]
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Tokenize with processor
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=processed_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            logger.debug(f"Inputs prepared: {inputs.input_ids.shape}")
            
            # Inference with generation parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=False if temperature == 0 else True,
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Inference complete. Output length: {len(generated_text)} chars")
            logger.debug(f"Raw output: {generated_text[:200]}...")
            
            return generated_text
        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, VLMInferenceError)):
                raise
            error_msg = f"Inference failed for images: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VLMInferenceError(error_msg) from e
    
    def extract_json(
        self,
        image_paths: list[str],
        custom_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Extract and parse JSON from multi-image inference output.
        
        Attempts to extract valid JSON from VLM output, handling common
        formatting issues (trailing text, invalid syntax).
        
        Args:
            image_paths: List of paths to image files.
            custom_prompt: Override default extraction prompt.
            max_tokens: Maximum tokens in generation.
            
        Returns:
            Parsed JSON dictionary.
            
        Raises:
            VLMInferenceError: If JSON parsing fails.
            
        Example:
            >>> extractor = QwenVLMExtractor()
            >>> data = extractor.extract_json(["page_1.jpg", "page_2.jpg"])
            >>> print(data["patient_details"]["name"])
        """
        try:
            raw_output = self.extract_from_images(
                image_paths,
                custom_prompt=custom_prompt,
                max_tokens=max_tokens,
            )
            
            # Try direct JSON parsing first
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                logger.warning("Direct JSON parsing failed, attempting recovery")
                
                # Fallback: Extract JSON object from text
                json_str = self._extract_json_from_text(raw_output)
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise VLMInferenceError(
                            f"Invalid JSON even after extraction: {str(e)[:100]}"
                        ) from e
                else:
                    raise VLMInferenceError(
                        "No JSON object found in VLM output"
                    )
        
        except VLMInferenceError:
            raise
        except Exception as e:
            error_msg = f"JSON extraction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VLMInferenceError(error_msg) from e
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract JSON object from text with error recovery.
        
        Finds first '{' and last '}' to isolate JSON object.
        Removes trailing text (VLM often adds explanations).
        
        Args:
            text: Raw VLM output text.
            
        Returns:
            JSON string or None if not found.
        """
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        
        return None
    
    def __del__(self):
        """Cleanup on object deletion."""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
            logger.debug("Model unloaded and GPU cache cleared")


def get_extractor() -> QwenVLMExtractor:
    """
    Singleton-like factory for VLM extractor.
    
    Reuse this to avoid reloading the model for multiple inferences.
    """
    if not hasattr(get_extractor, '_instance'):
        get_extractor._instance = QwenVLMExtractor()
    return get_extractor._instance


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <image_file>")
        sys.exit(1)
    
    try:
        extractor = QwenVLMExtractor()
        result = extractor.extract_json(sys.argv[1:])
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
