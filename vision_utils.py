import google.generativeai as genai
from PIL import Image
import config
from typing import Optional
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from utils import setup_logger

logger = setup_logger("VisionUtils")

@retry(
    wait=wait_exponential(multiplier=config.RETRY_MIN_SECONDS, max=config.RETRY_MAX_SECONDS),
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _analyze_image_with_retry(model, prompt, image):
    return model.generate_content([prompt, image])

def analyze_image_with_vision(image: Image.Image, prompt: Optional[str] = None) -> str:
    """
    Use Gemini Vision (1.5 Flash) to describe an image, diagram, or flowchart.
    Includes robust retries for production usage.
    """
    if not config.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY missing in config")
        return "[Vision Error: No API Key]"

    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        if not prompt:
            prompt = (
                "Describe this image in detail for a RAG knowledge base. "
                "Include: "
                "1. Main objects and subjects. "
                "2. Labels, text, and annotations visible. "
                "3. If it is a flowchart: describe the steps, arrows, and logic flow. "
                "4. If it is a diagram: describe relationships and components. "
                "5. If it is a table: extract the data in a markdown format. "
                "6. If it is a chart/graph: describe the data trends and values. "
                "Return a clean, searchable text summary."
            )
            
        logger.info("Calling Gemini Vision API...")
        response = _analyze_image_with_retry(model, prompt, image)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Vision Analysis Error: {e}")
        return f"[Vision Analysis Failed: {str(e)}]"
