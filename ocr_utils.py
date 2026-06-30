import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import config
import os
import logging
from utils import setup_logger

logger = setup_logger("OCR")

# Configure tesseract path
if os.path.exists(config.TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
else:
    logger.warning(f"Tesseract path not found: {config.TESSERACT_PATH}. OCR might fail.")

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    """
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Resize if too small (OCR likes at least 300 DPI equivalent)
    if image.width < 1000:
        ratio = 2000 / image.width
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    return image

def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from an image using pytesseract.
    """
    try:
        processed_img = preprocess_image(image)
        text = pytesseract.image_to_string(processed_img)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""

def is_scanned_pdf(text_content: str, threshold: int = 50) -> bool:
    """
    Helper to determine if a PDF page is likely scanned based on text length.
    """
    return len(text_content.strip()) < threshold
