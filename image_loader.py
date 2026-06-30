from PIL import Image
from langchain_core.documents import Document
from ocr_utils import extract_text_from_image
from vision_utils import analyze_image_with_vision
import os
import logging
from utils import setup_logger

logger = setup_logger("ImageLoader")

def load_image_multimodal(file_path: str) -> list[Document]:
    """
    Process an image using OCR and Gemini Vision.
    """
    file_name = os.path.basename(file_path)
    documents = []
    
    try:
        logger.info(f"Processing image: {file_name}")
        img = Image.open(file_path)
        
        # 1. OCR for text extraction
        ocr_text = extract_text_from_image(img)
        
        # 2. Vision for deep understanding
        vision_text = analyze_image_with_vision(img)
        
        combined_content = (
            f"--- OCR TEXT FROM IMAGE ---\n{ocr_text}\n\n"
            f"--- VISUAL DESCRIPTION & ANALYSIS ---\n{vision_text}"
        )
        
        metadata = {
            "source": file_name,
            "file_name": file_name,
            "page_number": 1,
            "file_type": "image",
            "content_type": "image_analysis",
            "method": "ocr_vision"
        }
        
        documents.append(Document(page_content=combined_content, metadata=metadata))
        logger.info(f"Successfully processed image {file_name}")
        
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        
    return documents
