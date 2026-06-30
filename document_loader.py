import os
from typing import List
from langchain_core.documents import Document
import shutil
import config
import logging
from utils import setup_logger, sanitize_filename

# Import specialized loaders
from pdf_loader import load_pdf_multimodal
from image_loader import load_image_multimodal
from docx_loader import load_docx
from pptx_loader import load_pptx

logger = setup_logger("DocumentLoader")

def save_uploaded_file(uploaded_file, name: str) -> str:
    """Save an uploaded file to the data directory with security checks."""
    safe_name = sanitize_filename(name)
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (Max: {config.MAX_FILE_SIZE_MB}MB)")

    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        
    file_path = os.path.join(config.DATA_DIR, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved secure file: {file_path}")
    return file_path

def load_any_document(file_path: str) -> List[Document]:
    """
    Dispatcher that selects the correct loader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    logger.info(f"Loading document: {os.path.basename(file_path)} (Ext: {ext})")
    
    try:
        if ext == ".pdf":
            return load_pdf_multimodal(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return load_image_multimodal(file_path)
        elif ext == ".docx":
            return load_docx(file_path)
        elif ext == ".pptx":
            return load_pptx(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return []

def load_multiple_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple documents and combine them into one list."""
    all_documents = []
    for path in file_paths:
        docs = load_any_document(path)
        if docs:
            all_documents.extend(docs)
    logger.info(f"Successfully loaded a total of {len(all_documents)} document pages/segments.")
    return all_documents
