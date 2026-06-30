import fitz  # PyMuPDF
from PIL import Image
import io
import pdfplumber
from langchain_core.documents import Document
from ocr_utils import extract_text_from_image, is_scanned_pdf
from vision_utils import analyze_image_with_vision
import os
import logging
from utils import setup_logger

logger = setup_logger("PDFLoader")

def extract_tables_from_page(file_path: str, page_num: int) -> str:
    """
    Extract tables from a specific page using pdfplumber.
    Returns a markdown-formatted string of tables.
    """
    tables_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            for i, table in enumerate(tables):
                tables_text += f"\n\n--- TABLE {i+1} ---\n"
                # Clean table data
                df_text = ""
                for row in table:
                    # Filter out None and join with pipe for markdown-like look
                    clean_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                    df_text += "| " + " | ".join(clean_row) + " |\n"
                tables_text += df_text
    except Exception as e:
        logger.error(f"Table extraction error on page {page_num}: {e}")
    return tables_text

def load_pdf_multimodal(file_path: str) -> list[Document]:
    """
    Load PDF using PyMuPDF for both text extraction and OCR if needed.
    Also extracts tables using pdfplumber.
    """
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        doc = fitz.open(file_path)
        logger.info(f"PDF {file_name} opened. total pages: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            
            # Metadata for the page
            metadata = {
                "source": file_name,
                "file_name": file_name,
                "page_number": page_num + 1,
                "file_type": "pdf",
                "method": "text_extraction"
            }
            
            # 1. Handle Tables
            tables_content = extract_tables_from_page(file_path, page_num)
            if tables_content.strip():
                logger.info(f"Tables found on {file_name} page {page_num+1}")
                table_metadata = metadata.copy()
                table_metadata["content_type"] = "table"
                documents.append(Document(page_content=tables_content, metadata=table_metadata))
            
            # 2. Handle Text / OCR
            if is_scanned_pdf(text):
                logger.info(f"Page {page_num + 1} of {file_name} looks scanned. Using OCR/Vision...")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # High res for OCR
                img = Image.open(io.BytesIO(pix.tobytes()))
                
                ocr_text = extract_text_from_image(img)
                vision_text = analyze_image_with_vision(img)
                
                full_text = f"--- OCR TEXT ---\n{ocr_text}\n\n--- VISUAL ANALYSIS ---\n{vision_text}"
                metadata["method"] = "ocr_vision"
                metadata["content_type"] = "scanned_page"
                documents.append(Document(page_content=full_text, metadata=metadata))
            else:
                # Normal text PDF
                metadata["content_type"] = "text"
                documents.append(Document(page_content=text, metadata=metadata))
                
                # 3. Handle Embedded Images
                image_list = page.get_images(full=True)
                if image_list:
                    logger.info(f"Found {len(image_list)} embedded images on {file_name} page {page_num+1}")
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            img = Image.open(io.BytesIO(image_bytes))
                            
                            vision_desc = analyze_image_with_vision(img)
                            img_metadata = metadata.copy()
                            img_metadata["content_type"] = "embedded_image"
                            img_metadata["image_index"] = img_index
                            documents.append(Document(page_content=vision_desc, metadata=img_metadata))
                        except Exception as e:
                            logger.error(f"Error processing embedded image {img_index}: {e}")
                            continue
                            
        doc.close()
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}", exc_info=True)
        
    return documents
