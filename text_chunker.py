import hashlib
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config
from utils import setup_logger

logger = setup_logger("TextChunker")

def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create and configure the text splitter with settings from config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

def generate_chunk_id(content: str, file_hash: str, index: int) -> str:
    """
    Generate a deterministic, globally unique ID for a chunk.
    Incorporates file_hash for document isolation and index for order.
    """
    # Hash the content itself to ensure identity detection
    content_hash = hashlib.sha256(content.strip().encode()).hexdigest()
    raw_id = f"{file_hash}_{index}_{content_hash}"
    return hashlib.md5(raw_id.encode()).hexdigest()

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into semantic chunks with Enterprise safeguards:
    1. Filters blank/whitespace-only pages/extractions.
    2. Assigns deterministic IDs for incremental indexing.
    3. Retains technical metadata.
    """
    # 1. Filter empty documents at the source
    valid_docs = [d for d in documents if d.page_content and d.page_content.strip()]
    skipped_empty = len(documents) - len(valid_docs)
    
    if skipped_empty > 0:
        logger.info(f"Filtered {skipped_empty} empty document pages/extractions.")

    splitter = create_text_splitter()
    raw_chunks = splitter.split_documents(valid_docs)
    
    processed_chunks = []
    seen_content_hashes = set()
    
    for i, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        
        # 2. Strict content filter (post-splitting)
        if not content:
            continue
            
        file_name = chunk.metadata.get("file_name", "unknown")
        file_hash = chunk.metadata.get("file_hash", "no_hash")

        # 3. Deterministic chunk ID
        chunk_id = generate_chunk_id(content, file_hash, i)
        
        # Metadata enrichment
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i
        chunk.metadata["file_hash"] = file_hash
        
        # Update content with stripped version for cleaner embeddings
        chunk.page_content = content
        processed_chunks.append(chunk)

    logger.info(f"Processed {len(documents)} docs into {len(processed_chunks)} valid chunks (Skipped: {skipped_empty} empty, {len(raw_chunks) - len(processed_chunks)} small/empty chunks).")
    return processed_chunks
