"""
============================================================
Utility Functions Module
============================================================

Contains helper functions used across the application:
  - File validation
  - Display formatting
  - Session management
  - Logging helpers

These utilities keep the main modules focused on their
core responsibilities (Single Responsibility Principle).
============================================================
"""

import os
import time
import logging
import hashlib
import json
import re
from datetime import datetime, timezone

import config


def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(config.LOG_LEVEL)
        
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


logger = setup_logger("RAG_Utils")


def validate_config() -> bool:
    """Validates the system environment and configuration on startup."""
    missing = []
    if not config.GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    
    if missing:
        msg = f"CRITICAL: Missing environment variables: {', '.join(missing)}"
        logger.error(msg)
        return False
    
    # Ensure directories exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    logger.info("System configuration validated successfully.")
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to prevent path traversal and shell injection."""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\.-]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename


def calculate_file_hash(file_path: str) -> str:
    """Calculate the SHA-256 hash of a file for change detection."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_indexed_hashes() -> dict[str, dict]:
    """Load the map of indexed file hashes from disk."""
    if not os.path.exists(config.HASH_TRACKER_FILE):
        return {}
    try:
        with open(config.HASH_TRACKER_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading hash tracker: {e}")
        return {}


def save_indexed_hash(file_hash: str, metadata: dict):
    """Save a file hash and its metadata to the tracker."""
    hashes = get_indexed_hashes()
    hashes[file_hash] = metadata
    os.makedirs(os.path.dirname(config.HASH_TRACKER_FILE), exist_ok=True)
    with open(config.HASH_TRACKER_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def validate_pdf_file(filename: str, file_size: int) -> tuple[bool, str]:
    """
    Validate an uploaded PDF file.

    Args:
        filename: The name of the uploaded file.
        file_size: Size of the file in bytes.

    Returns:
        Tuple of (is_valid, message).
        If invalid, message explains why.
    """
    # ── Check file extension ───────────────────────────────────
    _, ext = os.path.splitext(filename)
    if ext.lower() not in config.SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {ext}. Only PDF files are accepted."

    # ── Check file size ────────────────────────────────────────
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        return False, (
            f"File too large: {file_size / (1024*1024):.1f} MB. "
            f"Maximum allowed: {config.MAX_FILE_SIZE_MB} MB."
        )

    # ── Check for empty files ──────────────────────────────────
    if file_size == 0:
        return False, "File is empty."

    return True, "File is valid."


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to a human-readable file size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_directory_size(directory: str) -> int:
    """Calculate total size of a directory in bytes."""
    total_size = 0
    if not os.path.exists(directory):
        return 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size


def format_timestamp(iso_timestamp: str) -> str:
    """
    Convert ISO timestamp to a readable format.

    Args:
        iso_timestamp: ISO format timestamp string.

    Returns:
        Human-readable timestamp string.
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return "Unknown"


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if needed.

    Args:
        text: The text to truncate.
        max_length: Maximum number of characters.

    Returns:
        Truncated text with "..." appended if it was shortened.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def format_similarity_score(score: float) -> str:
    """
    Format a similarity score with a visual indicator.

    Args:
        score: Similarity score between 0 and 1.

    Returns:
        Formatted string with emoji indicator.
    """
    if score >= 0.7:
        indicator = "🟢"  # High relevance
        label = "High"
    elif score >= 0.4:
        indicator = "🟡"  # Medium relevance
        label = "Medium"
    else:
        indicator = "🔴"  # Low relevance
        label = "Low"

    return f"{indicator} {score:.4f} ({label})"


def get_elapsed_time(start_time: float) -> str:
    """
    Calculate elapsed time from a start timestamp.

    Args:
        start_time: The time.time() value when processing started.

    Returns:
        Formatted elapsed time string like "2.34 seconds".
    """
    elapsed = time.time() - start_time
    return f"{elapsed:.2f} seconds"


def clean_data_directory() -> bool:
    """
    Remove all files from the data directory.

    Returns:
        True if successful, False otherwise.
    """
    try:
        if os.path.exists(config.DATA_DIR):
            for file in os.listdir(config.DATA_DIR):
                file_path = os.path.join(config.DATA_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        return True
    except Exception as e:
        print(f"⚠️ Error cleaning data directory: {e}")
        return False
