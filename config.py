import os
from typing import List
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────
load_dotenv()

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Enterprise Model Priority (Google AI Studio)
# The system will automatically select the first available model from these lists
LLM_PRIORITY: List[str] = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash", 
    "models/gemini-flash-latest"
]
LLM_FALLBACK: str = "models/gemini-1.5-flash"

EMBEDDING_PRIORITY: List[str] = [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2",
]
EMBEDDING_FALLBACK: str = "models/gemini-embedding-001"

# LLM Runtime Settings
LLM_TEMPERATURE: float = 0.2
LLM_MAX_OUTPUT_TOKENS: int = 4096

# Chunking Configuration
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 200
CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

# ChromaDB / Vector Store Configuration
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "enterprise_rag_v1"

# Retriever Configuration
RETRIEVER_TOP_K: int = 8
SIMILARITY_THRESHOLD: float = 0.3

# Application Configuration
APP_TITLE: str = "🚀 Enterprise Multimodal RAG"
APP_SUBTITLE: str = "Document Intelligence & Vision Analysis"
MAX_FILE_SIZE_MB: int = 50  # Enterprise limit
MAX_FILES: int = 50

SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".docx", ".pptx"]

# Tesseract OCR path for Windows
TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DATA_DIR: str = "./data"

# ── Optimization & Resilience ───────────────────
CACHE_DIR: str = os.path.join(os.getcwd(), ".cache")
EMBEDDING_CACHE_FILE: str = os.path.join(CACHE_DIR, "embedding_cache.json")
HASH_TRACKER_FILE: str = os.path.join(CACHE_DIR, "indexed_hashes.json")

RETRY_MAX_ATTEMPTS: int = 6
RETRY_MIN_SECONDS: int = 2
RETRY_MAX_SECONDS: int = 60

EMBEDDING_BATCH_SIZE: int = 16 

# Logging Configuration
LOG_FILE: str = "enterprise_app.log"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL: str = "INFO"
