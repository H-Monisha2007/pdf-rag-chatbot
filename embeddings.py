import os
import json
import logging
from typing import List, Dict, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config
from utils import setup_logger
from model_manager import GeminiModelManager

logger = setup_logger("Embeddings")

class GeminiEmbeddingsManager:
    """
    Enterprise-Grade Embedding Manager.
    Features: 
    - Deterministic Caching
    - Exponential Backoff (429/Network)
    - Batching
    - Stats Isolation
    """
    def __init__(self):
        self.model_name = GeminiModelManager.get_best_embedding()
        self.api_key = config.GOOGLE_API_KEY
        self.cache_file = config.EMBEDDING_CACHE_FILE
        self.cache = self._load_cache()
        self.stats = {}
        self.reset_stats()
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")

        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
        )

    def reset_stats(self):
        """Clears stats for a fresh indexing run."""
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "new_embeddings": 0,
            "batch_failures": 0,
            "model_used": self.model_name
        }

    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Cache failed to load: {e}")
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Cache failed to save: {e}")

    @retry(
        wait=wait_exponential(multiplier=config.RETRY_MIN_SECONDS, max=config.RETRY_MAX_SECONDS),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        self.stats["api_calls"] += 1
        try:
            return self.embeddings_model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Embedding attempt failed: {str(e)[:100]}... Retrying...")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Main entry point for batch embedding with intelligent caching."""
        results = [None] * len(texts)
        to_embed_indices = []
        to_embed_texts = []

        # 1. Deduplicate & Cache Check
        for i, text in enumerate(texts):
            content = text.strip()
            if content in self.cache:
                results[i] = self.cache[content]
                self.stats["cache_hits"] += 1
            else:
                to_embed_indices.append(i)
                to_embed_texts.append(content)

        # 2. Batch Execution
        if to_embed_texts:
            batch_size = config.EMBEDDING_BATCH_SIZE
            logger.info(f"Embedding {len(to_embed_texts)} new chunks in batches of {batch_size}...")
            
            for i in range(0, len(to_embed_texts), batch_size):
                batch = to_embed_texts[i : i + batch_size]
                batch_indices = to_embed_indices[i : i + batch_size]
                
                try:
                    embeddings = self._embed_with_retry(batch)
                    for idx, emb in zip(batch_indices, embeddings):
                        results[idx] = emb
                        self.cache[to_embed_texts[to_embed_indices.index(idx)]] = emb 
                        self.stats["new_embeddings"] += 1
                except Exception as e:
                    self.stats["batch_failures"] += 1
                    logger.error(f"Critical failure in embedding batch: {e}")
                    raise # Never swallow exceptions

            self._save_cache()

        return results

    def embed_query(self, query: str) -> List[float]:
        return self._embed_with_retry([query])[0]

_manager = None
def get_embedding_manager() -> GeminiEmbeddingsManager:
    global _manager
    if _manager is None:
        _manager = GeminiEmbeddingsManager()
    return _manager

def get_embedding_model():
    return get_embedding_manager().embeddings_model
