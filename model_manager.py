import google.generativeai as genai
import logging
import config
from typing import List, Optional
from utils import setup_logger

logger = setup_logger("ModelManager")

class GeminiModelManager:
    """
    Handles dynamic discovery and validation of Gemini models.
    Ensures the application never hardcodes deprecated models by 
    validating against the live API.
    """
    
    _cached_models: List[str] = []

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Fetch and cache the list of supported models from the API."""
        if cls._cached_models:
            return cls._cached_models
        
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            models = [m.name for m in genai.list_models()]
            # Clean names (ensure 'models/' prefix)
            cls._cached_models = [m if m.startswith("models/") else f"models/{m}" for m in models]
            return cls._cached_models
        except Exception as e:
            logger.error(f"Failed to fetch models from Gemini API: {e}")
            return []

    @classmethod
    def get_best_model(cls, priority_list: List[str], fallback: str) -> str:
        """
        Returns the first model from the priority list that is actually supported.
        Falls back to a default if none match.
        """
        supported = cls.get_supported_models()
        if not supported:
            logger.warning("No supported models found from API. Using fallback.")
            return fallback

        for model in priority_list:
            full_name = model if model.startswith("models/") else f"models/{model}"
            if full_name in supported:
                logger.info(f"Selected model: {full_name}")
                return full_name
        
        logger.warning(f"None of the priority models {priority_list} are supported. Using fallback: {fallback}")
        return fallback

    @classmethod
    def get_best_llm(cls) -> str:
        return cls.get_best_model(config.LLM_PRIORITY, config.LLM_FALLBACK)

    @classmethod
    def get_best_embedding(cls) -> str:
        return cls.get_best_model(config.EMBEDDING_PRIORITY, config.EMBEDDING_FALLBACK)
