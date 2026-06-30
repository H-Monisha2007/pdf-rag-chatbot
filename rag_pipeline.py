import time
import logging
import os
from typing import Any, List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

import config
from document_loader import load_multiple_documents, save_uploaded_file
from text_chunker import chunk_documents
from vector_store import add_documents_to_store, get_collection_stats, is_hash_indexed
from retriever import (
    retrieve_relevant_chunks,
    get_valid_chunks,
    format_context_for_llm,
)
from prompts import format_prompt, CONVERSATIONAL_PROMPT
from utils import setup_logger, calculate_file_hash, save_indexed_hash, get_indexed_hashes, get_directory_size
from embeddings import get_embedding_manager
from model_manager import GeminiModelManager

logger = setup_logger("RAG_Pipeline")

class RAGPipeline:
    def __init__(self):
        self.llm_model_name = GeminiModelManager.get_best_llm()
        self.emb_manager = get_embedding_manager()
        
    def get_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        )

    def process_uploads(self, uploaded_files: List[Any], status_callback=None) -> Dict[str, Any]:
        """
        Enterprise Indexing Pipeline with Multi-Stage Verification.
        Verification Path: Files -> Loaded Pages -> Semantic Chunks -> Stored Vectors.
        """
        start_time = time.time()
        self.emb_manager.reset_stats()  # Isolation: Reset stats for this specific run
        
        metrics = {
            "success": False,
            "processed_files": [],
            "skipped_files": 0,
            "pages_loaded": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "storage_delta_mb": 0.0,
            "duration": 0.0,
            "errors": [],
            "llm_active": self.llm_model_name,
            "emb_active": self.emb_manager.model_name
        }

        try:
            # 1. Document Loading & Stage 1 Verification
            all_new_docs = []
            for uploaded_file in uploaded_files:
                f_name = uploaded_file.name
                try:
                    if status_callback: status_callback(f"🛡️ Hashing {f_name}...")
                    t_path = save_uploaded_file(uploaded_file, f_name)
                    f_hash = calculate_file_hash(t_path)
                    
                    if is_hash_indexed(f_hash):
                        metrics["skipped_files"] += 1
                        metrics["processed_files"].append({"name": f_name, "status": "Already Indexed"})
                        continue

                    if status_callback: status_callback(f"📄 Extracting {f_name}...")
                    load_start = time.time()
                    docs = load_multiple_documents([t_path])
                    metrics["pages_loaded"] += len(docs)
                    
                    for doc in docs: doc.metadata["file_hash"] = f_hash
                    all_new_docs.extend(docs)
                    
                    metrics["processed_files"].append({"name": f_name, "status": "Success", "pages": len(docs)})
                    save_indexed_hash(f_hash, {"name": f_name, "indexed_at": time.time()})
                    logger.info(f"Stage 1 Success: {f_name} loaded in {time.time()-load_start:.2f}s")
                except Exception as e:
                    metrics["errors"].append(f"{f_name}: {str(e)}")

            if not all_new_docs:
                if metrics["skipped_files"] > 0:
                    metrics["success"] = True
                else:
                    metrics["errors"].append("No valid content found to index.")
                return metrics

            # 2. Semantic Chunking & Stage 2 Verification
            if status_callback: status_callback("🧩 Generating Semantic Chunks...")
            chunk_start = time.time()
            chunks = chunk_documents(all_new_docs)
            metrics["chunks_created"] = len(chunks)
            if len(chunks) == 0:
                raise ValueError("Content extraction yielded Zero Chunks. Pipeline Aborted.")
            logger.info(f"Stage 2 Success: {len(chunks)} chunks created in {time.time()-chunk_start:.2f}s")

            # 3. Vector Storage & Stage 3 Verification
            if status_callback: status_callback("🧠 Indexing Vectors (Batching + Cache)...")
            store_start = time.time()
            
            stats_before = get_collection_stats()
            add_documents_to_store(chunks)
            stats_after = get_collection_stats()
            
            metrics["chunks_stored"] = stats_after["total_documents"] - stats_before["total_documents"]
            metrics["cache_hits"] = self.emb_manager.stats["cache_hits"]
            metrics["api_calls"] = self.emb_manager.stats["api_calls"]
            
            # Post-Insertion Integrity Check
            if metrics["chunks_stored"] == 0 and metrics["cache_hits"] == 0:
                logger.warning("Zero new vectors stored. Possible collision or duplicate extraction.")

            logger.info(f"Stage 3 Success: Vector store updated in {time.time()-store_start:.2f}s")

            # Calculate Final Analytics
            db_size = get_directory_size(config.CHROMA_PERSIST_DIR)
            metrics["storage_delta_mb"] = round(db_size / (1024 * 1024), 2)
            metrics["duration"] = round(time.time() - start_time, 2)
            metrics["success"] = True

        except Exception as e:
            logger.error(f"PIPELINE CRITICAL FAILURE: {e}", exc_info=True)
            metrics["errors"].append(f"Global Pipeline Failure: {str(e)}")

        return metrics

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Enterprise Query Engine with Conversation Memory and Source Validation.
        """
        q_start = time.time()
        res = {"success": False, "answer": "", "chunks": [], "time": 0.0, "error": None}
        
        try:
            # 1. Semantic Retrieval
            all_chunks = retrieve_relevant_chunks(question)
            valid = get_valid_chunks(all_chunks)
            res["chunks"] = valid

            if not valid:
                res["answer"] = config.NO_CONTEXT_RESPONSE
                res["success"] = True
                return res

            # 2. Multi-Context Building
            context = format_context_for_llm(valid)
            
            # 3. Memory-Aware Prompts
            if chat_history:
                hist_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in chat_history[-3:]])
                prompt = CONVERSATIONAL_PROMPT.format(chat_history=hist_text, context=context, question=question)
            else:
                prompt = format_prompt(context=context, question=question)

            # 4. LLM Generation
            llm = self.get_llm()
            response = llm.invoke(prompt)
            res["answer"] = response.content
            res["success"] = True
            
        except Exception as e:
            logger.error(f"Query Service Failure: {e}")
            res["error"] = str(e)
            
        res["time"] = round(time.time() - q_start, 3)
        return res

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
