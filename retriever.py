"""
============================================================
Retriever Module
============================================================
Responsible for performing semantic search and quality validation.
============================================================
"""

from __future__ import annotations
from langchain_core.documents import Document
import config
from vector_store import get_vector_store

def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[dict]:
    if top_k is None:
        top_k = config.RETRIEVER_TOP_K
    if similarity_threshold is None:
        similarity_threshold = config.SIMILARITY_THRESHOLD

    vector_store = get_vector_store()

    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
    )

    processed_results = []
    for doc, score in results_with_scores:
        processed_results.append({
            "document": doc,
            "content": doc.page_content,
            "score": round(score, 4),
            "metadata": doc.metadata,
            "passed_threshold": score >= similarity_threshold,
        })

    return processed_results

def get_valid_chunks(results: list[dict]) -> list[dict]:
    return [r for r in results if r["passed_threshold"]]

def format_context_for_llm(valid_chunks: list[dict]) -> str:
    """
    Format validated chunks into a context string with multimodal metadata.
    """
    if not valid_chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(valid_chunks, 1):
        meta = chunk["metadata"]
        file_name = meta.get("file_name", "Unknown")
        page_num = meta.get("page_number", "N/A")
        content_type = meta.get("content_type", "text").replace("_", " ").title()
        
        context_parts.append(
            f"--- SOURCE {i} [{file_name}, Page {page_num}, Type: {content_type}] ---\n"
            f"{chunk['content']}\n"
        )

    return "\n".join(context_parts)
