import os
import time
import streamlit as st
import pandas as pd
import config
from rag_pipeline import get_pipeline
from vector_store import get_collection_stats, clear_vector_store
from utils import (
    setup_logger, 
    clean_data_directory, 
    validate_config, 
    format_file_size, 
    get_directory_size,
    get_indexed_hashes
)

logger = setup_logger("App")

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_styles():
    """Enterprise-grade CSS with Theme-Aware Variable Injection."""
    st.markdown("""
    <style>
    .hero-container {
        padding: 2.5rem;
        border-radius: 16px;
        background-color: rgba(128, 128, 138, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2.5rem;
    }
    .hero-title-row { display: flex; align-items: center; gap: 15px; margin-bottom: 0.8rem; }
    .hero-title { font-size: 3rem; font-weight: 850; margin: 0; line-height: 1; letter-spacing: -0.04em; }
    .version-pill {
        background-color: #3b82f6;
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    .hero-subtitle { font-size: 1.25rem; opacity: 0.85; margin-bottom: 2rem; font-weight: 300; }
    .chip-container { display: flex; flex-wrap: wrap; gap: 10px; }
    .status-chip {
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        background-color: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-weight: 500;
    }
    .stMetric {
        border-radius: 14px !important;
        background-color: rgba(128, 128, 128, 0.07) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        padding: 1.2rem !important;
    }
    .source-tag {
        display: inline-block;
        font-size: 0.75rem;
        background: rgba(59, 130, 246, 0.1);
        padding: 5px 12px;
        border-radius: 8px;
        margin: 6px 8px 6px 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        font-weight: 600;
        color: #3b82f6;
    }
    .file-card {
        padding: 12px;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 10px;
    }
    .dashboard-section-header {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #3b82f6;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(59, 130, 246, 0.1);
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title-row">
            <h1 class="hero-title">🧠 Enterprise RAG</h1>
            <span class="version-pill">v2.1</span>
        </div>
        <div class="hero-subtitle">Production-Grade Multimodal Intelligence</div>
        <div class="chip-container">
            <div class="status-chip">🤖 Gemini 1.5</div>
            <div class="status-chip">📄 Multimodal</div>
            <div class="status-chip">🔍 Adaptive RAG</div>
            <div class="status-chip">💾 ChromaDB 0.6</div>
            <div class="status-chip">⚡ Production Ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def init_state():
    if "chat" not in st.session_state: st.session_state.chat = []
    if "last_metrics" not in st.session_state: st.session_state.last_metrics = None
    if "chat_stats" not in st.session_state: 
        st.session_state.chat_stats = {"questions": 0, "avg_time": 0.0, "total_time": 0.0}
    if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

def sidebar_ui():
    with st.sidebar:
        st.markdown(f'<div style="text-align: center; padding-bottom: 1rem;"><span style="font-size: 1.6rem; font-weight: 900;">🧠 Enterprise RAG</span></div>', unsafe_allow_html=True)
        st.divider()

        with st.expander("⬆️ UPLOAD KNOWLEDGE", expanded=True):
            files = st.file_uploader(
                "Knowledge Base", 
                accept_multiple_files=True, 
                type=["pdf", "png", "jpg", "docx", "pptx"],
                key=f"uploader_{st.session_state.uploader_key}"
            )
            
            if files:
                for f in files:
                    st.markdown(f'<div class="file-card"><b>{f.name}</b><br/><small>{format_file_size(f.size)}</small></div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 Process", type="primary", use_container_width=True):
                        status = st.empty()
                        metrics = get_pipeline().process_uploads(files, status_callback=lambda t: status.info(t))
                        st.session_state.last_metrics = metrics
                        status.empty()
                        if metrics["success"]: st.success("Knowledge Base Updated.")
                with c2:
                    if st.button("🗑 Clear List", use_container_width=True):
                        st.session_state.uploader_key += 1
                        st.rerun()

        stats = get_collection_stats()
        with st.expander("📊 SYSTEM ANALYTICS"):
            st.metric("Indexed Vectors", stats.get("total_documents", 0))
            st.metric("Total Documents", len(get_indexed_hashes()))
            st.write(f"**DB Size:** {format_file_size(get_directory_size(config.CHROMA_PERSIST_DIR))}")
            st.write(f"**Cache:** {format_file_size(get_directory_size(config.CACHE_DIR))}")

        with st.expander("💬 CONVERSATION"):
            st.write(f"**Interactions:** {st.session_state.chat_stats['questions']}")
            st.write(f"**Latency:** {round(st.session_state.chat_stats['avg_time'], 2)}s")
            if st.button("🗑 Clear Conversation", use_container_width=True):
                st.session_state.chat = []
                st.session_state.chat_stats = {"questions": 0, "avg_time": 0.0, "total_time": 0.0}
                st.rerun()

        with st.expander("📖 HELP"):
            st.info("Upload documents to start. The AI indexes semantic chunks for factual retrieval.")

        with st.expander("⚙ MAINTENANCE"):
            if st.button("⚠️ Factory Reset", use_container_width=True):
                if clear_vector_store():
                    clean_data_directory()
                    if os.path.exists(config.HASH_TRACKER_FILE): os.remove(config.HASH_TRACKER_FILE)
                    st.session_state.chat = []
                    st.session_state.last_metrics = None
                    st.rerun()

def dashboard_ui():
    m = st.session_state.last_metrics
    if not m: return

    with st.expander("📈 ENTERPRISE PERFORMANCE DASHBOARD", expanded=True):
        st.markdown('<p class="dashboard-section-header">Stage 1: Document Processing</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processing Time", f"{m['duration']}s")
        c2.metric("Pages Loaded", m['pages_loaded'])
        c3.metric("Files Skipped", m['skipped_files'])
        c4.metric("Success Rate", "100%" if not m['errors'] else "Partial")

        st.markdown('<p class="dashboard-section-header">Stage 2: Core Embedding Engine</p>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Chunks Created", m['chunks_created'])
        e2.metric("Cache Hits", m['cache_hits'])
        e3.metric("API Requests", m['api_calls'])
        
        total = m['chunks_created']
        saving = round((m['cache_hits'] / total * 100), 1) if total > 0 else 0
        e4.metric("API Savings", f"{saving}%")

        st.markdown('<p class="dashboard-section-header">Infrastructure Status</p>', unsafe_allow_html=True)
        st.info(f"**LLM:** {m['llm_active']} | **Embedding:** {m['emb_active']} | **Storage:** {m['storage_delta_mb']}MB")

        if m["errors"]:
            st.error("#### Critical Errata")
            for err in m["errors"]: st.warning(err)

def chat_ui():
    render_hero()
    stats = get_collection_stats()
    
    if stats.get("total_documents", 0) == 0:
        st.warning("⚠️ Intelligent Agent Inactive: Knowledge Base is currently empty.")
        return

    for msg in st.session_state.chat:
        with st.chat_message("user"): st.write(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            if msg.get("sources"):
                src_html = "".join([f'<span class="source-tag">📄 {s["file_name"]} | P{s["page_number"]}</span>' for s in msg["sources"]])
                st.markdown(src_html, unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a technical question..."):
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                start_q = time.time()
                pipeline = get_pipeline()
                res = pipeline.query(prompt, chat_history=[{"question": c["question"], "answer": c["answer"]} for c in st.session_state.chat])
                elapsed = time.time() - start_q
                
                st.session_state.chat_stats["questions"] += 1
                st.session_state.chat_stats["total_time"] += elapsed
                st.session_state.chat_stats["avg_time"] = st.session_state.chat_stats["total_time"] / st.session_state.chat_stats["questions"]
                
                if res["success"]:
                    st.markdown(res["answer"])
                    if res["chunks"]:
                        st.divider()
                        seen = set()
                        sources = []
                        for c in res["chunks"]:
                            meta = c["metadata"]
                            s_key = f"{meta.get('file_name')}_{meta.get('page_number')}"
                            if s_key not in seen:
                                seen.add(s_key)
                                sources.append({"file_name": meta.get("file_name"), "page_number": meta.get("page_number")})
                                st.markdown(f'<span class="source-tag">📄 {meta.get("file_name")} | P{meta.get("page_number")}</span>', unsafe_allow_html=True)
                        st.session_state.chat.append({"question": prompt, "answer": res["answer"], "sources": sources})
                else: st.error(f"Error: {res['error']}")

def main():
    apply_styles()
    init_state()
    if not validate_config():
        st.error("System Configuration Failed. Verify .env file.")
        st.stop()
    sidebar_ui()
    dashboard_ui()
    chat_ui()

if __name__ == "__main__":
    main()
