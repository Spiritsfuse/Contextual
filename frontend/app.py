"""
app.py — Streamlit UI for the PDF-Constrained Conversational Agent
------------------------------------------------------------------
Features:
  - PDF upload via sidebar
  - Full chat interface with conversation history
  - Per-response expandable panel: retrieved chunks + similarity scores
  - Citation display
  - Language detection indicator
  - Index reset button
  - System status indicator
"""

import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Lightweight Ping Mode (for uptime monitoring & cron-job compatibility)
# -----------------------------------------------------------------------
if st.query_params.get("ping") == "1":
    st.write("ok")
    st.stop()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TOP_K = int(os.getenv("TOP_K", 5))

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Document QA Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom CSS — clean, readable, professional
# -----------------------------------------------------------------------
st.markdown("""
<style>
/* Main typography */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Chat message styling */
.user-message {
    background: #1e3a5f;
    color: #e8f4fd;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    float: right;
    clear: both;
    word-wrap: break-word;
}

.assistant-message {
    background: #1a1a2e;
    color: #e0e0e0;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    float: left;
    clear: both;
    border-left: 3px solid #4a90d9;
    word-wrap: break-word;
}

.refusal-message {
    background: #3d1a1a;
    color: #ff9999;
    border-left-color: #cc3333;
}

/* Citation badges */
.citation-badge {
    display: inline-block;
    background: #2a5298;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
    margin: 0 3px;
}

/* Chunk preview */
.chunk-card {
    background: #0f0f23;
    border: 1px solid #2d2d50;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
}

.chunk-meta {
    color: #7090b0;
    font-size: 0.8em;
    margin-bottom: 4px;
    font-weight: 600;
}

.score-bar-container {
    background: #1a1a2e;
    border-radius: 4px;
    height: 6px;
    margin: 4px 0;
    overflow: hidden;
}

.score-bar {
    height: 100%;
    background: linear-gradient(90deg, #4a90d9, #7bb3e8);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* Status indicators */
.status-ok { color: #4caf50; font-weight: bold; }
.status-warn { color: #ff9800; font-weight: bold; }
.status-err { color: #f44336; font-weight: bold; }

/* Language pill */
.lang-pill {
    display: inline-block;
    background: #1a3a4a;
    color: #66ccff;
    border: 1px solid #336688;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.78em;
}

.clearfix::after { content: ""; display: table; clear: both; }

/* Sidebar styling */
section[data-testid="stSidebar"] { background: #0d0d1a; }
section[data-testid="stSidebar"] .stMarkdown { color: #aaaacc; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta}

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# -----------------------------------------------------------------------
# Startup Ping — ensures backend is awake
# -----------------------------------------------------------------------
if "has_pinged_backend" not in st.session_state:
    try:
        # Silently ping the backend to wake it up
        requests.get(f"{BACKEND_URL}/ping", timeout=5)
        st.session_state.has_pinged_backend = True
    except Exception:
        # If it fails, we'll try again on the next run or via the sidebar check
        pass


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def get_health() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json() if resp.ok else {}
    except Exception:
        return {}


def upload_pdf(file_bytes: bytes, filename: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/upload_pdf",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
        )
        if resp.ok:
            return resp.json()
        else:
            st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None


def query_backend(question: str, top_k: int = TOP_K) -> dict | None:
    try:
        payload = {
            "query": question,
            "top_k": top_k,
            "session_id": st.session_state.session_id,
        }
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=60,
        )
        if resp.ok:
            return resp.json()
        else:
            detail = resp.json().get("detail", "Unknown error")
            st.error(f"Query failed: {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Query error: {e}")
        return None


def reset_index() -> bool:
    try:
        resp = requests.post(f"{BACKEND_URL}/reset_index", timeout=10)
        return resp.ok
    except Exception:
        return False


def render_score_bar(score: float) -> str:
    """Render an HTML progress bar for similarity score."""
    pct = max(0, min(100, int(score * 100)))
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar" style="width:{pct}%"></div>'
        f'</div>'
    )


def render_citations(citations: list[int]) -> str:
    if not citations:
        return ""
    badges = "".join(f'<span class="citation-badge">📄 Page {p}</span>' for p in citations)
    return f"<div style='margin-top:8px'>{badges}</div>"


LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
}


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 PDF Document QA")
    st.markdown("---")

    # Backend status
    health = get_health()
    if health:
        idx_size = health.get("index_size", 0)
        current_pdf = health.get("current_pdf")
        st.markdown(
            f'<span class="status-ok">● Backend Online</span>',
            unsafe_allow_html=True,
        )
        if current_pdf:
            st.markdown(f"**Loaded PDF:** `{current_pdf}`")
            st.markdown(f"**Indexed chunks:** `{idx_size}`")
        else:
            st.markdown('<span class="status-warn">⚠ No PDF indexed</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-err">● Backend Offline</span>',
            unsafe_allow_html=True,
        )
        st.info("Please check if the backend service is running.")

    st.markdown("---")

    # PDF upload
    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the document you want to query",
        label_visibility="collapsed",
    )

    top_k_slider = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        step=1,
        help="Number of document chunks used as context for each answer",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"📁 `{uploaded_file.name}`")
        with col2:
            process_btn = st.button("Process", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing PDF…"):
                result = upload_pdf(uploaded_file.getvalue(), uploaded_file.name)
            if result:
                st.session_state.pdf_uploaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.success(
                    f"✅ Ready!\n\n"
                    f"📊 {result['num_chunks']} chunks | "
                    f"📄 {result['pages_processed']} pages | "
                    f"🔤 {result['total_tokens']:,} tokens"
                )
                st.session_state.messages = []  # clear chat for new PDF
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # Reset
    if st.button("🗑️ Reset Index", use_container_width=True, type="secondary"):
        if reset_index():
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_name = None
            st.session_state.messages = []
            st.success("Index cleared")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Reset failed")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    # Test queries
    with st.expander("📋 Sample Test Queries", expanded=False):
        st.markdown("""
**Valid queries** (for sample PDF):
1. What are the main applications of AI in healthcare?
2. What ethical challenges does AI face in medical diagnosis?
3. Describe a case study mentioned in the document.
4. What role does machine learning play in drug discovery?
5. What are the limitations of AI systems discussed?

**Invalid queries** (should refuse):
1. What is the capital of France?
2. Write a Python script to sort a list.
3. What happened in World War II?
""")


# -----------------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------------
st.markdown("## 🤖 PDF Document QA Agent")
st.markdown(
    "Ask questions about your uploaded document. "
    "All answers are strictly grounded in the document content."
)

if not health:
    st.error("⚠️ **Backend is not running.** Please check if the backend service is up.")
elif not health.get("current_pdf"):
    st.info("👈 **Upload a PDF** using the sidebar to get started.")

# Render conversation history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f'<div class="clearfix"><div class="user-message">👤 {content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        is_refusal = meta.get("is_refusal", False)
        extra_class = "refusal-message" if is_refusal else ""
        citations = meta.get("citations", [])
        detected_lang = meta.get("detected_language", "en")
        lang_name = LANG_NAMES.get(detected_lang, detected_lang.upper())

        st.markdown(
            f'<div class="clearfix">'
            f'<div class="assistant-message {extra_class}">'
            f'<div style="font-size:0.8em;color:#7090b0;margin-bottom:6px">'
            f'🤖 Agent &nbsp;|&nbsp; <span class="lang-pill">🌐 {lang_name}</span>'
            f'</div>'
            f'{content}'
            f'{render_citations(citations)}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Expandable retrieval details
        if meta.get("retrieved_chunks"):
            with st.expander(
                f"🔍 Retrieved chunks ({len(meta['retrieved_chunks'])}) — click to inspect",
                expanded=False
            ):
                if meta.get("query_translated") and meta.get("query_translated") != meta.get("query_original"):
                    st.markdown(
                        f"**Translated query:** `{meta['query_translated']}`"
                    )
                for i, chunk in enumerate(meta["retrieved_chunks"], 1):
                    score = chunk["similarity_score"]
                    page = chunk["page_number"]
                    text_preview = chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")

                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta">'
                        f'#{i} | chunk_id={chunk["chunk_id"]} | '
                        f'📄 Page {page} | '
                        f'Score: {score:.4f} | '
                        f'{chunk["token_count"]} tokens'
                        f'</div>'
                        f'{render_score_bar(score)}'
                        f'<div style="color:#b0c4d8;margin-top:6px">{text_preview}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input(
    "Ask a question about the document…",
    disabled=not health.get("current_pdf"),
):
    if not health.get("current_pdf"):
        st.warning("Please upload and process a PDF first.")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        # Query backend
        with st.spinner("Retrieving and generating answer…"):
            response = query_backend(prompt, top_k=top_k_slider)

        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "meta": {
                    "citations": response.get("citations", []),
                    "is_refusal": response.get("is_refusal", False),
                    "retrieved_chunks": response.get("retrieved_chunks", []),
                    "detected_language": response.get("detected_language", "en"),
                    "query_original": response.get("query_original", ""),
                    "query_translated": response.get("query_translated", ""),
                },
            })

        st.rerun()
