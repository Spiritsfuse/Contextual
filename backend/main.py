"""
main.py
-------
FastAPI backend for the PDF-Constrained Conversational Agent.

Endpoints:
  POST /upload_pdf     — Upload and process a PDF
  POST /query          — Run a RAG query against the indexed PDF
  POST /reset_index    — Clear the FAISS index
  GET  /health         — Health check + index status
  GET  /index_info     — Detailed index information

Application state:
  A single VectorStore instance is shared across requests via FastAPI's
  application state (app.state). This is safe for single-worker deployments.
  For multi-worker deployments, use a shared Redis/PostgreSQL-backed store.

Observability:
  All queries, retrieved chunks, scores, and final answers are logged to:
  - stdout (structured)
  - logs/session_{date}.log (append-only file log)
"""

import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import embed_texts, get_embedding_dim
from .llm import REFUSAL_PHRASE, generate_answer
from .pdf_processor import process_pdf
from .retriever import retrieve, translate_from_english
from .vector_store import VectorStore

# -----------------------------------------------------------------------
# Load env + configure logging
# -----------------------------------------------------------------------

load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------

app = FastAPI(
    title="PDF-Constrained Conversational Agent",
    description=(
        "A production-grade RAG system that strictly answers from PDF content. "
        "Uses FAISS + sentence-transformers + Gemini API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Application startup — initialize shared VectorStore
# -----------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize shared VectorStore and attempt to load existing index."""
    dim = get_embedding_dim()
    store = VectorStore(dim=dim)

    # Try to restore persisted index from previous session
    loaded = store.load()
    app.state.vector_store = store
    app.state.current_pdf_name = None

    if loaded:
        logger.info(
            f"Restored existing index with {len(store)} vectors on startup"
        )
    else:
        logger.info("Fresh VectorStore initialized")


# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for log correlation"
    )


class RetrievedChunkResponse(BaseModel):
    chunk_id: int
    page_number: int
    text: str
    similarity_score: float
    token_count: int


class QueryResponse(BaseModel):
    answer: str
    citations: List[int]
    is_refusal: bool
    retrieved_chunks: List[RetrievedChunkResponse]
    detected_language: str
    query_original: str
    query_translated: str
    session_id: str
    log_file: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    num_chunks: int
    pages_processed: int
    total_tokens: int


class ResetResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    index_size: int
    current_pdf: Optional[str]
    model: str
    log_file: str


# -----------------------------------------------------------------------
# Helper: observability logger
# -----------------------------------------------------------------------

def _log_query_event(
    session_id: str,
    query_original: str,
    query_translated: str,
    detected_lang: str,
    chunks,
    answer: str,
    citations: List[int],
    is_refusal: bool,
) -> None:
    """Structured log entry for a full query–response cycle."""
    separator = "=" * 70
    log_lines = [
        f"\n{separator}",
        f"SESSION:          {session_id}",
        f"TIMESTAMP:        {datetime.now().isoformat()}",
        f"QUERY (original): {query_original}",
        f"LANG DETECTED:    {detected_lang}",
        f"QUERY (English):  {query_translated}",
        f"--- RETRIEVED CHUNKS ({len(chunks)}) ---",
    ]

    for i, chunk in enumerate(chunks, 1):
        log_lines.append(
            f"  [{i}] chunk_id={chunk.chunk_id} | page={chunk.page_number} "
            f"| score={chunk.similarity_score:.4f} | tokens={chunk.token_count}"
        )
        log_lines.append(f"      TEXT PREVIEW: {chunk.text[:200].replace(chr(10), ' ')}...")

    log_lines.extend([
        f"--- FINAL ANSWER ---",
        f"IS_REFUSAL:  {is_refusal}",
        f"CITATIONS:   Pages {citations}",
        f"ANSWER:\n{answer}",
        separator,
    ])

    full_log = "\n".join(log_lines)
    logger.info(full_log)


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — returns index status and configuration."""
    store: VectorStore = app.state.vector_store
    return HealthResponse(
        status="ok",
        index_size=len(store),
        current_pdf=app.state.current_pdf_name,
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        log_file=str(LOG_FILE),
    )


@app.get("/index_info", tags=["System"])
async def index_info():
    """Return detailed FAISS index information."""
    store: VectorStore = app.state.vector_store
    return store.info()


@app.post("/upload_pdf", response_model=UploadResponse, tags=["PDF"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, chunk it, embed chunks, and store in FAISS.

    - Accepts: multipart/form-data with a PDF file.
    - Processing: extraction → chunking (600 tokens, 100 overlap) → embedding → FAISS.
    - Previous index is replaced on each upload.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(f"PDF upload received: {file.filename} ({len(content)} bytes)")

    try:
        # 1. Process PDF → chunks
        chunk_size = int(os.getenv("CHUNK_SIZE", 600))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        chunks = process_pdf(tmp_path, chunk_size=chunk_size, overlap=chunk_overlap)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract any text from the PDF. "
                       "The file may be scanned or image-based."
            )

        # 2. Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, show_progress=True)

        # 3. Reset existing index and add new embeddings
        store: VectorStore = app.state.vector_store
        store.reset()
        store.add_chunks(chunks, embeddings)
        store.save()

        app.state.current_pdf_name = file.filename

        pages_processed = len(set(c.page_number for c in chunks))
        total_tokens = sum(c.token_count for c in chunks)

        logger.info(
            f"PDF processed: {file.filename} | "
            f"{len(chunks)} chunks | {pages_processed} pages | "
            f"{total_tokens} total tokens"
        )

        return UploadResponse(
            status="success",
            filename=file.filename,
            num_chunks=len(chunks),
            pages_processed=pages_processed,
            total_tokens=total_tokens,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF processing failed: {str(e)}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Run a RAG query against the indexed PDF.

    Pipeline:
    1. Detect language of query.
    2. Translate query to English if needed.
    3. Embed query and retrieve top-k chunks from FAISS.
    4. Generate grounded answer via Gemini with strict context-only prompt.
    5. Translate answer back to original language if needed.
    6. Log the full query–retrieval–response cycle.
    7. Return structured response.
    """
    store: VectorStore = app.state.vector_store

    if store.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document indexed. Please upload a PDF first via POST /upload_pdf."
        )

    session_id = request.session_id or str(uuid.uuid4())[:8]
    top_k = int(os.getenv("TOP_K", request.top_k))

    try:
        # Step 1–3: Retrieve
        retrieval = retrieve(
            query=request.query,
            vector_store=store,
            top_k=top_k,
        )

        # Step 4: Generate answer (in English)
        llm_resp = generate_answer(
            question=retrieval.query_translated,
            retrieved_chunks=retrieval.retrieved_chunks,
        )

        # Step 5: Translate answer back to original language
        final_answer = translate_from_english(
            llm_resp.answer,
            target_lang=retrieval.detected_language,
        )

        # Step 6: Log the full cycle
        _log_query_event(
            session_id=session_id,
            query_original=retrieval.query_original,
            query_translated=retrieval.query_translated,
            detected_lang=retrieval.detected_language,
            chunks=retrieval.retrieved_chunks,
            answer=final_answer,
            citations=llm_resp.citations,
            is_refusal=llm_resp.is_refusal,
        )

        # Step 7: Return
        return QueryResponse(
            answer=final_answer,
            citations=llm_resp.citations,
            is_refusal=llm_resp.is_refusal,
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=c.chunk_id,
                    page_number=c.page_number,
                    text=c.text,
                    similarity_score=round(c.similarity_score, 4),
                    token_count=c.token_count,
                )
                for c in retrieval.retrieved_chunks
            ],
            detected_language=retrieval.detected_language,
            query_original=retrieval.query_original,
            query_translated=retrieval.query_translated,
            session_id=session_id,
            log_file=str(LOG_FILE),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/reset_index", response_model=ResetResponse, tags=["System"])
async def reset_index():
    """
    Clear the FAISS index and all stored chunk metadata.

    Use this when you want to upload a new PDF and start fresh.
    """
    store: VectorStore = app.state.vector_store
    store.reset()
    app.state.current_pdf_name = None
    logger.info("Index reset by API call")

    return ResetResponse(
        status="success",
        message="FAISS index and metadata cleared. Ready for new PDF upload.",
    )
