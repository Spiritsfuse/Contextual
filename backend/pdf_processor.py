"""
pdf_processor.py
----------------
Handles PDF text extraction and chunking.

Strategy:
- Extract text per page using PyMuPDF (fitz) for reliable extraction.
- Apply sliding-window chunking at the word level (approximating tokens).
- Preserve page_number and chunk_id as metadata for citations.
- Chunk size: 500–800 tokens with overlap.

Tokenization note:
  We use a simple whitespace-based word tokenizer rather than tiktoken
  (which requires Rust/MSVC to compile). The approximation is:
      1 token ≈ 1 word (slightly conservative — GPT tokenizers split on
      subwords, averaging ~1.3 tokens/word for English). For chunking
      purposes (512–800 token windows), word-count boundaries are
      accurate enough: a 600-word chunk ≈ 750-800 GPT tokens.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a single chunk of text with provenance metadata."""
    chunk_id: int
    text: str
    page_number: int           # 1-indexed page number from the PDF
    token_count: int
    char_start: int = 0        # reserved for future character-level tracing
    char_end: int = 0


def _tokenize(text: str) -> List[str]:
    """Split text into word-level tokens (whitespace + punctuation aware)."""
    # Split on whitespace; keep hyphenated words together
    return text.split()


def _decode_tokens(tokens: List[str]) -> str:
    """Rejoin word tokens into text."""
    return " ".join(tokens)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving sentence boundaries."""
    text = re.sub(r'\n{3,}', '\n\n', text)    # collapse triple+ newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)     # collapse multiple spaces/tabs
    text = text.strip()
    return text


def extract_pages(pdf_path: str) -> List[dict]:
    """
    Extract text from each page of the PDF.

    Returns:
        List of dicts: [{ 'page_number': int, 'text': str }, ...]
    """
    pages = []
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}") from e

    logger.info(f"Extracting text from PDF: {path.name} ({doc.page_count} pages)")

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        raw_text = page.get_text("text")  # plain text extraction
        cleaned = _clean_text(raw_text)
        if cleaned:
            pages.append({
                "page_number": page_idx + 1,  # 1-indexed
                "text": cleaned,
            })

    doc.close()
    logger.info(f"Extracted text from {len(pages)} non-empty pages")
    return pages


def chunk_pages(
    pages: List[dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[TextChunk]:
    """
    Sliding-window token-level chunking across page boundaries.

    Each chunk carries the page_number of the page where its text STARTS.
    If a chunk spans multiple pages, the starting page is recorded,
    and the text itself contains natural breaks making cross-page context clear.

    Args:
        pages:      Output of extract_pages().
        chunk_size: Target token count per chunk (500–800 recommended).
        overlap:    Token overlap between consecutive chunks for context continuity.

    Returns:
        List of TextChunk objects sorted by chunk_id.
    """
    if chunk_size < 100 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap values.")

    chunks: List[TextChunk] = []
    chunk_id = 0

    # Build a combined token stream preserving page boundaries
    # We interleave a special marker so we can recover page numbers later.
    all_tokens: List[int] = []
    token_page_map: List[int] = []  # maps each token index → page_number

    for page in pages:
        page_tokens = _tokenize(page["text"])
        all_tokens.extend(page_tokens)
        token_page_map.extend([page["page_number"]] * len(page_tokens))

    if not all_tokens:
        logger.warning("No tokens extracted from PDF — empty document?")
        return chunks

    total = len(all_tokens)
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        window_tokens = all_tokens[start:end]
        page_num = token_page_map[start]  # page where this chunk starts

        text = _decode_tokens(window_tokens)
        token_count = len(window_tokens)

        chunks.append(TextChunk(
            chunk_id=chunk_id,
            text=text,
            page_number=page_num,
            token_count=token_count,
        ))

        chunk_id += 1
        start += chunk_size - overlap  # advance with overlap

    logger.info(
        f"Created {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap}, "
        f"total_tokens={total})"
    )
    return chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[TextChunk]:
    """
    Full pipeline: PDF path → List[TextChunk].

    Args:
        pdf_path:   Path to the PDF file.
        chunk_size: Token count per chunk.
        overlap:    Token overlap between chunks.

    Returns:
        Ordered list of TextChunk objects ready for embedding.
    """
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return chunks
