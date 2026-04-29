"""
retriever.py
------------
Query pipeline: language detection → translation → embedding → FAISS search.

Multi-language Support:
1. Detect query language using langdetect.
2. If not English, translate to English using deep-translator (GoogleTranslator).
3. Embed the (now English) query.
4. Search FAISS for top-k chunks.
5. Return results + detected language + translated query for logging.

Note on determinism:
- langdetect uses a probabilistic model internally. For maximum determinism,
  we seed DetectorFactory before each detection call.
- Translation output depends on the external Google Translate service and
  can vary slightly across network calls; this is the only non-deterministic step.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from langdetect import DetectorFactory, detect as _detect_lang
from deep_translator import GoogleTranslator

from .embeddings import embed_query, DEFAULT_MODEL
from .pdf_processor import TextChunk
from .vector_store import VectorStore

# Seed langdetect for reproducibility
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with similarity score."""
    chunk_id: int
    page_number: int
    text: str
    similarity_score: float
    token_count: int


@dataclass
class RetrievalResult:
    """Full output of a retrieval operation."""
    query_original: str
    query_translated: str
    detected_language: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    top_k: int = 5


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Returns:
        ISO 639-1 language code (e.g. 'en', 'fr', 'es').
        Falls back to 'en' on error.
    """
    try:
        DetectorFactory.seed = 0
        lang = _detect_lang(text)
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
        return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from source_lang to English.

    Uses deep-translator's GoogleTranslator which wraps the free Google
    Translate web API — no API key required.

    Args:
        text:        Text to translate.
        source_lang: ISO 639-1 source language code.

    Returns:
        Translated English text. Returns original text if translation fails.
    """
    if source_lang == "en":
        return text

    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        translated = translator.translate(text)
        logger.info(f"Translated [{source_lang}→en]: '{text[:80]}' → '{translated[:80]}'")
        return translated
    except Exception as e:
        logger.warning(f"Translation failed ({source_lang}→en): {e}. Using original text.")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate an English response back to the target language.

    Args:
        text:        English text to translate back.
        target_lang: ISO 639-1 target language code.

    Returns:
        Translated text. Falls back to English text on error.
    """
    if target_lang == "en":
        return text

    try:
        translator = GoogleTranslator(source="en", target=target_lang)
        translated = translator.translate(text)
        logger.info(f"Translated [en→{target_lang}]: response translated back")
        return translated
    except Exception as e:
        logger.warning(f"Back-translation failed (en→{target_lang}): {e}. Returning English.")
        return text


def retrieve(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> RetrievalResult:
    """
    Full retrieval pipeline for a single query.

    Steps:
    1. Detect language.
    2. Translate to English if needed.
    3. Embed the English query.
    4. Search FAISS for top-k chunks.
    5. Return structured RetrievalResult.

    Args:
        query:        Raw user query (any language).
        vector_store: Initialized VectorStore instance.
        top_k:        Number of chunks to retrieve.
        model_name:   Embedding model to use.

    Returns:
        RetrievalResult with all metadata.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if vector_store.is_empty:
        raise RuntimeError(
            "Vector store is empty. Please upload and process a PDF first."
        )

    # Step 1 & 2: Language detection + translation
    detected_lang = detect_language(query)
    logger.info(f"Query language detected: {detected_lang}")

    query_en = translate_to_english(query, source_lang=detected_lang)

    # Step 3: Embed the English query
    query_embedding = embed_query(query_en, model_name=model_name)

    # Step 4: FAISS search
    raw_results: List[Tuple[TextChunk, float]] = vector_store.search(
        query_embedding, top_k=top_k
    )

    # Step 5: Package results
    retrieved = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity_score=score,
            token_count=chunk.token_count,
        )
        for chunk, score in raw_results
    ]

    result = RetrievalResult(
        query_original=query,
        query_translated=query_en,
        detected_language=detected_lang,
        retrieved_chunks=retrieved,
        top_k=top_k,
    )

    logger.info(
        f"Retrieved {len(retrieved)} chunks. "
        f"Scores: {[round(r.similarity_score, 4) for r in retrieved]}"
    )

    return result
