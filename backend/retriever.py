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
from .crag import CRAGPipeline, CRAG_INITIAL_TOPK_MULTIPLIER, CRAG_RELEVANCE_THRESHOLD

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
    use_crag: bool = True,
) -> RetrievalResult:
    """
    Full retrieval pipeline for a single query with optional Corrective RAG (CRAG).

    Steps (Standard RAG):
    1. Detect language.
    2. Translate to English if needed.
    3. Embed the English query.
    4. Search FAISS for top-k chunks.
    5. Return structured RetrievalResult.

    Steps (Corrective RAG):
    1. Detect language & Translate to English.
    2. Retrieve top-k * multiplier chunks from FAISS (Initial dense search).
    3. Evaluate retrieved chunks via Gemini (Parallel batch grading).
    4. If no relevant chunks are found, trigger Fallback Query Rewriter.
    5. Retrieve secondary chunks with the rewritten query and evaluate them.
    6. Merge, deduplicate by chunk_id, and filter out irrelevant chunks.
    7. Sort/rerank the final chunks using a hybrid score (0.7 * relevance + 0.3 * similarity).
    8. Truncate to top_k and return.
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

    if not use_crag:
        logger.info("CRAG is disabled. Using standard RAG dense vector search.")
        # Embed the English query
        query_embedding = embed_query(query_en, model_name=model_name)

        # FAISS search
        raw_results: List[Tuple[TextChunk, float]] = vector_store.search(
            query_embedding, top_k=top_k
        )

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
            f"Standard Retrieval: retrieved {len(retrieved)} chunks. "
            f"Scores: {[round(r.similarity_score, 4) for r in retrieved]}"
        )
        return result

    # --- Corrective RAG (CRAG) Pipeline ---
    logger.info("CRAG is enabled. Executing Corrective RAG pipeline.")
    
    # Step 1: Embed query and search FAISS with scaled top_k
    query_embedding = embed_query(query_en, model_name=model_name)
    initial_top_k = max(top_k, top_k * CRAG_INITIAL_TOPK_MULTIPLIER)
    
    logger.info(f"CRAG: Performing initial dense search (top_k={initial_top_k})")
    raw_results = vector_store.search(query_embedding, top_k=initial_top_k)
    
    initial_chunks = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity_score=score,
            token_count=chunk.token_count,
        )
        for chunk, score in raw_results
    ]
    
    # Step 2: Evaluate the relevance of the retrieved chunks
    crag = CRAGPipeline()
    graded_chunks = crag.evaluate_relevance(query_en, initial_chunks)
    
    # Count how many initial chunks are relevant
    num_relevant = sum(1 for _, grade in graded_chunks if grade.is_relevant)
    logger.info(f"CRAG: Initial evaluation found {num_relevant}/{len(graded_chunks)} relevant chunks.")
    
    # Step 3: Trigger Query Rewrite & Fallback Search if context quality is low (0 relevant chunks)
    rewritten_graded_chunks = []
    if num_relevant == 0:
        logger.warning("CRAG: Low-confidence retrieval detected (0 relevant chunks). Triggering fallback rewriter...")
        rewritten_query_en = crag.rewrite_query(query_en)
        
        if rewritten_query_en != query_en:
            logger.info("CRAG: Performing fallback search with rewritten query.")
            rewritten_embedding = embed_query(rewritten_query_en, model_name=model_name)
            rewritten_raw_results = vector_store.search(rewritten_embedding, top_k=initial_top_k)
            
            rewritten_chunks = [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    similarity_score=score,
                    token_count=chunk.token_count,
                )
                for chunk, score in rewritten_raw_results
            ]
            
            # Evaluate new chunks against the rewritten query
            rewritten_graded_chunks = crag.evaluate_relevance(rewritten_query_en, rewritten_chunks)
            num_rewritten_relevant = sum(1 for _, grade in rewritten_graded_chunks if grade.is_relevant)
            logger.info(f"CRAG: Fallback evaluation found {num_rewritten_relevant}/{len(rewritten_graded_chunks)} relevant chunks.")
        else:
            logger.info("CRAG: Rewriter returned identical query. Skipping secondary search.")

    # Step 4: Merge, Deduplicate, and Filter
    merged_graded: Dict[int, Tuple[RetrievedChunk, Any]] = {}
    
    # Process original graded chunks
    for chunk, grade in graded_chunks:
        merged_graded[chunk.chunk_id] = (chunk, grade)
        
    # Process rewritten graded chunks (higher score replaces lower score on duplicate)
    for chunk, grade in rewritten_graded_chunks:
        cid = chunk.chunk_id
        if cid not in merged_graded:
            merged_graded[cid] = (chunk, grade)
        else:
            _, existing_grade = merged_graded[cid]
            if grade.relevance_score > existing_grade.relevance_score:
                merged_graded[cid] = (chunk, grade)

    all_graded = list(merged_graded.values())
    logger.info(f"CRAG: Deduplicated merged set size: {len(all_graded)} chunks.")

    # Keep only relevant chunks
    relevant_graded = [
        (chunk, grade) for chunk, grade in all_graded 
        if grade.is_relevant or grade.relevance_score >= CRAG_RELEVANCE_THRESHOLD
    ]
    
    logger.info(f"CRAG: Filtered out {len(all_graded) - len(relevant_graded)} irrelevant chunks.")

    # Fallback: if STILL no relevant chunks are found after rewrite & filtering,
    # keep the top-2 chunks from the original search to guarantee downstream compatibility (refusal flow).
    if not relevant_graded:
        logger.warning("CRAG: No relevant chunks found in the entire pipeline. Proceeding with best irrelevant chunks for refusal flow.")
        # Mark them as not relevant but keep them so LLM has the context to form a proper refusal
        relevant_graded = graded_chunks[:2]

    # Step 5: Rerank the surviving chunks using the hybrid formula
    reranked_chunks = crag.rerank(relevant_graded)
    
    # Step 6: Truncate to top_k
    final_chunks = reranked_chunks[:top_k]
    logger.info(f"CRAG: Pipeline completed. Truncated {len(reranked_chunks)} chunks down to final top_k={len(final_chunks)}.")

    result = RetrievalResult(
        query_original=query,
        query_translated=query_en,
        detected_language=detected_lang,
        retrieved_chunks=final_chunks,
        top_k=top_k,
    )
    
    return result
