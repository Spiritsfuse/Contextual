"""
backend/crag.py
---------------
Corrective Retrieval-Augmented Generation (CRAG) pipeline components.

Features:
1. Relevance Evaluator: Grades retrieved chunks as RELEVANT or IRRELEVANT with a confidence score.
2. Query Rewriter: Reformulates vague queries into optimized keyword search terms for fallback retrieval.
3. Reranker: Deterministically sorts chunks by relevance and similarity with stable tie-breaking.
4. Observability: Emits detailed, structured logs of all CRAG execution steps.
5. Latency Cache: Caches evaluations and rewritten queries in memory to avoid redundant API calls.
6. Failure Resilience: Gracefully falls back to standard RAG if the Gemini API fails or times out.
"""

import logging
import os
import time
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from google.genai import types
from dotenv import load_dotenv

# Import client loader from LLM module to share the connection pool
from .llm import _get_client, GEMINI_PRIMARY_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

# Configurable parameters with fallbacks
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "0.3"))
CRAG_INITIAL_TOPK_MULTIPLIER = int(os.getenv("CRAG_INITIAL_TOPK_MULTIPLIER", "2"))
CRAG_ENABLE_QUERY_REWRITE = os.getenv("CRAG_ENABLE_QUERY_REWRITE", "True").lower() == "true"

# -----------------------------------------------------------------------
# Caches to minimize API calls and latency
# -----------------------------------------------------------------------
# Cache for relevance evaluation: Key = (query, chunk_text) -> Value = ChunkRelevance
_evaluation_cache: Dict[Tuple[str, str], Any] = {}
# Cache for rewritten queries: Key = original_query -> Value = rewritten_query
_rewrite_cache: Dict[str, str] = {}


def clear_crag_caches() -> None:
    """Clear all memory caches used by the CRAG pipeline. (Useful for testing)"""
    _evaluation_cache.clear()
    _rewrite_cache.clear()
    logger.info("CRAG in-memory caches cleared.")


# -----------------------------------------------------------------------
# Pydantic Models for Structured Output
# -----------------------------------------------------------------------

class ChunkRelevance(BaseModel):
    """Relevance evaluation for a single document chunk."""
    chunk_id: int = Field(..., description="The ID of the chunk being evaluated.")
    relevance_score: float = Field(
        ...,
        description="A score from 0.0 (completely irrelevant) to 1.0 (highly relevant) indicating if this chunk helps answer the query."
    )
    is_relevant: bool = Field(
        ...,
        description="True if the chunk contains information directly relevant or partially relevant to the query (score >= 0.3); False if entirely irrelevant."
    )
    reason: str = Field(
        ...,
        description="A brief, one-sentence explanation of why the chunk is or is not relevant, citing specific keywords."
    )


class RetrievalEvaluation(BaseModel):
    """Batch evaluation of multiple document chunks."""
    evaluations: List[ChunkRelevance] = Field(..., description="List of chunk evaluations.")


# -----------------------------------------------------------------------
# Core Evaluator Prompts
# -----------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = f"""You are a highly precise document retrieval evaluator for a strict question-answering system.
Your task is to evaluate the relevance of the retrieved document chunks to the user's search query.

For each chunk:
1. Carefully read the chunk text and compare it to the user's query.
2. Determine if the chunk contains information that directly helps answer the query (RELEVANT), contains secondary or partial context (PARTIALLY RELEVANT), or contains nothing related to the query (IRRELEVANT).
3. Assign a relevance_score:
   - 0.8 to 1.0 for highly relevant chunks that directly answer the query or major parts of it.
   - 0.3 to 0.7 for partially relevant chunks that provide useful background, related context, or partial details.
   - 0.0 to 0.2 for irrelevant chunks that do not mention the search topic or are completely out of context.
4. Set `is_relevant` to True if the score is >= {CRAG_RELEVANCE_THRESHOLD}, otherwise set it to False.
5. Provide a short, one-sentence reason for your decision.

CRITICAL HALLUCINATION PREVENTION:
- Only evaluate the relevance based on the EXPLICIT text in the provided chunk.
- Do NOT assume, infer, or use outside knowledge to justify relevance.
- If the chunk does not explicitly relate, it must be graded as irrelevant.
- Do NOT hallucinate relevance under any circumstances."""


EVALUATOR_USER_TEMPLATE = """User Query: {query}

Retrieved Chunks to Evaluate:
{chunks_formatted}"""


REWRITER_SYSTEM_PROMPT = """You are a search query optimizer. The user's initial search query yielded poor or irrelevant results from a document index.
Your goal is to reformulate the user's query into a concise, high-impact search term or set of terms optimized for keyword/concept matching in a PDF document.

Rules:
1. Focus on the core factual entities, medical terms, and concepts in the query.
2. Remove conversational filler, polite phrasing, and generic question words (like "please tell me", "what is", "do you know").
3. Keep it brief -- ideally 3 to 6 keywords or a clean factual phrase.
4. Return ONLY the optimized query string. Do not include any explanation or introduction."""


# -----------------------------------------------------------------------
# CRAG Pipeline Orchestrator Class
# -----------------------------------------------------------------------

class CRAGPipeline:
    """Orchestrates the Corrective RAG pipeline."""

    def __init__(self, model_name: str = GEMINI_PRIMARY_MODEL):
        self.model_name = model_name

    def evaluate_relevance(
        self,
        query: str,
        chunks: List[Any],
    ) -> List[Tuple[Any, ChunkRelevance]]:
        """
        Evaluate the relevance of retrieved chunks to the query.
        Uses in-memory caching for repeat queries/chunks, and batches new evaluations to Gemini.

        Args:
            query:  The user query (in English).
            chunks: List of RetrievedChunk objects.

        Returns:
            List of (chunk, ChunkRelevance) tuples.
        """
        if not chunks:
            return []

        logger.info(f"CRAG: Evaluating relevance of {len(chunks)} chunks for query: '{query[:60]}...'")

        # Step 1: Check cache and divide into cached vs to-evaluate
        results: List[Optional[Tuple[Any, ChunkRelevance]]] = [None] * len(chunks)
        chunks_to_eval_idx: List[int] = []
        chunks_to_eval_list: List[Any] = []

        for idx, chunk in enumerate(chunks):
            cache_key = (query, chunk.text)
            if cache_key in _evaluation_cache:
                logger.debug(f"CRAG: Evaluation cache HIT for chunk_id={chunk.chunk_id}")
                results[idx] = (chunk, _evaluation_cache[cache_key])
            else:
                chunks_to_eval_idx.append(idx)
                chunks_to_eval_list.append(chunk)

        # Step 2: Batch evaluate remaining chunks via Gemini
        if chunks_to_eval_list:
            logger.info(f"CRAG: Cache miss for {len(chunks_to_eval_list)} chunks. Calling Gemini batch evaluator...")
            
            # Format chunks for prompt
            chunks_formatted = ""
            for chunk in chunks_to_eval_list:
                chunks_formatted += f"--- Chunk ID: {chunk.chunk_id} (Page: {chunk.page_number}) ---\n{chunk.text}\n\n"

            user_prompt = EVALUATOR_USER_TEMPLATE.format(query=query, chunks_formatted=chunks_formatted)

            try:
                client = _get_client()
                
                # Add strict timeout handling (e.g. 15 seconds) using a request configuration if available,
                # or fallback to general try/except.
                start_time = time.time()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=EVALUATOR_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=RetrievalEvaluation,
                        temperature=0.0,
                    )
                )
                latency = time.time() - start_time
                logger.info(f"CRAG: Gemini evaluation completed in {latency:.3f}s")

                # Parse the structured response
                eval_data = RetrievalEvaluation.model_validate_json(response.text.strip())
                
                # Create a map for quick access
                eval_map: Dict[int, ChunkRelevance] = {e.chunk_id: e for e in eval_data.evaluations}

                # Map back to initial chunks and update cache
                for idx, chunk_idx in enumerate(chunks_to_eval_idx):
                    chunk = chunks_to_eval_list[idx]
                    # Find evaluation by ID, or create a safe fallback if LLM omitted it
                    eval_obj = eval_map.get(chunk.chunk_id)
                    if not eval_obj:
                        logger.warning(f"CRAG: LLM omitted evaluation for chunk_id={chunk.chunk_id}. Using low-relevance fallback.")
                        eval_obj = ChunkRelevance(
                            chunk_id=chunk.chunk_id,
                            relevance_score=0.1,
                            is_relevant=False,
                            reason="Evaluation omitted by model."
                        )
                    
                    # Store in cache
                    _evaluation_cache[(query, chunk.text)] = eval_obj
                    results[chunk_idx] = (chunk, eval_obj)

            except Exception as e:
                logger.error(f"CRAG: Gemini evaluation failed: {e}. Falling back to standard RAG grades.", exc_info=True)
                # Fail-safe: Mark all chunks as relevant with safe similarity scores to avoid breaking downstream generation
                for idx, chunk_idx in enumerate(chunks_to_eval_idx):
                    chunk = chunks_to_eval_list[idx]
                    fallback_eval = ChunkRelevance(
                        chunk_id=chunk.chunk_id,
                        relevance_score=chunk.similarity_score,
                        is_relevant=True,
                        reason="Fallback due to evaluation timeout/failure."
                    )
                    results[chunk_idx] = (chunk, fallback_eval)

        # Assemble final results (guaranteed to be fully populated)
        final_results = [r for r in results if r is not None]
        
        # Log evaluation results summary
        log_grades = [
            f"[ID={r[0].chunk_id}|Page={r[0].page_number}|Score={r[1].relevance_score:.2f}|Rel={r[1].is_relevant}]" 
            for r in final_results
        ]
        logger.info(f"CRAG: Evaluation results: {', '.join(log_grades)}")

        return final_results

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query to optimize it for FAISS retrieval.
        Utilizes in-memory caching.

        Args:
            query: The original user query.

        Returns:
            An optimized keyword-based query.
        """
        if not CRAG_ENABLE_QUERY_REWRITE:
            logger.info("CRAG: Query rewrite is disabled by config. Returning original query.")
            return query

        if query in _rewrite_cache:
            rewritten = _rewrite_cache[query]
            logger.info(f"CRAG: Query rewrite cache HIT: '{query}' -> '{rewritten}'")
            return rewritten

        logger.info(f"CRAG: Rewriting low-confidence query: '{query}'")
        try:
            client = _get_client()
            start_time = time.time()
            response = client.models.generate_content(
                model=self.model_name,
                contents=f"Optimize this query: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=REWRITER_SYSTEM_PROMPT,
                    temperature=0.0,
                )
            )
            latency = time.time() - start_time
            rewritten = response.text.strip()
            
            # Sanitization (ensure it is not empty)
            if not rewritten:
                rewritten = query
                
            logger.info(f"CRAG: Query rewritten in {latency:.3f}s. Original: '{query}' -> Rewritten: '{rewritten}'")
            _rewrite_cache[query] = rewritten
            return rewritten

        except Exception as e:
            logger.warning(f"CRAG: Query rewrite failed: {e}. Falling back to original query.")
            return query

    def rerank(self, graded_chunks: List[Tuple[Any, ChunkRelevance]]) -> List[Any]:
        """
        Sort chunks descending based on a hybrid relevance + similarity score,
        with stable tie-breaking using chunk_id.

        Rerank Score Formula:
            score = (relevance_score * 0.7) + (similarity_score * 0.3)

        Tie-breakers:
            1. relevance_score (descending)
            2. similarity_score (descending)
            3. chunk_id (ascending) -> strictly stable and deterministic!

        Args:
            graded_chunks: List of (chunk, ChunkRelevance) tuples.

        Returns:
            Sorted list of chunk objects.
        """
        if not graded_chunks:
            return []

        # List to store (chunk, sort_key)
        # We want descending order for scores, so we negate them in Python's default ascending sort,
        # or we sort descending. Let's build a deterministic sort key and sort:
        # key = (is_relevant, rerank_score, similarity_score, -chunk_id)
        # Let's sort using a lambda key to ensure deterministic order:
        def get_sort_key(item: Tuple[Any, ChunkRelevance]) -> Tuple[int, float, float, int]:
            chunk, grade = item
            is_relevant_val = 1 if grade.is_relevant else 0
            hybrid_score = (grade.relevance_score * 0.7) + (chunk.similarity_score * 0.3)
            # Python's sort is stable. By returning (is_relevant, hybrid_score, similarity_score, -chunk_id)
            # and sorting reverse=True, we get:
            # 1. is_relevant = True (1) before False (0)
            # 2. Higher hybrid_score first
            # 3. Higher similarity_score first
            # 4. Lower chunk_id first (since we negate it, a lower chunk_id e.g. 2 -> -2 is larger than 5 -> -5)
            return (is_relevant_val, hybrid_score, chunk.similarity_score, -chunk.chunk_id)

        sorted_items = sorted(graded_chunks, key=get_sort_key, reverse=True)
        
        # Log reranking adjustments
        logger.info("CRAG: Reranked chunk ordering:")
        for idx, (chunk, grade) in enumerate(sorted_items, 1):
            hybrid = (grade.relevance_score * 0.7) + (chunk.similarity_score * 0.3)
            logger.info(
                f"  [{idx}] chunk_id={chunk.chunk_id} | page={chunk.page_number} | "
                f"hybrid_score={hybrid:.4f} (rel={grade.relevance_score:.2f}, sim={chunk.similarity_score:.4f}) | "
                f"is_relevant={grade.is_relevant}"
            )

        return [item[0] for item in sorted_items]
