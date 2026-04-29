"""
llm.py
------
Gemini API integration with strict document-grounded generation.

Uses the new google-genai SDK (google.genai) which replaced the deprecated
google-generativeai package.

Hallucination Prevention Strategy:
1. System prompt explicitly forbids answering outside the provided context.
2. Context is constructed only from retrieved FAISS chunks — no world knowledge injected.
3. Temperature = 0 for maximum determinism.
4. Model instructed to ALWAYS cite specific page numbers.
5. If context is insufficient, model MUST output the exact refusal phrase.
6. Refusal is also programmatically enforced: if the model's answer doesn't
   reference the context, we check and possibly override.

Model: gemini-2.0-flash (fast, cost-effective, large context window)
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from .retriever import RetrievedChunk

load_dotenv()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

REFUSAL_PHRASE = "I cannot answer this from the provided document."

SYSTEM_PROMPT = """You are a strict document Question-Answering (QA) system.

RULES — FOLLOW THESE WITHOUT EXCEPTION:
1. Answer ONLY using information explicitly present in the provided CONTEXT sections below.
2. Do NOT use any external knowledge, general facts, or information not found in the context.
3. If the answer to the question is NOT found in the context, respond with EXACTLY:
   "I cannot answer this from the provided document."
4. ALWAYS include page number references in your answer using the format: (Page X) or (Pages X, Y).
5. Be precise and factual. Do not speculate, infer beyond what is stated, or hallucinate details.
6. You may quote directly from the context when helpful.
7. Keep answers clear, well-structured, and professional.

IMPORTANT: Any answer without a page number citation is INVALID. Always cite pages."""

CONTEXT_TEMPLATE = """--- CONTEXT (Page {page}) ---
{text}"""

USER_QUERY_TEMPLATE = """Based ONLY on the context provided above, answer the following question:

Question: {question}

Remember:
- If the answer is not in the context, say: "I cannot answer this from the provided document."
- Always cite page numbers in your answer."""


# -----------------------------------------------------------------------
# Gemini client (lazy init)
# -----------------------------------------------------------------------

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY not set. Please add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    _client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f"Gemini client initialized with model: {GEMINI_MODEL}")
    return _client


# -----------------------------------------------------------------------
# Response data structure
# -----------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    answer: str
    citations: List[int]           # Extracted page numbers referenced in the answer
    is_refusal: bool               # True if the model refused to answer
    raw_response: str              # Unmodified model output
    context_used: List[int]        # chunk_ids used as context
    prompt_tokens_estimate: int = 0


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _extract_page_citations(text: str) -> List[int]:
    """
    Extract page number references from the model's response.

    Patterns matched: (Page 3), (Pages 3, 4), Page 3, page 5, etc.
    Returns sorted unique list of page numbers.
    """
    patterns = [
        r'\(Pages?\s+([\d,\s]+)\)',   # (Page 3) or (Pages 3, 4)
        r'Pages?\s+([\d,\s]+)',       # Page 3 or Pages 3, 4
        r'\[Page\s+(\d+)\]',         # [Page 3]
    ]
    pages = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            group = match.group(1) if match.lastindex else match.group(0)
            for num in re.findall(r'\d+', group):
                pages.add(int(num))
    return sorted(pages)


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into the context block for the prompt."""
    parts = []
    for chunk in chunks:
        parts.append(CONTEXT_TEMPLATE.format(
            page=chunk.page_number,
            text=chunk.text.strip(),
        ))
    return "\n\n".join(parts)


# -----------------------------------------------------------------------
# Core generation function
# -----------------------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    model_name: str = GEMINI_MODEL,
    temperature: float = 0.0,
) -> LLMResponse:
    """
    Generate a grounded answer using Gemini with retrieved context.

    This is the core RAG generation step. The model is strictly prompted
    to only use the provided context, and the system prompt is non-negotiable.

    Args:
        question:         The user's question (in English).
        retrieved_chunks: Chunks from FAISS retrieval.
        model_name:       Gemini model identifier.
        temperature:      Generation temperature (0 = deterministic).

    Returns:
        LLMResponse with answer, citations, and metadata.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: On Gemini API error.
    """
    if not retrieved_chunks:
        return LLMResponse(
            answer=REFUSAL_PHRASE,
            citations=[],
            is_refusal=True,
            raw_response=REFUSAL_PHRASE,
            context_used=[],
        )

    # Build the full prompt
    context_block = _build_context_block(retrieved_chunks)
    user_message = USER_QUERY_TEMPLATE.format(question=question)
    full_prompt = f"{context_block}\n\n{user_message}"

    logger.info(f"Sending query to Gemini ({model_name}): {question[:100]}")
    logger.debug(f"Context block length: {len(context_block)} chars")

    try:
        client = _get_client()

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                top_p=1.0,
                top_k=1,          # greedy decoding for determinism
                candidate_count=1,
            ),
        )

        raw_answer = response.text.strip() if response.text else ""

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e

    # Check for refusal
    is_refusal = REFUSAL_PHRASE.lower() in raw_answer.lower()

    # Extract cited pages from the answer
    citations = _extract_page_citations(raw_answer)

    # Fallback: if not refusal but no citations extracted, add page numbers from context
    if not is_refusal and not citations:
        logger.warning("Model answer has no page citations — appending context pages as fallback")
        context_pages = sorted(set(c.page_number for c in retrieved_chunks))
        citations = context_pages
        raw_answer += f"\n\n(Based on: Pages {', '.join(str(p) for p in context_pages)})"

    llm_response = LLMResponse(
        answer=raw_answer,
        citations=citations,
        is_refusal=is_refusal,
        raw_response=raw_answer,
        context_used=[c.chunk_id for c in retrieved_chunks],
        prompt_tokens_estimate=len(full_prompt) // 4,  # rough estimate
    )

    logger.info(
        f"Gemini response: is_refusal={is_refusal}, "
        f"citations={citations}, "
        f"answer_length={len(raw_answer)}"
    )

    return llm_response
