"""
embeddings.py
-------------
Manages embeddings via Google Gemini's Embedding API.

Design decisions:
- Model: text-embedding-004 (768-dim, high quality, cloud-based)
- Efficiency: Cloud-based, removing need for massive Torch/local model dependencies (fixing Render OOM).
- L2-normalization: Embeddings are normalized to support dot product similarity.
"""

import logging
import os
from typing import List

import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "models/gemini-embedding-001"

# Global client for reuse
_client_instance: genai.Client | None = None

def get_client() -> genai.Client:
    """Singleton for the GenAI client."""
    global _client_instance
    if _client_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _client_instance = genai.Client(api_key=api_key)
    return _client_instance

def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 100,  # Gemini supports large batches
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings using Google's API.
    Returns: np.ndarray of shape (len(texts), 768), dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    client = get_client()
    logger.info(f"Embedding {len(texts)} texts via Google API")

    all_embeddings = []
    
    # Process in batches (Gemini has limits on total tokens/items per call)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        
        # Extract vectors
        for emb in response.embeddings:
            v = np.array(emb.values, dtype=np.float32)
            # L2-normalize
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            all_embeddings.append(v)

    embeddings = np.stack(all_embeddings)
    logger.info(f"Embeddings generated: {embeddings.shape}")
    return embeddings

def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Embed a single query string."""
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    client = get_client()
    response = client.models.embed_content(
        model=model_name,
        contents=query.strip(),
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    
    v = np.array(response.embeddings[0].values, dtype=np.float32)
    # L2-normalize
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
        
    return v.reshape(1, -1)

def get_embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the dimensionality (768 for text-embedding-004)."""
    return 768

