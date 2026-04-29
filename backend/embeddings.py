"""
embeddings.py
-------------
Manages the sentence-transformers embedding model.

Design decisions:
- Model: all-MiniLM-L6-v2 (384-dim, ~80MB, fast inference, strong retrieval quality)
- Singleton pattern: model loaded once, reused across requests.
- Embeddings are L2-normalized before return, enabling cosine similarity
  via simple dot product (FAISS IndexFlatIP).
- Deterministic: same text always produces same embedding (no randomness).
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model — change here to upgrade (e.g. all-mpnet-base-v2 for higher accuracy)
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Module-level singleton
_model_instance: SentenceTransformer | None = None
_model_name_loaded: str | None = None


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Return the singleton embedding model, loading it on first call.

    Thread-safe note: In a multi-worker server, each worker process has
    its own singleton (standard Python behavior). The model is read-only
    after loading so this is safe.
    """
    global _model_instance, _model_name_loaded

    if _model_instance is None or _model_name_loaded != model_name:
        logger.info(f"Loading sentence-transformer model: {model_name}")
        _model_instance = SentenceTransformer(model_name)
        _model_name_loaded = model_name
        logger.info("Embedding model loaded successfully")

    return _model_instance


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of text strings and return L2-normalized vectors.

    Args:
        texts:        List of strings to embed.
        model_name:   sentence-transformers model name.
        batch_size:   Encoding batch size (tune for memory/speed trade-off).
        show_progress: Show tqdm progress bar (useful for large corpora).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_model(model_name)
    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,    # L2-normalize → cosine sim = dot product
        convert_to_numpy=True,
    )

    # Ensure float32 (FAISS requirement)
    embeddings = embeddings.astype(np.float32)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        np.ndarray of shape (1, embedding_dim), float32, L2-normalized.
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    result = embed_texts([query.strip()], model_name=model_name, batch_size=1)
    return result  # shape (1, dim)


def get_embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the embedding dimensionality for the given model."""
    model = get_model(model_name)
    # API changed in sentence-transformers 3.x: prefer get_embedding_dimension
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()  # legacy fallback
