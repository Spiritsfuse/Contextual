"""
vector_store.py
---------------
FAISS-based vector store with persistence.

Design:
- Uses faiss.IndexFlatIP (exact inner product search).
- Since embeddings are L2-normalized, inner product == cosine similarity.
- Stores chunk metadata (text, page_number, chunk_id) in a parallel list.
- Supports save/load to disk for session persistence across restarts.
- IndexFlatIP is chosen over HNSW for determinism and small corpora
  (PDFs rarely exceed 10K chunks). For larger scale, swap to IndexHNSWFlat.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from .pdf_processor import TextChunk

logger = logging.getLogger(__name__)

# Persistent storage paths
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "metadata.pkl"


class VectorStore:
    """
    Thread-safe FAISS vector store for PDF chunk embeddings.

    Attributes:
        index:     FAISS index (IndexFlatIP).
        metadata:  Parallel list of TextChunk objects aligned to index vectors.
        dim:       Embedding dimensionality.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.metadata: List[TextChunk] = []
        logger.info(f"VectorStore initialized (dim={dim})")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks:     List of TextChunk objects.
            embeddings: np.ndarray (N, dim), float32, L2-normalized.

        Raises:
            ValueError: If shapes are mismatched or embeddings are wrong dtype.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({embeddings.shape[0]})"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        # Ensure embeddings are C-contiguous (FAISS requirement)
        embeddings = np.ascontiguousarray(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search for the top-k most similar chunks.

        Args:
            query_embedding: np.ndarray of shape (1, dim), float32, L2-normalized.
            top_k:           Number of results to return.

        Returns:
            List of (TextChunk, similarity_score) tuples, sorted descending by score.
            Similarity scores are cosine similarities in [-1, 1].
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        k = min(top_k, self.index.ntotal)
        query_embedding = np.ascontiguousarray(
            query_embedding.astype(np.float32)
        )

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for no result
                continue
            chunk = self.metadata[idx]
            results.append((chunk, float(score)))

        logger.debug(
            f"Search top-{k}: {[(r[0].chunk_id, round(r[1], 4)) for r in results]}"
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = index_path or INDEX_FILE
        meta_path = meta_path or META_FILE

        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Index saved: {index_path} ({self.index.ntotal} vectors)")

    def load(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> bool:
        """
        Load a persisted index from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        index_path = index_path or INDEX_FILE
        meta_path = meta_path or META_FILE

        if not index_path.exists() or not meta_path.exists():
            logger.info("No persisted index found — starting fresh")
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"Index loaded: {self.index.ntotal} vectors from {index_path}")
        return True

    def reset(self) -> None:
        """Clear the index and metadata in memory and on disk."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []

        # Remove persisted files if they exist
        for f in [INDEX_FILE, META_FILE]:
            if f.exists():
                f.unlink()

        logger.info("VectorStore reset: index cleared")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.index.ntotal

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def info(self) -> dict:
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.dim,
            "index_type": type(self.index).__name__,
        }
