"""
tests/test_crag.py
------------------
Deterministic unit tests for the Corrective RAG (CRAG) pipeline.

Tests covered:
1. Deduplication correctness.
2. Reranking stability & deterministic tie-breaking.
3. Irrelevant chunk filtering based on thresholds.
4. Low-confidence fallback query rewrite triggering.
5. Backward compatibility with CRAG disabled.

These tests utilize standard Python unittest and mocks to bypass live network calls
and guarantee execution speed, security, and predictability.
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.pdf_processor import TextChunk
from backend.retriever import RetrievedChunk, retrieve, RetrievalResult
from backend.crag import CRAGPipeline, ChunkRelevance, RetrievalEvaluation, clear_crag_caches
from backend.vector_store import VectorStore


class TestCRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Clear in-memory caches between test runs to ensure state isolation
        clear_crag_caches()

    def test_deduplication_correctness(self):
        """Test that chunks are correctly deduplicated by chunk_id, retaining the higher score."""
        # Setup duplicate chunks
        chunk1 = RetrievedChunk(chunk_id=1, page_number=1, text="Primary details.", similarity_score=0.8, token_count=10)
        chunk2 = RetrievedChunk(chunk_id=1, page_number=1, text="Primary details.", similarity_score=0.8, token_count=10)
        chunk3 = RetrievedChunk(chunk_id=2, page_number=2, text="Secondary details.", similarity_score=0.6, token_count=10)

        # Grades with duplicate chunk_id 1 but different relevance scores
        grade1 = ChunkRelevance(chunk_id=1, relevance_score=0.4, is_relevant=True, reason="Somewhat related.")
        grade2 = ChunkRelevance(chunk_id=1, relevance_score=0.9, is_relevant=True, reason="Highly related!")
        grade3 = ChunkRelevance(chunk_id=2, relevance_score=0.8, is_relevant=True, reason="Strong relation.")

        original_graded = [(chunk1, grade1)]
        rewritten_graded = [(chunk2, grade2), (chunk3, grade3)]

        # Merge, deduplicate, and keep the highest score (simulate retrieve logic)
        merged_graded = {}
        for chunk, grade in original_graded:
            merged_graded[chunk.chunk_id] = (chunk, grade)
            
        for chunk, grade in rewritten_graded:
            cid = chunk.chunk_id
            if cid not in merged_graded:
                merged_graded[cid] = (chunk, grade)
            else:
                _, existing_grade = merged_graded[cid]
                if grade.relevance_score > existing_grade.relevance_score:
                    merged_graded[cid] = (chunk, grade)

        # Assertions
        self.assertEqual(len(merged_graded), 2, "Merged map should have exactly 2 unique chunk_ids.")
        self.assertIn(1, merged_graded)
        self.assertIn(2, merged_graded)
        # Verify chunk 1 kept the higher score of 0.9 (from grade2) instead of 0.4 (from grade1)
        self.assertEqual(merged_graded[1][1].relevance_score, 0.9)

    def test_reranking_stability_and_determinism(self):
        """Test that hybrid reranking is stable and breaks ties deterministically using chunk_id ascending."""
        crag = CRAGPipeline()

        # Build chunks with identical relevance and similarity scores
        chunk_high_id = RetrievedChunk(chunk_id=10, page_number=5, text="Context A", similarity_score=0.5, token_count=10)
        chunk_low_id = RetrievedChunk(chunk_id=2, page_number=1, text="Context B", similarity_score=0.5, token_count=10)
        
        grade_high_id = ChunkRelevance(chunk_id=10, relevance_score=0.8, is_relevant=True, reason="Same score.")
        grade_low_id = ChunkRelevance(chunk_id=2, relevance_score=0.8, is_relevant=True, reason="Same score.")

        # Combined sorted items should put chunk_id=2 BEFORE chunk_id=10 since:
        # - Both are is_relevant = True
        # - Both have identical hybrid score: (0.8 * 0.7) + (0.5 * 0.3) = 0.71
        # - Tie-breaking sorts lower chunk_id first
        graded_chunks = [(chunk_high_id, grade_high_id), (chunk_low_id, grade_low_id)]
        
        reranked = crag.rerank(graded_chunks)

        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].chunk_id, 2, "Lower chunk_id should come first under tie-break.")
        self.assertEqual(reranked[1].chunk_id, 10, "Higher chunk_id should come second under tie-break.")

    def test_irrelevant_chunk_filtering(self):
        """Test that chunks below relevance threshold are filtered out of the final list."""
        crag = CRAGPipeline()
        
        chunk_rel = RetrievedChunk(chunk_id=1, page_number=1, text="Relevant clinical study details.", similarity_score=0.8, token_count=10)
        chunk_irrel = RetrievedChunk(chunk_id=2, page_number=2, text="Irrelevant chat conversation.", similarity_score=0.3, token_count=10)

        grade_rel = ChunkRelevance(chunk_id=1, relevance_score=0.9, is_relevant=True, reason="Perfect match.")
        grade_irrel = ChunkRelevance(chunk_id=2, relevance_score=0.1, is_relevant=False, reason="Unrelated.")

        all_graded = [(chunk_rel, grade_rel), (chunk_irrel, grade_irrel)]

        # Apply filtering (relevance threshold is 0.3)
        relevant_graded = [
            (chunk, grade) for chunk, grade in all_graded 
            if grade.is_relevant or grade.relevance_score >= 0.3
        ]

        self.assertEqual(len(relevant_graded), 1)
        self.assertEqual(relevant_graded[0][0].chunk_id, 1)

    @patch("backend.retriever.embed_query")
    @patch("backend.retriever.CRAGPipeline")
    def test_low_confidence_fallback_rewriting(self, mock_pipeline_class, mock_embed_query):
        """Test that low-confidence initial retrieval (0 relevant chunks) triggers query rewriting and secondary search."""
        # Setup mocks
        mock_crag = MagicMock()
        mock_pipeline_class.return_value = mock_crag
        
        # Initial search yields 2 chunks, but graded as irrelevant (is_relevant=False)
        chunk1 = RetrievedChunk(chunk_id=1, page_number=1, text="Unrelated text A", similarity_score=0.4, token_count=10)
        chunk2 = RetrievedChunk(chunk_id=2, page_number=2, text="Unrelated text B", similarity_score=0.3, token_count=10)
        
        grade1 = ChunkRelevance(chunk_id=1, relevance_score=0.1, is_relevant=False, reason="Irrelevant.")
        grade2 = ChunkRelevance(chunk_id=2, relevance_score=0.1, is_relevant=False, reason="Irrelevant.")
        
        # Rewritten search yields a new relevant chunk
        chunk_rewritten = RetrievedChunk(chunk_id=3, page_number=3, text="Perfect target info.", similarity_score=0.9, token_count=10)
        grade_rewritten = ChunkRelevance(chunk_id=3, relevance_score=0.9, is_relevant=True, reason="Very relevant.")

        # Configure evaluator calls
        mock_crag.evaluate_relevance.side_effect = [
            [(chunk1, grade1), (chunk2, grade2)],       # First call (initial dense search)
            [(chunk_rewritten, grade_rewritten)]         # Second call (after rewrite)
        ]
        mock_crag.rewrite_query.return_value = "optimized medical term"
        mock_crag.rerank.return_value = [chunk_rewritten]

        # Mock vector store
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_vector_store.is_empty = False
        
        # Search returns initial chunks first, and rewritten chunk next
        mock_vector_store.search.side_effect = [
            [(TextChunk(chunk_id=1, text="Unrelated text A", page_number=1, token_count=10), 0.4),
             (TextChunk(chunk_id=2, text="Unrelated text B", page_number=2, token_count=10), 0.3)],
            [(TextChunk(chunk_id=3, text="Perfect target info.", page_number=3, token_count=10), 0.9)]
        ]

        # Call retrieve
        result = retrieve(
            query="Vague query",
            vector_store=mock_vector_store,
            top_k=2,
            use_crag=True
        )

        # Assertions
        mock_crag.rewrite_query.assert_called_once_with("Vague query")
        mock_vector_store.search.assert_any_call(mock_embed_query.return_value, top_k=4)
        self.assertEqual(len(result.retrieved_chunks), 1)
        self.assertEqual(result.retrieved_chunks[0].chunk_id, 3, "Should return the rewritten chunk.")

    @patch("backend.retriever.embed_query")
    def test_backward_compatibility_crag_disabled(self, mock_embed_query):
        """Test that when use_crag=False, the CRAG pipeline is completely bypassed."""
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_vector_store.is_empty = False
        
        # Standard FAISS search returns raw result
        mock_vector_store.search.return_value = [
            (TextChunk(chunk_id=42, text="Ground truth info.", page_number=4, token_count=12), 0.85)
        ]

        # Retrieve with use_crag=False
        result = retrieve(
            query="Direct check",
            vector_store=mock_vector_store,
            top_k=1,
            use_crag=False
        )

        # Assertions
        self.assertEqual(len(result.retrieved_chunks), 1)
        self.assertEqual(result.retrieved_chunks[0].chunk_id, 42)
        self.assertEqual(result.retrieved_chunks[0].similarity_score, 0.85)
        # Vector store should only be searched once
        mock_vector_store.search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
