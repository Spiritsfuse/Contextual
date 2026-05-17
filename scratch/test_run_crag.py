"""
scratch/test_run_crag.py
------------------------
End-to-end manual verification of the Corrective RAG (CRAG) pipeline.
It loads, processes, and indexes the sample PDF, then runs tests to demonstrate:
1. High-confidence path (relevant query).
2. Fallback query-rewriting path (vague/obfuscated query).
3. Out-of-scope query handling.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("crag_test_run")

from backend.pdf_processor import process_pdf
from backend.embeddings import embed_texts, get_embedding_dim
from backend.vector_store import VectorStore
from backend.retriever import retrieve
from backend.llm import generate_answer

def run_verification():
    pdf_path = "data/sample_ai_healthcare.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"Sample PDF not found at {pdf_path}. Please run generate_sample_pdf.py first.")
        return

    logger.info("1. Processing PDF...")
    chunks = process_pdf(pdf_path, chunk_size=200, overlap=30)
    logger.info(f"Generated {len(chunks)} chunks.")

    logger.info("2. Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, show_progress=False)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    logger.info("3. Populating FAISS VectorStore...")
    dim = get_embedding_dim()
    store = VectorStore(dim=dim)
    store.reset()
    store.add_chunks(chunks, embeddings)
    logger.info("Vector store populated.")

    # Test Query 1: Direct Relevant Query (Expected: High relevance, direct answer)
    query_1 = "What are the main applications of AI in healthcare?"
    logger.info(f"\n==================================================")
    logger.info(f"TEST 1: HIGH RELEVANCE QUERY")
    logger.info(f"Query: '{query_1}'")
    logger.info(f"==================================================")
    
    retrieval_1 = retrieve(query_1, store, top_k=3, use_crag=True)
    logger.info(f"Retrieved {len(retrieval_1.retrieved_chunks)} final chunks.")
    
    answer_1 = generate_answer(retrieval_1.query_translated, retrieval_1.retrieved_chunks)
    logger.info(f"Grounded Answer:\n{answer_1.answer}\n")

    # Test Query 2: Obfuscated Query targeting deep concepts (Expected: Fallback rewrite -> successful retrieval)
    # The document describes Insilico Medicine's drug design. Let's make it vague to trigger rewrite.
    query_2 = "tell me about that company that made a compound for fibrosis"
    logger.info(f"\n==================================================")
    logger.info(f"TEST 2: OBFUSCATED QUERY (EXPECTED REWRITE FALLBACK)")
    logger.info(f"Query: '{query_2}'")
    logger.info(f"==================================================")
    
    retrieval_2 = retrieve(query_2, store, top_k=3, use_crag=True)
    logger.info(f"Retrieved {len(retrieval_2.retrieved_chunks)} final chunks.")
    
    answer_2 = generate_answer(retrieval_2.query_translated, retrieval_2.retrieved_chunks)
    logger.info(f"Grounded Answer:\n{answer_2.answer}\n")

    # Test Query 3: Completely Out of Scope Query (Expected: Irrelevant chunks -> rewrite -> still irrelevant -> refusal)
    query_3 = "What is the capital of France?"
    logger.info(f"\n==================================================")
    logger.info(f"TEST 3: OUT OF SCOPE QUERY (EXPECTED REFUSAL)")
    logger.info(f"Query: '{query_3}'")
    logger.info(f"==================================================")
    
    retrieval_3 = retrieve(query_3, store, top_k=3, use_crag=True)
    logger.info(f"Retrieved {len(retrieval_3.retrieved_chunks)} final chunks.")
    
    answer_3 = generate_answer(retrieval_3.query_translated, retrieval_3.retrieved_chunks)
    logger.info(f"Grounded Answer:\n{answer_3.answer}\n")

if __name__ == "__main__":
    run_verification()
