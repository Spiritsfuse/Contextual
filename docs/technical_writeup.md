# Technical Write-Up: PDF-Constrained Conversational Agent

## 1. Architecture Design

The system implements a Retrieval-Augmented Generation (RAG) pipeline — a widely-adopted
architecture that grounds language model outputs in a specific knowledge source rather than
relying on the model's parametric (trained) knowledge.

```
PDF Upload
    │
    ▼
pdf_processor.py
  - PyMuPDF extracts text per page
  - Sliding-window tokenization (tiktoken cl100k_base)
  - Chunk size: 600 tokens, overlap: 100 tokens
  - Metadata: { chunk_id, page_number, token_count }
    │
    ▼
embeddings.py
  - sentence-transformers all-MiniLM-L6-v2 (384-dim)
  - L2-normalized embeddings (enables cosine via dot product)
  - Batch encoding for efficiency
    │
    ▼
vector_store.py
  - FAISS IndexFlatIP (exact search, inner product)
  - Persisted to disk: data/index/faiss.index + metadata.pkl
  - Reset/reload supported
    │
    └──────────────────────────────────────┐
                                           │
Query Input (any language)                 │
    │                                      │
    ▼                                      │
retriever.py                               │
  - langdetect: language identification    │
  - deep-translator: translate → English   │
  - Embed query (same model as corpus)     │
  - FAISS search → top-k chunks + scores   │
    │                                      │
    ▼                                      │
llm.py                                     │
  - Build context block from chunks ←──────┘
  - Strict system prompt
  - Gemini 1.5 Flash: temperature=0, top_k=1
  - Extract page citations from response
  - Back-translate response if needed
    │
    ▼
Structured Response
{ answer, citations, retrieved_chunks, scores, language, logs }
```

---

## 2. Retrieval Pipeline

### Step 1: Text Extraction
PyMuPDF (`fitz`) is used over alternatives like pdfplumber or pypdf because it:
- Handles complex layouts (multi-column, tables) better
- Preserves text order more reliably
- Has lower memory footprint for large PDFs

### Step 2: Chunking Strategy
**Sliding-window token-level chunking:**
- Tokenize the entire document using tiktoken (cl100k_base)
- Split into windows of 600 tokens with 100-token overlap
- Overlap ensures context-critical information spanning chunk boundaries is captured

Token-level chunking is preferred over character-level or sentence-level because:
- Direct control over context window size
- More uniform chunk sizes improve embedding quality
- Compatible with transformer token limits

### Step 3: Embedding
`all-MiniLM-L6-v2` produces 384-dimensional dense embeddings. The model is a distilled
version of larger transformers optimized for semantic similarity tasks. It achieves >99%
of BERT-large performance at 5x the speed with 1/4 the parameters.

All embeddings are L2-normalized: `||e|| = 1`. This means:
```
cosine_similarity(a, b) = dot_product(a, b)  [when ||a||=||b||=1]
```
This allows using FAISS's fast IndexFlatIP (inner product) for cosine similarity search.

### Step 4: FAISS Indexing
`IndexFlatIP` performs exhaustive exact search over all stored vectors. For typical PDF
corpora (tens to thousands of chunks), exhaustive search is fast enough (<5ms) and
guarantees finding the true nearest neighbors (no approximation error).

### Step 5: Answer Generation
The retrieved chunks are formatted into a structured context block and passed to
Gemini 1.5 Flash with a strict system prompt. The prompt architecture:
- System instruction: defines the QA role with explicit refusal rules
- Context block: per-chunk text with page number headers
- User message: question + reminder of constraints

---

## 3. How Hallucination is Prevented

Hallucination prevention operates at **three independent layers**:

### Layer 1: Architectural (Context-Only Retrieval)
The LLM receives **only** the retrieved chunks as its information source. No web access,
no training data fallback. The prompt contains only: context + question.

### Layer 2: Prompt Engineering (Strict System Instructions)
The system prompt explicitly states:
- "Answer ONLY using information explicitly present in the provided CONTEXT"
- "Do NOT use any external knowledge"
- "If the answer is NOT found in the context, respond with EXACTLY: 'I cannot answer...'"
- "ALWAYS include page number references"

Temperature = 0 and top_k = 1 (greedy decoding) eliminate sampling randomness.

### Layer 3: Programmatic Enforcement
The code post-processes Gemini's output:
- Checks if the refusal phrase is present
- Verifies citation presence; if missing, appends context page numbers
- In future versions: semantic similarity check between answer and context can be added
  as a consistency validator

---

## 4. Why FAISS + sentence-transformers?

### FAISS
| Factor | FAISS | Alternative (Pinecone, Weaviate) |
|--------|-------|----------------------------------|
| Cost | Free, local | Paid API calls |
| Latency | <5ms for PDFs | 50-200ms network overhead |
| Privacy | Fully local, no data leaves system | Data sent to cloud |
| Determinism | Exact search (IndexFlatIP) | May use ANN approximation |
| Scalability | Supports billions of vectors with ANN | Managed scaling |

For a document QA system with single-PDF corpora (typically 50-2000 chunks), FAISS
IndexFlatIP is optimal: exact results, zero cost, millisecond response.

### sentence-transformers
| Factor | all-MiniLM-L6-v2 | OpenAI text-embedding-ada-002 |
|--------|------------------|-------------------------------|
| Cost | Free, local | $0.10/1M tokens (ongoing) |
| Privacy | Fully local | Data sent to OpenAI |
| Speed | ~50ms/batch CPU | 200-500ms API roundtrip |
| Quality | 80+ BEIR benchmarks | 85+ BEIR benchmarks |
| Control | Full | Limited (black box) |

The quality gap is minimal for single-domain document retrieval (within 5% on BEIR).
The cost and privacy advantages of local embeddings are significant.

---

## 5. Trade-offs

### Accuracy vs Speed
- IndexFlatIP: exact, O(n*d) per query. For n<10K chunks, CPU speed is acceptable.
- For millions of vectors: switch to IndexHNSWFlat (approximate, O(log n), ~5% accuracy loss).
- The current architecture prioritizes accuracy (exact search) over maximum throughput.

### Local vs API (Embeddings)
- Local (sentence-transformers): no cost, private, but requires CPU resources
- API (OpenAI/Cohere): higher accuracy potential, but cost scales with usage
- Recommendation: Use local for prototyping/regulated environments; API for production at scale

### Chunk Size
- Smaller chunks (200-300 tokens): higher precision, risk of missing context
- Larger chunks (800-1000 tokens): more context, noisier embeddings
- Current (600 + 100 overlap): balanced for document QA

### Multi-Language Translation
- Free GoogleTranslator (deep-translator): no API key, but rate-limited
- For production: use DeepL API or Google Cloud Translation for reliability
- Translation introduces non-determinism (slight rephrasing possible)

---

## 6. Limitations

1. **Scanned/Image PDFs**: The system requires text-based PDFs. Scanned documents
   need OCR preprocessing (e.g., pytesseract + pdf2image).

2. **Cross-Chunk Reasoning**: If an answer requires synthesizing information from
   distant sections (e.g., comparing Chapter 1 and Chapter 7), retrieval may miss
   relevant chunks if they have low similarity to the query embedding.

3. **Tables and Figures**: PyMuPDF extracts table text as plain strings, losing
   structural information. Complex numerical tables may not be answerable.

4. **Translation Quality**: Deep-translator's free service can produce poor translations
   for low-resource languages or technical terminology.

5. **Context Window**: Very long retrieved contexts may approach Gemini's context limits
   for dense technical documents with high top-k settings.

6. **Temporal Freshness**: The PDF content is static; the system cannot answer questions
   about events after the PDF was written.

---

## 7. Deployment Guide

### Local Deployment (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key

# Start backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start UI
streamlit run frontend/app.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501"]
```

### Cloud Deployment (GCP)
- **Backend**: Cloud Run (serverless, scales to zero)
- **FAISS index**: Store on Cloud Storage, load into memory on startup
- **Streamlit**: Cloud Run or App Engine
- **Secrets**: Secret Manager for GEMINI_API_KEY

### Scaling FAISS
For multi-document or enterprise use:
- Replace IndexFlatIP with `IndexIVFFlat` (clustering-based) or `IndexHNSWFlat` (graph-based)
- Use FAISS GPU for 10-100x throughput
- Consider dedicated vector databases (Milvus, Qdrant) for horizontal scaling
- Implement per-document namespacing for multi-user deployments

### API Hosting Considerations
- Add authentication (API key or JWT) to FastAPI endpoints
- Rate limiting with `slowapi`
- Multi-worker uvicorn with shared Redis for VectorStore state
- Health monitoring with Prometheus + Grafana


