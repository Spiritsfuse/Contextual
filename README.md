# PDF-Constrained Conversational Agent

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions **strictly from uploaded PDF content** using FAISS, sentence-transformers, and Google Gemini.

---

## 🏗️ Architecture

```
PDF → Chunking → Embeddings → FAISS
Query → Embedding → Retrieval → Gemini → Response
```

**Backend modules:**
| Module | Purpose |
|--------|---------|
| `pdf_processor.py` | Extract text + sliding-window chunking |
| `embeddings.py` | Local sentence-transformers model |
| `vector_store.py` | FAISS IndexFlatIP with persistence |
| `retriever.py` | Multi-language query pipeline |
| `llm.py` | Gemini with strict grounding prompt |
| `main.py` | FastAPI REST API |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key: https://aistudio.google.com/app/apikey
```

### 3. Generate sample PDF
```bash
python generate_sample_pdf.py
```

### 4. Start the backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```
Backend runs at: http://localhost:10000  
API docs: http://localhost:10000/docs

### 5. Start the Streamlit UI
```bash
streamlit run frontend/app.py
```
UI runs at: http://localhost:8501

---

## 📁 Project Structure

```
Contextual/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + endpoints
│   ├── pdf_processor.py     # PDF extraction + chunking
│   ├── embeddings.py        # sentence-transformers singleton
│   ├── vector_store.py      # FAISS index management
│   ├── retriever.py         # Query pipeline + multi-lang
│   └── llm.py               # Gemini integration
├── frontend/
│   └── app.py               # Streamlit chat UI
├── data/
│   └── sample_ai_healthcare.pdf  # Generated test document
├── data/index/              # FAISS index (auto-created)
├── logs/                    # Session log files (auto-created)
├── tests/
│   └── test_cases.md        # 5 valid + 3 invalid test cases
├── docs/
│   ├── technical_writeup.md # Architecture, design, trade-offs, deployment
│   └── demo_script.md       # Step-by-step recording guide (gitignored)
├── generate_sample_pdf.py   # Sample PDF generator
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload_pdf` | Upload and process a PDF |
| `POST` | `/query` | Run RAG query |
| `POST` | `/reset_index` | Clear FAISS index |
| `GET` | `/health` | System health check |
| `GET` | `/docs` | Interactive API documentation |

### Example: Upload PDF
```bash
curl -X POST http://localhost:10000/upload_pdf \
  -F "file=@data/sample_ai_healthcare.pdf"
```

### Example: Query
```bash
curl -X POST http://localhost:10000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main applications of AI in healthcare?", "top_k": 5}'
```

---

## ✅ Test Cases

**Valid queries** (expect grounded answer + page citations):
1. "What are the main applications of AI in healthcare?"
2. "What ethical challenges does AI face in medical diagnosis?"
3. "Describe a case study mentioned in the document."
4. "What role does machine learning play in drug discovery?"
5. "What are the limitations of AI systems discussed?"

**Invalid queries** (expect exact refusal):
1. "What is the capital of France?" → `"I cannot answer this from the provided document."`
2. "Write a Python script to sort a list." → refusal
3. "What happened in World War II?" → refusal

See `tests/test_cases.md` for full evaluation instructions.

---

## 🔒 Key Design Decisions

- **No hallucinations**: Temperature=0, strict system prompt, context-only architecture
- **Deterministic**: Same query → same answer on every run
- **Multi-language**: Detects language, translates → English, processes, translates back
- **Fully local embeddings**: No OpenAI/Cohere API keys needed for retrieval
- **Persistent index**: FAISS index survives server restarts
- **Full observability**: Every query logged with retrieved chunks + similarity scores

---

## 📊 Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Backend | FastAPI | Fast, async, auto-docs |
| Frontend | Streamlit | Rapid prototyping, interactive |
| Embeddings | all-MiniLM-L6-v2 | Free, local, high quality |
| Vector DB | FAISS IndexFlatIP | Exact cosine search, deterministic |
| LLM | Gemini 3.1 Flash Lite (primary) + 2.5 Flash (fallback) | High quota, fast, and robust with fallback |
| PDF | PyMuPDF | Best text extraction quality |
| Multi-lang | langdetect + deep-translator | No API key required |

---

## 📖 Documentation

| File | Contents |
|------|----------|
| [`docs/technical_writeup.md`](docs/technical_writeup.md) | Architecture, retrieval pipeline, anti-hallucination, trade-offs, deployment |
| [`tests/test_cases.md`](tests/test_cases.md) | 5 valid queries, 3 invalid (refusal), reproducibility + multi-language tests |
| API reference | http://localhost:10000/docs (when backend is running) |

---

## ⚠️ Important Notes

1. **Scanned PDFs are not supported** — PDF must contain text layers
2. **Gemini API key is required** — Get a free key at https://aistudio.google.com/app/apikey
3. **First run downloads the embedding model** (~80MB, one-time)
4. **Logs location**: `logs/session_YYYYMMDD.log`
