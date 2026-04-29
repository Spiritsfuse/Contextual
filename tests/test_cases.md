# Test Cases — PDF-Constrained Conversational Agent

## Document Used: `data/sample_ai_healthcare.pdf`

> Upload this PDF before running any test queries.
> The system MUST be running: `uvicorn backend.main:app --reload`

---

## ✅ Valid Queries (Expected: Grounded Answer + Page Citations)

### Test 1: Main Applications
**Query:** `What are the main applications of AI in healthcare?`

**Expected behavior:**
- Answer references medical imaging, predictive analytics, NLP documentation
- Cites Page 3 (Section 2)
- Includes specific examples (CNN, Google DeepMind, sepsis prediction)

**Sample expected answer excerpt:**
> "AI has several main applications in healthcare including medical imaging and diagnostics (using CNNs for X-rays, CT scans), predictive analytics for early warning systems, and NLP for clinical documentation. (Page 3)"

---

### Test 2: Ethical Challenges
**Query:** `What ethical challenges does AI face in medical diagnosis?`

**Expected behavior:**
- Covers: algorithmic bias, black box problem, privacy, liability
- Cites Page 5 (Section 4)
- References the 2019 Science study on health algorithm bias

**Sample expected answer excerpt:**
> "Key ethical challenges include algorithmic bias (e.g., a 2019 study showed AI underestimated Black patients' needs), explainability issues with black-box models, privacy concerns, and unclear liability. (Page 5)"

---

### Test 3: Case Study
**Query:** `Describe a case study mentioned in the document.`

**Expected behavior:**
- Describes the Google diabetic retinopathy screening in India/Thailand
- Cites Page 6 (Section 5)
- Mentions specific numbers: 128,175 images, 54 ophthalmologists, AUC 0.991

**Sample expected answer excerpt:**
> "The document describes Google's diabetic retinopathy screening program in India. The deep learning model was trained on 128,175 retinal images graded by 54 ophthalmologists, achieving an AUC of 0.991. (Page 6)"

---

### Test 4: Machine Learning in Drug Discovery
**Query:** `What role does machine learning play in drug discovery?`

**Expected behavior:**
- Covers: target identification, molecular design, clinical trial optimization, drug repurposing
- Cites Page 4 (Section 3)
- Mentions Insilico Medicine 18-month drug design example
- Mentions COVID-19 drug repurposing

---

### Test 5: Limitations
**Query:** `What are the limitations of AI systems discussed in this document?`

**Expected behavior:**
- Lists: data quality, distribution shift, limited generalization, regulatory barriers, physician adoption
- Cites Page 7 (Section 6)
- Specific example: chest X-ray model US vs Southeast Asia

---

## ❌ Invalid Queries (Expected: Refusal Response)

### Invalid Test 1: Out-of-scope general knowledge
**Query:** `What is the capital of France?`

**Expected response (EXACT):**
> `"I cannot answer this from the provided document."`

**Reason for refusal:** This factual question has no connection to the uploaded healthcare document.

---

### Invalid Test 2: Coding request
**Query:** `Write a Python script to sort a list.`

**Expected response (EXACT):**
> `"I cannot answer this from the provided document."`

**Reason for refusal:** A programming task; entirely unrelated to the document content.

---

### Invalid Test 3: Historical event
**Query:** `What happened during World War II?`

**Expected response (EXACT):**
> `"I cannot answer this from the provided document."`

**Reason for refusal:** Historical event with no connection to AI healthcare content.

---

## 🔁 Reproducibility Test

**Instructions:** Run Test 1 three times in a row with identical query text.

**Expected result:** Same answer, same citations, same retrieved chunks on every run.

**Why deterministic:**
- `temperature=0` + `top_k=1` (greedy decoding) in Gemini generation config
- FAISS IndexFlatIP is exact search (no approximation)
- sentence-transformers produces identical embeddings for identical input
- langdetect seeded with `DetectorFactory.seed = 0`

---

## 🌍 Multi-Language Test (Bonus)

### Spanish Query
**Query:** `¿Cuáles son las principales aplicaciones de la IA en la salud?`

**Expected behavior:**
- Language detected: Spanish (`es`)
- Query translated to English: "What are the main applications of AI in healthcare?"
- Processing: same as Test 1
- Response translated back to Spanish with page citations

### French Query
**Query:** `Quels sont les défis éthiques de l'IA dans le diagnostic médical ?`

**Expected behavior:**
- Language detected: French (`fr`)
- Answer returned in French
- Same citations as Test 2

---

## 📊 Evaluator Instructions

1. **Setup:**
   ```
   pip install -r requirements.txt
   cp .env.example .env
   # Add your GEMINI_API_KEY to .env
   python generate_sample_pdf.py
   ```

2. **Start backend:**
   ```
   uvicorn backend.main:app --reload
   ```
   Verify at: http://localhost:8000/health

3. **Start Streamlit UI:**
   ```
   streamlit run frontend/app.py
   ```

4. **Upload PDF:** Use sidebar → "Choose a PDF file" → select `data/sample_ai_healthcare.pdf` → click "Process"

5. **Run test queries:** Type each query from the test cases above into the chat input

6. **Verify:**
   - ✅ Valid queries: Answer with page citations visible
   - ✅ Invalid queries: Exact refusal phrase
   - ✅ Logs: Check `logs/session_YYYYMMDD.log` for full observability output

7. **Check API directly (optional):**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main applications of AI in healthcare?", "top_k": 5}'
   ```
