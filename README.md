# Breast Cancer Treatment RAG

A **Retrieval-Augmented Generation (RAG)** system that answers clinical questions
about breast cancer treatment by grounding every claim in PubMed abstracts and
citing the source PMID.

---

## What is RAG and why does it matter here?

A large language model (LLM) like Gemini is trained on a fixed snapshot of the
internet. It can hallucinate facts, especially for niche medical topics, and it
can't cite a specific paper for a specific claim.

RAG fixes this by *retrieving* real documents first, then asking the LLM to answer
using only those documents. Every sentence in the answer can be traced back to a
source — in this project, a PubMed abstract with its PMID.

---

## Architecture

```
User question
      │
      ▼
 [Retrieval layer]
 ├── Dense retrieval  → Chroma vector DB (sentence-transformers embeddings)
 └── Sparse retrieval → BM25 index (keyword matching)
      │
      ▼
 [Reranker]
 └── Cross-encoder (ms-marco-MiniLM-L-6-v2) scores each candidate
      │
      ▼
 [Generator]
 └── Google Gemini — given top-k abstracts, produces a cited answer
      │
      ▼
 Answer + PMIDs  →  FastAPI  →  Streamlit UI
```

**Hybrid retrieval** (dense + BM25) outperforms either alone: dense embeddings
catch semantic similarity while BM25 catches exact keyword matches. The
cross-encoder reranker then does a slower-but-more-accurate pass over the
combined candidates.

---

## Stack

| Tool | Role |
|------|------|
| `sentence-transformers` | Generate dense vector embeddings from abstract text |
| `chromadb` | Persist and query the embedding index (HNSW search) |
| `rank-bm25` | BM25 sparse index for keyword-based retrieval |
| `sentence-transformers` (cross-encoder) | Rerank candidates; trained on MS MARCO passage ranking |
| `google-generativeai` | Gemini API — the LLM that synthesizes the final answer |
| `fastapi` + `uvicorn` | Async REST backend; exposes `POST /query` |
| `streamlit` | Zero-boilerplate web UI that calls the FastAPI backend |
| `biopython` | `Bio.Entrez` — fetches PubMed records via NCBI Entrez API |
| `scikit-learn` | Evaluation utilities; we implement recall@k and MRR ourselves |
| `python-dotenv` | Loads secrets from `.env` so they never appear in code |

---

## Planned Phases

| # | Phase | Status |
|---|-------|--------|
| 0 | Project scaffolding + README | ✅ Done |
| 1 | Ingest ~2,500 PubMed abstracts via Entrez → JSON | 🔜 |
| 2 | Chunk, embed, and store abstracts in Chroma | 🔜 |
| 3 | Dense retrieval: top-k from Chroma | 🔜 |
| 4 | Generation: grounded, PMID-cited answers via Gemini | 🔜 |
| 5 | FastAPI backend — `POST /query` endpoint | 🔜 |
| 6 | Streamlit UI calling the API | 🔜 |
| 7 | Hybrid retrieval (dense + BM25) + cross-encoder reranking | 🔜 |
| 8 | Evaluation harness: recall@k, MRR, faithfulness | 🔜 |
| 9 | Polish: finalize README, limitations, ship | 🔜 |

---

## Repository layout

```
breast-cancer-rag/
├── .env.example        # Copy to .env and add your GOOGLE_API_KEY
├── .gitignore
├── requirements.txt
├── README.md
├── data/               # Downloaded PubMed abstracts (gitignored)
├── src/
│   ├── ingest.py       # Phase 1 — fetch abstracts from PubMed
│   ├── index.py        # Phase 2 — embed + store in Chroma
│   ├── retrieve.py     # Phase 3 & 7 — dense, BM25, hybrid, rerank
│   ├── generate.py     # Phase 4 — call Gemini with retrieved context
│   ├── rag.py          # Phase 4 — orchestrate retrieve → generate
│   └── eval.py         # Phase 8 — metrics
├── api/
│   └── main.py         # Phase 5 — FastAPI app
├── app.py              # Phase 6 — Streamlit UI
└── eval/
    └── questions.json  # Phase 8 — hand-crafted evaluation questions
```

---

## How to run

> **Prerequisites:** Python 3.10+, a Google Gemini API key.

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd breast-cancer-rag

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Open .env and paste your GOOGLE_API_KEY
```

### 3. Ingest data (Phase 1)

```bash
python src/ingest.py   # fetches ~2500 abstracts → data/abstracts.json
```

### 4. Build the index (Phase 2)

```bash
python src/index.py    # embeds and stores in chroma_db/
```

### 5. Start the API (Phase 5)

```bash
uvicorn api.main:app --reload
```

### 6. Start the UI (Phase 6)

```bash
streamlit run app.py
```

---

## Limitations

- Coverage is limited to abstracts (no full-text), so long methods sections are absent.
- Embeddings from `all-MiniLM-L6-v2` are competent but not biomedical-domain-tuned
  (PubMedBERT would improve recall at the cost of slower inference).
- Gemini answers are grounded in retrieved abstracts but the model can still mis-read
  or over-generalize a finding — always verify citations.
- This is a learning project, not a clinical tool.
