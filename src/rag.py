"""
rag.py — Phase 4 (generation) updated in Phase 7 (hybrid retrieval default)
Orchestrates the full RAG pipeline: retrieve → generate → return cited answer.

This is the single entry point used by:
  - Phase 5: FastAPI endpoint  (api/main.py calls answer())
  - Phase 6: Streamlit UI      (app.py calls answer() via the API)
  - Phase 8: Eval harness      (eval.py calls answer() in a loop, varying mode)

Keeping orchestration here (not inside generate.py or retrieve.py) means each
component stays independently testable: you can test retrieve() without hitting
Gemini, and test generate_answer() with mock chunks.
"""

import sys
from pathlib import Path

# Make sibling src/ modules importable whether this file is run directly
# (python src/rag.py — Python adds src/ to sys.path automatically) or
# imported from the project root (api/main.py — only project root is on path).
sys.path.insert(0, str(Path(__file__).parent))

from generate import generate_answer
from retrieve import retrieve


def answer(question: str, k: int = 5, mode: str = "hybrid") -> dict:
    """
    Full RAG pipeline: retrieve k chunks, generate a cited answer.

    Parameters
    ----------
    question : str
        A clinical question, e.g. "What are first-line treatments for
        HER2-positive breast cancer?"
    k : int
        Number of abstracts to retrieve and pass to Gemini as context.
        See retrieve.py module docstring for the recall/noise trade-off.
    mode : str
        Retrieval strategy — "hybrid" (default) or "dense".
        "hybrid" = dense top-20 + BM25 top-20 → cross-encoder rerank → top-k.
        "dense"  = embedding nearest-neighbour from Chroma only.
        Pass mode="dense" in Phase 8 eval to compare against the hybrid baseline.

    Returns
    -------
    dict with two keys:
        "answer"  : str        — Gemini's grounded, cited answer
        "sources" : list[dict] — the retrieved chunks used as context;
                                 each has pmid, title, text, score
    """
    chunks      = retrieve(question, k=k, mode=mode)
    answer_text = generate_answer(question, chunks)

    sources = [
        {
            "pmid":    c["pmid"],
            "title":   c["title"],
            "snippet": c["text"][:300],   # first 300 chars shown as a preview
            "score":   c["score"],
        }
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    SAMPLE_QUESTION = "What are first-line treatments for HER2-positive breast cancer?"

    print(f"Question: {SAMPLE_QUESTION}\n")
    result = answer(SAMPLE_QUESTION, k=5, mode="hybrid")

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")
    for i, src in enumerate(result["sources"], start=1):
        print(f"  [{i}] PMID {src['pmid']}  score={src['score']:+.4f}")
        print(f"       {src['title'][:80]}")
