"""
rag.py — Phase 4
Orchestrates the full RAG pipeline: retrieve → generate → return cited answer.

This is the single entry point used by:
  - Phase 5: FastAPI endpoint  (api/main.py calls answer())
  - Phase 6: Streamlit UI      (app.py calls answer())
  - Phase 8: Eval harness      (eval.py calls answer() in a loop)

Keeping orchestration here (not inside generate.py or retrieve.py) means each
component stays independently testable: you can unit-test dense_retrieve()
without hitting Gemini, and unit-test generate_answer() with mock chunks.
"""

import sys
from pathlib import Path

# Make sibling src/ modules importable whether this file is run directly
# (python src/rag.py — Python adds src/ to sys.path automatically) or
# imported from the project root (api/main.py — only project root is on path).
sys.path.insert(0, str(Path(__file__).parent))

from generate import generate_answer
from retrieve import dense_retrieve


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def answer(question: str, k: int = 5) -> dict:
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

    Returns
    -------
    dict with two keys:
        "answer"  : str        — Gemini's grounded, cited answer
        "sources" : list[dict] — the retrieved chunks used as context;
                                 each has pmid, title, text, score
    """
    # Step 1 — dense retrieval: embed query → nearest-neighbour search in Chroma
    chunks = dense_retrieve(question, k=k)

    # Step 2 — generation: build prompt with labelled context → call Gemini
    answer_text = generate_answer(question, chunks)

    # Step 3 — package result; sources let the UI render clickable PMID links
    sources = [
        {"pmid": c["pmid"], "title": c["title"], "score": c["score"]}
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK VERIFICATION — run: python src/rag.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_QUESTION = "What are first-line treatments for HER2-positive breast cancer?"

    print(f"Question: {SAMPLE_QUESTION}\n")
    print("Running pipeline (retrieve -> generate) ...\n")

    result = answer(SAMPLE_QUESTION, k=5)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES RETRIEVED")
    print("=" * 60)
    for i, src in enumerate(result["sources"], start=1):
        print(f"  [{i}] PMID {src['pmid']}  score={src['score']:.4f}")
        print(f"       {src['title'][:80]}")

    print("\nPhase 4 smoke test complete.")
