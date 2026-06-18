"""
retrieve.py — Phase 3 (dense retrieval) + stub for Phase 7 (hybrid + rerank)

CONCEPTS — read this before the code
──────────────────────────────────────────────────────────────────────────────

Why the query must be embedded with the SAME model used for indexing
────────────────────────────────────────────────────────────────────
Every embedding model defines its own "semantic space" — a high-dimensional
coordinate system where similar meanings land near each other.  The geometry
is arbitrary and model-specific: the direction that encodes "chemotherapy" in
all-MiniLM-L6-v2 is completely different from that direction in, say, PubMedBERT.

When you index documents you lock in a particular geometry.  If you embed the
query with a different model the query vector lands in a different (incompatible)
coordinate system.  The cosine similarity you compute is then meaningless —
like asking "is 3 km close to 3 miles?" without a conversion factor.
Rule: index model == query model, always.

What the similarity score represents
─────────────────────────────────────
Cosine similarity measures the angle between two vectors, ignoring their
magnitudes.  A score of 1.0 means the vectors point in exactly the same
direction (semantically identical).  0.0 means orthogonal (unrelated).
Negative values mean "opposite meaning" — rare in practice for text.

Chroma stores distances, not similarities, where distance = 1 − cosine_similarity.
We convert back: score = 1 − distance, so higher score = better match.

How k trades off recall vs. noise
──────────────────────────────────
k is how many chunks you hand to the LLM as context.

  Small k (e.g. 3):  only the most relevant passages → high precision, but you
      might miss a paper that uses different phrasing → lower recall.

  Large k (e.g. 20): broader coverage → higher recall, but now the LLM has to
      read more text, some of it loosely relevant.  This inflates cost (tokens),
      can distract the model, and in a fixed-context window pushes out better docs.

  Sweet spot for this corpus (~2,500 docs): k=5 for fast answers, k=10 for
  evaluation runs.  Phase 7's reranker lets us retrieve k=20 but then rerank
  down to 5, getting recall without the noise.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL INITIALISATION — loaded once when this module is first imported
# ─────────────────────────────────────────────────────────────────────────────
# Why module-level?
#   Loading a SentenceTransformer model takes ~1–2 s and allocates ~300 MB RAM.
#   Connecting to Chroma opens the SQLite file and loads the HNSW index.
#   Doing either of those inside dense_retrieve() would pay that cost on EVERY
#   query call — unacceptable for an API endpoint or interactive UI.
#   By initialising at import time, all callers share one model and one client.

_ROOT        = Path(__file__).parent.parent
_CHROMA_DIR  = _ROOT / "chroma_db"
_EMBED_MODEL = "all-MiniLM-L6-v2"   # MUST match the model used in index.py
_COLLECTION  = "breast_cancer"       # MUST match the name used in index.py

print(f"[retrieve] Loading embedding model '{_EMBED_MODEL}' ...")
_model: SentenceTransformer = SentenceTransformer(_EMBED_MODEL)

print(f"[retrieve] Connecting to Chroma collection '{_COLLECTION}' ...")
_client: chromadb.PersistentClient = chromadb.PersistentClient(path=str(_CHROMA_DIR))
_collection = _client.get_collection(name=_COLLECTION)
print(f"[retrieve] Ready — {_collection.count():,} documents indexed.")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def dense_retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed *query* and return the top-k most similar documents from Chroma.

    Parameters
    ----------
    query : str
        A natural-language clinical question, e.g.
        "What are treatment options for HER2-positive breast cancer?"
    k : int
        Number of results to return.  See module docstring for the recall/noise
        trade-off discussion.

    Returns
    -------
    list[dict] — each element has:
        pmid    : str   — PubMed identifier (used for citations)
        title   : str   — article title
        text    : str   — abstract text (the passage passed to the LLM)
        score   : float — cosine similarity in [0, 1]; higher = more relevant
    """
    # Step 1 — embed the query with the SAME model used during indexing.
    # encode() returns a (384,) float32 ndarray; .tolist() converts to plain Python
    # list[float] which is what chromadb.Collection.query() expects.
    query_vec: list[float] = _model.encode(query, convert_to_numpy=True).tolist()

    # Step 2 — ask Chroma for the k nearest neighbours by cosine distance.
    # include= controls what Chroma sends back; we need documents (abstract text),
    # metadatas (title, pmid, year, journal), and distances (to compute scores).
    results = _collection.query(
        query_embeddings=[query_vec],   # outer list = batch of 1 query
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Step 3 — unpack Chroma's response.
    # Chroma returns batch-indexed lists (results["documents"][0] = list for
    # the first query in the batch).  We sent one query, so index [0] everywhere.
    docs       = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    # Step 4 — convert distances → similarity scores and build clean dicts.
    # Chroma cosine distance = 1 − cosine_similarity, so:
    #   distance 0.0 → score 1.0 (perfect match)
    #   distance 1.0 → score 0.0 (unrelated)
    hits = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        hits.append(
            {
                "pmid":  meta["pmid"],
                "title": meta["title"],
                "text":  doc,
                "score": round(1.0 - dist, 4),
            }
        )

    return hits


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 STUBS — implemented later
# ─────────────────────────────────────────────────────────────────────────────

def bm25_retrieve(query: str, k: int = 5) -> list[dict]:
    """Phase 7 — keyword-based BM25 retrieval (not yet implemented)."""
    raise NotImplementedError("BM25 retrieval is planned for Phase 7.")


def hybrid_retrieve(query: str, k: int = 5) -> list[dict]:
    """Phase 7 — merge dense + BM25 results via Reciprocal Rank Fusion."""
    raise NotImplementedError("Hybrid retrieval is planned for Phase 7.")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Phase 7 — cross-encoder reranking of a candidate list."""
    raise NotImplementedError("Reranking is planned for Phase 7.")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK VERIFICATION — run: python src/retrieve.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_QUERIES = [
        "treatment options for HER2-positive breast cancer",
        "side effects of tamoxifen",
    ]

    for query in SAMPLE_QUERIES:
        print(f"\n{'═' * 60}")
        print(f"Query: {query}")
        print("═" * 60)

        hits = dense_retrieve(query, k=5)

        for rank, hit in enumerate(hits, start=1):
            snippet = hit["text"][:160].replace("\n", " ")
            print(
                f"\n  #{rank}  PMID {hit['pmid']}  score={hit['score']:.4f}\n"
                f"       {hit['title'][:80]}\n"
                f"       {snippet}..."
            )

    print(f"\n{'═' * 60}")
    print("Phase 3 smoke test complete.")
