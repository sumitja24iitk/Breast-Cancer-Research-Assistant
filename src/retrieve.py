"""
retrieve.py — Phase 3 (dense) + Phase 7 (hybrid + rerank)

Phase 3 — Dense retrieval:
  Embed the user's question, then ask Chroma for the k nearest vectors.
  "Nearest" = cosine similarity between the question embedding and stored embeddings.

Phase 7 — Hybrid + reranking:
  1. Dense:  top-N from Chroma
  2. Sparse: top-N from BM25 (keyword overlap)
  3. Merge both candidate lists (Reciprocal Rank Fusion or simple union)
  4. Cross-encoder: re-score every candidate — a slower but more accurate model
     that looks at (question, passage) jointly rather than independently.
  5. Return top-k after reranking.
"""

# TODO (Phase 3): implement dense_retrieve(query, k) -> list[dict]
# TODO (Phase 7): implement bm25_retrieve(), hybrid_retrieve(), rerank()
