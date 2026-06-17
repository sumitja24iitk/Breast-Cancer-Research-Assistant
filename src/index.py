"""
index.py — Phase 2
Reads data/abstracts.json, embeds each abstract with sentence-transformers,
and stores the vectors + metadata in a persistent Chroma collection.

Key concepts:
  - Embedding: convert text → fixed-length float vector (semantic meaning compressed)
  - Chroma collection: like a table; stores (id, vector, metadata, document)
  - Persistence: Chroma writes to disk (chroma_db/) so we don't re-embed on every run
"""

# TODO (Phase 2): implement load_abstracts(), build_index(), main()
