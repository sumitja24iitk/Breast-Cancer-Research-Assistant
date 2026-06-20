"""
retrieve.py — Phase 3 (dense retrieval) upgraded in Phase 7 (hybrid + rerank)

CONCEPTS — read this before the code

Why dense retrieval alone is not enough
----------------------------------------
Dense retrieval embeds the query into a vector and finds the nearest abstract
vectors by cosine similarity.  It captures *semantic* similarity beautifully —
"trastuzumab" and "HER2-targeted therapy" land close in the embedding space.

But it has a blind spot: rare, precise terms.  The word "cardiotoxicity" might
appear verbatim in only three abstracts.  If the embedding model smears its
meaning across a broad "heart side effects" neighbourhood, a query for
"trastuzumab cardiotoxicity" may miss those three exact-match papers and return
vaguely relevant oncology papers instead.

BM25 does the opposite: it is purely keyword-based.  It counts term frequency
(TF) in each document, penalises common words (IDF), and normalises for
document length.  It has no understanding of meaning, but it is very precise
about exact terms.  "cardiotoxicity" matches "cardiotoxicity", full stop.

Combining both:
  • Dense retrieval = high recall for paraphrase / semantic matches.
  • BM25            = high precision for rare, exact terms.
  • Hybrid          = better recall AND precision than either alone.

What the cross-encoder reranker adds
--------------------------------------
Initial retrieval (dense or BM25) scores each document *independently* of the
others — the query and document never "see" each other at full attention depth.
A bi-encoder like all-MiniLM-L6-v2 compresses each text to a single 384-d
vector and then compares vectors; fast, but lossy.

A cross-encoder sees the query and document *together* as one input:
    "[query] [SEP] [document]"
and outputs a single relevance score.  Because every token in the query can
attend to every token in the document (and vice versa), it can catch fine-
grained signals like exact drug names co-occurring with exact side effects.
The catch: it is O(n) — you can't pre-compute anything.  So we use it only
to *rerank a small candidate set* (≤40 docs), not to score the full corpus.

Why making the mode switchable matters for evaluation
------------------------------------------------------
Phase 8 measures recall@k and MRR: does the correct abstract appear in the
top-k results?  To know whether hybrid+rerank actually *helps*, we need to
compare it against the dense-only baseline on the same set of questions.
A `mode` parameter lets the eval harness call retrieve() twice — once per
mode — on the same query and produce a side-by-side table.

Why the query must be embedded with the SAME model used for indexing
---------------------------------------------------------------------
Every embedding model defines its own "semantic space".  Indexing with one
model and querying with another produces vectors in incompatible coordinate
systems — the cosine similarity you compute is meaningless.
Rule: index model == query model, always.

How Chroma distances relate to similarity scores
-------------------------------------------------
Chroma returns *cosine distance* = 1 − cosine_similarity.
We convert: score = 1 − distance, so higher score = better match ∈ [0, 1].
Cross-encoder scores are raw logits (any float); higher = more relevant.
The two score types are NOT comparable across modes.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


_ROOT         = Path(__file__).parent.parent
_CHROMA_DIR   = _ROOT / "chroma_db"
_EMBED_MODEL  = "all-MiniLM-L6-v2"                      # MUST match index.py
_COLLECTION   = "breast_cancer"                          # MUST match index.py
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ms-marco = web passage ranking

# Dense top-20 + BM25 top-20 → up to 40 unique candidates → reranked → top-k.
# Raising this number improves recall at the cost of reranker latency.
_HYBRID_CANDIDATE_K = 20


# Models and indexes are loaded once at module import time. Every subsequent
# call to retrieve() reuses the already-loaded objects at zero re-loading cost.

print(f"[retrieve] Loading embedding model '{_EMBED_MODEL}' ...")
_model: SentenceTransformer = SentenceTransformer(_EMBED_MODEL)

print(f"[retrieve] Connecting to Chroma collection '{_COLLECTION}' ...")
_client: chromadb.PersistentClient = chromadb.PersistentClient(path=str(_CHROMA_DIR))
_collection = _client.get_collection(name=_COLLECTION)
print(f"[retrieve] Chroma ready — {_collection.count():,} documents indexed.")

# Pull all documents from Chroma once to build the in-memory BM25 index.
# One-time O(n) cost at startup: for 2,500 abstracts <2 s and ~50 MB RAM,
# worth it for every subsequent fast keyword lookup.
print("[retrieve] Fetching all documents from Chroma to build BM25 index ...")
_corpus = _collection.get(include=["documents", "metadatas"])
_all_texts: list[str] = _corpus["documents"]
_all_metas: list[dict] = _corpus["metadatas"]

_tokenized_corpus: list[list[str]] = [text.lower().split() for text in _all_texts]
_bm25: BM25Okapi = BM25Okapi(_tokenized_corpus)
print(f"[retrieve] BM25 index built over {len(_all_texts):,} documents.")

# ms-marco-MiniLM-L-6-v2 was fine-tuned on the MS MARCO passage-ranking
# dataset (~500 k query-passage pairs from Bing search). It generalises well
# to biomedical text for re-ranking short passages like PubMed abstracts.
# Loading takes ~1–2 s and ~100 MB RAM.
print(f"[retrieve] Loading cross-encoder reranker '{_RERANK_MODEL}' ...")
_reranker: CrossEncoder = CrossEncoder(_RERANK_MODEL)
print("[retrieve] Reranker ready. All models loaded.")


def _bm25_retrieve(query: str, k: int) -> list[dict]:
    """
    Keyword-based retrieval using the in-memory BM25Okapi index.

    Scores every document in the corpus and returns the top-k by BM25 score.
    The BM25 score is NOT a probability; it is a relative ranking signal —
    higher = more keyword overlap with the query, scaled by term rarity.

    Parameters
    ----------
    query : str  — same natural-language question fed to the dense retriever
    k     : int  — number of top results to return

    Returns
    -------
    list[dict] — same shape as dense_retrieve: pmid, title, text, score
    """
    tokenized_query = query.lower().split()
    scores          = _bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    hits = []
    for idx in top_indices:
        hits.append(
            {
                "pmid":  _all_metas[idx]["pmid"],
                "title": _all_metas[idx]["title"],
                "text":  _all_texts[idx],
                "score": round(float(scores[idx]), 4),
            }
        )
    return hits


def dense_retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed *query* and return the top-k most similar documents from Chroma.

    Parameters
    ----------
    query : str
        A natural-language clinical question, e.g.
        "What are treatment options for HER2-positive breast cancer?"
    k : int
        Number of results to return.

    Returns
    -------
    list[dict] — each element has:
        pmid    : str   — PubMed identifier (used for citations)
        title   : str   — article title
        text    : str   — abstract text (the passage passed to the LLM)
        score   : float — cosine similarity in [0, 1]; higher = more relevant
    """
    # Embed with the same model used during indexing — mixing models produces
    # vectors in incompatible spaces and makes cosine similarity meaningless.
    query_vec: list[float] = _model.encode(query, convert_to_numpy=True).tolist()

    results = _collection.query(
        query_embeddings=[query_vec],   # outer list = batch of 1 query
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns batch-indexed lists (results["documents"][0] for the first
    # query in the batch). We sent one query, so index [0] everywhere.
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Chroma cosine distance = 1 − cosine_similarity, so higher score = better match.
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


def retrieve(query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
    """
    Unified retrieval entry point.  Choose mode based on the quality/speed
    trade-off you need:

      mode="dense"  — embed query → nearest-neighbour search in Chroma.
                      Fast (~50 ms), good semantic recall, weaker on rare terms.

      mode="hybrid" — dense top-20 + BM25 top-20 → merge → cross-encoder
                      rerank → top-k.  Slower (~1–3 s for the reranker) but
                      meaningfully better recall+precision, especially for
                      drug names, gene symbols, and other rare clinical terms.

    Parameters
    ----------
    query : str   — natural-language clinical question
    k     : int   — number of results to return (after reranking for hybrid)
    mode  : str   — "dense" or "hybrid"

    Returns
    -------
    list[dict] — each element has:
        pmid    : str   — PubMed identifier
        title   : str   — article title
        text    : str   — abstract text
        score   : float — cosine similarity (dense) or reranker logit (hybrid);
                          scores are NOT comparable across modes
    """
    if mode == "dense":
        return dense_retrieve(query, k=k)

    if mode == "hybrid":
        # Gather a broad candidate pool from both methods. Each contributes
        # _HYBRID_CANDIDATE_K docs so the reranker has more to choose from;
        # the final top-k is carved out after reranking.
        dense_hits = dense_retrieve(query, k=_HYBRID_CANDIDATE_K)
        bm25_hits  = _bm25_retrieve(query, k=_HYBRID_CANDIDATE_K)

        # Merge and deduplicate by PMID. When both methods surface the same
        # abstract we keep the first occurrence; initial scores are on different
        # scales (cosine vs BM25) so we discard them and let the cross-encoder
        # assign the definitive relevance score.
        seen: dict[str, dict] = {}
        for hit in dense_hits + bm25_hits:
            if hit["pmid"] not in seen:
                seen[hit["pmid"]] = hit
        candidates = list(seen.values())

        # The cross-encoder sees each [query, document] pair as a single
        # sequence and outputs a relevance logit — more accurate than the
        # bi-encoder but O(n), so we only run it on the small candidate pool.
        pairs: list[list[str]] = [[query, c["text"]] for c in candidates]
        rerank_scores = _reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["score"] = round(float(score), 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    raise ValueError(f"Unknown retrieval mode '{mode}'. Choose 'dense' or 'hybrid'.")


if __name__ == "__main__":
    TEST_QUERIES = [
        "trastuzumab cardiotoxicity",
        "BRCA1 mutation treatment",
    ]

    for query in TEST_QUERIES:
        print(f"\nQUERY: {query}")
        for mode in ("dense", "hybrid"):
            hits        = retrieve(query, k=5, mode=mode)
            score_label = "cosine" if mode == "dense" else "rerank logit"
            print(f"\n  [{mode.upper()}]  score = {score_label}")
            for rank, hit in enumerate(hits, start=1):
                print(
                    f"    #{rank}  PMID {hit['pmid']}  score={hit['score']:+.4f}\n"
                    f"         {hit['title'][:65]}"
                )
