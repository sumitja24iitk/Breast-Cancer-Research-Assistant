"""
index.py — Phase 2
Loads every abstract from data/abstracts.json, generates dense embeddings
using all-MiniLM-L6-v2, and persists them in a Chroma collection.

Usage:
    python src/index.py

Output:
    chroma_db/  — Chroma writes its HNSW index and sqlite metadata here.
                  The directory is created automatically if it doesn't exist.

Why we embed title + abstract together:
    The title usually contains the most specific clinical signal
    ("CDK4/6 inhibitor in HR+ metastatic breast cancer"). Prepending it to
    the abstract gives the embedding model richer context and improves
    recall for short, keyword-style queries.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim; fast on CPU; ~80 MB download
COLLECTION_NAME = "breast_cancer"
EMBED_BATCH     = 64                   # sentences per encode() call; fits in ~1 GB RAM

ROOT_DIR    = Path(__file__).parent.parent
DATA_FILE   = ROOT_DIR / "data" / "abstracts.json"
CHROMA_DIR  = ROOT_DIR / "chroma_db"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_abstracts(path: Path) -> list[dict]:
    """Read and return the list of abstract dicts from abstracts.json."""
    print(f"Loading abstracts from {path} ...")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records):,} records.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EMBED
# ─────────────────────────────────────────────────────────────────────────────

def build_embed_texts(records: list[dict]) -> list[str]:
    """
    Construct the string that will be embedded for each record.

    We concatenate title and abstract so the embedding captures both the
    high-level topic (title) and the detailed methodology/results (abstract).
    A newline separates them so they read naturally to the model's tokeniser.
    """
    return [f"{r['title']}\n{r['abstract']}" for r in records]


def embed_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """
    Encode all texts in fixed-size batches and return a flat list of vectors.

    Why batch?
      SentenceTransformer.encode() loads the whole list into GPU/CPU memory at
      once. For 2,500 abstracts that's fine, but batching makes progress visible
      and keeps peak memory predictable.

    show_progress_bar=False because we print our own batch-level progress.
    """
    all_vectors: list[list[float]] = []
    total     = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        batch     = texts[i : i + batch_size]
        print(
            f"  Embedding batch {batch_num:>3}/{n_batches} "
            f"(docs {i + 1}–{min(i + batch_size, total)})...",
            end=" ",
            flush=True,
        )
        vecs = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,   # returns float32 ndarray; Chroma accepts it
        )
        # .tolist() converts each numpy row → plain Python list[float]
        all_vectors.extend(vecs.tolist())
        print("done.")

    return all_vectors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — STORE
# ─────────────────────────────────────────────────────────────────────────────

def store_in_chroma(
    records: list[dict],
    vectors: list[list[float]],
    chroma_dir: Path,
    collection_name: str,
) -> None:
    """
    Persist documents, embeddings, and metadata in a Chroma collection.

    Chroma's PersistentClient writes an SQLite file + HNSW index files under
    chroma_dir/. On subsequent runs, opening the same path reconnects to the
    existing index — no re-embedding needed.

    Collection metadata {"hnsw:space": "cosine"}:
      all-MiniLM-L6-v2 produces L2-normalised vectors, so cosine similarity
      and dot-product give the same ranking. We request cosine explicitly so
      the distance values (0 = identical, 2 = opposite) are human-readable
      if we ever inspect them directly.

    What we store per document:
      id        → PMID string (unique, stable, and meaningful)
      document  → abstract text (shown to the user / passed to the LLM)
      embedding → 384-dim vector (used for ANN search)
      metadata  → title, year, journal, pmid (filtering + citation display)
    """
    print(f"\nConnecting to Chroma at {chroma_dir} ...")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # get_or_create_collection is idempotent — safe to call on repeated runs.
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        print(f"  Collection '{collection_name}' already has {existing:,} docs.")
        print("  Delete chroma_db/ and re-run to rebuild from scratch.")
        return

    # ── Prepare parallel lists (Chroma's add() takes separate lists) ──────────
    ids       = [r["pmid"]     for r in records]
    documents = [r["abstract"] for r in records]
    metadatas = [
        {
            "pmid":    r["pmid"],
            "title":   r["title"],
            "year":    r["year"],
            "journal": r["journal"],
        }
        for r in records
    ]

    # ── Add in one call — Chroma batches internally ───────────────────────────
    print(f"  Adding {len(ids):,} documents to collection '{collection_name}'...")
    collection.add(
        ids        = ids,
        documents  = documents,
        embeddings = vectors,
        metadatas  = metadatas,
    )
    print(f"  Done. Collection now has {collection.count():,} documents.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load ───────────────────────────────────────────────────────────────
    records = load_abstracts(DATA_FILE)

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    print(f"\nLoading embedding model '{EMBED_MODEL}' ...")
    print(
        "  (First run downloads ~80 MB to ~/.cache/huggingface/; "
        "subsequent runs load from cache.)"
    )
    model = SentenceTransformer(EMBED_MODEL)

    texts   = build_embed_texts(records)
    print(f"\nEmbedding {len(texts):,} documents in batches of {EMBED_BATCH}...")
    vectors = embed_in_batches(model, texts, EMBED_BATCH)

    # ── 3. Store ──────────────────────────────────────────────────────────────
    store_in_chroma(records, vectors, CHROMA_DIR, COLLECTION_NAME)

    # ── Summary ───────────────────────────────────────────────────────────────
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  Documents embedded   : {len(vectors):>6,}")
    print(f"  Vector dimensions    : {len(vectors[0]):>6,}")
    print(f"  Chroma collection    : {COLLECTION_NAME}")
    print(f"  Persisted to         : {CHROMA_DIR}")
    print(f"{bar}")
    print("\nPhase 2 complete. You can now run Phase 3 (retrieve.py).")


if __name__ == "__main__":
    main()
