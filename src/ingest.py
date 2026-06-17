"""
ingest.py — Phase 1
Fetches breast-cancer-treatment abstracts from PubMed via NCBI Entrez
and saves them as data/abstracts.json.

Usage:
    python src/ingest.py

Output:
    data/abstracts.json — a JSON array where every element looks like:
    {
        "pmid":     "38123456",
        "title":    "Trastuzumab plus pertuzumab in HER2-positive breast cancer...",
        "abstract": "Background: ... Methods: ... Results: ...",
        "year":     "2023",
        "journal":  "Journal of Clinical Oncology"
    }
"""

import json
import os
import time
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (safe no-op if .env is absent)


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

# NCBI requires an email address so they can contact you if your queries
# cause server-side issues. It is NOT a secret — just a courtesy identifier.
# Replace the string below with your real email before running.
ENTREZ_EMAIL = "sumitjaincis@gmail.com"   # <-- REQUIRED: set this before running

# Optional: get a free NCBI API key at https://www.ncbi.nlm.nih.gov/account/
# It raises your rate limit from 3 to 10 requests/sec.
# Set NCBI_API_KEY=your_key in .env and the line below activates it.
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

TARGET_COUNT   = 2500   # max PMIDs to retrieve from esearch
BATCH_SIZE     = 200    # records per efetch call (200 is a safe, stable value)
REQUEST_DELAY  = 0.4    # seconds between HTTP requests (keeps us under 3 req/sec)

DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "abstracts.json"


# ────────────────────────────────────────────────────────────────────────────
# PUBMED SEARCH QUERY — explained in plain English
#
# PubMed uses a tagged query language where each term has a [Field] suffix.
# Here is what each clause does:
#
#  "breast neoplasms"[MeSH Terms]
#      MeSH = Medical Subject Headings, a controlled vocabulary maintained
#      by the US National Library of Medicine. Papers are manually tagged
#      with MeSH terms after publication. Using MeSH catches synonyms that
#      free-text search would miss: "mammary carcinoma", "breast tumor",
#      "breast cancer" all map to "breast neoplasms".
#
#  AND "therapy"[Subheading]
#      MeSH Subheadings narrow the main heading to a specific aspect.
#      Appending /therapy means breast neoplasms must be a major topic OF
#      the paper AND treated/discussed in a treatment context. Without this
#      we'd get thousands of epidemiology-only papers that happen to mention
#      breast cancer but offer no treatment content.
#
#  AND hasabstract
#      A filter field (no brackets needed). Excludes records that were
#      indexed by NCBI but lack abstract text — common for old citations,
#      conference abstracts, and editorial letters.
#
#  AND English[lang]
#      Language filter. Our embedding model (all-MiniLM-L6-v2) was trained
#      primarily on English text, so non-English abstracts would produce
#      poor-quality embeddings.
#
#  AND ("2015/01/01"[PDAT] : "2025/12/31"[PDAT])
#      PDAT = Publication DATe. Restricts to the last ~10 years, which keeps
#      the corpus focused on current treatment approaches (checkpoint
#      inhibitors, CDK4/6 inhibitors, ADCs) and avoids outdated regimens.
# ────────────────────────────────────────────────────────────────────────────

SEARCH_QUERY = (
    '"breast neoplasms"[MeSH Terms] '
    'AND "therapy"[Subheading] '
    'AND hasabstract '
    'AND English[lang] '
    'AND ("2015/01/01"[PDAT] : "2025/12/31"[PDAT])'
)


# ────────────────────────────────────────────────────────────────────────────
# STEP 1: SEARCH — get a list of matching PMIDs
# ────────────────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int) -> list[str]:
    """
    Run esearch against PubMed and return a list of PMID strings.

    esearch is the "search" E-utility. It does a full-text query and
    returns IDs only — no abstract text yet. Think of it like a database
    SELECT that returns primary keys; efetch (below) fetches the rows.

    retmax caps how many IDs come back in one call. The NCBI hard max is
    100,000, but we never need more than TARGET_COUNT.
    """
    print(f"Searching PubMed (retmax={max_results})...")
    handle  = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    results = Entrez.read(handle)
    handle.close()

    pmids          = results["IdList"]
    total_in_pubmed = int(results["Count"])   # total hits in PubMed (often >> retmax)

    print(f"  PubMed total matches : {total_in_pubmed:,}")
    print(f"  PMIDs we will fetch  : {len(pmids):,}")
    return pmids


# ────────────────────────────────────────────────────────────────────────────
# STEP 2: FETCH — pull full records in batches
# ────────────────────────────────────────────────────────────────────────────

def fetch_records_in_batches(pmids: list[str]) -> list[dict]:
    """
    Fetch full XML records from PubMed for all PMIDs, BATCH_SIZE at a time.

    Why batch?
      Sending 2,500 IDs in one HTTP request is unreliable (URL length limits,
      server timeouts). Batching at 200 keeps each request fast and allows
      graceful recovery if one batch fails.

    Why sleep between batches?
      NCBI allows 3 requests/second without an API key. Exceeding this causes
      HTTP 429 errors and can get your IP temporarily blocked. We sleep 0.4 s
      between each request, giving a safe ~2.5 req/sec.

    Returns a flat list of raw Biopython article dicts.
    """
    all_articles    = []
    total_batches   = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(pmids), BATCH_SIZE), start=1):
        batch_ids = pmids[start : start + BATCH_SIZE]
        print(
            f"  Batch {batch_num:>2}/{total_batches} "
            f"(records {start + 1}–{start + len(batch_ids)})...",
            end=" ",
            flush=True,
        )

        handle     = Entrez.efetch(
            db      = "pubmed",
            id      = ",".join(batch_ids),
            rettype = "xml",
            retmode = "xml",
        )
        batch_data = Entrez.read(handle)
        handle.close()

        articles = batch_data.get("PubmedArticle", [])
        all_articles.extend(articles)
        print(f"got {len(articles)} articles.")

        # Throttle — skip the sleep after the final batch.
        if batch_num < total_batches:
            time.sleep(REQUEST_DELAY)

    return all_articles


# ────────────────────────────────────────────────────────────────────────────
# STEP 3: PARSE — extract fields from each raw record
# ────────────────────────────────────────────────────────────────────────────

def parse_article(article: dict) -> dict | None:
    """
    Extract our five fields from a single PubmedArticle dict.

    Biopython's Entrez.read() returns a nested dict structure that mirrors
    the PubMed XML schema. The nesting is:
        article
          └─ MedlineCitation
               ├─ PMID
               └─ Article
                    ├─ ArticleTitle
                    ├─ Abstract.AbstractText   ← plain string OR structured list
                    └─ Journal
                         ├─ Title
                         └─ JournalIssue.PubDate

    Returns None if:
      - The abstract is empty/missing (we don't want empty documents in our index)
      - The record is malformed (rare; we log and skip rather than crash)
    """
    try:
        medline  = article["MedlineCitation"]
        article_ = medline["Article"]

        pmid    = str(medline["PMID"])
        title   = str(article_["ArticleTitle"])
        journal = str(article_["Journal"]["Title"])

        # ── Abstract ─────────────────────────────────────────────────────────
        # PubMed abstracts come in two shapes:
        #
        #   Shape A — Plain string (most common):
        #       AbstractText = "This study examined the effect of..."
        #
        #   Shape B — Structured abstract (clinical trials, systematic reviews):
        #       AbstractText = [
        #           StringElement("We investigated...", {"Label": "BACKGROUND"}),
        #           StringElement("Patients were...",  {"Label": "METHODS"}),
        #           ...
        #       ]
        #
        # We normalise both into a single readable string.
        abstract_node = article_.get("Abstract", {})
        raw           = abstract_node.get("AbstractText", "")

        if isinstance(raw, list):
            # Structured: join labelled sections with a blank line separator
            parts = []
            for section in raw:
                label = getattr(section, "attributes", {}).get("Label", "")
                text  = str(section).strip()
                parts.append(f"{label}: {text}" if label else text)
            abstract = "\n".join(parts).strip()
        else:
            abstract = str(raw).strip()

        if not abstract:
            return None   # Skip — empty abstracts are useless for retrieval

        # ── Publication year ──────────────────────────────────────────────────
        # PubDate can hold a structured date (Year/Month/Day) or a freeform
        # MedlineDate string like "2023 Jan-Feb". We take the first 4 characters,
        # which always gives us the year regardless of format.
        pub_date = article_["Journal"]["JournalIssue"]["PubDate"]
        year_raw = pub_date.get("Year") or pub_date.get("MedlineDate", "unknown")
        year     = str(year_raw)[:4]

        return {
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "year":     year,
            "journal":  journal,
        }

    except (KeyError, IndexError, AttributeError) as exc:
        # Malformed record — skip rather than crashing the whole run.
        print(f"\n    Warning: skipping malformed record ({exc})")
        return None


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Wire up Entrez credentials.
    Entrez.email   = ENTREZ_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY
        print("NCBI API key detected — using 10 req/sec rate limit.")

    DATA_DIR.mkdir(exist_ok=True)   # create data/ if it doesn't exist yet

    # ── Step 1: search ───────────────────────────────────────────────────────
    pmids = search_pubmed(SEARCH_QUERY, TARGET_COUNT)
    if not pmids:
        print("ERROR: esearch returned 0 PMIDs. Check your query and connection.")
        return

    # ── Step 2: fetch ────────────────────────────────────────────────────────
    print(f"\nFetching {len(pmids):,} records in batches of {BATCH_SIZE}...")
    raw_articles = fetch_records_in_batches(pmids)

    # ── Step 3: parse and filter ─────────────────────────────────────────────
    print("\nParsing and filtering records...")
    parsed    = []
    skipped   = 0
    for raw in raw_articles:
        record = parse_article(raw)
        if record:
            parsed.append(record)
        else:
            skipped += 1

    # ── Step 4: save ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  Fetched from PubMed  : {len(raw_articles):>6,} records")
    print(f"  Skipped (no abstract): {skipped:>6,} records")
    print(f"  Saved to JSON        : {len(parsed):>6,} records")
    print(f"  Output file          : {OUTPUT_FILE}")
    print(f"{bar}")


if __name__ == "__main__":
    main()
