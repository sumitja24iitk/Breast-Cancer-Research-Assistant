"""
ingest.py — Phase 1
Fetches breast-cancer-treatment abstracts from PubMed via NCBI Entrez
and saves them as data/abstracts.json.

Each record will be a dict:
    {
        "pmid":     "12345678",
        "title":    "...",
        "abstract": "...",
        "authors":  ["Last FM", ...],
        "year":     "2023"
    }
"""

# TODO (Phase 1): implement fetch_abstracts() and main()
