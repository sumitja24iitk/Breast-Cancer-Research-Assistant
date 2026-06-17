"""
eval.py — Phase 8
Evaluation harness measuring:

  Retrieval quality (does the right abstract show up in top-k?):
    - Recall@k:  fraction of questions where the gold PMID appears in top-k results
    - MRR:       Mean Reciprocal Rank — rewards finding the gold doc higher up

  Answer quality (does the answer only use what's in the retrieved docs?):
    - Faithfulness: heuristic check that cited PMIDs actually appear in retrieval set

Supports comparing dense-only vs. hybrid+rerank pipelines side by side.
"""

# TODO (Phase 8): implement recall_at_k(), mrr(), faithfulness(), run_eval()
