"""
rag.py — Phase 4
Thin orchestration layer that wires together retrieve → generate.

This module is the single entry point for both the FastAPI backend (api/main.py)
and ad-hoc command-line testing. Keeping it separate from retrieve.py and
generate.py makes each component independently testable.
"""

# TODO (Phase 4): implement answer(question, mode="dense", top_k=5) -> dict
#   Returns: {"answer": str, "sources": list[dict]}
