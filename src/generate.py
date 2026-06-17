"""
generate.py — Phase 4
Calls the Google Gemini API to produce a grounded answer.

The prompt template:
  - Provides the retrieved abstract texts as numbered context passages
  - Instructs Gemini to cite [PMID: XXXXXXXX] inline for every claim
  - Instructs Gemini NOT to invent facts outside the provided passages

Why cite PMIDs?
  Medical claims need provenance. By forcing inline citations we make it
  easy for users (and the eval harness) to verify every statement.
"""

# TODO (Phase 4): implement generate_answer(question, retrieved_docs) -> str
