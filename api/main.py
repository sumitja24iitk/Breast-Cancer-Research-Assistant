"""
api/main.py — Phase 5
FastAPI application exposing a single endpoint:

    POST /query
    Body: {"question": "...", "mode": "dense"|"hybrid", "top_k": 5}
    Response: {"answer": "...", "sources": [...]}

Run with:
    uvicorn api.main:app --reload
"""

# TODO (Phase 5): implement FastAPI app, QueryRequest model, /query endpoint
