"""
label_eval.py — Interactive ground-truth labeler for eval/questions.json

For each unlabeled question (relevant_pmids == []) this script:
  1. Retrieves a wide candidate pool: dense top-25 + BM25 top-25, merged and
     SHUFFLED so the display order carries no retriever-rank signal.
  2. Prints each candidate with its index, PMID, title, and a short snippet.
  3. Prompts you to type the index numbers of genuinely relevant abstracts.
  4. Writes your choices back to eval/questions.json immediately after each
     question so progress survives if you quit early.

───────────────────────────────────────────────────────────────────────────────
WHY YOU MUST JUDGE BY CONTENT, NOT BY RETRIEVER RANK
───────────────────────────────────────────────────────────────────────────────
Both dense and BM25 rank documents by their own scoring heuristic — cosine
similarity and keyword frequency, respectively.  Neither score is a reliable
proxy for "does this abstract actually answer the question?"

If you label only the top-ranked candidates as relevant and skip lower-ranked
ones without reading them, you build an evaluation set that only reflects what
your retriever already does well — every recall failure stays hidden.  The eval
numbers will look great because the "gold standard" was derived from the same
system you're measuring.  This is circular and useless.

An unbiased gold standard requires human judgment that is INDEPENDENT of the
system under test.  Shuffling the display order breaks rank-anchoring (people
unconsciously trust whatever appears first) and forces you to evaluate each
abstract on its own merits.  Same reason clinical trials use blinded reviewers:
the rater must not know which treatment arm the patient was assigned to.
───────────────────────────────────────────────────────────────────────────────

Usage:
    python src/label_eval.py
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

# Both files live in src/, so a direct import works when the script is run as
#     python src/label_eval.py
# from the project root (Python prepends the script's directory to sys.path).
#
# Importing _bm25_retrieve by its private name is intentional — this is an
# internal development tool and we need both retrievers without triggering the
# cross-encoder reranker (which would re-introduce a ranking bias).
from retrieve import _bm25_retrieve, dense_retrieve  # noqa: PLC2701

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_QUESTIONS_FILE = _ROOT / "eval" / "questions.json"

# ── Configuration ──────────────────────────────────────────────────────────────
CANDIDATE_K = 25    # top-k from each retriever; merged pool ≤ 2 × this
SNIPPET_CHARS = 220 # abstract characters shown per candidate


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_pool(question: str) -> list[dict]:
    """
    Gather a wide set of candidates from BOTH retrievers, then shuffle.

    Dense and BM25 each contribute up to CANDIDATE_K hits.  After merging
    (duplicate PMIDs kept once), we shuffle so the labeler sees a random order
    that is not correlated with either retriever's confidence score.
    """
    dense_hits = dense_retrieve(question, k=CANDIDATE_K)
    bm25_hits  = _bm25_retrieve(question, k=CANDIDATE_K)

    # Merge — dense goes first, so if a PMID appears in both we keep the
    # dense dict (title/text are identical; only the score differs).
    seen: dict[str, dict] = {}
    for hit in dense_hits + bm25_hits:
        if hit["pmid"] not in seen:
            seen[hit["pmid"]] = hit

    pool = list(seen.values())
    random.shuffle(pool)  # break retriever-rank ordering
    return pool


def _print_candidates(pool: list[dict]) -> None:
    """Print the numbered candidate list with PMID, title, and snippet."""
    for i, c in enumerate(pool, start=1):
        snippet = textwrap.shorten(c["text"], width=SNIPPET_CHARS, placeholder="…")
        print(f"\n  [{i:2d}]  PMID {c['pmid']}")
        print(f"        {c['title']}")
        print(f"        {snippet}")


def _save(questions: list[dict]) -> None:
    """Atomically overwrite questions.json with updated data."""
    _QUESTIONS_FILE.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_indices(raw: str, pool_size: int) -> tuple[list[int], list[str]]:
    """
    Parse a space-separated string of 1-based indices.

    Returns (valid_indices, bad_tokens) — bad_tokens is non-empty when the
    user typed something out of range or non-numeric.
    """
    valid, bad = [], []
    for token in raw.split():
        try:
            n = int(token)
            if 1 <= n <= pool_size:
                valid.append(n)
            else:
                bad.append(token)
        except ValueError:
            bad.append(token)
    return valid, bad


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LABELING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    questions: list[dict] = json.loads(_QUESTIONS_FILE.read_text(encoding="utf-8"))
    unlabeled = [q for q in questions if not q["relevant_pmids"]]

    if not unlabeled:
        print("All questions already have labels — nothing to do.")
        return

    print(f"\n{len(unlabeled)} unlabeled question(s) out of {len(questions)} total.\n")
    print("─" * 72)
    print("INSTRUCTIONS")
    print("  Read each abstract title and snippet.  Select every abstract that")
    print("  genuinely helps answer the question — ignore where it appeared in")
    print("  the list, since the order is randomised on purpose.")
    print()
    print("  Numbers (e.g. '2 5 11') — mark those abstracts as relevant")
    print("  's'                      — skip this question for now")
    print("  'q'                      — save progress and quit")
    print("─" * 72)

    # Note: importing retrieve.py runs its module-level init (loads embedding
    # model + builds BM25 index).  Expect ~5-10 s of loading messages above.

    for q_num, q in enumerate(unlabeled, start=1):
        question = q["question"]
        print(f"\n{'═' * 72}")
        print(f"  [{q_num}/{len(unlabeled)}]  {question}")
        print("─" * 72)

        print("  Retrieving candidates (dense + BM25, then shuffled) …")
        pool = _candidate_pool(question)
        print(f"  {len(pool)} unique abstracts in pool")

        _print_candidates(pool)

        # ── Input loop for this question ────────────────────────────────────
        while True:
            print(f"\n  Enter numbers, 's' to skip, or 'q' to quit: ", end="")
            raw = input().strip().lower()

            # ── Quit ─────────────────────────────────────────────────────────
            if raw == "q":
                _save(questions)
                print("  Progress saved. Goodbye.")
                return

            # ── Skip ─────────────────────────────────────────────────────────
            if raw in ("s", ""):
                print("  Skipped — will appear again on next run.")
                break

            # ── Parse index selection ─────────────────────────────────────────
            indices, bad = _parse_indices(raw, len(pool))

            if bad:
                print(f"  Ignored (out of range or not a number): {', '.join(bad)}")

            if not indices:
                print("  No valid numbers entered.  Try again or type 's' to skip.")
                continue

            # ── Confirm selection ─────────────────────────────────────────────
            chosen_pmids = [pool[i - 1]["pmid"] for i in sorted(set(indices))]

            print("\n  You selected:")
            for pmid in chosen_pmids:
                title = next(c["title"] for c in pool if c["pmid"] == pmid)
                print(f"    PMID {pmid} — {title[:65]}")

            print("  Confirm? (y to save / n to re-enter): ", end="")
            if input().strip().lower() != "y":
                # Re-print the candidates so the user can see the list again
                print()
                _print_candidates(pool)
                continue

            # ── Persist ───────────────────────────────────────────────────────
            for master_q in questions:
                if master_q["question"] == question:
                    master_q["relevant_pmids"] = chosen_pmids
                    break

            _save(questions)
            print(f"  Saved {len(chosen_pmids)} PMID(s) for this question.")
            break  # move to next question

    print(f"\n{'═' * 72}")
    print("All done. eval/questions.json is fully labeled.")


if __name__ == "__main__":
    main()
