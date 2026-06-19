"""
eval.py — Phase 8
Evaluation harness measuring retrieval quality and answer faithfulness.

CONCEPTS — read before the code
──────────────────────────────────────────────────────────────────────────────

Why measuring retrieval quality matters
────────────────────────────────────────
The retriever is the first component in the pipeline.  If it misses the
relevant abstract, the LLM never sees the evidence it needs and either
hallucinates an answer or admits it can't help.  A great generator cannot
rescue a bad retriever.  So before trusting any answer, we must know: "how
reliably does the retriever surface the right documents?"

Hit@k vs Recall@k
─────────────────
  Hit@k (also called success@k):
    Binary — 1.0 if ANY gold PMID appears in the top-k results, else 0.0.
    Tells you: "what fraction of questions have ≥1 useful abstract in top-k?"
    Lenient: you get full credit even if you only found one of three gold docs.

  Recall@k:
    Fractional — |gold PMIDs found in top-k| / |gold PMIDs total|.
    More demanding when a question has multiple gold docs.  If there are 4
    gold PMIDs and only 2 appear in top-5, recall@5 = 0.5 but hit@5 = 1.0.
    Use recall@k when you need the LLM to see ALL the evidence for a complete
    answer, not just one supporting abstract.

MRR — Mean Reciprocal Rank
───────────────────────────
  For each question, compute the Reciprocal Rank (RR):
    RR = 1 / rank of the first gold PMID found in the retrieved list.
    Rank 1 → 1.0,  rank 5 → 0.2,  not found → 0.0.
  MRR = average RR across all questions.
  MRR penalises retrievers that bury the relevant document at rank 10 even if
  it technically appears within the retrieved set.  A model limited to top-5
  context needs MRR > 0.2 (i.e., gold doc found by rank 5 on average).

Faithfulness
─────────────
  Fraction of [PMID: XXXXXXXX] citations in the generated answer that were
  actually in the retrieved context passed to the model.  A score < 1.0 means
  the model cited papers it didn't see in context — hallucinated references,
  possibly real PMIDs recalled from training data but applied incorrectly.
  Requires generating answers (--faithfulness flag), which calls Gemini.

Why side-by-side mode comparison matters
─────────────────────────────────────────
  Running both "dense" and "hybrid" on the same labeled questions produces a
  controlled experiment.  Same questions, same gold labels, only the retrieval
  strategy changes.  Any metric difference is attributable to the retrieval
  method, not the questions or the evaluation set.

Usage
─────
    python src/eval.py                   # both modes, k = [5, 10]
    python src/eval.py --mode dense      # dense only
    python src/eval.py --mode hybrid     # hybrid only
    python src/eval.py --k 5 10 20       # custom k values
    python src/eval.py --faithfulness    # also call Gemini (~30s per question)

Prerequisites
─────────────
    eval/questions.json must contain at least one question with a non-empty
    "relevant_pmids" list.  Run  python src/label_eval.py  first.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

# Make sibling src/ modules importable whether run directly (python src/eval.py)
# or imported from another module.
sys.path.insert(0, str(Path(__file__).parent))

from retrieve import retrieve

_ROOT = Path(__file__).parent.parent
_QUESTIONS_FILE = _ROOT / "eval" / "questions.json"


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS — each takes pre-computed lists, no I/O
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int) -> float:
    """
    1.0 if at least one gold PMID is in the top-k retrieved results, else 0.0.

    The most lenient retrieval metric.  Averaged across questions, it answers:
    "what fraction of the time did the retriever include at least one relevant
    abstract in its top-k list?"
    """
    top_k_set = set(retrieved_pmids[:k])
    return 1.0 if any(p in top_k_set for p in gold_pmids) else 0.0


def recall_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int) -> float:
    """
    Fraction of gold PMIDs that appear in the top-k retrieved results.

    More demanding than hit_at_k when a question has multiple gold documents.
    Example: 4 gold PMIDs, 2 found in top-5 → recall@5 = 0.5 but hit@5 = 1.0.
    """
    if not gold_pmids:
        return 0.0
    top_k_set = set(retrieved_pmids[:k])
    return sum(1 for p in gold_pmids if p in top_k_set) / len(gold_pmids)


def reciprocal_rank(retrieved_pmids: list[str], gold_pmids: list[str]) -> float:
    """
    1 / rank of the first gold PMID in the retrieved list; 0.0 if not found.

    RR = 1.0 means a gold doc was ranked #1.
    RR = 0.2 means the first gold doc appeared at rank 5.
    MRR averages this across questions — it rewards putting a relevant doc
    near the top, not just somewhere in the page.
    """
    gold_set = set(gold_pmids)
    for rank, pmid in enumerate(retrieved_pmids, start=1):
        if pmid in gold_set:
            return 1.0 / rank
    return 0.0


def faithfulness_score(answer_text: str, retrieved_pmids: list[str]) -> float:
    """
    Fraction of [PMID: XXXXXXXX] citations in the answer that were in the
    retrieved context actually passed to the model.

    A score of 1.0 = fully grounded.  A score below 1.0 means the model cited
    PMIDs it never saw in context — hallucinated references.  These may be real
    papers from the model's training data but are being cited incorrectly.

    Returns 1.0 when the answer contains no PMID citations (vacuously faithful;
    flag these answers for manual inspection — a citation-free answer may itself
    be a problem).
    """
    cited = set(re.findall(r'\[PMID:\s*(\d+)\]', answer_text))
    if not cited:
        return 1.0  # no citations → can't measure; treated as vacuously faithful
    retrieved_set = set(retrieved_pmids)
    return sum(1 for p in cited if p in retrieved_set) / len(cited)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    questions: list[dict],
    modes: list[str],
    k_values: list[int],
    include_faithfulness: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """
    Iterate over every labeled question and retrieval mode, collecting
    per-question metric scores.

    Parameters
    ----------
    questions : list[dict]
        Labeled questions from questions.json.  Each must have a non-empty
        "relevant_pmids" list.
    modes : list[str]
        Retrieval modes to compare, e.g. ["dense", "hybrid"].
    k_values : list[int]
        Which k thresholds to compute hit@k and recall@k for.
    include_faithfulness : bool
        If True, also calls Gemini for each question/mode pair and computes
        citation faithfulness.  ~30 s per question per mode; uses API quota.

    Returns
    -------
    dict — results[mode][metric_name] = list[float], one float per question.
    """
    # Retrieve enough docs to cover all k values AND give MRR headroom to
    # find gold docs that might fall below the largest k threshold.
    retrieval_k = max(max(k_values), 25)

    # Delay-import rag only if faithfulness is requested, because importing
    # rag.py also imports generate.py which validates GOOGLE_API_KEY at module
    # load time — no need to require the key if only measuring retrieval.
    if include_faithfulness:
        from rag import answer as rag_answer  # noqa: PLC0415

    metric_keys: list[str] = (
        [f"hit@{k}" for k in k_values]
        + [f"recall@{k}" for k in k_values]
        + ["mrr"]
        + (["faithfulness"] if include_faithfulness else [])
    )
    results: dict[str, dict[str, list[float]]] = {
        mode: {key: [] for key in metric_keys}
        for mode in modes
    }

    n = len(questions)
    for q_idx, q in enumerate(questions, start=1):
        question   = q["question"]
        gold_pmids = q["relevant_pmids"]

        if not gold_pmids:
            print(f"\n[{q_idx}/{n}] SKIP (no labels): {question[:65]}")
            continue

        print(f"\n[{q_idx}/{n}] {question[:72]}")

        for mode in modes:
            t0   = time.perf_counter()
            hits = retrieve(question, k=retrieval_k, mode=mode)
            elapsed = time.perf_counter() - t0

            retrieved_pmids = [h["pmid"] for h in hits]

            rr = reciprocal_rank(retrieved_pmids, gold_pmids)
            results[mode]["mrr"].append(rr)

            for k in k_values:
                results[mode][f"hit@{k}"].append(
                    hit_at_k(retrieved_pmids, gold_pmids, k)
                )
                results[mode][f"recall@{k}"].append(
                    recall_at_k(retrieved_pmids, gold_pmids, k)
                )

            faith_str = ""
            if include_faithfulness:
                rag_result = rag_answer(question, k=max(k_values), mode=mode)
                faith = faithfulness_score(rag_result["answer"], retrieved_pmids)
                results[mode]["faithfulness"].append(faith)
                faith_str = f"  faith={faith:.3f}"

            hit_at_first_k = results[mode][f"hit@{k_values[0]}"][-1]
            print(
                f"    {mode:6s}  RR={rr:.3f}"
                f"  hit@{k_values[0]}={'✓' if hit_at_first_k else '✗'}"
                f"  ({elapsed:.1f}s){faith_str}"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    results: dict[str, dict[str, list[float]]],
    k_values: list[int],
    n_questions: int,
    include_faithfulness: bool,
) -> None:
    """
    Print a formatted side-by-side comparison table.

    Each cell shows mean ± standard deviation.  SD matters: if hybrid is 0.05
    better than dense but SD is 0.20, the difference is within noise for a
    26-question eval set.  You'd need more questions or a paired t-test to
    claim significance.
    """
    modes = list(results.keys())
    col_w = 18  # characters per mode column

    def _cell(vals: list[float]) -> str:
        if not vals:
            return "—".rjust(col_w)
        mean = statistics.mean(vals)
        sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.3f} (±{sd:.2f})".rjust(col_w)

    def _row(label: str, key: str) -> str:
        return f"{label:<16}" + "".join(_cell(results[m].get(key, [])) for m in modes)

    header = f"{'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in modes)
    bar    = "─" * len(header)
    dbar   = "═" * len(header)

    print(f"\n{dbar}")
    print(f"Retrieval Evaluation — {n_questions} labeled questions  (k = {k_values})")
    print(dbar)
    print(header)
    print(bar)

    for k in k_values:
        print(_row(f"Hit@{k}", f"hit@{k}"))
    print(bar)
    for k in k_values:
        print(_row(f"Recall@{k}", f"recall@{k}"))
    print(bar)
    print(_row("MRR", "mrr"))

    if include_faithfulness:
        print(bar)
        print(_row("Faithfulness", "faithfulness"))

    print(dbar)
    print()
    print("Hit@k      — fraction of questions with ≥1 gold PMID in top-k (binary)")
    print("Recall@k   — avg fraction of ALL gold PMIDs found in top-k")
    print("MRR        — Mean Reciprocal Rank (1/rank of first gold PMID found)")
    if include_faithfulness:
        print("Faithfulness — fraction of cited PMIDs that were in retrieved context")
    print()
    print("Scores shown as mean (± 1 standard deviation across questions).")
    if len(modes) > 1:
        print("Dense scores use cosine similarity; hybrid uses reranker logits.")
        print("The two score types are not comparable — only the averaged metrics above are.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 evaluation harness — retrieval quality and faithfulness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "both"],
        default="both",
        help="Retrieval mode(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10],
        metavar="K",
        help="k values for Hit@k and Recall@k (default: 5 10).",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help=(
            "Also generate answers with Gemini and measure citation faithfulness. "
            "Slow (~30 s per question per mode) and uses Gemini API quota."
        ),
    )
    args = parser.parse_args()

    modes  = ["dense", "hybrid"] if args.mode == "both" else [args.mode]
    k_vals = sorted(set(args.k))

    # ── Load questions ────────────────────────────────────────────────────────
    all_questions: list[dict] = json.loads(
        _QUESTIONS_FILE.read_text(encoding="utf-8")
    )
    labeled = [q for q in all_questions if q["relevant_pmids"]]

    if not labeled:
        print(
            "\nNo labeled questions found in eval/questions.json.\n"
            "Run  python src/label_eval.py  first to add ground-truth PMIDs.\n"
        )
        sys.exit(1)

    skipped = len(all_questions) - len(labeled)
    if skipped:
        print(
            f"Note: {skipped}/{len(all_questions)} questions have no labels "
            "and will be skipped.  Run label_eval.py to complete them."
        )

    print(f"\nEvaluating {len(labeled)} questions — modes={modes}  k={k_vals}")
    if args.faithfulness:
        print("Faithfulness ON — will call Gemini for each question (slow).")
    print("Loading retrieval models …")

    # ── Run ───────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = run_eval(labeled, modes, k_vals, include_faithfulness=args.faithfulness)
    total   = time.perf_counter() - t_start

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(results, k_vals, len(labeled), args.faithfulness)
    print(f"Total wall time: {total:.1f}s\n")


if __name__ == "__main__":
    main()
