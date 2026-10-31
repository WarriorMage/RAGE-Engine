"""Step 8: retrieval evaluation. Score vector, text and hybrid search on eval/test_questions.json.

Run from the repo root (after scripts/create_indexes.py):
    python scripts/evaluate.py                 # all modes, top 5 sources
    python scripts/evaluate.py -k 3            # stricter: top 3 sources
    python scripts/evaluate.py --mode hybrid   # one mode only
    python scripts/evaluate.py --details       # per-question breakdown

Metrics (computed over the unique source documents retrieved, best first):
    Recall@k : share of the expected sources that appear in the top k        (did we find the evidence?)
    Hit@k    : share of questions with at least one expected source in top k (did we find anything right?)
    MRR      : average of 1 / rank of the first expected source              (how high was it?)
Unanswerable questions have no expected sources, so they are not scored here; their best
vector score is reported instead (Step 10 uses it to decide when to say "I don't know").

Each run is saved to eval/results/ so you can compare runs as you change things.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from casefile.db import get_db  # noqa: E402
from casefile.embeddings import embed_query, model_name  # noqa: E402
from casefile.search import Filters, search  # noqa: E402

QUESTIONS = ROOT / "eval" / "test_questions.json"
CACHE = ROOT / "eval" / "query_cache.json"
RESULTS_DIR = ROOT / "eval" / "results"
MODES = ["vector", "text", "hybrid"]
CHUNK_DEPTH = 20  # chunks fetched per question, then collapsed to unique sources


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def query_vector(question: str, cache: dict) -> list[float]:
    """Embed a question once per model and reuse it on later runs (saves Voyage requests)."""
    key = f"{model_name()}::{question}"
    if key not in cache:
        cache[key] = embed_query(question)
        CACHE.write_text(json.dumps(cache))
    return cache[key]


def to_filters(spec: dict | None) -> Filters:
    spec = spec or {}
    doc_type = spec.get("doc_type")
    return Filters(
        doc_types=[doc_type] if isinstance(doc_type, str) else list(doc_type or []),
        date_from=spec.get("date_from"),
        date_to=spec.get("date_to"),
        people=spec.get("people", []),
        authors=spec.get("authors", []),
    )


def unique_sources(results: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r["source_id"] not in seen:
            seen.append(r["source_id"])
    return seen


def score(expected: list[str], ranked: list[str], k: int) -> dict:
    top = ranked[:k]
    found = [s for s in expected if s in top]
    first = next((i for i, s in enumerate(ranked, 1) if s in expected), None)
    return {
        "recall": len(found) / len(expected),
        "hit": 1.0 if found else 0.0,
        "rr": 1.0 / first if first else 0.0,
        "found": found,
        "missed": [s for s in expected if s not in top],
        "top": top,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=5, help="score the top k unique sources (default 5)")
    parser.add_argument("--mode", choices=MODES, help="evaluate one mode only")
    parser.add_argument("--details", action="store_true", help="print per-question results")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    modes = [args.mode] if args.mode else MODES
    db = get_db()
    cache = load_cache()

    rows: dict[str, list[dict]] = {m: [] for m in modes}
    unanswerable: list[dict] = []

    print(f"Evaluating {len(questions)} questions, modes: {', '.join(modes)}, k={args.k}")
    for q in questions:
        qvec = query_vector(q["question"], cache) if ("vector" in modes or "hybrid" in modes) else None
        filters = to_filters(q.get("filter"))
        if not q["expected_source_ids"]:
            best = search(db, q["question"], k=1, mode="vector", query_vector=qvec or query_vector(q["question"], cache))
            unanswerable.append({"id": q["id"], "question": q["question"],
                                 "best_vector_score": round(best[0]["score"], 4) if best else None,
                                 "best_source": best[0]["source_id"] if best else None})
            continue
        for m in modes:
            results = search(db, q["question"], k=CHUNK_DEPTH, filters=filters, mode=m, query_vector=qvec)
            s = score(q["expected_source_ids"], unique_sources(results), args.k)
            rows[m].append({"id": q["id"], "type": q["type"], **s})
        print(f"  {q['id']} done")

    # ---- summary table
    types = sorted({r["type"] for r in rows[modes[0]]})
    print(f"\n=== Retrieval scoreboard (top {args.k} unique sources) ===\n")
    header = f"{'mode':8} | {'Recall@' + str(args.k):9} | {'Hit@' + str(args.k):6} | {'MRR':5} | " + " | ".join(f"{t[:12]:12}" for t in types)
    print(header)
    print("-" * len(header))
    summary = {}
    for m in modes:
        r = rows[m]
        per_type = {t: mean([x["recall"] for x in r if x["type"] == t]) for t in types}
        summary[m] = {"recall": mean([x["recall"] for x in r]), "hit": mean([x["hit"] for x in r]),
                      "mrr": mean([x["rr"] for x in r]), "recall_by_type": per_type}
        print(f"{m:8} | {summary[m]['recall']:9.0%} | {summary[m]['hit']:6.0%} | {summary[m]['mrr']:5.2f} | "
              + " | ".join(f"{per_type[t]:12.0%}" for t in types))
    print("\n(per-type columns show Recall for that question type)")

    if args.details:
        for m in modes:
            print(f"\n--- {m}: per question ---")
            for x in rows[m]:
                flag = "✅" if not x["missed"] else ("🟡" if x["found"] else "❌")
                miss = f"  missed {x['missed']}" if x["missed"] else ""
                print(f"{flag} {x['id']} {x['type']:12} recall {x['recall']:.0%}  top={x['top']}{miss}")

    if unanswerable:
        print("\nUnanswerable questions (no right source exists); best vector score found:")
        for u in unanswerable:
            print(f"  {u['id']}  best={u['best_vector_score']} ({u['best_source']})  {u['question']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"k": args.k, "model": model_name(), "summary": summary,
                               "questions": rows, "unanswerable": unanswerable}, indent=2))
    print(f"\nSaved to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
