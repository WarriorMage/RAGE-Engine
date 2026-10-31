"""Step 10: evaluate whole ANSWERS (not just retrieval) on eval/test_questions.json.

Checks, per question:
  status   : conflict questions -> "conflicting", unanswerable -> "insufficient",
             everything else -> any status except "insufficient"
  cited    : share of the expected source ids that the answer actually cites
  quotes   : share of quotes found word-for-word in their source

Run from the repo root:
    python scripts/evaluate_answers.py                               # all 20 (slow on free tiers)
    python scripts/evaluate_answers.py --types conflict unanswerable # just the honesty cases
    python scripts/evaluate_answers.py --ids Q07 Q11                 # specific questions

Free tiers allow only a few AI calls per minute, so there is a pause between questions
(--pause, default 20 seconds). Results are saved to eval/results/answers-*.json.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_cache, query_vector, to_filters  # noqa: E402

from casefile.answer import answer_question  # noqa: E402
from casefile.db import get_db  # noqa: E402
from casefile.llm import model_name  # noqa: E402

QUESTIONS = ROOT / "eval" / "test_questions.json"
RESULTS_DIR = ROOT / "eval" / "results"


def expected_status(qtype: str) -> set[str]:
    if qtype == "conflict":
        return {"conflicting"}
    if qtype == "unanswerable":
        return {"insufficient"}
    return {"answered", "partial", "conflicting"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--types", nargs="+", help="only these question types")
    parser.add_argument("--ids", nargs="+", help="only these question ids, e.g. Q07 Q11")
    parser.add_argument("--pause", type=int, default=20, help="seconds between questions (free tiers)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.types:
        questions = [q for q in questions if q["type"] in args.types]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    if not questions:
        print("No questions match.")
        return 1

    db = get_db()
    cache = load_cache()
    rows = []
    print(f"Evaluating {len(questions)} answers with {model_name()} (pause {args.pause}s between questions)\n")
    for i, q in enumerate(questions):
        if i:
            time.sleep(args.pause)
        try:
            a = answer_question(db, q["question"], to_filters(q.get("filter")), query_vector(q["question"], cache))
        except RuntimeError as e:
            print(f"  {q['id']} ❌ error: {e}")
            rows.append({"id": q["id"], "type": q["type"], "error": str(e)})
            continue
        expected = q["expected_source_ids"]
        cited = a.cited_sources
        cited_recall = (len([s for s in expected if s in cited]) / len(expected)) if expected else None
        quotes_ok = (sum(c.quote_verified for c in a.citations) / len(a.citations)) if a.citations else None
        status_ok = a.status in expected_status(q["type"])
        rows.append({
            "id": q["id"], "type": q["type"], "status": a.status, "status_ok": status_ok,
            "cited_recall": cited_recall, "quotes_verified": quotes_ok, "cited": cited,
            "missed": [s for s in expected if s not in cited], "warnings": a.warnings, "answer": a.answer,
        })
        recall_txt = f"cited {cited_recall:.0%}" if cited_recall is not None else "cited  n/a"
        quotes_txt = f"quotes {quotes_ok:.0%}" if quotes_ok is not None else "quotes n/a"
        print(f"  {q['id']} {'✅' if status_ok else '❌'} {q['type']:12} status={a.status:12} {recall_txt:11} {quotes_txt}")

    done = [r for r in rows if "error" not in r]
    if not done:
        return 1

    def avg(key):
        vals = [r[key] for r in done if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== Answer scoreboard ===")
    print(f"  Correct status        : {sum(r['status_ok'] for r in done)}/{len(done)}")
    for t in ("conflict", "unanswerable"):
        sub = [r for r in done if r["type"] == t]
        if sub:
            print(f"    {t:20}: {sum(r['status_ok'] for r in sub)}/{len(sub)}")
    print(f"  Expected sources cited: {avg('cited_recall'):.0%} (answerable questions)")
    print(f"  Quotes verified       : {avg('quotes_verified'):.0%}")
    errors = len(rows) - len(done)
    if errors:
        print(f"  Errors                : {errors} (rate limits? try a longer --pause)")

    wrong = [r for r in done if not r["status_ok"]]
    if wrong:
        print("\nWrong status:")
        for r in wrong:
            print(f"  {r['id']} ({r['type']}) got '{r['status']}', expected {sorted(expected_status(r['type']))}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"model": model_name(), "rows": rows}, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
