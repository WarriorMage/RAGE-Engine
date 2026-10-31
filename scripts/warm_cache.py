"""Pre-answer the demo questions and demo reports, so they appear instantly in the app.

Run from the repo root, some time before presenting (takes a few minutes on free tiers):
    python scripts/warm_cache.py                 # answer anything not cached yet
    python scripts/warm_cache.py --refresh       # redo everything (e.g. once the main model works again)
    python scripts/warm_cache.py --clear         # empty the cache
    python scripts/warm_cache.py --no-reports    # questions only (reports are the slowest)
    python scripts/warm_cache.py --reports-only --refresh   # redo just the demo reports

The cache is in MongoDB, so warming it from any laptop also speeds up the deployed site.
Note: the hotel question is cached for the CURRENT collection. After you add the night porter's
statement in the demo, the collection changes and that question is answered fresh (as it should be).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.cache import cached_answer, cached_report, clear_cache  # noqa: E402
from casefile.db import get_db  # noqa: E402
from casefile.demo import DEMO_QUESTIONS, DEMO_REPORT_FOCUSES, IMPORT_DEMO_QUESTION  # noqa: E402
from casefile.llm import model_name  # noqa: E402
from casefile.search import Filters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="recompute even if cached")
    parser.add_argument("--clear", action="store_true", help="delete every cached answer and report, then stop")
    parser.add_argument("--no-reports", action="store_true", help="skip the (slow) demo reports")
    parser.add_argument("--reports-only", action="store_true", help="only the demo reports, no questions")
    parser.add_argument("--pause", type=int, default=20, help="seconds between AI calls (free tiers)")
    args = parser.parse_args()

    db = get_db()
    if args.clear:
        print(f"Removed {clear_cache(db)} cached item(s).")
        return 0

    jobs = [] if args.reports_only else [("answer", q) for q in [*DEMO_QUESTIONS, IMPORT_DEMO_QUESTION]]
    if not args.no_reports:
        jobs += [("report", f) for f in DEMO_REPORT_FOCUSES]

    weak = []
    print(f"Warming {len(jobs)} item(s); configured model: {model_name()}\n")
    for kind, text in jobs:
        label = text or "(whole case)"
        try:
            if kind == "answer":
                result, cached_at = cached_answer(db, text, Filters(), refresh=args.refresh)
                detail = f"status={result.status}"
            else:
                result, cached_at = cached_report(db, text, Filters(), refresh=args.refresh)
                detail = f"{len(result.timeline)} timeline entries"
                if result.removed_uncited:
                    detail += f", {result.removed_uncited} uncited removed"
        except RuntimeError as e:
            print(f"  ❌ {kind:6} {label}\n       {e}")
            continue
        if cached_at:
            print(f"  ⚡ {kind:6} already cached  {label}")
            continue
        print(f"  ✅ {kind:6} {detail:22} model={result.model:24} {label}")
        if "lite" in result.model:
            weak.append(label)
        time.sleep(args.pause)

    if weak:
        print(f"\n⚠️  {len(weak)} item(s) were answered by a 'lite' backup model (main model busy or rate-limited).")
        print("   When the main model works again, run:  python scripts/warm_cache.py --refresh")
    print("\nDone. These now load instantly in the app (local and deployed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
