"""Search the case files: hybrid (default), vector-only or keyword-only, with metadata filters.

Run from the repo root:
    python scripts/search.py "who used badge HB-9042?"
    python scripts/search.py "who used badge HB-9042?" --mode vector     # meaning only
    python scripts/search.py "who used badge HB-9042?" --mode text       # keywords only
    python scripts/search.py "who came in during the night?" --type door_access_log
    python scripts/search.py "case status" --from 2026-10-10
    python scripts/search.py "the car park" --person "Femi Adeyemi"
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.db import get_db  # noqa: E402
from casefile.search import DOC_TYPES, Filters, search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the case files.",
        epilog=f"Document types: {', '.join(DOC_TYPES)}",
    )
    parser.add_argument("query", help="your question, in quotes")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    parser.add_argument("--mode", choices=["hybrid", "vector", "text"], default="hybrid",
                        help="hybrid = meaning + keywords (default)")
    parser.add_argument("--type", nargs="+", dest="doc_types", default=[], metavar="TYPE", help="only these document types")
    parser.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD", help="only documents dated on/after")
    parser.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD", help="only documents dated on/before")
    parser.add_argument("--person", nargs="+", dest="people", default=[], metavar="NAME", help="only documents mentioning")
    parser.add_argument("--author", nargs="+", dest="authors", default=[], metavar="NAME", help="only documents written by")
    args = parser.parse_args()

    try:
        filters = Filters(args.doc_types, args.date_from, args.date_to, args.people, args.authors)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    results = search(get_db(), args.query, k=args.k, filters=filters, mode=args.mode)

    print(f'\n[{args.mode}] top {len(results)} chunks for: "{args.query}"')
    if not filters.is_empty():
        print(f"Filter: {filters.to_vector_filter()}")
    print()
    if not results:
        print("No results. Try loosening the filters (names must match exactly, e.g. \"Rohan Mehta\").")
        return 1

    for rank, r in enumerate(results, 1):
        date = r["date"].strftime("%Y-%m-%d") if hasattr(r.get("date"), "strftime") else r.get("date")
        extra = ""
        if "ranks" in r:  # hybrid: show where each search ranked it
            extra = "  (" + ", ".join(f"{name} #{pos}" for name, pos in r["ranks"].items()) + ")"
        print(f"{rank}. [{r['source_id']}] {r['title']}  ({r['doc_type']}, {date})  score={r['score']:.4f}{extra}")
        snippet = " ".join(r["text"].split())
        print(textwrap.indent(textwrap.fill(snippet[:300] + ("..." if len(snippet) > 300 else ""), 90), "     "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
