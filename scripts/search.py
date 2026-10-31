"""Step 5: search the case files by meaning.

Run from the repo root:
    python scripts/search.py "who was near the back door at 2am?"
    python scripts/search.py "the guard's phone call" -k 8
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.db import get_db  # noqa: E402
from casefile.search import vector_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over the case files.")
    parser.add_argument("query", help="your question, in quotes")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    args = parser.parse_args()

    results = vector_search(get_db(), args.query, k=args.k)
    if not results:
        print("No results. Did you run scripts/embed.py and scripts/create_indexes.py?")
        return 1

    print(f'\nTop {len(results)} chunks for: "{args.query}"\n')
    for rank, r in enumerate(results, 1):
        date = r["date"].strftime("%Y-%m-%d") if hasattr(r.get("date"), "strftime") else r.get("date")
        print(f"{rank}. [{r['source_id']}] {r['title']}  ({r['doc_type']}, {date})  score={r['score']:.3f}")
        snippet = " ".join(r["text"].split())
        print(textwrap.indent(textwrap.fill(snippet[:300] + ("..." if len(snippet) > 300 else ""), 90), "     "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
