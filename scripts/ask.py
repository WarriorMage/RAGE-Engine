"""Step 9: ask the case files a question and get an answer with verified citations.

Run from the repo root:
    python scripts/ask.py "Did the security guard leave his desk?"
    python scripts/ask.py "What did the cameras show after 01:40?" --type cctv_log
    python scripts/ask.py "Who used badge HB-9042?" --show-evidence
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.answer import answer_question  # noqa: E402
from casefile.db import get_db  # noqa: E402
from casefile.search import DOC_TYPES, Filters  # noqa: E402


def _ids(source_ids: list[str]) -> str:
    return "".join(f"[{s}]" for s in source_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the case files a question.",
                                     epilog=f"Document types: {', '.join(DOC_TYPES)}")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--type", nargs="+", dest="doc_types", default=[], metavar="TYPE", help="only these document types")
    parser.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD", help="only documents dated on/after")
    parser.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD", help="only documents dated on/before")
    parser.add_argument("--person", nargs="+", dest="people", default=[], metavar="NAME", help="only documents mentioning")
    parser.add_argument("--author", nargs="+", dest="authors", default=[], metavar="NAME", help="only documents written by")
    parser.add_argument("--show-evidence", action="store_true", help="also print every evidence chunk the AI saw")
    args = parser.parse_args()

    try:
        filters = Filters(args.doc_types, args.date_from, args.date_to, args.people, args.authors)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    print("Searching the case files and asking the AI...")
    try:
        result = answer_question(get_db(), args.question, filters)
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1

    print(f"\nQ: {result.question}")
    print(f"\n{result.status_label}\n")
    print(textwrap.fill(result.answer, 95, replace_whitespace=False))

    if result.conflicts:
        print("\nContradictions")
        print("--------------")
        for i, c in enumerate(result.conflicts, 1):
            print(f"  {i}. {c.topic}")
            print(textwrap.indent(textwrap.fill(f"Claim: {c.claim} {_ids(c.claim_sources)}", 88), "     "))
            print(textwrap.indent(textwrap.fill(f"But:   {c.contradiction} {_ids(c.contradiction_sources)}", 88), "     "))
            print(textwrap.indent(textwrap.fill(f"Assessment: {c.assessment}", 88), "     "))

    if result.missing:
        print("\nNot in the case files")
        print("---------------------")
        for m in result.missing:
            print(f"  - {m}")

    for w in result.warnings:
        print(f"\n⚠️  Check: {w}")

    if result.follow_up_searches:
        print(f"\n(follow-up searches: {', '.join(result.follow_up_searches)})")

    print("\nSources")
    print("-------")
    if not result.citations:
        print("  (no citations)")
    for c in result.citations:
        mark = "✅" if c.quote_verified else "⚠️ quote not found in source"
        print(f"  [{c.source_id}] {c.title}  {mark}")
        print(textwrap.indent(textwrap.fill(f'"{c.quote}"', 88), "      "))

    if args.show_evidence:
        print(f"\nEvidence shown to the AI ({len(result.evidence)} chunks)")
        print("------------------------------------")
        for e in result.evidence:
            snippet = " ".join(e["text"].split())
            print(f"  {e['label']:>3} [{e['source_id']}] {snippet[:160]}...")

    print(f"\nmodel: {result.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
