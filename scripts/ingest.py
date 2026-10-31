"""Step 3: import case documents into MongoDB (sources + chunks).

Run from the repo root:
    python scripts/ingest.py                 # import data/case_files/*.md
    python scripts/ingest.py --dry-run       # only parse and chunk, no database
    python scripts/ingest.py --reset         # wipe sources + chunks first
    python scripts/ingest.py path/to/file.md # import specific files or folders
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from casefile.chunking import chunk_text  # noqa: E402
from casefile.parsing import load_document  # noqa: E402

DEFAULT_PATH = ROOT / "data" / "case_files"


def collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths or [str(DEFAULT_PATH)]):
        files.extend(sorted(p.glob("*.md")) if p.is_dir() else [p])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="files or folders (default: data/case_files)")
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only; do not touch the database")
    parser.add_argument("--reset", action="store_true", help="delete all sources and chunks before importing")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("❌ No .md files found.")
        return 1

    docs, failed = [], 0
    for f in files:
        try:
            docs.append(load_document(f))
        except (ValueError, OSError) as e:
            print(f"  ❌ {f.name}: {e}")
            failed += 1

    if args.dry_run:
        total = 0
        for d in docs:
            n = len(chunk_text(d.body))
            total += n
            print(f"  {d.source_id:12} {d.doc_type:22} {n:2} chunk(s)")
        print(f"\nDry run: {len(docs)} documents parsed, {total} chunks, {failed} failed. Database not touched.")
        return 1 if failed else 0

    from casefile.db import get_db
    from casefile.ingest import CHUNKS, SOURCES, ensure_indexes, ingest_document

    db = get_db()
    if args.reset:
        db[SOURCES].delete_many({})
        db[CHUNKS].delete_many({})
        print("Reset: removed all existing sources and chunks.")
    ensure_indexes(db)

    counts = {"added": 0, "updated": 0, "unchanged": 0}
    for d in docs:
        result = ingest_document(db, d)
        counts[result.status] += 1
        print(f"  {result.status:9} {result.source_id:12} {result.chunk_count:2} chunk(s)")

    print()
    print(f"Done ✅  added {counts['added']}, updated {counts['updated']}, "
          f"unchanged {counts['unchanged']}, failed {failed}")
    print(f"Database '{db.name}': {db[SOURCES].count_documents({})} sources, "
          f"{db[CHUNKS].count_documents({})} chunks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
