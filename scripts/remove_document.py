"""Remove a document (its source record and all its chunks) from MongoDB.

Handy for resetting after a live demo of the "Add documents" tab.

Run from the repo root:
    python scripts/remove_document.py STMT-011
    python scripts/remove_document.py STMT-011 STMT-012 --yes   # no confirmation question
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from casefile.db import get_db  # noqa: E402
from casefile.ingest import CHUNKS, SOURCES  # noqa: E402

PROTECTED_DIR = Path(__file__).resolve().parent.parent / "data" / "case_files"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_ids", nargs="+", help="document ids to remove, e.g. STMT-011")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    db = get_db()
    for sid in args.source_ids:
        src = db[SOURCES].find_one({"_id": sid}, {"title": 1})
        if not src:
            print(f"  {sid}: not found, skipped")
            continue
        if (PROTECTED_DIR / f"{sid}.md").exists():
            print(f"  {sid}: is one of the original 40 case files; refusing to remove it")
            continue
        if not args.yes and input(f"Remove [{sid}] {src['title']}? (y/n) ").strip().lower() != "y":
            print(f"  {sid}: kept")
            continue
        chunks = db[CHUNKS].delete_many({"source_id": sid}).deleted_count
        db[SOURCES].delete_one({"_id": sid})
        print(f"  {sid}: removed (1 source, {chunks} chunks)")
    print(f"Collection now has {db[SOURCES].count_documents({})} sources, {db[CHUNKS].count_documents({})} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
