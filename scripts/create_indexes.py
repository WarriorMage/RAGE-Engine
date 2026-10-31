"""Create the Atlas indexes on the chunks collection:
  - chunks_vector_index : Vector Search (meaning)
  - chunks_text_index   : Atlas Search full-text (exact words and codes)

Run from the repo root (after scripts/embed.py):
    python scripts/create_indexes.py

Safe to run again: it only creates or updates what is missing or changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo.errors import OperationFailure  # noqa: E402

from casefile.db import get_db  # noqa: E402
from casefile.search import TEXT_INDEX, VECTOR_INDEX, ensure_text_index, ensure_vector_index, wait_until_ready  # noqa: E402


def main() -> int:
    db = get_db()
    try:
        ensure_vector_index(db)
        ensure_text_index(db)
    except OperationFailure as e:
        print(f"❌ Atlas refused to create an index: {e.details.get('errmsg', e) if e.details else e}")
        print("   Check the database user has the 'Atlas admin' or 'readWriteAnyDatabase' role.")
        return 1

    statuses = wait_until_ready(db, [VECTOR_INDEX, TEXT_INDEX])
    if all(s == "READY" for s in statuses.values()):
        print(f"Done ✅  {VECTOR_INDEX} and {TEXT_INDEX} are READY. Try: python scripts/search.py \"HB-9042\"")
        return 0
    print(f"⏳ Not ready yet: {statuses}. Atlas is still building; run this script again in a minute.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
