"""Step 5: create the Atlas Vector Search index on the chunks collection.

Run from the repo root (after scripts/embed.py):
    python scripts/create_indexes.py

Safe to run again: it only creates or updates what is missing or changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo.errors import OperationFailure  # noqa: E402

from casefile.db import get_db  # noqa: E402
from casefile.search import VECTOR_INDEX, ensure_vector_index  # noqa: E402


def main() -> int:
    db = get_db()
    try:
        status = ensure_vector_index(db)
    except OperationFailure as e:
        print(f"❌ Atlas refused to create the index: {e.details.get('errmsg', e) if e.details else e}")
        print("   Check that the database user has the 'Atlas admin' or 'readWriteAnyDatabase' role.")
        return 1

    if status == "READY":
        print(f"Done ✅  vector index '{VECTOR_INDEX}' is READY. Try: python scripts/search.py \"your question\"")
        return 0
    print(f"⏳ Index status is {status}. Atlas is still building it; run this script again in a minute.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
