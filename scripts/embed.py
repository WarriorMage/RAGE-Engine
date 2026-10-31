"""Step 4: add a Voyage embedding to every chunk in MongoDB that does not have one yet.

Run from the repo root (after scripts/ingest.py):
    python scripts/embed.py          # embed chunks that are missing an embedding
    python scripts/embed.py --all    # re-embed every chunk (e.g. after changing model)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import UpdateOne  # noqa: E402

from casefile.db import get_db  # noqa: E402
from casefile.embeddings import chunk_embedding_text, embed_documents, model_name  # noqa: E402
from casefile.ingest import CHUNKS  # noqa: E402

SAVE_EVERY = 12  # chunks embedded and saved per step, so progress survives rate limits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="re-embed every chunk, not just missing ones")
    args = parser.parse_args()

    db = get_db()
    model = model_name()
    query = {} if args.all else {"$or": [{"embedding": {"$exists": False}}, {"embedding_model": {"$ne": model}}]}
    chunks = list(db[CHUNKS].find(query, {"embedding": 0}))
    total = db[CHUNKS].count_documents({})

    if total == 0:
        print("❌ No chunks in the database. Run: python scripts/ingest.py")
        return 1
    if not chunks:
        print(f"Nothing to do ✅  all {total} chunks already have {model} embeddings.")
        return 0

    print(f"Embedding {len(chunks)} of {total} chunks with {model}...")
    dims = 0
    # Save batch by batch, so progress is kept even if a later batch fails.
    for i in range(0, len(chunks), SAVE_EVERY):
        batch = chunks[i:i + SAVE_EVERY]
        vectors = embed_documents([chunk_embedding_text(c) for c in batch])
        now = datetime.now(timezone.utc)
        db[CHUNKS].bulk_write([
            UpdateOne({"_id": c["_id"]}, {"$set": {"embedding": v, "embedding_model": model, "embedded_at": now}})
            for c, v in zip(batch, vectors)
        ])
        dims = len(vectors[0])
        print(f"  saved {min(i + SAVE_EVERY, len(chunks))}/{len(chunks)}")

    done = db[CHUNKS].count_documents({"embedding_model": model})
    print(f"Done ✅  {done}/{total} chunks have embeddings ({model}, {dims} numbers each)")
    return 0 if done == total else 1


if __name__ == "__main__":
    sys.exit(main())
