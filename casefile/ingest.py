"""Store documents and their chunks in MongoDB.

Collections:
  sources : one record per original document (full text + metadata), _id = source_id
  chunks  : one record per chunk, _id = "<source_id>#<index>", with the source's metadata
            copied in so retrieval can filter on it directly.

Re-running is safe: unchanged documents are skipped (content hash), changed ones are replaced.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from pymongo.database import Database

from casefile.chunking import chunk_text
from casefile.parsing import Document

SOURCES = "sources"
CHUNKS = "chunks"


@dataclass
class IngestResult:
    source_id: str
    status: str        # "added", "updated" or "unchanged"
    chunk_count: int


def content_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_records(doc: Document) -> tuple[dict, list[dict]]:
    """Turn a parsed document into one source record and its chunk records."""
    now = datetime.now(timezone.utc)
    metadata = {
        "source_id": doc.source_id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "author": doc.author,
        "date": doc.date,
        "timestamp": doc.timestamp,
        "people_mentioned": doc.people_mentioned,
    }
    chunks = chunk_text(doc.body)
    source = {
        "_id": doc.source_id,
        **metadata,
        "time": doc.time,
        "body": doc.body,
        "raw": doc.raw,
        "content_hash": content_hash(doc.raw),
        "chunk_count": len(chunks),
        "ingested_at": now,
    }
    chunk_records = [
        {
            "_id": f"{doc.source_id}#{c.index:03d}",
            **metadata,
            "chunk_index": c.index,
            "text": c.text,
            "start_char": c.start,
            "end_char": c.end,
            "word_count": c.word_count,
            "ingested_at": now,
        }
        for c in chunks
    ]
    return source, chunk_records


def ensure_indexes(db: Database) -> None:
    db[CHUNKS].create_index("source_id")
    db[CHUNKS].create_index([("doc_type", 1), ("date", 1)])
    db[SOURCES].create_index("doc_type")


def ingest_document(db: Database, doc: Document) -> IngestResult:
    source, chunk_records = build_records(doc)
    existing = db[SOURCES].find_one({"_id": doc.source_id}, {"content_hash": 1})
    if existing and existing.get("content_hash") == source["content_hash"]:
        return IngestResult(doc.source_id, "unchanged", source["chunk_count"])

    db[CHUNKS].delete_many({"source_id": doc.source_id})
    if chunk_records:
        db[CHUNKS].insert_many(chunk_records)
    db[SOURCES].replace_one({"_id": doc.source_id}, source, upsert=True)
    return IngestResult(doc.source_id, "updated" if existing else "added", len(chunk_records))
