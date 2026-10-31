"""One-call helpers used by the web app: add a document end to end, collection stats.

add_document() = parse -> store source + chunks (Step 3) -> embed new chunks (Step 4).
The Atlas Search and Vector Search indexes update themselves automatically, so a new
document becomes searchable within a few seconds.
"""

import re
from datetime import datetime, timezone

from pymongo import UpdateOne
from pymongo.database import Database

from casefile.embeddings import chunk_embedding_text, embed_documents, model_name
from casefile.ingest import CHUNKS, SOURCES, IngestResult, ensure_indexes, ingest_document
from casefile.parsing import parse_document

# Source id prefix for each document type (same scheme as data/case_files).
ID_PREFIXES = {
    "witness_statement": "STMT",
    "interview_transcript": "INTV",
    "cctv_log": "CCTV",
    "door_access_log": "ACCESS",
    "phone_record": "PHONE",
    "police_note": "NOTE",
    "forensic_report": "FOREN",
    "email": "EMAIL",
    "chat_export": "CHAT",
    "receipt": "RCPT",
}


def next_source_id(db: Database, doc_type: str) -> str:
    """Next free id for a type, e.g. STMT-011 after STMT-010."""
    prefix = ID_PREFIXES[doc_type]
    numbers = [int(m.group(1)) for sid in db[SOURCES].distinct("_id", {"_id": {"$regex": f"^{prefix}-"}})
               if (m := re.match(rf"^{prefix}-(\d+)$", sid))]
    return f"{prefix}-{(max(numbers) + 1 if numbers else 1):03d}"


def build_raw(source_id: str, title: str, doc_type: str, author: str, date: str, time: str,
              people: list[str], body: str) -> str:
    """Create a document in the case-file format (metadata header + body)."""
    header = [
        "---",
        f"source_id: {source_id}",
        f"title: {title.strip()}",
        f"doc_type: {doc_type}",
        f"author: {author.strip()}",
        f"date: {date}",
        f"time: {time}",
        f"people_mentioned: {', '.join(p.strip() for p in people if p.strip()) or 'none'}",
        "---",
    ]
    return "\n".join(header) + "\n" + body.strip() + "\n"


def embed_source(db: Database, source_id: str) -> int:
    """Embed the chunks of one source that do not have an embedding yet. Returns how many."""
    model = model_name()
    chunks = list(db[CHUNKS].find({"source_id": source_id, "embedding_model": {"$ne": model}}, {"embedding": 0}))
    if not chunks:
        return 0
    vectors = embed_documents([chunk_embedding_text(c) for c in chunks])
    now = datetime.now(timezone.utc)
    db[CHUNKS].bulk_write([
        UpdateOne({"_id": c["_id"]}, {"$set": {"embedding": v, "embedding_model": model, "embedded_at": now}})
        for c, v in zip(chunks, vectors)
    ])
    return len(chunks)


def add_document(db: Database, raw: str) -> tuple[IngestResult, int]:
    """Store, chunk and embed one document. Raises ValueError if the document is malformed."""
    doc = parse_document(raw)
    if doc.doc_type not in ID_PREFIXES:
        raise ValueError(f"unknown doc_type '{doc.doc_type}'. Valid: {', '.join(ID_PREFIXES)}")
    ensure_indexes(db)
    result = ingest_document(db, doc)
    embedded = embed_source(db, doc.source_id)
    return result, embedded


def collection_stats(db: Database) -> dict:
    by_type = {r["_id"]: r["n"] for r in db[SOURCES].aggregate([{"$group": {"_id": "$doc_type", "n": {"$sum": 1}}}])}
    return {
        "sources": db[SOURCES].count_documents({}),
        "chunks": db[CHUNKS].count_documents({}),
        "embedded": db[CHUNKS].count_documents({"embedding": {"$exists": True}}),
        "by_type": dict(sorted(by_type.items())),
    }
