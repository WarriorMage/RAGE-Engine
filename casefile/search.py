"""Semantic search over chunks with MongoDB Atlas Vector Search."""

import time
from datetime import datetime, timedelta

from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from casefile.embeddings import EMBEDDING_DIMS, embed_query
from casefile.ingest import CHUNKS

VECTOR_INDEX = "chunks_vector_index"

# Filter fields are declared now so later steps can filter by type, date, source or person
# without rebuilding the index.
VECTOR_INDEX_DEFINITION = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "doc_type"},
        {"type": "filter", "path": "date"},
        {"type": "filter", "path": "source_id"},
        {"type": "filter", "path": "people_mentioned"},
        {"type": "filter", "path": "author"},
    ]
}

DOC_TYPES = [
    "witness_statement", "interview_transcript", "cctv_log", "door_access_log", "phone_record",
    "police_note", "forensic_report", "email", "chat_export", "receipt",
]


def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date '{value}' must look like YYYY-MM-DD") from None


def build_filter(
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    people: list[str] | None = None,
    authors: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> dict | None:
    """Build a $vectorSearch pre-filter from simple options. Returns None when nothing is set.

    date_from / date_to are inclusive days ("YYYY-MM-DD") matched against the document date.
    people matches documents whose people_mentioned list contains any of the names.
    """
    conditions: list[dict] = []
    if doc_types:
        unknown = sorted(set(doc_types) - set(DOC_TYPES))
        if unknown:
            raise ValueError(f"unknown doc type(s): {', '.join(unknown)}. Valid: {', '.join(DOC_TYPES)}")
        conditions.append({"doc_type": {"$in": list(doc_types)}})
    if date_from:
        conditions.append({"date": {"$gte": _parse_day(date_from)}})
    if date_to:
        conditions.append({"date": {"$lt": _parse_day(date_to) + timedelta(days=1)}})
    if people:
        conditions.append({"people_mentioned": {"$in": list(people)}})
    if authors:
        conditions.append({"author": {"$in": list(authors)}})
    if source_ids:
        conditions.append({"source_id": {"$in": list(source_ids)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


# Fields returned for each result (never the big embedding array).
RESULT_FIELDS = {
    "_id": 1, "source_id": 1, "title": 1, "doc_type": 1, "author": 1, "date": 1,
    "chunk_index": 1, "text": 1, "start_char": 1, "end_char": 1,
}


def _field_set(definition: dict | None) -> set:
    """Comparable summary of an index definition (Atlas may reorder or add defaults)."""
    return {(f.get("type"), f.get("path"), f.get("numDimensions"), f.get("similarity"))
            for f in (definition or {}).get("fields", [])}


def _find_index(db: Database, name: str) -> dict | None:
    return next(iter(db[CHUNKS].list_search_indexes(name)), None)


def ensure_vector_index(db: Database, wait: bool = True, timeout_s: int = 300) -> str:
    """Create the vector index if missing (or update it if the definition changed). Returns its status."""
    existing = _find_index(db, VECTOR_INDEX)
    if existing is None:
        db[CHUNKS].create_search_index(
            SearchIndexModel(definition=VECTOR_INDEX_DEFINITION, name=VECTOR_INDEX, type="vectorSearch")
        )
        print(f"Created vector index '{VECTOR_INDEX}'.")
    elif _field_set(existing.get("latestDefinition")) != _field_set(VECTOR_INDEX_DEFINITION):
        db[CHUNKS].update_search_index(VECTOR_INDEX, VECTOR_INDEX_DEFINITION)
        print(f"Updated vector index '{VECTOR_INDEX}' to the new definition.")
    else:
        print(f"Vector index '{VECTOR_INDEX}' already exists.")

    if not wait:
        return (_find_index(db, VECTOR_INDEX) or {}).get("status", "UNKNOWN")

    start = time.monotonic()
    while True:
        info = _find_index(db, VECTOR_INDEX) or {}
        status = info.get("status", "PENDING")
        if info.get("queryable") and status == "READY":
            return status
        if status == "FAILED":
            raise RuntimeError(f"Atlas reported the index FAILED: {info}")
        if time.monotonic() - start > timeout_s:
            return status
        print(f"  waiting for index to be ready (status: {status})...")
        time.sleep(5)


def vector_search(
    db: Database,
    query: str,
    k: int = 5,
    filters: dict | None = None,
    num_candidates: int | None = None,
) -> list[dict]:
    """Return the k chunks closest in meaning to the query, best first, each with a 'score'.

    filters: an optional MQL filter on indexed fields, e.g. {"doc_type": "cctv_log"}.
    """
    stage = {
        "index": VECTOR_INDEX,
        "path": "embedding",
        "queryVector": embed_query(query),
        "numCandidates": num_candidates or max(100, k * 20),
        "limit": k,
    }
    if filters:
        stage["filter"] = filters
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {**RESULT_FIELDS, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(db[CHUNKS].aggregate(pipeline))
