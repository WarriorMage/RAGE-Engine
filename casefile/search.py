"""Retrieval over chunks with MongoDB Atlas.

Three ways to search, all sharing the same metadata filters:
  vector_search : Atlas Vector Search ($vectorSearch) - matches meaning
  text_search   : Atlas Search ($search, BM25)        - matches exact words and codes (HB-9042, DOOR-E2)
  hybrid_search : both, merged with Reciprocal Rank Fusion (RRF)
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from casefile.embeddings import EMBEDDING_DIMS, embed_query
from casefile.ingest import CHUNKS

VECTOR_INDEX = "chunks_vector_index"
TEXT_INDEX = "chunks_text_index"

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

# Full-text (BM25) index. 'text' and 'title' are searchable words; the rest are filter fields.
TEXT_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "text": {"type": "string", "analyzer": "lucene.standard"},
            "title": {"type": "string", "analyzer": "lucene.standard"},
            "doc_type": {"type": "token"},
            "source_id": {"type": "token"},
            "author": {"type": "token"},
            "people_mentioned": {"type": "token"},
            "date": {"type": "date"},
        },
    }
}

DOC_TYPES = [
    "witness_statement", "interview_transcript", "cctv_log", "door_access_log", "phone_record",
    "police_note", "forensic_report", "email", "chat_export", "receipt",
]

# Fields returned for each result (never the big embedding array).
RESULT_FIELDS = {
    "_id": 1, "source_id": 1, "title": 1, "doc_type": 1, "author": 1, "date": 1,
    "chunk_index": 1, "text": 1, "start_char": 1, "end_char": 1,
}

RRF_K = 60  # standard Reciprocal Rank Fusion constant

# Codes such as HB-9042, 231-D-55120, DOOR-E2, SOD-48213, LT-7731: letters/digits joined by dashes
# and containing at least one digit. Meaning-based search is weak on these, so hybrid search
# gives the keyword side more weight when a query contains one.
CODE_PATTERN = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
CODE_TEXT_WEIGHT = 2.0


def has_code(query: str) -> bool:
    return bool(CODE_PATTERN.search(query))


# ---------------------------------------------------------------- filters

def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date '{value}' must look like YYYY-MM-DD") from None


@dataclass
class Filters:
    """Metadata filters, usable by both vector and text search."""
    doc_types: list[str] = field(default_factory=list)
    date_from: str | None = None   # inclusive, YYYY-MM-DD
    date_to: str | None = None     # inclusive, YYYY-MM-DD
    people: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        unknown = sorted(set(self.doc_types) - set(DOC_TYPES))
        if unknown:
            raise ValueError(f"unknown doc type(s): {', '.join(unknown)}. Valid: {', '.join(DOC_TYPES)}")
        self._from = _parse_day(self.date_from) if self.date_from else None
        self._to = _parse_day(self.date_to) + timedelta(days=1) if self.date_to else None

    def is_empty(self) -> bool:
        return not (self.doc_types or self._from or self._to or self.people or self.authors or self.source_ids)

    def to_vector_filter(self) -> dict | None:
        """MQL-style filter for $vectorSearch."""
        c: list[dict] = []
        if self.doc_types:
            c.append({"doc_type": {"$in": self.doc_types}})
        if self._from:
            c.append({"date": {"$gte": self._from}})
        if self._to:
            c.append({"date": {"$lt": self._to}})
        if self.people:
            c.append({"people_mentioned": {"$in": self.people}})
        if self.authors:
            c.append({"author": {"$in": self.authors}})
        if self.source_ids:
            c.append({"source_id": {"$in": self.source_ids}})
        if not c:
            return None
        return c[0] if len(c) == 1 else {"$and": c}

    def to_search_clauses(self) -> list[dict]:
        """Atlas Search 'filter' clauses for $search compound queries."""
        c: list[dict] = []
        if self.doc_types:
            c.append({"in": {"path": "doc_type", "value": self.doc_types}})
        if self._from or self._to:
            rng: dict = {"path": "date"}
            if self._from:
                rng["gte"] = self._from
            if self._to:
                rng["lt"] = self._to
            c.append({"range": rng})
        if self.people:
            c.append({"in": {"path": "people_mentioned", "value": self.people}})
        if self.authors:
            c.append({"in": {"path": "author", "value": self.authors}})
        if self.source_ids:
            c.append({"in": {"path": "source_id", "value": self.source_ids}})
        return c


def build_filter(doc_types=None, date_from=None, date_to=None, people=None, authors=None, source_ids=None):
    """Backwards-compatible helper: returns the $vectorSearch filter for these options."""
    return Filters(doc_types or [], date_from, date_to, people or [], authors or [], source_ids or []).to_vector_filter()


# ---------------------------------------------------------------- indexes

def _find_index(db: Database, name: str) -> dict | None:
    return next(iter(db[CHUNKS].list_search_indexes(name)), None)


def _vector_signature(definition: dict | None) -> set:
    return {(f.get("type"), f.get("path"), f.get("numDimensions"), f.get("similarity"))
            for f in (definition or {}).get("fields", [])}


def _text_signature(definition: dict | None) -> set:
    fields = ((definition or {}).get("mappings") or {}).get("fields") or {}
    return {(name, spec.get("type")) for name, spec in fields.items()}


def _ensure_index(db: Database, name: str, definition: dict, index_type: str, signature) -> None:
    existing = _find_index(db, name)
    if existing is None:
        db[CHUNKS].create_search_index(SearchIndexModel(definition=definition, name=name, type=index_type))
        print(f"Created index '{name}'.")
    elif signature(existing.get("latestDefinition")) != signature(definition):
        db[CHUNKS].update_search_index(name, definition)
        print(f"Updated index '{name}' to the new definition.")
    else:
        print(f"Index '{name}' already exists.")


def wait_until_ready(db: Database, names: list[str], timeout_s: int = 300) -> dict[str, str]:
    """Poll Atlas until every named index is READY (or timeout). Returns name -> status."""
    start = time.monotonic()
    while True:
        statuses = {}
        for n in names:
            info = _find_index(db, n) or {}
            statuses[n] = "READY" if info.get("queryable") and info.get("status") == "READY" else info.get("status", "PENDING")
        if any(s == "FAILED" for s in statuses.values()):
            raise RuntimeError(f"Atlas reported an index FAILED: {statuses}")
        if all(s == "READY" for s in statuses.values()) or time.monotonic() - start > timeout_s:
            return statuses
        print("  waiting for indexes: " + ", ".join(f"{n}={s}" for n, s in statuses.items()))
        time.sleep(5)


def ensure_vector_index(db: Database) -> None:
    _ensure_index(db, VECTOR_INDEX, VECTOR_INDEX_DEFINITION, "vectorSearch", _vector_signature)


def ensure_text_index(db: Database) -> None:
    _ensure_index(db, TEXT_INDEX, TEXT_INDEX_DEFINITION, "search", _text_signature)


# ---------------------------------------------------------------- search

def _as_filters(filters) -> Filters:
    if filters is None:
        return Filters()
    if isinstance(filters, Filters):
        return filters
    raise TypeError("filters must be a Filters object (see casefile.search.Filters)")


def vector_search(db: Database, query: str, k: int = 5, filters: Filters | None = None,
                  num_candidates: int | None = None, query_vector: list[float] | None = None) -> list[dict]:
    """The k chunks closest in meaning to the query, best first, each with 'score'."""
    f = _as_filters(filters)
    stage = {
        "index": VECTOR_INDEX,
        "path": "embedding",
        "queryVector": query_vector or embed_query(query),
        "numCandidates": num_candidates or max(100, k * 20),
        "limit": k,
    }
    vf = f.to_vector_filter()
    if vf:
        stage["filter"] = vf
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {**RESULT_FIELDS, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(db[CHUNKS].aggregate(pipeline))


def text_search(db: Database, query: str, k: int = 5, filters: Filters | None = None) -> list[dict]:
    """The k chunks that best match the query's words (BM25), best first, each with 'score'.

    Exact phrases (e.g. a code like "HB-9042" whose parts appear side by side) get a boost.
    """
    f = _as_filters(filters)
    compound: dict = {
        "should": [
            {"text": {"query": query, "path": ["text", "title"]}},
            {"phrase": {"query": query, "path": "text", "slop": 0, "score": {"boost": {"value": 3}}}},
        ],
        "minimumShouldMatch": 1,
    }
    clauses = f.to_search_clauses()
    if clauses:
        compound["filter"] = clauses
    pipeline = [
        {"$search": {"index": TEXT_INDEX, "compound": compound}},
        {"$limit": k},
        {"$project": {**RESULT_FIELDS, "score": {"$meta": "searchScore"}}},
    ]
    return list(db[CHUNKS].aggregate(pipeline))


def rrf_fuse(ranked_lists: dict[str, list[dict]], k: int, weights: dict[str, float] | None = None) -> list[dict]:
    """Merge ranked result lists with Reciprocal Rank Fusion.

    Each result gets sum(weight / (RRF_K + rank)) over the lists it appears in.
    The output keeps, per result, its rank in each list (e.g. {"vector": 1, "text": 4}).
    """
    weights = weights or {}
    merged: dict[str, dict] = {}
    for name, results in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, r in enumerate(results, 1):
            item = merged.setdefault(r["_id"], {**{x: y for x, y in r.items() if x != "score"},
                                                "score": 0.0, "ranks": {}})
            item["score"] += w / (RRF_K + rank)
            item["ranks"][name] = rank
    return sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:k]


def hybrid_search(db: Database, query: str, k: int = 5, filters: Filters | None = None,
                  depth: int = 20, vector_weight: float = 1.0, text_weight: float | None = None,
                  query_vector: list[float] | None = None) -> list[dict]:
    """Run vector and text search (each `depth` deep) and fuse them with RRF.

    text_weight defaults to 1.0, or CODE_TEXT_WEIGHT when the query contains a code like HB-9042.
    """
    f = _as_filters(filters)
    if text_weight is None:
        text_weight = CODE_TEXT_WEIGHT if has_code(query) else 1.0
    return rrf_fuse(
        {"vector": vector_search(db, query, depth, f, query_vector=query_vector),
         "text": text_search(db, query, depth, f)},
        k,
        {"vector": vector_weight, "text": text_weight},
    )


def search(db: Database, query: str, k: int = 5, filters: Filters | None = None, mode: str = "hybrid",
           query_vector: list[float] | None = None) -> list[dict]:
    if mode == "vector":
        return vector_search(db, query, k, filters, query_vector=query_vector)
    if mode == "text":
        return text_search(db, query, k, filters)
    if mode == "hybrid":
        return hybrid_search(db, query, k, filters, query_vector=query_vector)
    raise ValueError("mode must be 'vector', 'text' or 'hybrid'")
