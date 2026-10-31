"""Answer and report cache, stored in MongoDB (collection "answer_cache").

Why: free AI tiers are slow and rate-limited. A question asked before is answered instantly
from the cache, which keeps the live demo fast and saves AI requests.

Correctness:
  - The cache key includes the question (normalised), the filters and a fingerprint of the
    collection (number of sources + latest ingestion time). Adding or removing a document
    changes the fingerprint, so old answers are not reused after the evidence changes.
  - Evaluation scripts call answer_question() directly, so scores always measure fresh answers.
"""

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone

from pymongo.database import Database

from casefile.answer import Answer, Citation, Conflict, answer_question
from casefile.ingest import SOURCES
from casefile.report import Report, generate_report
from casefile.search import Filters

CACHE = "answer_cache"


def normalise(text: str) -> str:
    """'  Who used badge HB-9042?? ' -> 'who used badge hb-9042'"""
    return re.sub(r"\s+", " ", text).strip().rstrip("?!. ").lower()


def collection_fingerprint(db: Database) -> str:
    """Changes whenever a document is added, updated or removed."""
    latest = db[SOURCES].find_one({}, {"ingested_at": 1}, sort=[("ingested_at", -1)])
    stamp = latest["ingested_at"].isoformat() if latest and latest.get("ingested_at") else "none"
    return f"{db[SOURCES].count_documents({})}:{stamp}"


def _filters_dict(filters: Filters | None) -> dict:
    return asdict(filters) if filters else {}


def cache_key(db: Database, kind: str, text: str, filters: Filters | None) -> str:
    payload = json.dumps({"kind": kind, "text": normalise(text), "filters": _filters_dict(filters),
                          "collection": collection_fingerprint(db)}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------- (de)serialising results

def _answer_from_dict(d: dict) -> Answer:
    d = dict(d)
    d["citations"] = [Citation(**c) for c in d.get("citations", [])]
    d["conflicts"] = [Conflict(**c) for c in d.get("conflicts", [])]
    return Answer(**d)


def _report_from_dict(d: dict) -> Report:
    d = dict(d)
    d["citations"] = [Citation(**c) for c in d.get("citations", [])]
    return Report(**d)


def _load(db: Database, key: str):
    return db[CACHE].find_one({"_id": key})


def _save(db: Database, key: str, kind: str, text: str, filters: Filters | None, result) -> None:
    db[CACHE].replace_one({"_id": key}, {
        "_id": key, "kind": kind, "text": text, "filters": _filters_dict(filters),
        "result": asdict(result), "created_at": datetime.now(timezone.utc),
    }, upsert=True)


# ---------------------------------------------------------------- public API

def cached_answer(db: Database, question: str, filters: Filters | None = None,
                  refresh: bool = False) -> tuple[Answer, datetime | None]:
    """Return (answer, cached_at). cached_at is None when the answer was just computed."""
    key = cache_key(db, "answer", question, filters)
    if not refresh and (hit := _load(db, key)):
        return _answer_from_dict(hit["result"]), hit["created_at"]
    answer = answer_question(db, question, filters)
    _save(db, key, "answer", question, filters, answer)
    return answer, None


def cached_report(db: Database, focus: str = "", filters: Filters | None = None,
                  refresh: bool = False) -> tuple[Report, datetime | None]:
    key = cache_key(db, "report", focus, filters)
    if not refresh and (hit := _load(db, key)):
        return _report_from_dict(hit["result"]), hit["created_at"]
    report = generate_report(db, focus, filters)
    _save(db, key, "report", focus, filters, report)
    return report, None


def clear_cache(db: Database) -> int:
    return db[CACHE].delete_many({}).deleted_count
