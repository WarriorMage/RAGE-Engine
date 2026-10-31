"""Check that the Case File dataset is complete and consistent.

Run from the repo root:
    python scripts/check_dataset.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "data" / "case_files"
QUESTIONS = ROOT / "eval" / "test_questions.json"

REQUIRED_FIELDS = ["source_id", "title", "doc_type", "author", "date", "time", "people_mentioned"]
DOC_TYPES = {
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
QUESTION_TYPES = {"multi_source", "conflict", "unanswerable", "exact_code", "filtered"}
EXPECTED_DOC_COUNT = 40

errors: list[str] = []


def parse_header(path: Path) -> dict[str, str] | None:
    """Return the fields of the '---' header block at the top of a file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    header = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return header
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()
    return None  # header never closed


def check_documents() -> set[str]:
    files = sorted(CASE_DIR.glob("*.md"))
    print(f"Documents found: {len(files)} (expected {EXPECTED_DOC_COUNT})")
    if len(files) != EXPECTED_DOC_COUNT:
        errors.append(f"Expected {EXPECTED_DOC_COUNT} documents, found {len(files)}")

    ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in files:
        header = parse_header(path)
        if header is None:
            errors.append(f"{path.name}: missing or unclosed '---' header")
            continue
        for field in REQUIRED_FIELDS:
            if not header.get(field):
                errors.append(f"{path.name}: header field '{field}' missing or empty")
        sid = header.get("source_id", "")
        if sid != path.stem:
            errors.append(f"{path.name}: source_id '{sid}' does not match filename")
        if sid in ids:
            errors.append(f"{path.name}: duplicate source_id '{sid}'")
        ids.add(sid)
        doc_type = header.get("doc_type", "")
        prefix = DOC_TYPES.get(doc_type)
        if prefix is None:
            errors.append(f"{path.name}: unknown doc_type '{doc_type}'")
        elif not sid.startswith(prefix + "-"):
            errors.append(f"{path.name}: doc_type '{doc_type}' should use prefix {prefix}-")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", header.get("date", "")):
            errors.append(f"{path.name}: date must look like YYYY-MM-DD")
        if not re.fullmatch(r"\d{2}:\d{2}", header.get("time", "")):
            errors.append(f"{path.name}: time must look like HH:MM")
        counts[doc_type] = counts.get(doc_type, 0) + 1

    print("Documents per type:")
    for doc_type, n in sorted(counts.items()):
        print(f"  {doc_type:22} {n}")
    return ids


def check_questions(doc_ids: set[str]) -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"Test questions found: {len(questions)}")
    type_counts: dict[str, int] = {}
    for q in questions:
        qid = q.get("id", "?")
        for field in ("id", "question", "type", "expected_source_ids", "correct_answer"):
            if field not in q:
                errors.append(f"{qid}: missing field '{field}'")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            errors.append(f"{qid}: unknown type '{qtype}'")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        expected = q.get("expected_source_ids", [])
        if qtype == "unanswerable" and expected:
            errors.append(f"{qid}: unanswerable questions must have no expected_source_ids")
        if qtype != "unanswerable" and not expected:
            errors.append(f"{qid}: expected_source_ids is empty")
        if qtype == "filtered" and not q.get("filter"):
            errors.append(f"{qid}: filtered question needs a 'filter'")
        for sid in expected:
            if sid not in doc_ids:
                errors.append(f"{qid}: expected source '{sid}' does not exist")

    print("Questions per type:")
    for qtype, n in sorted(type_counts.items()):
        print(f"  {qtype:22} {n}")


def main() -> int:
    doc_ids = check_documents()
    check_questions(doc_ids)
    print()
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
