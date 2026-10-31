"""Read a case document: the '---' metadata header plus the body text."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = ("source_id", "title", "doc_type", "author", "date", "time", "people_mentioned")


@dataclass
class Document:
    source_id: str
    title: str
    doc_type: str
    author: str
    date: datetime          # date only (midnight), for date filters
    timestamp: datetime     # date + time
    time: str               # "HH:MM" as written in the header
    people_mentioned: list[str] = field(default_factory=list)
    body: str = ""          # the text after the header; chunk offsets point into this
    raw: str = ""           # the whole original file, kept as the source of truth


def parse_document(raw: str) -> Document:
    """Parse a document string. Raises ValueError with a clear message if it is malformed."""
    lines = raw.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("document must start with a '---' header line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("header is never closed with a second '---' line") from None

    header: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not header.get(f)]
    if missing:
        raise ValueError(f"header is missing: {', '.join(missing)}")

    try:
        timestamp = datetime.strptime(f"{header['date']} {header['time']}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("date must be YYYY-MM-DD and time must be HH:MM") from None

    people = header["people_mentioned"]
    people_list = [] if people.lower() == "none" else [p.strip() for p in people.split(",") if p.strip()]

    return Document(
        source_id=header["source_id"],
        title=header["title"],
        doc_type=header["doc_type"],
        author=header["author"],
        date=datetime(timestamp.year, timestamp.month, timestamp.day),
        timestamp=timestamp,
        time=header["time"],
        people_mentioned=people_list,
        body="\n".join(lines[end + 1:]).strip("\n"),
        raw=raw,
    )


def load_document(path: Path) -> Document:
    doc = parse_document(path.read_text(encoding="utf-8"))
    if doc.source_id != path.stem:
        raise ValueError(f"source_id '{doc.source_id}' does not match filename '{path.name}'")
    return doc
